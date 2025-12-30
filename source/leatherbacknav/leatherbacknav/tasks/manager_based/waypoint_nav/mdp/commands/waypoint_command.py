# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint command generator for navigation tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import WaypointCommandCfg


class WaypointCommand(CommandTerm):
    """Command generator that generates a sequence of 2D waypoints for navigation.
    
    This command generator samples multiple waypoints within a configurable radius
    around the robot's starting position. The waypoints are sampled sequentially,
    with each new waypoint being generated relative to the previous one.
    
    The command provides:
    - Current waypoint position (x, y, z) relative to the robot (in robot's base frame)
    - Optionally, the next few waypoints for look-ahead planning
    
    For wheeled robots, z is kept at ground level.
    
    The waypoint index advances when the robot reaches within a threshold distance
    of the current waypoint (using 2D distance).
    
    Dynamic Spacing:
        Supports multiple spacing scenarios for diverse training environments:
        - Close waypoints: Tests precise maneuvering
        - Far waypoints: Tests sustained locomotion
        - Mixed: Natural combination when per_waypoint_spacing=True
    
    Visualization uses colored spheres:
    - Red sphere: current target waypoint
    - Green spheres: future waypoints
    """

    cfg: WaypointCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: WaypointCommandCfg, env: ManagerBasedEnv):
        """Initialize the waypoint command generator."""
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.total_waypoints_tracked = 1 + cfg.num_lookahead_waypoints

        # Buffers: waypoints in world frame (num_envs, num_waypoints, 3)
        self.waypoints_w = torch.zeros(self.num_envs, cfg.num_waypoints, 3, device=self.device)
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.all_waypoints_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.num_waypoints_per_env = torch.full(
            (self.num_envs,), cfg.num_waypoints, dtype=torch.long, device=self.device
        )
        # Command output: waypoints in robot's base frame (num_envs, total_waypoints_tracked * 3)
        self.waypoints_b = torch.zeros(
            self.num_envs, self.total_waypoints_tracked * 3, device=self.device
        )
        self._markers_pos = torch.zeros(self.num_envs, cfg.num_waypoints, 3, device=self.device)

        self._setup_spacing_scenarios()
        self.metrics["waypoints_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["distance_to_waypoint"] = torch.zeros(self.num_envs, device=self.device)
    
    def _setup_spacing_scenarios(self):
        """Set up spacing scenario probability distribution and ranges."""
        if self.cfg.spacing_scenarios:
            self._spacing_ranges = torch.tensor(
                [s.spacing_range for s in self.cfg.spacing_scenarios],
                device=self.device, dtype=torch.float32
            )
            weights = torch.tensor(
                [s.weight for s in self.cfg.spacing_scenarios],
                device=self.device, dtype=torch.float32
            )
            self._spacing_probs = weights / weights.sum()
            self._use_scenarios = True
        else:
            self._use_scenarios = False

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "WaypointCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tNumber of waypoints: {self.cfg.num_waypoints}\n"
        if self._use_scenarios:
            msg += f"\tSpacing scenarios: {len(self.cfg.spacing_scenarios)} configured\n"
            for i, s in enumerate(self.cfg.spacing_scenarios):
                msg += f"\t  [{i}] range={s.spacing_range}, weight={s.weight}\n"
            msg += f"\tPer-waypoint spacing: {self.cfg.per_waypoint_spacing}\n"
        else:
            msg += f"\tWaypoint spacing: {self.cfg.waypoint_spacing} (legacy mode)\n"
        msg += f"\tReach threshold: {self.cfg.waypoint_reach_threshold}\n"
        msg += f"\tLookahead waypoints: {self.cfg.num_lookahead_waypoints}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The waypoint commands in the robot's base frame."""
        return self.waypoints_b

    def set_num_waypoints(self, env_idx: int, num_waypoints: int) -> None:
        """Set the number of active waypoints for a specific environment."""
        self.num_waypoints_per_env[env_idx] = max(1, min(num_waypoints, self.cfg.num_waypoints))

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for logging."""
        robot_pos_w = self.robot.data.root_pos_w
        current_wp_indices = self.current_waypoint_idx.clamp(max=self.cfg.num_waypoints - 1)
        current_waypoints = self.waypoints_w[
            torch.arange(self.num_envs, device=self.device), current_wp_indices
        ]
        self.metrics["distance_to_waypoint"] = torch.norm(
            current_waypoints[:, :2] - robot_pos_w[:, :2], dim=1
        )
        self.metrics["waypoints_reached"] = self.current_waypoint_idx.float()

    def _sample_spacing_distance(self, num_samples: int, per_waypoint: bool = True) -> torch.Tensor:
        """Sample distances from spacing scenarios."""
        if not self._use_scenarios:
            return torch.empty(num_samples, device=self.device).uniform_(*self.cfg.waypoint_spacing)
        
        if per_waypoint:
            scenario_indices = torch.multinomial(self._spacing_probs, num_samples, replacement=True)
        else:
            scenario_indices = torch.full(
                (num_samples,), torch.multinomial(self._spacing_probs, 1).item(),
                device=self.device, dtype=torch.long
            )
        
        selected_ranges = self._spacing_ranges[scenario_indices]
        uniform_samples = torch.rand(num_samples, device=self.device)
        return selected_ranges[:, 0] + (selected_ranges[:, 1] - selected_ranges[:, 0]) * uniform_samples

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new waypoints for the specified environments."""
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return

        robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2].clone()
        robot_pos_z = self.robot.data.root_pos_w[env_ids, 2].clone()
        robot_heading = self.robot.data.heading_w[env_ids].clone()

        self.current_waypoint_idx[env_ids] = 0
        self.all_waypoints_reached[env_ids] = False
        self.num_waypoints_per_env[env_ids] = self.cfg.num_waypoints
        
        if self._use_scenarios and not self.cfg.per_waypoint_spacing:
            env_scenario_indices = torch.multinomial(
                self._spacing_probs, num_envs_to_reset, replacement=True
            )

        for wp_idx in range(self.cfg.num_waypoints):
            if wp_idx == 0:
                distance = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.initial_waypoint_distance
                )
                angle = robot_heading + torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.ranges.angle_range
                )
                base_pos = robot_pos_xy
            else:
                if self._use_scenarios:
                    if self.cfg.per_waypoint_spacing:
                        distance = self._sample_spacing_distance(num_envs_to_reset, per_waypoint=True)
                    else:
                        selected_ranges = self._spacing_ranges[env_scenario_indices]
                        uniform_samples = torch.rand(num_envs_to_reset, device=self.device)
                        distance = selected_ranges[:, 0] + (selected_ranges[:, 1] - selected_ranges[:, 0]) * uniform_samples
                else:
                    distance = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                        *self.cfg.waypoint_spacing
                    )
                
                if wp_idx == 1:
                    prev_heading = robot_heading
                else:
                    direction = self.waypoints_w[env_ids, wp_idx - 1] - self.waypoints_w[env_ids, wp_idx - 2]
                    prev_heading = torch.atan2(direction[:, 1], direction[:, 0])
                
                angle = prev_heading + torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.ranges.angle_range
                )
                base_pos = self.waypoints_w[env_ids, wp_idx - 1, :2]

            wp_xy = base_pos + distance.unsqueeze(1) * torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
            self.waypoints_w[env_ids, wp_idx, :2] = wp_xy
            self.waypoints_w[env_ids, wp_idx, 2] = 0.1

        self._markers_pos[env_ids] = self.waypoints_w[env_ids].clone()
        if self.cfg.debug_vis and hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.visualize(translations=self._markers_pos.view(-1, 3))

    def _update_command(self):
        """Update the command based on the current robot state."""
        robot_pos_w = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w

        current_wp_indices = self.current_waypoint_idx.clamp(max=self.cfg.num_waypoints - 1)
        current_waypoints = self.waypoints_w[
            torch.arange(self.num_envs, device=self.device), current_wp_indices
        ]
        distance_to_current = torch.norm(current_waypoints[:, :2] - robot_pos_w[:, :2], dim=1)
        
        reached_mask = distance_to_current < self.cfg.waypoint_reach_threshold
        can_advance = self.current_waypoint_idx < self.num_waypoints_per_env - 1
        self.current_waypoint_idx[reached_mask & can_advance] += 1
        
        current_wp_indices_after = torch.min(self.current_waypoint_idx, self.num_waypoints_per_env - 1)
        current_waypoints_after = self.waypoints_w[
            torch.arange(self.num_envs, device=self.device), current_wp_indices_after
        ]
        distance_to_current_after = torch.norm(current_waypoints_after[:, :2] - robot_pos_w[:, :2], dim=1)
        at_last_waypoint = self.current_waypoint_idx >= self.num_waypoints_per_env - 1
        self.all_waypoints_reached = at_last_waypoint & (distance_to_current_after < self.cfg.waypoint_reach_threshold)

        yaw_q = yaw_quat(robot_quat)
        for i in range(self.total_waypoints_tracked):
            wp_idx = (self.current_waypoint_idx + i).clamp(max=self.cfg.num_waypoints - 1)
            waypoint_w = self.waypoints_w[torch.arange(self.num_envs, device=self.device), wp_idx]
            rel_pos_b = quat_apply_inverse(yaw_q, waypoint_w - robot_pos_w)
            self.waypoints_b[:, i * 3:(i + 1) * 3] = rel_pos_b

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization for waypoints."""
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer = VisualizationMarkers(self.cfg.waypoint_visualizer_cfg)
            self.waypoint_visualizer.set_visibility(True)
        elif hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        if not self.robot.is_initialized:
            return

        self._markers_pos[:, :, :3] = self.waypoints_w[:, :, :3]
        one_hot_encoded = torch.nn.functional.one_hot(
            self.current_waypoint_idx.long(), num_classes=self.cfg.num_waypoints
        )
        marker_indices = (1 - one_hot_encoded).view(-1).tolist()
        self.waypoint_visualizer.visualize(
            translations=self._markers_pos.view(-1, 3), marker_indices=marker_indices
        )

