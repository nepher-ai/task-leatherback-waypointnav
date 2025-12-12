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
        """Initialize the waypoint command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Number of waypoints to track (current + lookahead)
        self.total_waypoints_tracked = 1 + cfg.num_lookahead_waypoints

        # Create buffers to store waypoints
        # Waypoints in world frame: (num_envs, num_waypoints, 3) for x, y, z
        self.waypoints_w = torch.zeros(
            self.num_envs, cfg.num_waypoints, 3, device=self.device
        )
        # Current waypoint index for each environment
        self.current_waypoint_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Flag indicating if all waypoints have been reached
        self.all_waypoints_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        
        # Command output: waypoints in robot's base frame
        # Shape: (num_envs, total_waypoints_tracked * 3)
        # Contains [current_wp_x, current_wp_y, current_wp_z, next_wp_x, next_wp_y, next_wp_z, ...]
        self.waypoints_b = torch.zeros(
            self.num_envs, self.total_waypoints_tracked * 3, device=self.device
        )

        # Visualization buffers
        # Store 3D marker positions: (num_envs, num_waypoints, 3)
        self._markers_pos = torch.zeros(
            self.num_envs, cfg.num_waypoints, 3, device=self.device
        )

        # Build spacing scenario sampling weights
        self._setup_spacing_scenarios()

        # Metrics for logging
        self.metrics["waypoints_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["distance_to_waypoint"] = torch.zeros(self.num_envs, device=self.device)
    
    def _setup_spacing_scenarios(self):
        """Set up spacing scenario probability distribution and ranges."""
        if self.cfg.spacing_scenarios:
            # Extract ranges and weights from scenarios
            self._spacing_ranges = torch.tensor(
                [s.spacing_range for s in self.cfg.spacing_scenarios],
                device=self.device, dtype=torch.float32
            )  # Shape: (num_scenarios, 2)
            
            weights = torch.tensor(
                [s.weight for s in self.cfg.spacing_scenarios],
                device=self.device, dtype=torch.float32
            )
            # Normalize weights to probabilities
            self._spacing_probs = weights / weights.sum()
            self._use_scenarios = True
        else:
            # Fallback to legacy single spacing range
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
        """The waypoint commands in the robot's base frame.
        
        Shape is (num_envs, total_waypoints_tracked * 3) containing
        [current_wp_x, current_wp_y, current_wp_z, next_wp_x, next_wp_y, next_wp_z, ...] 
        in the robot's local frame.
        """
        return self.waypoints_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for logging."""
        # Compute 2D distance to current waypoint
        robot_pos_w = self.robot.data.root_pos_w
        current_wp_indices = self.current_waypoint_idx.clamp(max=self.cfg.num_waypoints - 1)
        current_waypoints = self.waypoints_w[
            torch.arange(self.num_envs, device=self.device), current_wp_indices
        ]
        # Use 2D distance for ground vehicles
        self.metrics["distance_to_waypoint"] = torch.norm(
            current_waypoints[:, :2] - robot_pos_w[:, :2], dim=1
        )
        self.metrics["waypoints_reached"] = self.current_waypoint_idx.float()

    def _sample_spacing_distance(self, num_samples: int, per_waypoint: bool = True) -> torch.Tensor:
        """Sample distances from spacing scenarios.
        
        Args:
            num_samples: Number of distance samples needed.
            per_waypoint: If True, sample independently for each sample.
                         If False, sample one scenario and apply to all.
        
        Returns:
            Tensor of sampled distances with shape (num_samples,).
        """
        if not self._use_scenarios:
            # Legacy mode: use single spacing range
            return torch.empty(num_samples, device=self.device).uniform_(
                *self.cfg.waypoint_spacing
            )
        
        num_scenarios = len(self.cfg.spacing_scenarios)
        
        if per_waypoint:
            # Sample scenario index for each sample independently
            scenario_indices = torch.multinomial(
                self._spacing_probs, 
                num_samples, 
                replacement=True
            )
        else:
            # Sample one scenario for all samples (uniform within batch)
            single_idx = torch.multinomial(self._spacing_probs, 1).item()
            scenario_indices = torch.full(
                (num_samples,), single_idx, 
                device=self.device, dtype=torch.long
            )
        
        # Get spacing ranges for selected scenarios
        selected_ranges = self._spacing_ranges[scenario_indices]  # (num_samples, 2)
        
        # Sample uniformly within each selected range
        # distance = min + (max - min) * uniform(0, 1)
        uniform_samples = torch.rand(num_samples, device=self.device)
        distances = selected_ranges[:, 0] + (
            selected_ranges[:, 1] - selected_ranges[:, 0]
        ) * uniform_samples
        
        return distances

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new waypoints for the specified environments.
        
        Waypoints are sampled sequentially, with each waypoint being generated
        relative to the previous one. The spacing between waypoints is determined
        by the configured spacing scenarios.
        """
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return

        # Get the robot's initial position for these environments
        robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2].clone()
        robot_pos_z = self.robot.data.root_pos_w[env_ids, 2].clone()  # Store z for waypoint height
        robot_heading = self.robot.data.heading_w[env_ids].clone()

        # Reset waypoint index and flags
        self.current_waypoint_idx[env_ids] = 0
        self.all_waypoints_reached[env_ids] = False
        
        # Pre-sample scenario indices for non-per-waypoint mode
        if self._use_scenarios and not self.cfg.per_waypoint_spacing:
            env_scenario_indices = torch.multinomial(
                self._spacing_probs, 
                num_envs_to_reset, 
                replacement=True
            )

        # Sample waypoints
        for wp_idx in range(self.cfg.num_waypoints):
            if wp_idx == 0:
                # First waypoint: sample within initial_waypoint_distance from robot
                distance = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.initial_waypoint_distance
                )
                # Sample angle relative to robot's current heading
                angle_offset = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.ranges.angle_range
                )
                angle = robot_heading + angle_offset
            else:
                # Subsequent waypoints: sample distance from spacing scenarios
                if self._use_scenarios:
                    if self.cfg.per_waypoint_spacing:
                        distance = self._sample_spacing_distance(
                            num_envs_to_reset, per_waypoint=True
                        )
                    else:
                        # Use pre-sampled scenario for this environment
                        selected_ranges = self._spacing_ranges[env_scenario_indices]
                        uniform_samples = torch.rand(num_envs_to_reset, device=self.device)
                        distance = selected_ranges[:, 0] + (
                            selected_ranges[:, 1] - selected_ranges[:, 0]
                        ) * uniform_samples
                else:
                    # Legacy mode
                    distance = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                        *self.cfg.waypoint_spacing
                    )
                
                # Compute heading from previous to current position for direction reference
                if wp_idx == 1:
                    prev_heading = robot_heading
                else:
                    prev_wp = self.waypoints_w[env_ids, wp_idx - 2]
                    curr_wp = self.waypoints_w[env_ids, wp_idx - 1]
                    direction = curr_wp - prev_wp
                    prev_heading = torch.atan2(direction[:, 1], direction[:, 0])
                
                angle_offset = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.ranges.angle_range
                )
                angle = prev_heading + angle_offset

            # Compute waypoint position
            if wp_idx == 0:
                base_pos = robot_pos_xy
            else:
                base_pos = self.waypoints_w[env_ids, wp_idx - 1, :2]

            wp_x = base_pos[:, 0] + distance * torch.cos(angle)
            wp_y = base_pos[:, 1] + distance * torch.sin(angle)
            # z is set to ground level (slightly above for visibility)
            wp_z = torch.zeros_like(robot_pos_z) + 0.1
            
            self.waypoints_w[env_ids, wp_idx, 0] = wp_x
            self.waypoints_w[env_ids, wp_idx, 1] = wp_y
            self.waypoints_w[env_ids, wp_idx, 2] = wp_z

        # Update marker positions for visualization
        self._markers_pos[env_ids] = self.waypoints_w[env_ids].clone()
        
        # Update visualization immediately after resampling
        if self.cfg.debug_vis and hasattr(self, "waypoint_visualizer"):
            visualize_pos = self._markers_pos.view(-1, 3)
            self.waypoint_visualizer.visualize(translations=visualize_pos)

    def _update_command(self):
        """Update the command based on the current robot state.
        
        This function:
        1. Checks if the current waypoint has been reached (using 2D distance)
        2. Advances the waypoint index if needed
        3. Transforms waypoints to the robot's local frame
        """
        # Get robot position in world frame
        robot_pos_w = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w

        # Check if current waypoint is reached (using 2D distance)
        current_wp_indices = self.current_waypoint_idx.clamp(max=self.cfg.num_waypoints - 1)
        current_waypoints = self.waypoints_w[
            torch.arange(self.num_envs, device=self.device), current_wp_indices
        ]
        # 2D distance for ground vehicles
        distance_to_current = torch.norm(current_waypoints[:, :2] - robot_pos_w[:, :2], dim=1)
        
        # Check which environments have reached their current waypoint
        reached_mask = distance_to_current < self.cfg.waypoint_reach_threshold
        # Only advance if not already at the last waypoint
        can_advance = self.current_waypoint_idx < self.cfg.num_waypoints - 1
        should_advance = reached_mask & can_advance
        
        # Advance waypoint index
        self.current_waypoint_idx[should_advance] += 1
        
        # Check if all waypoints reached
        current_wp_indices_after = self.current_waypoint_idx.clamp(max=self.cfg.num_waypoints - 1)
        current_waypoints_after = self.waypoints_w[
            torch.arange(self.num_envs, device=self.device), current_wp_indices_after
        ]
        distance_to_current_after = torch.norm(current_waypoints_after[:, :2] - robot_pos_w[:, :2], dim=1)
        reached_current_after = distance_to_current_after < self.cfg.waypoint_reach_threshold
        
        at_last_waypoint = self.current_waypoint_idx >= self.cfg.num_waypoints - 1
        self.all_waypoints_reached = at_last_waypoint & reached_current_after

        # Transform waypoints to robot's local frame
        yaw_q = yaw_quat(robot_quat)
        
        for i in range(self.total_waypoints_tracked):
            # Get waypoint index (clamped to valid range)
            wp_idx = (self.current_waypoint_idx + i).clamp(max=self.cfg.num_waypoints - 1)
            waypoint_w = self.waypoints_w[
                torch.arange(self.num_envs, device=self.device), wp_idx
            ]
            
            # Compute relative position in world frame
            rel_pos_w = waypoint_w - robot_pos_w
            
            # Transform to robot's local frame
            rel_pos_b = quat_apply_inverse(yaw_q, rel_pos_w)
            
            # Store in command buffer (x, y, z)
            self.waypoints_b[:, i * 3:(i + 1) * 3] = rel_pos_b

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization for waypoints.
        
        Uses colored spheres:
        - Red sphere (marker0): current target waypoint
        - Green spheres (marker1): future waypoints
        """
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer = VisualizationMarkers(
                    self.cfg.waypoint_visualizer_cfg
                )
            self.waypoint_visualizer.set_visibility(True)
        else:
            if hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.
        Updates marker positions for all waypoints in all environments
        """
        if not self.robot.is_initialized:
            return

        # Copy waypoint positions for visualization
        self._markers_pos[:, :, :3] = self.waypoints_w[:, :, :3]

        visualize_pos = self._markers_pos.view(-1, 3)
        
        one_hot_encoded = torch.nn.functional.one_hot(
            self.current_waypoint_idx.long(), 
            num_classes=self.cfg.num_waypoints
        )
        marker_indices = (1 - one_hot_encoded).view(-1).tolist()
        
        self.waypoint_visualizer.visualize(translations=visualize_pos, marker_indices=marker_indices)

