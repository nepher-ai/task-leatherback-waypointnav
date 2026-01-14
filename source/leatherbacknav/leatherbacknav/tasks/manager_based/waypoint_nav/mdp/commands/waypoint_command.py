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

from .waypoint_sampler import RandomWaypointSampler, EnvNavWaypointSampler, WaypointSampler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import WaypointCommandCfg


# ============================================================================
# Internal Helper Classes
# ============================================================================


class SpacingManager:
    """Manages spacing scenarios and distance sampling for waypoints."""

    def __init__(self, cfg: WaypointCommandCfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._use_scenarios = bool(cfg.spacing_scenarios)
        self._spacing_ranges = None
        self._spacing_probs = None
        
        if self._use_scenarios:
            ranges = torch.tensor([s.spacing_range for s in cfg.spacing_scenarios], device=device, dtype=torch.float32)
            weights = torch.tensor([s.weight for s in cfg.spacing_scenarios], device=device, dtype=torch.float32)
            self._spacing_ranges = ranges
            self._spacing_probs = weights / weights.sum()

    def sample_distance(
        self, 
        num_samples: int, 
        per_waypoint: bool = True,
        scenario_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample distances from spacing scenarios."""
        if not self._use_scenarios or self._spacing_probs is None or self._spacing_ranges is None:
            return torch.empty(num_samples, device=self.device).uniform_(*self.cfg.waypoint_spacing)
        
        if per_waypoint:
            scenario_indices = torch.multinomial(self._spacing_probs, num_samples, replacement=True)
        elif scenario_indices is None:
            scenario_indices = torch.full(
                (num_samples,), torch.multinomial(self._spacing_probs, 1).item(), device=self.device, dtype=torch.long
            )
        
        selected_ranges = self._spacing_ranges[scenario_indices]
        uniform_samples = torch.rand(num_samples, device=self.device)
        return selected_ranges[:, 0] + (selected_ranges[:, 1] - selected_ranges[:, 0]) * uniform_samples

    def sample_episode_scenario(self, num_envs: int) -> torch.Tensor | None:
        """Sample scenario indices for each environment in an episode."""
        if not self._use_scenarios or self._spacing_probs is None:
            return None
        return torch.multinomial(self._spacing_probs, num_envs, replacement=True)

    @property
    def use_scenarios(self) -> bool:
        return self._use_scenarios


# ============================================================================
# Main Command Class
# ============================================================================


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
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.spacing_manager = SpacingManager(cfg, self.device)
        self.total_waypoints_tracked = 1 + cfg.num_lookahead_waypoints
        self.sampler = self._create_sampler(cfg, env)

        # Tracking buffers
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.all_waypoints_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.num_waypoints_per_env = torch.full((self.num_envs,), cfg.num_waypoints, dtype=torch.long, device=self.device)

        # Buffers: waypoints in world frame (num_envs, num_waypoints, 3)
        self.waypoints_w = torch.zeros(self.num_envs, cfg.num_waypoints, 3, device=self.device)
        # Command output: waypoints in robot's base frame (num_envs, total_waypoints_tracked * 3)
        self.waypoints_b = torch.zeros(self.num_envs, self.total_waypoints_tracked * 3, device=self.device)
        self._markers_pos = torch.zeros(self.num_envs, cfg.num_waypoints, 3, device=self.device)

        # Initialize metrics
        self.metrics["waypoints_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["distance_to_waypoint"] = torch.zeros(self.num_envs, device=self.device)

    def _create_sampler(self, cfg: WaypointCommandCfg, env: ManagerBasedEnv) -> WaypointSampler:
        """Create the appropriate waypoint sampler based on configuration."""
        if cfg.use_envs_nav_waypoints:
            scene_cfg = getattr(env.cfg, "_scene_cfg", None)
            if scene_cfg is None or not (hasattr(scene_cfg, "gen_random_waypoints") or hasattr(scene_cfg, "gen_waypoints")):
                raise ValueError(
                    "use_envs_nav_waypoints=True requires environment to have _scene_cfg "
                    "with gen_random_waypoints or gen_waypoints method. "
                    "Make sure to call cfg.load_scene() before creating the environment."
                )
            return EnvNavWaypointSampler(cfg, self.robot, scene_cfg, lambda env_ids: env.scene.env_origins[env_ids])
        return RandomWaypointSampler(cfg, self.robot, self.spacing_manager)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = f"WaypointCommand:\n\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tNumber of waypoints: {self.cfg.num_waypoints}\n"
        
        if self.cfg.use_envs_nav_waypoints:
            msg += "\tMode: envhub (nepher) waypoints (from scene config)\n"
            if isinstance(self.sampler, EnvNavWaypointSampler):
                msg += f"\tScene config: {getattr(self.sampler.scene_cfg, 'name', 'unknown')}\n"
        elif self.spacing_manager.use_scenarios:
            msg += f"\tSpacing scenarios: {len(self.cfg.spacing_scenarios)} configured\n"
            msg += "\n".join(f"\t  [{i}] range={s.spacing_range}, weight={s.weight}" for i, s in enumerate(self.cfg.spacing_scenarios)) + "\n"
            msg += f"\tPer-waypoint spacing: {self.cfg.per_waypoint_spacing}\n"
        else:
            msg += f"\tWaypoint spacing: {self.cfg.waypoint_spacing} (legacy mode)\n"
        
        msg += f"\tReach threshold: {self.cfg.waypoint_reach_threshold}\n"
        msg += f"\tLookahead waypoints: {self.cfg.num_lookahead_waypoints}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The waypoint commands in the robot's base frame."""
        return self.waypoints_b

    def _reset_tracking(self, env_ids: Sequence[int] | torch.Tensor):
        """Reset waypoint tracking for specified environments."""
        if isinstance(env_ids, Sequence):
            env_ids = torch.tensor(env_ids, device=self.device)
        self.current_waypoint_idx[env_ids] = 0
        self.all_waypoints_reached[env_ids] = False
        self.num_waypoints_per_env[env_ids] = self.cfg.num_waypoints

    def update(self, waypoints_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update waypoint progression based on robot position."""
        robot_pos_w = self.robot.data.root_pos_w
        env_indices = torch.arange(self.num_envs, device=self.device)
        
        # Get current waypoint indices (clamped to valid range per environment)
        current_wp_indices = self._clamp_waypoint_indices(self.current_waypoint_idx)
        current_waypoints = waypoints_w[env_indices, current_wp_indices]
        
        # Compute distance to current waypoint (2D distance) and advance if reached
        distance_to_current = torch.norm(current_waypoints[:, :2] - robot_pos_w[:, :2], dim=1)
        reached_mask = distance_to_current < self.cfg.waypoint_reach_threshold
        can_advance = self.current_waypoint_idx < (self.num_waypoints_per_env - 1)
        self.current_waypoint_idx[reached_mask & can_advance] += 1
        
        # Update all_waypoints_reached flag
        final_wp_indices = self._clamp_waypoint_indices(self.current_waypoint_idx)
        final_waypoints = waypoints_w[env_indices, final_wp_indices]
        final_distance = torch.norm(final_waypoints[:, :2] - robot_pos_w[:, :2], dim=1)
        at_last_waypoint = self.current_waypoint_idx >= (self.num_waypoints_per_env - 1)
        self.all_waypoints_reached = at_last_waypoint & (final_distance < self.cfg.waypoint_reach_threshold)
        
        return final_wp_indices, final_waypoints, final_distance

    def _clamp_waypoint_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Clamp waypoint indices per environment based on num_waypoints_per_env."""
        max_valid_indices = (self.num_waypoints_per_env - 1).clamp(min=0)
        buffer_limit = torch.tensor(self.cfg.num_waypoints - 1, device=self.device)
        return indices.clamp(max=torch.min(max_valid_indices, buffer_limit))
    
    def get_current_indices(self) -> torch.Tensor:
        """Get current waypoint indices for all environments, clamped per environment."""
        return self._clamp_waypoint_indices(self.current_waypoint_idx)

    def _update_metrics(self):
        """Update metrics for logging."""
        _, current_waypoints, distance_to_current = self.update(self.waypoints_w)
        self.metrics["distance_to_waypoint"] = distance_to_current
        self.metrics["waypoints_reached"] = self.current_waypoint_idx.float()

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new waypoints for the specified environments."""
        if len(env_ids) == 0:
            return
        self._reset_tracking(env_ids)
        self.sampler.sample_waypoints(env_ids, self.waypoints_w, self.num_waypoints_per_env)
        self._update_visualization(env_ids)

    def _update_command(self):
        """Update the command based on the current robot state."""
        self.update(self.waypoints_w)
        
        if self.cfg.debug_vis and hasattr(self, "waypoint_visualizer"):
            self._update_marker_visualization()
        
        # Transform waypoints to robot base frame
        robot_pos_w = self.robot.data.root_pos_w
        yaw_q = yaw_quat(self.robot.data.root_quat_w)
        env_indices = torch.arange(self.num_envs, device=self.device)
        current_wp_indices = self.get_current_indices()
        
        for i in range(self.total_waypoints_tracked):
            wp_idx = self._clamp_waypoint_indices(current_wp_indices + i)
            waypoint_w = self.waypoints_w[env_indices, wp_idx]
            rel_pos_b = quat_apply_inverse(yaw_q, waypoint_w - robot_pos_w)
            self.waypoints_b[:, i * 3:(i + 1) * 3] = rel_pos_b

    def _update_visualization(self, env_ids: Sequence[int]):
        """Update visualization markers for the specified environments."""
        self._markers_pos[env_ids] = self.waypoints_w[env_ids].clone()
        if self.cfg.debug_vis and hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.visualize(translations=self._markers_pos.view(-1, 3))

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization for waypoints."""
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer = VisualizationMarkers(self.cfg.waypoint_visualizer_cfg)
            self.waypoint_visualizer.set_visibility(True)
        elif hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.set_visibility(False)

    def _update_marker_visualization(self):
        """Update marker visualization with current waypoint colors."""
        if not self.robot.is_initialized:
            return

        self._markers_pos[:, :, :3] = self.waypoints_w[:, :, :3]
        
        # marker0=red (current), marker1=green (future)
        current_idx = self.get_current_indices()
        one_hot = torch.nn.functional.one_hot(current_idx.long(), num_classes=self.cfg.num_waypoints)
        marker_indices = (1 - one_hot).view(-1)
        
        # Mark invalid waypoints as green
        wp_idx = torch.arange(self.cfg.num_waypoints, device=self.device).repeat(self.num_envs)
        env_idx = torch.arange(self.num_envs, device=self.device).repeat_interleave(self.cfg.num_waypoints)
        marker_indices[wp_idx >= self.num_waypoints_per_env[env_idx]] = 1
        
        self.waypoint_visualizer.visualize(translations=self._markers_pos.view(-1, 3), marker_indices=marker_indices.tolist())

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        if self.cfg.debug_vis and hasattr(self, "waypoint_visualizer"):
            self._update_marker_visualization()
