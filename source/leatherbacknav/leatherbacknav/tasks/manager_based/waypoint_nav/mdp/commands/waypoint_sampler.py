# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint samplers for generating waypoint sequences."""

from __future__ import annotations

import warnings

import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import WaypointCommandCfg
    from .waypoint_command import SpacingManager


class WaypointSampler(ABC):
    """Base class for waypoint sampling strategies."""

    def __init__(
        self, 
        cfg: WaypointCommandCfg, 
        robot: Articulation,
        spacing_manager: SpacingManager
    ):
        """Initialize the waypoint sampler.
        
        Args:
            cfg: Configuration for waypoint command generator.
            robot: Robot articulation asset.
            spacing_manager: Spacing manager for distance sampling.
        """
        self.cfg = cfg
        self.robot = robot
        self.spacing_manager = spacing_manager
        self.device = robot.device

    @abstractmethod
    def sample_waypoints(
        self, 
        env_ids: Sequence[int], 
        waypoints_w: torch.Tensor,
        num_waypoints_per_env: torch.Tensor
    ) -> None:
        """Sample waypoints for the specified environments.
        
        Args:
            env_ids: Environment indices to sample waypoints for.
            waypoints_w: Output tensor of shape (num_envs, num_waypoints, 3) to fill.
                        Waypoints should be in world frame.
            num_waypoints_per_env: Number of waypoints per environment.
        """
        pass


class RandomWaypointSampler(WaypointSampler):
    """Random waypoint sampler that generates sequential waypoints."""

    def sample_waypoints(
        self, 
        env_ids: Sequence[int], 
        waypoints_w: torch.Tensor,
        num_waypoints_per_env: torch.Tensor
    ) -> None:
        """Sample random waypoints sequentially.
        
        Args:
            env_ids: Environment indices to sample waypoints for.
            waypoints_w: Output tensor to fill with waypoints in world frame.
            num_waypoints_per_env: Number of waypoints per environment.
        """
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return


        robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2].clone()
        robot_heading = self.robot.data.heading_w[env_ids].clone()
        
        # Sample episode-level scenario if not using per-waypoint spacing
        episode_scenario_indices = None
        if self.spacing_manager.use_scenarios and not self.cfg.per_waypoint_spacing:
            episode_scenario_indices = self.spacing_manager.sample_episode_scenario(
                num_envs_to_reset
            )

        for wp_idx in range(self.cfg.num_waypoints):
            if wp_idx == 0:
                # First waypoint: sample from initial distance range
                distance = torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.initial_waypoint_distance
                )
                angle = robot_heading + torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.ranges.angle_range
                )
                base_pos = robot_pos_xy
            else:
                # Subsequent waypoints: use spacing manager
                if self.spacing_manager.use_scenarios:
                    if self.cfg.per_waypoint_spacing:
                        distance = self.spacing_manager.sample_distance(
                            num_envs_to_reset, per_waypoint=True
                        )
                    else:
                        distance = self.spacing_manager.sample_distance(
                            num_envs_to_reset, 
                            per_waypoint=False,
                            scenario_indices=episode_scenario_indices
                        )
                else:
                    distance = self.spacing_manager.sample_distance(num_envs_to_reset)
                
                # Compute heading from previous waypoint
                if wp_idx == 1:
                    prev_heading = robot_heading
                else:
                    direction = waypoints_w[env_ids, wp_idx - 1] - waypoints_w[env_ids, wp_idx - 2]
                    prev_heading = torch.atan2(direction[:, 1], direction[:, 0])
                
                angle = prev_heading + torch.empty(num_envs_to_reset, device=self.device).uniform_(
                    *self.cfg.ranges.angle_range
                )
                base_pos = waypoints_w[env_ids, wp_idx - 1, :2]

            # Compute waypoint position
            wp_xy = base_pos + distance.unsqueeze(1) * torch.stack(
                [torch.cos(angle), torch.sin(angle)], dim=1
            )
            waypoints_w[env_ids, wp_idx, :2] = wp_xy
            waypoints_w[env_ids, wp_idx, 2] = 0.1

        print("Waypoints w: ", waypoints_w)
        num_waypoints_per_env[env_ids] = torch.tensor(self.cfg.num_waypoints, device=self.device)

class EnvNavWaypointSampler(WaypointSampler):
    """Waypoint sampler that uses waypoints from envs-nav scene config."""

    def __init__(
        self, 
        cfg: WaypointCommandCfg, 
        robot: Articulation,
        scene_cfg,
        get_env_origins,
    ):
        """Initialize the envs-nav waypoint sampler.
        
        Args:
            cfg: Configuration for waypoint command generator.
            robot: Robot articulation asset.
            scene_cfg: Scene configuration with gen_waypoints or gen_random_waypoints method.
            get_env_origins: Callable that takes env_ids and returns env_origins tensor.
        """
        # Initialize base attributes without spacing_manager (not needed for this sampler)
        self.cfg = cfg
        self.robot = robot
        self.device = robot.device
        self.scene_cfg = scene_cfg
        self.get_env_origins = get_env_origins
        # Check for gen_waypoints method (preferred) or gen_random_waypoints (fallback)
        if not (hasattr(scene_cfg, "gen_waypoints") or hasattr(scene_cfg, "gen_random_waypoints")):
            raise ValueError(
                "scene_cfg must have gen_waypoints or gen_random_waypoints method "
                "for EnvNavWaypointSampler"
            )
        

    def sample_waypoints(
        self, 
        env_ids: Sequence[int], 
        waypoints_w: torch.Tensor,
        num_waypoints_per_env: torch.Tensor
    ) -> None:
        """Sample waypoints from envs-nav scene config.
        
        Args:
            env_ids: Environment indices to sample waypoints for.
            waypoints_w: Output tensor to fill with waypoints in world frame.
            num_waypoints_per_env: Number of waypoints per environment.
        """

        
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return

        # Convert env_ids to tensor (handle both list and tensor inputs)
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        env_origins = self.get_env_origins(env_ids)

        # Reset robot initial position if gen_bot_init_pos or gen_bot_random_pos is available
        if hasattr(self.scene_cfg, "gen_bot_init_pos") or hasattr(self.scene_cfg, "gen_bot_random_pos"):
            try:
                # Try gen_bot_init_pos first (user's preferred name), fallback to gen_bot_random_pos
                if hasattr(self.scene_cfg, "gen_bot_init_pos"):
                    positions, yaws = self.scene_cfg.gen_bot_init_pos(
                        env_ids=env_ids_tensor,
                        env_origins=env_origins,
                        device=self.device,
                    )
                else:
                    positions, yaws = self.scene_cfg.gen_bot_random_pos(
                        env_ids=env_ids_tensor,
                        env_origins=env_origins,
                        device=self.device,
                    )
                
                # Convert yaw to quaternion (w, x, y, z) for z-axis rotation
                half_yaw = yaws / 2.0
                quats = torch.zeros((num_envs_to_reset, 4), device=self.device)
                quats[:, 0] = torch.cos(half_yaw)  # w
                quats[:, 3] = torch.sin(half_yaw)   # z (yaw rotation)
                
                # Combine positions and quaternions into root pose (shape: num_envs_to_reset, 7)
                root_pose = torch.cat([positions, quats], dim=1)
                
                # Set positions and orientations
                self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids_tensor)
                
                # Reset velocities to zero (linear + angular, shape: num_envs_to_reset, 6)
                zero_vel = torch.zeros((num_envs_to_reset, 6), device=self.device)
                self.robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_tensor)
            except Exception as e:
                # Log warning but continue with waypoint sampling
                warnings.warn(f"Failed to reset robot position using gen_bot_init_pos/gen_bot_random_pos: {e}")

        robot_pos_z = self.robot.data.root_pos_w[env_ids_tensor, 2].clone()

        if hasattr(self.scene_cfg, "gen_waypoints"):
            generated_waypoints, num_waypoints_per_env_tensor = self.scene_cfg.gen_waypoints(
                env_ids=env_ids_tensor,
                env_origins=env_origins,
                device=self.device,
            )

        elif hasattr(self.scene_cfg, "gen_random_waypoints"):
            generated_waypoints = self.scene_cfg.gen_random_waypoints(
                env_ids=env_ids_tensor,
                env_origins=env_origins,
                num_waypoints=self.cfg.num_waypoints,
                device=self.device,
            )
            num_waypoints_per_env_tensor = torch.tensor(self.cfg.num_waypoints, device=self.device, dtype=torch.long)
        else:
            raise RuntimeError(
                "scene_cfg does not have gen_waypoints or gen_random_waypoints method"
            )
        

        # Copy generated waypoints to output buffer
        max_num_waypoints = generated_waypoints.shape[1]
        generated_waypoints[:, :, 2] = 0.1
        generated_waypoints = generated_waypoints[:, :max_num_waypoints]
        waypoints_w[env_ids_tensor, :max_num_waypoints, :3] = generated_waypoints
        num_waypoints_per_env[env_ids_tensor] = num_waypoints_per_env_tensor