# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for Leatherback waypoint navigation task.

These reward functions encourage the wheeled robot to navigate through waypoints
while maintaining stable driving behavior.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Waypoint Navigation Rewards
##


def waypoint_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    bonus: float = 10.0,
) -> torch.Tensor:
    """Bonus reward when a waypoint is reached.
    
    Provides a large positive reward when the robot reaches within the threshold
    distance of the current waypoint and advances to the next one.
    Uses 2D distance for waypoint reach detection (appropriate for ground vehicles).
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        bonus: Bonus reward value. Defaults to 10.0.
        
    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    waypoint_term = env.command_manager.get_term(command_name)
    
    # Get current waypoint in robot frame (x, y for 2D distance)
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]  # x, y for 2D navigation
    distance = torch.norm(current_wp_b, dim=1)
    
    # Reward when within threshold
    reached = distance < waypoint_term.cfg.waypoint_reach_threshold
    return reached.float() * bonus


def waypoint_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    std: float = 2.0,
) -> torch.Tensor:
    """Reward for being close to the current waypoint using exponential kernel.
    
    Uses an exponential decay based on 2D distance: exp(-distance / std)
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        std: Standard deviation for exponential kernel. Defaults to 2.0.
        
    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]  # x, y for 2D distance
    distance = torch.norm(current_wp_b, dim=1)
    
    return torch.exp(-distance / std)


def waypoint_heading_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for facing towards the current waypoint using exponential kernel.
    
    Encourages the robot to orient itself towards the waypoint (yaw heading).
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        std: Standard deviation for exponential kernel. Defaults to 0.5.
        
    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]  # x, y for yaw heading
    
    # Heading error (yaw angle to waypoint in robot frame)
    heading_error = torch.abs(torch.atan2(current_wp_b[:, 1], current_wp_b[:, 0]))
    
    return torch.exp(-heading_error / std)


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_velocity: float = 1.0,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for moving towards the waypoint.
    
    Encourages the robot to move in the direction of the waypoint.
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        asset_cfg: Configuration for the robot asset.
        target_velocity: Target velocity when heading towards waypoint. Defaults to 1.0.
        std: Standard deviation for exponential kernel. Defaults to 0.5.
        
    Returns:
        Tensor of shape (num_envs,) containing the reward.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]  # x, y for 2D direction
    
    # Compute desired velocity direction (normalized 2D waypoint direction)
    wp_distance = torch.norm(current_wp_b, dim=1, keepdim=True).clamp(min=1e-6)
    wp_direction = current_wp_b / wp_distance
    
    # Get robot's velocity in base frame (2D horizontal)
    vel_b = asset.data.root_lin_vel_b[:, :2]
    
    # Project velocity onto waypoint direction (2D dot product)
    velocity_towards_wp = torch.sum(vel_b * wp_direction, dim=1)
    
    # Reward tracking target velocity towards waypoint
    vel_error = torch.abs(velocity_towards_wp - target_velocity)
    return torch.exp(-vel_error / std)


def progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for making progress towards the current waypoint.
    
    This is computed as the velocity component in the direction of the waypoint,
    rewarding the robot for getting closer.
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs,) containing the reward (positive when getting closer).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]  # x, y for 2D progress
    
    # Compute direction to waypoint (normalized 2D)
    wp_distance = torch.norm(current_wp_b, dim=1, keepdim=True).clamp(min=1e-6)
    wp_direction = current_wp_b / wp_distance
    
    # Get robot's velocity in base frame (2D horizontal)
    vel_b = asset.data.root_lin_vel_b[:, :2]
    
    # Velocity component towards waypoint (positive means getting closer)
    velocity_towards_wp = torch.sum(vel_b * wp_direction, dim=1)
    
    # Clip to avoid large negative rewards when moving away
    return torch.clamp(velocity_towards_wp, min=-0.5)


##
# Wheeled Robot Regularization Penalties
##


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output.
    
    This encourages smooth steering and throttle control.
    """
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def base_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity.
    
    Encourages the wheeled robot to maintain stable driving without bouncing.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def steering_penalty(
    env: ManagerBasedRLEnv,
    steering_action_index: int = 1,
) -> torch.Tensor:
    """Penalize excessive steering angle.
    
    Encourages the robot to drive more straight when possible.
    
    Args:
        env: The environment instance.
        steering_action_index: Index of steering in the action vector. Defaults to 1.
        
    Returns:
        Tensor of shape (num_envs,) containing the penalty.
    """
    # Get steering action (assuming action format is [throttle, steering])
    steering = env.action_manager.action[:, steering_action_index]
    return torch.square(steering)


def alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant reward for staying alive (not terminated)."""
    return torch.ones(env.num_envs, device=env.device)

