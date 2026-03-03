# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for Leatherback waypoint navigation task.

These functions provide observations related to waypoint navigation, including
the current waypoint position, distance to waypoint, and heading towards waypoint.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def waypoint_commands(env: ManagerBasedRLEnv, command_name: str = "waypoints") -> torch.Tensor:
    """Get the waypoint commands (current and lookahead waypoints in robot frame).
    
    This returns the waypoint positions relative to the robot's base frame,
    as provided by the WaypointCommand generator.
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        
    Returns:
        Tensor of shape (num_envs, num_waypoints * 3) containing [x, y, z] for each waypoint.
    """
    return env.command_manager.get_command(command_name)


def waypoint_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute the 2D distance to the current waypoint.
    
    Uses (x, y) components for 2D distance calculation (appropriate for ground vehicles).
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs, 1) containing the 2D distance to current waypoint.
    """
    # Get the waypoint command (first 3 elements are current waypoint in robot frame)
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]  # x, y in robot frame
    
    # 2D distance for ground vehicle navigation
    distance = torch.norm(current_wp_b, dim=1, keepdim=True)
    return distance


def waypoint_heading_error(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute the yaw (horizontal) heading error towards the current waypoint.
    
    The heading error is the angle between the robot's forward direction
    and the horizontal direction to the current waypoint, normalized to [-pi, pi].
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs, 1) containing the yaw heading error in radians.
    """
    # Get the waypoint command (first 2 elements are current waypoint x, y in robot frame)
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]  # x, y in robot frame
    
    # In robot frame, forward is +x, so heading to waypoint is atan2(y, x)
    heading_to_wp = torch.atan2(current_wp_b[:, 1], current_wp_b[:, 0])
    
    return heading_to_wp.unsqueeze(1)


def waypoint_progress_indicator(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
) -> torch.Tensor:
    """Get normalized progress through the waypoint sequence.
    
    Returns a value between 0 and 1 indicating how many waypoints
    have been reached relative to the total number.
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        
    Returns:
        Tensor of shape (num_envs, 1) containing the progress indicator.
    """
    waypoint_term = env.command_manager.get_term(command_name)
    progress = waypoint_term.current_waypoint_idx.float() / waypoint_term.cfg.num_waypoints
    return progress.unsqueeze(1)


def base_position_2d(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot's 2D position (x, y) in the world frame.
    
    Note: This is relative to the environment origin for proper multi-env support.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs, 2) containing [x, y] position.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Return position relative to environment origin
    pos_w = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return pos_w


def base_heading(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot's heading angle (yaw) in the world frame.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs, 1) containing the heading angle in radians.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.heading_w.unsqueeze(1)


def base_lin_vel_2d(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot's 2D linear velocity in the robot's base frame.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs, 2) containing [vx, vy] velocity in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]


def base_ang_vel_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the robot's angular velocity around the z-axis (yaw rate).
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Tensor of shape (num_envs, 1) containing the yaw rate in rad/s.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2:3]

