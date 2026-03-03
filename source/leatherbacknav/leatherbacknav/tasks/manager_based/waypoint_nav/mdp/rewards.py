# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for Leatherback waypoint navigation task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def waypoint_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    bonus: float = 10.0,
) -> torch.Tensor:
    """Bonus reward when a waypoint is reached."""
    waypoint_term = env.command_manager.get_term(command_name)
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]
    distance = torch.norm(current_wp_b, dim=1)
    reached = distance < waypoint_term.cfg.waypoint_reach_threshold
    return reached.float() * bonus


def waypoint_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    std: float = 2.0,
) -> torch.Tensor:
    """Reward for being close to the waypoint: exp(-distance / std)."""
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]
    distance = torch.norm(current_wp_b, dim=1)
    return torch.exp(-distance / std)


def waypoint_heading_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for facing towards the current waypoint."""
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]
    heading_error = torch.abs(torch.atan2(current_wp_b[:, 1], current_wp_b[:, 0]))
    return torch.exp(-heading_error / std)


def progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for making progress towards the current waypoint.
    
    Uses DELTA-based reward: (previous_distance - current_distance).
    This directly rewards reducing distance to waypoint, matching original Leatherback.
    
    Handles waypoint transitions by tracking the waypoint index and resetting
    the previous distance when a new waypoint becomes active.
    """
    waypoint_term = env.command_manager.get_term(command_name)
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]
    current_distance = torch.norm(current_wp_b, dim=1)
    
    # Get or initialize tracking buffers
    dist_key = f"_prev_waypoint_distance_{command_name}"
    idx_key = f"_prev_waypoint_index_{command_name}"
    
    if not hasattr(env, dist_key):
        setattr(env, dist_key, current_distance.clone())
        setattr(env, idx_key, waypoint_term.current_waypoint_idx.clone())
    
    prev_distance = getattr(env, dist_key)
    prev_waypoint_idx = getattr(env, idx_key)
    current_waypoint_idx = waypoint_term.current_waypoint_idx
    
    # Detect waypoint transitions (index changed)
    waypoint_changed = current_waypoint_idx != prev_waypoint_idx
    
    # Delta-based progress: positive when distance decreases (moving closer)
    # When waypoint changes, use 0 progress (the waypoint_reached bonus handles that)
    progress = torch.where(
        waypoint_changed,
        torch.zeros_like(current_distance),  # No progress penalty when waypoint changes
        prev_distance - current_distance     # Normal delta-based progress
    )
    
    # Update tracking buffers for next step
    setattr(env, dist_key, current_distance.clone())
    setattr(env, idx_key, current_waypoint_idx.clone())
    
    return progress


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large changes in action output for smooth control."""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant reward per step (use negative weight for time penalty)."""
    return torch.ones(env.num_envs, device=env.device)


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving forward in the robot's body frame.
    
    This prevents the robot from learning to drive backward.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Forward velocity in body frame (x-axis is forward)
    forward_vel = asset.data.root_lin_vel_b[:, 0]
    # Reward forward, penalize backward
    return forward_vel


def backward_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for moving backward in the robot's body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    forward_vel = asset.data.root_lin_vel_b[:, 0]
    # Only return negative values (backward motion), clamped to 0 otherwise
    return torch.clamp(forward_vel, max=0.0)
