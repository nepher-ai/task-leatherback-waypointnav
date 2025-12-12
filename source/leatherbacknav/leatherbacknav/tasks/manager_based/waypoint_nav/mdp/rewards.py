# Copyright (c) 2025, Nepher Team
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
    
    Only rewards forward progress, doesn't punish reversing (car may need to back up to steer).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    waypoint_cmd = env.command_manager.get_command(command_name)
    current_wp_b = waypoint_cmd[:, :2]
    
    wp_distance = torch.norm(current_wp_b, dim=1, keepdim=True).clamp(min=1e-6)
    wp_direction = current_wp_b / wp_distance
    vel_b = asset.data.root_lin_vel_b[:, :2]
    velocity_towards_wp = torch.sum(vel_b * wp_direction, dim=1)
    
    # Only reward forward progress, don't punish reversing
    return torch.clamp(velocity_towards_wp, min=0.0)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large changes in action output for smooth control."""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant reward per step (use negative weight for time penalty)."""
    return torch.ones(env.num_envs, device=env.device)
