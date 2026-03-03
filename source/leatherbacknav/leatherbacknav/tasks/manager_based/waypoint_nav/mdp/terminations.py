# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for Leatherback waypoint navigation task.

These termination conditions determine when an episode should end,
including reaching all waypoints, flipping over, or timing out.
"""

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_waypoints_reached(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoints",
) -> torch.Tensor:
    """Terminate when all waypoints have been reached.
    
    This is a "success" termination - the episode ends because the task
    is complete.
    
    Args:
        env: The environment instance.
        command_name: Name of the waypoint command term. Defaults to "waypoints".
        
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    waypoint_term = env.command_manager.get_term(command_name)
    return waypoint_term.all_waypoints_reached


def flipped_over(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt_angle: float = 10.0,
) -> torch.Tensor:
    """Terminate when the car tilts beyond the maximum allowed angle.
    
    This checks the tilt via the z-component of the projected gravity vector:
    - When upright: projected_gravity_b[:, 2] ≈ -1 (gravity points down in body frame)
    - At max_tilt_angle degrees: projected_gravity_b[:, 2] = -cos(max_tilt_angle)
    - When upside down: projected_gravity_b[:, 2] ≈ +1 (gravity points up in body frame)
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        max_tilt_angle: Maximum tilt angle in degrees before termination. Defaults to 60.
        
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    threshold = -math.cos(math.radians(max_tilt_angle))
    return asset.data.projected_gravity_b[:, 2] > threshold


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when the episode length exceeds the maximum allowed length.
    
    Args:
        env: The environment instance.
        
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    return env.episode_length_buf >= env.max_episode_length
