# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP module for Leatherback waypoint navigation task.

This module contains custom observations, rewards, terminations, and commands
for the waypoint navigation task with a wheeled robot.
"""

# Import all standard MDP functions from isaaclab explicitly for type checking
from isaaclab.envs.mdp import *  # noqa: F401, F403

# Re-export commonly used functions explicitly for type checking
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.observations import (
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)
from isaaclab.envs.mdp.events import (
    randomize_rigid_body_material,
    randomize_rigid_body_mass,
    reset_root_state_uniform,
    reset_joints_by_scale,
)

# Import custom observations
from .observations import (
    waypoint_commands,
    waypoint_distance,
    waypoint_heading_error,
    waypoint_progress_indicator,
    base_position_2d,
    base_heading,
    base_lin_vel_2d,
    base_ang_vel_yaw,
)

# Import custom rewards
from .rewards import (
    # Waypoint navigation rewards
    waypoint_reached_bonus,
    waypoint_distance_reward,
    waypoint_heading_reward,
    forward_velocity_reward,
    progress_reward,
    # Wheeled robot regularization penalties
    action_smoothness_penalty,
    base_motion_penalty,
    steering_penalty,
    alive_reward,
)

# Import custom terminations
from .terminations import (
    all_waypoints_reached,
    flipped_over,
    time_out,
)

# Import custom events
from .events import (
    randomize_action_scale,
)

# Import custom commands
from .commands import WaypointCommand, WaypointCommandCfg, SpacingScenarioCfg

__all__ = [
    # Actions (from isaaclab)
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    # Observations (from isaaclab)
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    # Events (from isaaclab)
    "randomize_rigid_body_material",
    "randomize_rigid_body_mass",
    "reset_root_state_uniform",
    "reset_joints_by_scale",
    # Custom Observations
    "waypoint_commands",
    "waypoint_distance",
    "waypoint_heading_error",
    "waypoint_progress_indicator",
    "base_position_2d",
    "base_heading",
    "base_lin_vel_2d",
    "base_ang_vel_yaw",
    # Custom Rewards
    "waypoint_reached_bonus",
    "waypoint_distance_reward",
    "waypoint_heading_reward",
    "forward_velocity_reward",
    "progress_reward",
    "action_smoothness_penalty",
    "base_motion_penalty",
    "steering_penalty",
    "alive_reward",
    # Custom Terminations
    "all_waypoints_reached",
    "flipped_over",
    "time_out",
    # Custom Events
    "randomize_action_scale",
    # Custom Commands
    "WaypointCommand",
    "WaypointCommandCfg",
    "SpacingScenarioCfg",
]
