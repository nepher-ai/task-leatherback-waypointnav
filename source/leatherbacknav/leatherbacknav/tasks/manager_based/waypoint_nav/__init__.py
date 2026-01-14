# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint navigation task for Leatherback wheeled robot.

This module registers the waypoint navigation environments with Gymnasium.
The task trains a wheeled robot to navigate through a sequence of 2D waypoints.
"""

import gymnasium as gym

from . import agents

# Export evaluation compatibility utilities
from .eval_compat import EvalCompatEnv, wrap_for_eval

__all__ = ["EvalCompatEnv", "wrap_for_eval"]

##
# Register Gym environments.
##

gym.register(
    id="Nepher-Leatherback-WaypointNav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.waypoint_nav_env_cfg:WaypointNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaypointNavPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Leatherback-WaypointNav-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.waypoint_nav_env_cfg:WaypointNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaypointNavPPORunnerCfg",
    },
)

##
# Navigation environments with envhub (nepher) integration
# Use --env_id and --scene args to select environment
##

gym.register(
    id="Nepher-Leatherback-WaypointNav-Envhub-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.waypoint_nav_env_cfg_envhub:WaypointNavEnvCfg_Envhub",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaypointNavPPORunnerCfg",
    },
)

gym.register(
    id="Nepher-Leatherback-WaypointNav-Envhub-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.waypoint_nav_env_cfg_envhub:WaypointNavEnvCfg_Envhub_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WaypointNavPPORunnerCfg",
    },
)

