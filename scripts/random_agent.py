# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a random agent in the Leatherback navigation environment."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run a random agent in the Leatherback navigation environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="Nepher-Leatherback-WaypointNav-v0", help="Name of the task."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import leatherbacknav.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv


def main():
    """Run a random agent in the environment."""
    # create environment configuration
    env_cfg = gym.spec(args_cli.task).make_kwargs["env_cfg_entry_point"]
    env_cfg = env_cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info
    print(f"[INFO]: Environment: {args_cli.task}")
    print(f"[INFO]: Number of environments: {env_cfg.scene.num_envs}")
    print(f"[INFO]: Observation space: {env.observation_space}")
    print(f"[INFO]: Action space: {env.action_space}")

    # reset environment
    env.reset()
    
    # simulate environment
    count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample random actions
            actions = 2.0 * torch.rand(env.unwrapped.num_envs, env.unwrapped.num_actions, device=env.unwrapped.device) - 1.0
            # step environment
            obs, rew, terminated, truncated, info = env.step(actions)
            # print mean reward
            if count % 100 == 0:
                print(f"[INFO] Step {count}: Mean reward: {rew.mean().item():.4f}")
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

