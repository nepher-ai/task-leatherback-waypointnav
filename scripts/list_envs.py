# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to list all registered environments in leatherbacknav."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym

# Import leatherbacknav to register environments
import leatherbacknav  # noqa: F401


def main():
    """List all registered Leatherback navigation environments."""
    print("\n" + "=" * 60)
    print("Registered Leatherback Navigation Environments")
    print("=" * 60)
    
    # Get all registered environments
    all_envs = gym.envs.registry.keys()
    
    # Filter for Leatherback environments
    leatherback_envs = [env for env in all_envs if "Leatherback" in env]
    
    if leatherback_envs:
        for env_name in sorted(leatherback_envs):
            print(f"  - {env_name}")
    else:
        print("  No Leatherback environments found.")
        print("  Make sure leatherbacknav is installed: pip install -e source/leatherbacknav")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()

