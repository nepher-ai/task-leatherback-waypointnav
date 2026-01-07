# Leatherback Waypoint Navigation for Isaac Lab

**Developed by Nepher Team**

## Overview

This project implements a **waypoint navigation task** for the Leatherback wheeled robot in Isaac Lab. The task trains a policy to navigate the robot through a sequence of 3D waypoints using low-level throttle and steering control.

## Installation

1. Install Isaac Lab following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. Install the extension in editable mode:

    ```bash
    python -m pip install -e source/leatherbacknav
    ```

3. Verify installation:

    ```bash
    python scripts/list_envs.py
    ```

## Usage

### Training

```bash
python scripts/rsl_rl/train.py --task=Nepher-Leatherback-WaypointNav-v0
```

### Playing/Testing

```bash
python scripts/rsl_rl/play.py --task=Nepher-Leatherback-WaypointNav-Play-v0 --checkpoint=/path/to/checkpoint.pt
```

### Testing with Random Actions

```bash
python scripts/random_agent.py --task=Nepher-Leatherback-WaypointNav-v0
```

## envs-nav Integration

This project integrates with the [envs-nav](../envs-nav/) framework, providing standardized navigation environments with predefined terrains, obstacles, and waypoint configurations.

### Usage

**Training:**
```bash
python scripts/rsl_rl/train.py --task=Nepher-Leatherback-WaypointNav-Envs-v0
```

**Playing/Testing:**
```bash
python scripts/rsl_rl/play.py --task=Nepher-Leatherback-WaypointNav-Envs-Play-v0 --checkpoint=/path/to/checkpoint.pt
```

**Customizing scenes:**
```python
from leatherbacknav.tasks.manager_based.waypoint_nav.waypoint_nav_env_cfg_envs_nav import WaypointNavEnvCfg_EnvsNav

cfg = WaypointNavEnvCfg_EnvsNav(nav_scene=1)  # Use scene 1
env = gym.make("Nepher-Leatherback-WaypointNav-Envs-v0", cfg=cfg)
```

## Environment Details

The Leatherback is a 4-wheeled rover-style robot with:
- **Throttle control**: Velocity control for all 4 wheels
- **Steering control**: Position control for front wheel knuckles (Ackermann-style steering)

The action space is 2D: `[throttle, steering]`

See the configuration files in `source/leatherbacknav/leatherbacknav/tasks/manager_based/waypoint_nav/` for full details including observations, actions, rewards, and termination conditions.

