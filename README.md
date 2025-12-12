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

### Evaluation

```bash
python scripts/rsl_rl/play.py --task=Nepher-Leatherback-WaypointNav-Play-v0 --checkpoint=/path/to/checkpoint.pt
```

### Testing with Random Actions

```bash
python scripts/random_agent.py --task=Nepher-Leatherback-WaypointNav-v0
```

## Environment Details

The Leatherback is a 4-wheeled rover-style robot with:
- **Throttle control**: Velocity control for all 4 wheels
- **Steering control**: Position control for front wheel knuckles (Ackermann-style steering)

The action space is 2D: `[throttle, steering]`

See `waypoint_nav_env_cfg.py` for the full configuration including observations, actions, rewards, and termination conditions.

