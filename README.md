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

## Evaluation with wheeled-waypoint-eval

This project is compatible with the [wheeled-waypoint-eval](../wheeled-waypoint-eval/) framework for standardized evaluation and benchmarking.

### Quick Start

```python
import gymnasium as gym
from leatherbacknav import wrap_for_eval
from wheeled_waypoint_eval import WaypointEvaluator
from wheeled_waypoint_eval.adapters import LeatherbackAdapter
from wheeled_waypoint_eval.scorers import get_scorer

# Create environment
env = gym.make("Nepher-Leatherback-WaypointNav-v0", cfg=env_cfg)

# Wrap for evaluation compatibility
eval_env = wrap_for_eval(env)

# Setup evaluator
adapter = LeatherbackAdapter()
scorers = [get_scorer("v1"), get_scorer("v2")]
evaluator = WaypointEvaluator(adapter=adapter, scorers=scorers)

# Run evaluation
results = evaluator.evaluate(
    env=eval_env,
    policy=policy,
    waypoint_config="path/to/waypoints.json",
    output_dir="eval_results/",
)
```

### Command-Line Evaluation

```bash
# From the wheeled-waypoint-eval directory
python scripts/evaluate.py \
    --task Nepher-Leatherback-WaypointNav-v0 \
    --checkpoint path/to/model.pt \
    --waypoints configs/sample_waypoints.json \
    --scorers v1 v2 \
    --output-dir results/
```

**Note**: The `wrap_for_eval()` function adapts the manager-based environment to expose the internal state attributes that the evaluation adapter expects. This is necessary because the evaluation framework was originally designed for direct-style environments with different attribute naming conventions.

## Environment Details

The Leatherback is a 4-wheeled rover-style robot with:
- **Throttle control**: Velocity control for all 4 wheels
- **Steering control**: Position control for front wheel knuckles (Ackermann-style steering)

The action space is 2D: `[throttle, steering]`

See `waypoint_nav_env_cfg.py` for the full configuration including observations, actions, rewards, and termination conditions.

