# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for Leatherback waypoint navigation task.

These event functions handle domain randomization during training.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_action_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str = "vehicle_actions",
    scale_range: tuple[float, float] = (0.8, 1.2),
):
    """Randomize the action scale for vehicle control actions.
    
    This provides domain randomization for the action scale, making the policy
    more robust to different action sensitivities.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to reset (unused, applies globally).
        action_name: Name of the action term to randomize. Defaults to "vehicle_actions".
        scale_range: Min and max scale values. Defaults to (0.8, 1.2).
    """
    # Get the action term
    action_term = env.action_manager.get_term(action_name)
    
    # Sample a new scale uniformly from the range
    new_scale = torch.empty(1, device=env.device).uniform_(*scale_range).item()
    
    # Update the scale
    if hasattr(action_term, "_scale"):
        current_scale = action_term._scale
        if isinstance(current_scale, torch.Tensor):
            action_term._scale = torch.full_like(current_scale, new_scale)
        else:
            # _scale is a float
            action_term._scale = new_scale

