# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation compatibility wrapper for wheeled-waypoint-eval framework.

This module provides a wrapper that exposes the manager-based environment's
internal state in a format compatible with the wheeled-waypoint-eval adapter.

Usage:
    from leatherbacknav.tasks.manager_based.waypoint_nav.eval_compat import EvalCompatEnv
    
    # Wrap your environment
    env = gym.make("Nepher-Leatherback-WaypointNav-v0", cfg=env_cfg)
    eval_env = EvalCompatEnv(env)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EvalCompatEnv:
    """Wrapper that adds evaluation framework compatibility attributes.
    
    This wrapper exposes internal state from the manager-based environment
    in the format expected by the wheeled-waypoint-eval LeatherbackAdapter.
    
    The adapter expects these attributes on env.unwrapped:
        - _target_positions: World-frame waypoint positions
        - _markers_pos: Marker positions for visualization
        - _target_index: Current waypoint index per environment
        - _num_waypoints: Number of waypoints per environment
        - _num_goals: Total waypoint capacity
        - task_completed: Success flag per environment
        - leatherback: Robot articulation reference
        - waypoints: Visualization markers
        - device: Torch device
    """
    
    def __init__(self, env: "ManagerBasedRLEnv"):
        """Initialize the evaluation compatibility wrapper.
        
        Args:
            env: The manager-based RL environment to wrap.
        """
        self._env = env
        self._waypoint_command = None
        # Store reference to self on the underlying env for adapter discovery
        self._env.unwrapped._eval_compat_wrapper = self
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped environment."""
        return getattr(self._env, name)
    
    @property
    def unwrapped(self):
        """Return the actual unwrapped environment (for RslRlVecEnvWrapper compatibility)."""
        return self._env.unwrapped
    
    @property
    def _waypoint_term(self):
        """Lazily get the waypoint command term."""
        if self._waypoint_command is None:
            # Try to get the waypoint command from the command manager
            try:
                self._waypoint_command = self._env.unwrapped.command_manager.get_term("waypoints")
            except (AttributeError, KeyError):
                # Fallback: search for waypoint-related command
                for term_name in self._env.unwrapped.command_manager.active_terms:
                    term = self._env.unwrapped.command_manager.get_term(term_name)
                    if hasattr(term, "waypoints_w"):
                        self._waypoint_command = term
                        break
        return self._waypoint_command
    
    # ============== Evaluation Adapter Compatibility Properties ==============
    
    @property
    def _target_positions(self) -> torch.Tensor:
        """World-frame waypoint target positions.
        
        Shape: (num_envs, num_waypoints, 3)
        """
        if self._waypoint_term is not None:
            return self._waypoint_term.waypoints_w
        return torch.zeros(self._env.num_envs, 1, 3, device=self.device)
    
    @_target_positions.setter
    def _target_positions(self, value: torch.Tensor):
        """Allow setting target positions for evaluation scenarios."""
        if self._waypoint_term is not None:
            self._waypoint_term.waypoints_w[:] = value
    
    @property
    def _markers_pos(self) -> torch.Tensor:
        """Marker positions for waypoint visualization.
        
        Shape: (num_envs, num_waypoints, 3)
        """
        if self._waypoint_term is not None:
            return self._waypoint_term._markers_pos
        return torch.zeros(self._env.num_envs, 1, 3, device=self.device)
    
    @_markers_pos.setter
    def _markers_pos(self, value: torch.Tensor):
        """Allow setting marker positions for visualization."""
        if self._waypoint_term is not None:
            self._waypoint_term._markers_pos[:] = value
    
    @property
    def _target_index(self) -> torch.Tensor:
        """Current target waypoint index for each environment.
        
        Shape: (num_envs,)
        """
        if self._waypoint_term is not None:
            return self._waypoint_term.current_waypoint_idx
        return torch.zeros(self._env.num_envs, dtype=torch.long, device=self.device)
    
    @_target_index.setter
    def _target_index(self, value: torch.Tensor):
        """Allow setting target index for evaluation scenarios."""
        if self._waypoint_term is not None:
            self._waypoint_term.current_waypoint_idx[:] = value
    
    @property
    def _num_waypoints(self) -> torch.Tensor:
        """Number of active waypoints per environment.
        
        Shape: (num_envs,)
        Returns the per-environment waypoint count from the waypoint command term.
        """
        if self._waypoint_term is not None:
            # Return the per-environment waypoint count tensor
            return self._waypoint_term.num_waypoints_per_env
        return torch.ones(self._env.num_envs, dtype=torch.long, device=self.device)
    
    @_num_waypoints.setter
    def _num_waypoints(self, value: torch.Tensor | int):
        """Set per-environment waypoint counts (for variable waypoint scenarios).
        
        Args:
            value: Either a tensor of per-env counts, or a single int to set for all envs.
        """
        if self._waypoint_term is not None:
            if isinstance(value, int):
                self._waypoint_term.num_waypoints_per_env[:] = value
            else:
                self._waypoint_term.num_waypoints_per_env[:] = value
    
    @property
    def _num_goals(self) -> int:
        """Total waypoint capacity (maximum number of waypoints supported)."""
        if self._waypoint_term is not None:
            return self._waypoint_term.cfg.num_waypoints
        return 1
    
    @property
    def task_completed(self) -> torch.Tensor:
        """Flag indicating task completion (all waypoints reached) per environment.
        
        Shape: (num_envs,)
        """
        if self._waypoint_term is not None:
            return self._waypoint_term.all_waypoints_reached
        return torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
    
    @property
    def leatherback(self):
        """Robot articulation reference (aliased for adapter compatibility).
        
        The adapter expects 'leatherback' but our environment uses 'robot'.
        """
        return self._env.unwrapped.scene["robot"]
    
    @property
    def waypoints(self):
        """Waypoint visualization markers."""
        if self._waypoint_term is not None and hasattr(self._waypoint_term, "waypoint_visualizer"):
            return self._waypoint_term.waypoint_visualizer
        return None
    
    @property
    def device(self) -> torch.device:
        """Torch device for the environment."""
        return self._env.unwrapped.device
    
    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._env.unwrapped.num_envs
    
    @property
    def step_dt(self) -> float:
        """Simulation time per environment step."""
        return self._env.unwrapped.step_dt
    
    @property
    def physics_dt(self) -> float:
        """Physics simulation timestep."""
        return self._env.unwrapped.physics_dt
    
    @property
    def cfg(self):
        """Environment configuration."""
        return self._env.unwrapped.cfg
    
    # ============== Environment Interface Methods ==============
    
    def reset(self, *args, **kwargs):
        """Reset the environment."""
        result = self._env.reset(*args, **kwargs)
        # Clear cached waypoint term to re-fetch after reset
        self._waypoint_command = None
        return result
    
    def step(self, action):
        """Step the environment."""
        return self._env.step(action)
    
    def close(self):
        """Close the environment."""
        return self._env.close()
    
    def render(self, *args, **kwargs):
        """Render the environment."""
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)


def wrap_for_eval(env: "ManagerBasedRLEnv") -> EvalCompatEnv:
    """Convenience function to wrap an environment for evaluation.
    
    Args:
        env: The manager-based RL environment to wrap.
        
    Returns:
        Wrapped environment with evaluation compatibility attributes.
        
    Example:
        import gymnasium as gym
        from leatherbacknav.tasks.manager_based.waypoint_nav.eval_compat import wrap_for_eval
        
        env = gym.make("Nepher-Leatherback-WaypointNav-v0", cfg=env_cfg)
        eval_env = wrap_for_eval(env)
        
        # Now eval_env is compatible with wheeled-waypoint-eval
    """
    return EvalCompatEnv(env)

