# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation compatibility wrapper for waypoint navigation environments.

This module provides a wrapper that exposes the manager-based environment's
internal state in a format compatible with evaluation frameworks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EvalCompatEnv:
    """Wrapper that exposes environment state for evaluation frameworks."""
    
    def __init__(self, env: "ManagerBasedRLEnv"):
        self._env = env
        self._waypoint_term = None
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped environment."""
        return getattr(self._env, name)
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self._env.unwrapped
    
    def _get_waypoint_term(self):
        """Get the waypoint command term."""
        if self._waypoint_term is None:
            try:
                cmd_manager = self._env.unwrapped.command_manager
                self._waypoint_term = cmd_manager.get_term("waypoints")
            except (AttributeError, KeyError):
                for term_name in cmd_manager.active_terms:
                    term = cmd_manager.get_term(term_name)
                    if hasattr(term, "waypoints_w"):
                        self._waypoint_term = term
                        break
        return self._waypoint_term
    
    def _get_attr(self, attr_name: str, default_factory):
        """Generic attribute getter with fallback."""
        term = self._get_waypoint_term()
        if term is not None and hasattr(term, attr_name):
            return getattr(term, attr_name)
        return default_factory()
    
    @property
    def _target_positions(self) -> torch.Tensor:
        """World-frame waypoint positions."""
        return self._get_attr("waypoints_w", lambda: torch.zeros(self._env.num_envs, 1, 3, device=self.device))
    
    @property
    def _markers_pos(self) -> torch.Tensor:
        """Marker positions for visualization."""
        return self._get_attr("_markers_pos", lambda: torch.zeros(self._env.num_envs, 1, 3, device=self.device))
    
    @property
    def _target_index(self) -> torch.Tensor:
        """Current waypoint index per environment."""
        return self._get_attr("current_waypoint_idx", lambda: torch.zeros(self._env.num_envs, dtype=torch.long, device=self.device))
    
    @property
    def _num_waypoints(self) -> torch.Tensor:
        """Number of waypoints per environment."""
        return self._get_attr("num_waypoints_per_env", lambda: torch.ones(self._env.num_envs, dtype=torch.long, device=self.device))
    
    @property
    def _num_goals(self) -> int:
        """Total waypoint capacity."""
        term = self._get_waypoint_term()
        return term.cfg.num_waypoints if term is not None else 1
    
    @property
    def task_completed(self) -> torch.Tensor:
        """Task completion flag per environment."""
        return self._get_attr("all_waypoints_reached", lambda: torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device))
    
    @property
    def task_failed(self) -> torch.Tensor:
        """Task failure flag per environment.
        
        Returns True if any failure condition is met (timeout, flipped_over, etc.)
        but task was not completed. This excludes success conditions.
        """
        failure = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
        
        try:
            term_manager = self._env.unwrapped.termination_manager
            if term_manager is not None:
                # Check for failure conditions (excluding success)
                failure_terms = ["time_out", "flipped_over"]
                for term_name in failure_terms:
                    try:
                        term_value = term_manager.get_term(term_name)
                        failure = failure | term_value
                    except (KeyError, AttributeError):
                        # Term doesn't exist, skip it
                        pass
        except (AttributeError, KeyError):
            # Termination manager not available, return zeros
            pass
        
        return failure
    
    @property
    def robot(self):
        """Robot articulation reference."""
        return self._env.unwrapped.scene["robot"]
    
    @property
    def waypoints(self):
        """Waypoint visualization markers."""
        term = self._get_waypoint_term()
        return getattr(term, "waypoint_visualizer", None) if term is not None else None
    
    @property
    def device(self) -> torch.device:
        """Torch device."""
        return self._env.unwrapped.device
    
    def reset(self, *args, **kwargs):
        """Reset the environment."""
        self._waypoint_term = None
        return self._env.reset(*args, **kwargs)
    
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
    
    def _log_state(self, env_idx: int | None = None, info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract complete robot state for logging.
        
        This method centralizes all state extraction logic for the evaluation framework.
        It extracts robot position/orientation, waypoints, goals, and additional info.
        
        Args:
            env_idx: Environment index (for vectorized envs). If None, uses env 0.
            info: Info dictionary from environment step.
            
        Returns:
            Dictionary containing all logged state data:
            - position: [x, y, z] in world frame
            - quat_w: quaternion w component
            - current_waypoint_idx: current waypoint index
            - all_waypoints_reached: whether all waypoints are reached
            - task_completed: task completion flag (all waypoints reached)
            - task_failed: task failure flag (timeout, flipped_over, etc.)
            - success, timeout: from info dict (if available)
        """
        state = {}
        idx = env_idx if env_idx is not None else 0
        
        try:
            # Extract robot position and orientation
            robot = self.robot
            if robot is not None:
                pos_w = robot.data.root_pos_w
                quat_w = robot.data.root_quat_w
                
                # Optimize: single indexing operation, batch CPU conversion
                if torch.is_tensor(pos_w):
                    state["position"] = pos_w[idx, :3].cpu().numpy()
                else:
                    state["position"] = pos_w[idx, :3]
                
                quat_w_val = quat_w[idx] if torch.is_tensor(quat_w) else quat_w
                if torch.is_tensor(quat_w_val):
                    state["quat_w"] = float(quat_w_val[0].cpu().item())
                else:
                    state["quat_w"] = float(quat_w_val[0])
            
            # Optimize: get waypoint term once and reuse
            term = self._get_waypoint_term()
            if term is not None:
                # Extract waypoint data directly (avoid redundant method call)
                if hasattr(term, "current_waypoint_idx"):
                    current_idx = term.current_waypoint_idx
                    if torch.is_tensor(current_idx):
                        state["current_waypoint_idx"] = int(current_idx[idx].cpu().item())
                    else:
                        state["current_waypoint_idx"] = int(current_idx[idx])
                
                if hasattr(term, "all_waypoints_reached"):
                    all_reached = term.all_waypoints_reached
                    if torch.is_tensor(all_reached):
                        if all_reached.numel() == 1:
                            state["all_waypoints_reached"] = bool(all_reached.cpu().item())
                        else:
                            state["all_waypoints_reached"] = bool(all_reached[idx].cpu().item())
                    else:
                        state["all_waypoints_reached"] = bool(all_reached[idx])
            
            # Extract task completion/failure status
            task_completed_val = self.task_completed
            if torch.is_tensor(task_completed_val):
                if task_completed_val.numel() == 1:
                    state["task_completed"] = bool(task_completed_val.cpu().item())
                else:
                    state["task_completed"] = bool(task_completed_val[idx].cpu().item())
            else:
                state["task_completed"] = bool(task_completed_val)
            
            task_failed_val = self.task_failed
            if torch.is_tensor(task_failed_val):
                if task_failed_val.numel() == 1:
                    state["task_failed"] = bool(task_failed_val.cpu().item())
                else:
                    state["task_failed"] = bool(task_failed_val[idx].cpu().item())
            else:
                state["task_failed"] = bool(task_failed_val)
            
            # Extract info fields
            if info:
                for key in ["success", "timeout"]:
                    if key in info:
                        val = info[key]
                        if torch.is_tensor(val):
                            if val.numel() == 1:
                                state[key] = float(val.cpu().item())
                            else:
                                state[key] = float(val[idx].cpu().item())
                        else:
                            state[key] = val
        except Exception:
            pass
        
        return state
    
    
    def _log_metadata(self, env_idx: int | None = None) -> dict[str, Any] | None:
        """Extract metadata information for logging.
        
        Args:
            env_idx: Environment index (for vectorized envs). If None, uses env 0.
            
        Returns:
            Dictionary containing metadata data or None.
        """
        waypoints = self._get_waypoint_pos(env_idx)
        return {"waypoints": waypoints} if waypoints else None

    def _get_waypoint_pos(self, env_idx: int | None = None) -> dict[str, Any] | None:
        """Extract waypoint data for logging.
        
        Args:
            env_idx: Environment index (for vectorized envs). If None, uses env 0.
            
        Returns:
            Dictionary containing waypoint data or None.
        """
        term = self._get_waypoint_term()
        if term is None:
            return None
        
        waypoints_data = {}
        idx = env_idx if env_idx is not None else 0
        
        # Get waypoint positions in world frame
        if hasattr(term, "waypoints_w"):
            waypoints_w = term.waypoints_w
            if torch.is_tensor(waypoints_w):
                waypoints_data["waypoints_world"] = waypoints_w[idx].cpu().numpy()
            else:
                waypoints_data["waypoints_world"] = waypoints_w[idx]
        
        return waypoints_data if waypoints_data else None

def wrap_for_eval(env: "ManagerBasedRLEnv") -> EvalCompatEnv:
    """Wrap an environment for evaluation."""
    return EvalCompatEnv(env)

