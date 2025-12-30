# Copyright (c) 2025, Nepher Team. SPDX-License-Identifier: BSD-3-Clause
"""Navigation environment with envs-nav integration (preset scenes)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .waypoint_nav_env_cfg import WaypointNavEnvCfg, WaypointNavSceneCfg

logger = logging.getLogger(__name__)


def _create_scene_class(terrain_cfg: Any, name: str, base_class: type = WaypointNavSceneCfg, **attrs) -> type:
    """Create a dynamic scene configuration class with terrain and additional attributes."""
    class_attrs = {"__annotations__": {}, "terrain": terrain_cfg, **attrs}
    for attr_name in attrs:
        class_attrs["__annotations__"][attr_name] = AssetBaseCfg
    return configclass(type(name, (base_class,), class_attrs))


def build_scene_with_preset(base_scene: WaypointNavSceneCfg, preset_cfg: Any, num_envs: int = 512) -> WaypointNavSceneCfg:
    """Build scene with obstacles from envs-nav preset config."""
    obstacle_cfgs, light_cfgs = preset_cfg.get_obstacle_cfgs(), preset_cfg.get_light_cfgs()
    
    # Build dynamic scene class with obstacles and lights as attributes
    attrs = {}
    for cfgs in (obstacle_cfgs, light_cfgs):
        attrs.update({name.lower().replace("-", "_"): cfg for name, cfg in cfgs.items()})
    
    SceneCfg = _create_scene_class(preset_cfg.get_terrain_cfg(), "PresetNavSceneCfg", **attrs)
    scene = SceneCfg(num_envs=num_envs, env_spacing=preset_cfg.env_spacing)
    
    # Update lidar mesh paths for obstacles
    if obstacle_cfgs:
        scene.front_lidar.mesh_prim_paths = ["/World/ground"] + [
            f"{{ENV_REGEX_NS}}/Obstacle{i+1}" for i in range(len(obstacle_cfgs))
        ]
        logger.info(f"Scene built with {len(obstacle_cfgs)} obstacles")
    
    return scene


class NavEnv(ManagerBasedRLEnv):
    """Navigation environment with envs-nav integration (preset scenes)."""
    
    cfg: "NavEnvCfg"
    
    def __init__(self, cfg: "NavEnvCfg", render_mode: str | None = None, **kwargs):
        num_envs, device = cfg.scene.num_envs, cfg.sim.device
        self.goal_pos_w = torch.zeros((num_envs, 3), device=device)
        self._goal_just_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._scene_cfg = getattr(cfg, "_scene_cfg", None)
        self._goal_x_range, self._goal_y_range = cfg.goal_x_range, cfg.goal_y_range
        self._goal_x_delta = self._goal_x_range[1] - self._goal_x_range[0]
        self._goal_y_delta = self._goal_y_range[1] - self._goal_y_range[0]
        
        super().__init__(cfg, render_mode, **kwargs)
        self._reset_goals(torch.arange(self.num_envs, device=self.device))
    
    def _reset_goals(self, env_ids: torch.Tensor):
        """Reset goal positions using scene config or fallback."""
        if self._scene_cfg and hasattr(self._scene_cfg, "gen_goal_random_pos"):
            try:
                self.goal_pos_w[env_ids] = self._scene_cfg.gen_goal_random_pos(
                    env_ids=env_ids, env_origins=self.scene.env_origins, device=self.device
                )
            except Exception as e:
                logger.warning(f"Scene goal generation failed: {e}")
                self._reset_goals_fallback(env_ids)
        else:
            self._reset_goals_fallback(env_ids)
        self._goal_just_reset_mask[env_ids] = True
    
    def _reset_goals_fallback(self, env_ids: torch.Tensor):
        """Fallback goal generation using configured ranges."""
        goal_slice = self.goal_pos_w[env_ids]
        origins = self.scene.env_origins[env_ids]
        
        torch.rand(len(env_ids), device=self.device, out=goal_slice[:, 0])
        goal_slice[:, 0].mul_(self._goal_x_delta).add_(self._goal_x_range[0]).add_(origins[:, 0])
        
        torch.rand(len(env_ids), device=self.device, out=goal_slice[:, 1])
        goal_slice[:, 1].mul_(self._goal_y_delta).add_(self._goal_y_range[0]).add_(origins[:, 1])
        
        goal_slice[:, 2] = 0.0
    
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._reset_goals(env_ids)


def reset_robot_from_scene(env: NavEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> None:
    """Reset robot position using scene config's gen_bot_init_pos() if available."""
    asset = env.scene[asset_cfg.name]
    scene_cfg = getattr(env.cfg, "_scene_cfg", None)
    root_state = asset.data.default_root_state[env_ids].clone()
    
    if scene_cfg and hasattr(scene_cfg, "gen_bot_init_pos"):
        try:
            positions, yaws = scene_cfg.gen_bot_init_pos(
                env_ids=env_ids, env_origins=env.scene.env_origins, device=env.device
            )
            root_state[:, :3].copy_(positions)
            
            # Convert yaw to quaternion (z-rotation)
            half_yaws = yaws * 0.5
            root_state[:, 3] = torch.cos(half_yaws)
            root_state[:, 4:6] = 0.0
            root_state[:, 6] = torch.sin(half_yaws)
            root_state[:, 7:13] = 0.0  # Zero velocities
        except Exception as e:
            logger.warning(f"Scene robot reset failed: {e}")
    
    asset.write_root_state_to_sim(root_state, env_ids)


@configclass 
class NavEnvCfg(WaypointNavEnvCfg):
    """Navigation environment config with envs-nav integration (preset scenes)."""
    
    class_type = NavEnv
    nav_env_id: str | None = None
    nav_scene: str | int | None = None
    _scene_cfg: Any = None
    
    def load_scene(self):
        """Load envs-nav scene (preset). Call AFTER setting nav_env_id and nav_scene."""
        if not self.nav_env_id or self._scene_cfg:
            return
        
        from envs_nav.lib.loader import load_cached_scene
        self._scene_cfg = load_cached_scene(self.nav_env_id, self.nav_scene)
        
        self._load_preset_scene()
        
        # Set robot reset event
        self.events.reset_base = EventTerm(
            func=reset_robot_from_scene, mode="reset", params={"asset_cfg": SceneEntityCfg("robot")}
        )
    
    def _load_preset_scene(self):
        """Load and configure a preset-based scene."""
        preset_cfg = self._scene_cfg
        
        env_spacing = max(self.scene.env_spacing, preset_cfg.env_spacing)
        self.scene = build_scene_with_preset(self.scene, preset_cfg, self.scene.num_envs)
        self.scene.env_spacing = env_spacing
        
        # Update goal ranges from playground bounds
        if playground := getattr(preset_cfg, "playground", None):
            margin = getattr(preset_cfg, "robot_safety_margin", 0.3)
            self.goal_x_range = (playground[0] + margin, playground[2] - margin)
            self.goal_y_range = (playground[1] + margin, playground[3] - margin)
        
        if hasattr(preset_cfg, "max_episode_length_s"):
            self.episode_length_s = preset_cfg.max_episode_length_s
        
        logger.info(f"Loaded preset: {preset_cfg.name} ({len(getattr(preset_cfg, 'obstacles', []))} obstacles)")


@configclass
class NavEnvCfg_PLAY(NavEnvCfg):
    """Play/evaluation config with fewer envs and no noise."""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs, self.scene.env_spacing = 16, 100.0
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 30.0

