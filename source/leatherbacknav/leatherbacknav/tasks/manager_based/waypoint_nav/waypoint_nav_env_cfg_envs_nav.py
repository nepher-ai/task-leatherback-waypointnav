# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for using envs-nav waypoint-benchmark-v1 environment.

This config is self-contained and automatically loads the waypoint-benchmark-v1
environment with terrain and waypoints from the preset configuration.

Usage:
    from leatherbacknav.tasks.manager_based.waypoint_nav.waypoint_nav_env_cfg_envs_nav import WaypointNavEnvCfg_EnvsNav
    
    cfg = WaypointNavEnvCfg_EnvsNav()
    env = gym.make("Nepher-Leatherback-WaypointNav-v0", cfg=cfg)
    
    # Or customize the scene:
    cfg = WaypointNavEnvCfg_EnvsNav(nav_scene=1)  # Use different scenario
"""

from __future__ import annotations

from typing import Any

from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .waypoint_nav_env_cfg import WaypointNavEnvCfg, WaypointNavSceneCfg


def _create_scene_class(terrain_cfg: Any, name: str, base_class: type = WaypointNavSceneCfg, **attrs) -> type:
    """Create a dynamic scene configuration class with terrain and additional attributes.
    
    Args:
        terrain_cfg: Terrain configuration to use.
        name: Name for the dynamically created class.
        base_class: Base class to inherit from.
        **attrs: Additional attributes to add to the class (e.g., obstacles, lights).
    
    Returns:
        A dynamically created configclass with terrain and attributes.
    """
    class_attrs = {"__annotations__": {}, "terrain": terrain_cfg, **attrs}
    for attr_name in attrs:
        class_attrs["__annotations__"][attr_name] = AssetBaseCfg
    return configclass(type(name, (base_class,), class_attrs))


def build_scene_with_preset(base_scene: WaypointNavSceneCfg, preset_cfg: Any, num_envs: int) -> WaypointNavSceneCfg:
    """Build scene with obstacles from envs-nav preset config.
    
    Args:
        base_scene: Base scene configuration to extend.
        preset_cfg: Preset configuration with terrain, obstacles, and lights.
        num_envs: Number of parallel environments.
    
    Returns:
        Scene configuration with terrain, obstacles, and lights from preset.
    """
    obstacle_cfgs, light_cfgs = preset_cfg.get_obstacle_cfgs(), preset_cfg.get_light_cfgs()
    
    # Build dynamic scene class with obstacles and lights as attributes
    attrs = {}
    for cfgs in (obstacle_cfgs, light_cfgs):
        attrs.update({name.lower().replace("-", "_"): cfg for name, cfg in cfgs.items()})
    
    SceneCfg = _create_scene_class(preset_cfg.get_terrain_cfg(), "PresetNavSceneCfg", **attrs)
    scene = SceneCfg(num_envs=num_envs, env_spacing=preset_cfg.env_spacing)
    
    return scene


@configclass
class WaypointNavEnvCfg_EnvsNav(WaypointNavEnvCfg):
    """Self-contained config for envs-nav waypoint-benchmark-v1.
    
    Automatically loads terrain, waypoints, and robot reset positions from preset.
    """
    
    nav_env_id: str = "waypoint-benchmark-v1"
    nav_scene: str | int = 0
    _scene_cfg: Any = None
    
    def __post_init__(self):
        super().__post_init__()
        self._load_scene()
        self.commands.waypoints.use_envs_nav_waypoints = True
        self.commands.waypoints.num_waypoints = self._scene_cfg.max_num_waypoints if self._scene_cfg else 5
    
    def _load_scene(self):
        """Load envs-nav scene and configure environment."""
        if not self.nav_env_id or self._scene_cfg:
            return
        
        from envs_nav.lib.loader import load_cached_scene
        self._scene_cfg = load_cached_scene(self.nav_env_id, self.nav_scene)
        preset_cfg = self._scene_cfg
        self.scene = build_scene_with_preset(self.scene, preset_cfg, self.scene.num_envs)
        self.scene.env_spacing = max(self.scene.env_spacing, preset_cfg.env_spacing)
        
        if hasattr(preset_cfg, "max_episode_length_s"):
            self.episode_length_s = preset_cfg.max_episode_length_s


@configclass
class WaypointNavEnvCfg_EnvsNav_PLAY(WaypointNavEnvCfg_EnvsNav):
    """Play/evaluation config with fewer envs and no noise."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.0
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 60.0
        self.commands.waypoints.use_envs_nav_waypoints = True


