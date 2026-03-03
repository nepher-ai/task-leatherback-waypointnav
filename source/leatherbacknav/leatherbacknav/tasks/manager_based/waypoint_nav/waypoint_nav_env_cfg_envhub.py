# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for using envhub (nepher) waypoint-benchmark-v1 environment.

This config is self-contained and automatically loads the waypoint-benchmark-v1
environment with terrain and waypoints from the preset configuration.

Usage:
    from leatherbacknav.tasks.manager_based.waypoint_nav.waypoint_nav_env_cfg_envhub import WaypointNavEnvCfg_Envhub
    
    cfg = WaypointNavEnvCfg_Envhub()
    env = gym.make("Nepher-Leatherback-WaypointNav-Envhub-v0", cfg=cfg)
    
    # Or customize the scene:
    cfg = WaypointNavEnvCfg_Envhub(scene_id=1)  # Use different scenario
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
    """Build scene with obstacles from envhub (nepher) preset config.
    
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
class WaypointNavEnvCfg_Envhub(WaypointNavEnvCfg):
    """Self-contained config for envhub (nepher) waypoint-benchmark-v1.
    
    Automatically loads terrain, waypoints, and robot reset positions from preset.
    """
    
    env_id: str = "waypoint-benchmark-v1"
    scene_id: str | int = 0  # Scene ID (separate from base class 'scene' field which is the scene config object)
    _scene_cfg: Any = None
    
    def __post_init__(self):
        # Store scene_id before calling super() to avoid conflict with base class 'scene' field
        scene_id_value = self.scene_id
        super().__post_init__()
        # Restore scene_id after super() may have overwritten it
        self.scene_id = scene_id_value
        self._load_scene()
        self.commands.waypoints.use_envs_nav_waypoints = True
        self.commands.waypoints.num_waypoints = self._scene_cfg.max_num_waypoints if self._scene_cfg else 5
    
    def _load_scene(self):
        """Load envhub (nepher) scene and configure environment."""
        if not self.env_id or self._scene_cfg:
            return
        
        from nepher import load_env, load_scene
        
        # Load environment from cache (will raise error if not cached)
        env = load_env(self.env_id, category="navigation")
        # Load scene config using scene_id (not self.scene which is the scene config object)
        self._scene_cfg = load_scene(env, self.scene_id, category="navigation")
        preset_cfg = self._scene_cfg
        # Now build the actual scene config object and assign it to self.scene
        self.scene = build_scene_with_preset(self.scene, preset_cfg, self.scene.num_envs)
        self.scene.env_spacing = max(self.scene.env_spacing, preset_cfg.env_spacing)
        
        if hasattr(preset_cfg, "max_episode_length_s"):
            self.episode_length_s = preset_cfg.max_episode_length_s


@configclass
class WaypointNavEnvCfg_Envhub_PLAY(WaypointNavEnvCfg_Envhub):
    """Play/evaluation config with fewer envs and no noise."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.0
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 60.0
        self.commands.waypoints.use_envs_nav_waypoints = True

