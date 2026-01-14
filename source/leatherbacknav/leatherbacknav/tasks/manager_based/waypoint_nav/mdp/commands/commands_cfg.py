# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for waypoint command generator."""

from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .waypoint_command import WaypointCommand


WAYPOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Waypoints",
    markers={
        "marker0": sim_utils.SphereCfg(  # Current target (red)
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker1": sim_utils.SphereCfg(  # Future targets (green)
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)


@dataclass
class SpacingScenarioCfg:
    """Spacing scenario: distance range between consecutive waypoints and sampling weight."""
    
    spacing_range: tuple[float, float] = (0.5, 1.0)
    weight: float = 1.0


@configclass
class WaypointCommandCfg(CommandTermCfg):
    """Configuration for waypoint command generator.
    
    Samples 2D waypoints (x, y) for navigation. Z-axis set to ground level.
    Supports multiple spacing scenarios for diverse training.
    """

    class_type: type = WaypointCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    num_waypoints: int = 5
    """Number of waypoints to generate per episode."""
    
    # Legacy spacing config (used as fallback if spacing_scenarios is empty)
    waypoint_spacing: tuple[float, float] = (0.5, 3.0)
    """Default waypoint spacing range. Used when spacing_scenarios is empty."""
    
    initial_waypoint_distance: tuple[float, float] = (0.5, 2.0)
    """Distance range for the first waypoint from the robot's starting position."""
    
    waypoint_reach_threshold: float = 0.25
    """Distance threshold (meters) to consider a waypoint reached."""
    
    num_lookahead_waypoints: int = 1
    """Number of future waypoints to include in the observation."""
    
    spacing_scenarios: list[SpacingScenarioCfg] = field(default_factory=lambda: [
        SpacingScenarioCfg(spacing_range=(0.5, 1.0), weight=1.0),
        SpacingScenarioCfg(spacing_range=(1.0, 2.0), weight=1.5),
        SpacingScenarioCfg(spacing_range=(2.0, 3.0), weight=1.0),
        SpacingScenarioCfg(spacing_range=(3.0, 5.0), weight=0.5),
        SpacingScenarioCfg(spacing_range=(5.5, 8.0), weight=100.0),
    ])
    """Spacing scenarios for diverse training (0.5-5.0m range). Empty list uses legacy waypoint_spacing."""
    
    per_waypoint_spacing: bool = True
    """If True, each waypoint samples its scenario independently. If False, one scenario per episode."""
    
    use_envs_nav_waypoints: bool = False
    """If True, use waypoints from envhub (nepher) scene config (gen_waypoints method) instead of random sampling.
    Requires the environment to have _scene_cfg with gen_waypoints method (e.g., waypoint-benchmark-v1)."""

    @configclass
    class Ranges:
        """Ranges for waypoint generation."""
        
        angle_range: tuple[float, float] = (-2.5, 2.5)
        """Angle range (radians) for next waypoint direction relative to heading."""
        
        z_offset_range: tuple[float, float] = (0.0, 0.0)
        """Z offset range (meters) relative to robot base height."""

    ranges: Ranges = Ranges()

    waypoint_visualizer_cfg: VisualizationMarkersCfg = WAYPOINT_MARKER_CFG
    """Waypoint visualization: red for current target, green for future waypoints."""

