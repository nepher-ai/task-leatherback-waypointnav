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
            radius=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker1": sim_utils.SphereCfg(  # Future targets (green)
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)


@dataclass
class SpacingScenarioCfg:
    """Configuration for a single spacing scenario.
    
    Each scenario defines a range of distances between consecutive waypoints
    and a weight for sampling probability.
    
    Args:
        spacing_range: Min and max distance (in meters) between consecutive waypoints.
        weight: Relative weight for sampling this scenario. Higher weight = more likely.
    """
    
    spacing_range: tuple[float, float] = (0.5, 1.0)
    weight: float = 1.0


@configclass
class WaypointCommandCfg(CommandTermCfg):
    """Configuration for the waypoint command generator.
    
    This command generator samples a sequence of 2D waypoints (x, y) within a 
    configurable radius and provides them to the policy for navigation.
    
    The z-axis is included for compatibility but set to ground level for 
    wheeled robots.
    
    Spacing Scenarios:
        The generator supports multiple spacing scenarios to create diverse training
        environments. Each scenario defines a spacing range and sampling weight.
        
        - When `per_waypoint_spacing=True`: Each waypoint independently samples a 
          scenario, creating natural mixed environments (some close, some far).
        - When `per_waypoint_spacing=False`: Each environment samples one scenario 
          applied to all its waypoints (uniform spacing within each episode).
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
    """Distance threshold (in meters) to consider a waypoint as reached.
    Set slightly larger for wheeled robots due to their larger turning radius."""
    
    num_lookahead_waypoints: int = 1
    """Number of future waypoints to include in the observation."""
    
    # Dynamic spacing configuration
    spacing_scenarios: list[SpacingScenarioCfg] = field(default_factory=lambda: [
        SpacingScenarioCfg(spacing_range=(0.5, 1.0), weight=1.0),     # Close: tight navigation
        SpacingScenarioCfg(spacing_range=(1.0, 2.0), weight=1.5),     # Medium: balanced
        SpacingScenarioCfg(spacing_range=(2.0, 3.0), weight=1.0),     # Far: longer distances
        SpacingScenarioCfg(spacing_range=(3.0, 5.0), weight=0.5),     # Very Far: challenging distance
        SpacingScenarioCfg(spacing_range=(0.5, 3.0), weight=1.5),     # Mixed: close-to-far range
    ])
    """List of spacing scenarios for diverse training environments.
    
    Default scenarios cover a wide range (0.5m - 5.0m) suitable for wheeled robots:
    - Close (0.5-1.0m): Tests precise maneuvering
    - Medium (1.0-2.0m): Balanced navigation challenges
    - Far (2.0-3.0m): Longer traversal between waypoints
    - Very Far (3.0-5.0m): Tests sustained driving
    - Mixed (0.5-3.0m): Combines close and far in a single range
    
    Set to empty list [] to use legacy waypoint_spacing instead.
    """
    
    per_waypoint_spacing: bool = True
    """Whether to sample spacing scenario independently for each waypoint.
    
    - True: Each waypoint samples its own scenario, creating mixed environments
            where some waypoints are close and others are far apart.
    - False: Each environment samples one scenario at reset, applied uniformly
             to all waypoints in that episode.
    """

    @configclass
    class Ranges:
        """Ranges for waypoint generation."""
        
        angle_range: tuple[float, float] = (-2.5, 2.5)
        """Angle range (in radians) for sampling next waypoint direction relative to heading. 
        Slightly restricted from full circle to encourage more forward-facing waypoints
        (better for wheeled vehicles with limited turning radius)."""
        
        z_offset_range: tuple[float, float] = (0.0, 0.0)
        """Z offset range (in meters) relative to robot base height. 
        Set to zero for flat ground navigation."""

    ranges: Ranges = Ranges()

    # Visualization settings - colored spheres
    waypoint_visualizer_cfg: VisualizationMarkersCfg = WAYPOINT_MARKER_CFG
    """The configuration for waypoint visualization markers.
    Uses red spheres for current target and green spheres for future waypoints."""

