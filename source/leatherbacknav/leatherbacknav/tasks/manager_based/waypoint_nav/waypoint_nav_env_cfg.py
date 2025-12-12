# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Leatherback waypoint navigation environment.

This environment trains a policy for a wheeled robot to navigate through
a sequence of 2D waypoints. The robot receives observations about its current state
and the waypoints, and must output throttle and steering commands to navigate.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.rewards import is_terminated_term
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from leatherbacknav.robots import LEATHERBACK_CFG  # isort: skip


##
# Terrain configuration
##

FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=15.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

##
# Scene definition
##


@configclass
class WaypointNavSceneCfg(InteractiveSceneCfg):
    """Configuration for the waypoint navigation scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


# Create waypoint ranges outside the class to avoid it becoming a class attribute
_WAYPOINT_RANGES = mdp.WaypointCommandCfg.Ranges()
_WAYPOINT_RANGES.angle_range = (-2.5, 2.5)  # ~±143 degrees (slightly restricted for wheeled vehicles)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    waypoints = mdp.WaypointCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),  # Effectively never auto-resample
        num_waypoints=5,
        waypoint_spacing=(0.5, 3.0),
        initial_waypoint_distance=(0.5, 2.0),
        waypoint_reach_threshold=0.25,  # Larger threshold for wheeled robots
        num_lookahead_waypoints=1,
        debug_vis=True,
        ranges=_WAYPOINT_RANGES,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    
    The Leatherback uses:
    - Throttle: Velocity control for all 4 wheels (joint velocity action)
    - Steering: Position control for front wheel knuckles (joint position action)
    """

    # Throttle control (wheel velocities)
    throttle = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["Wheel.*"],
        scale=10.0,  # Scale for wheel velocity
    )
    
    # Steering control (front wheel angles)
    steering = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Knuckle__Upright__Front.*"],
        scale=0.5,  # Scale for steering angle
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state observations
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        # Waypoint observations
        waypoint_commands = ObsTerm(
            func=mdp.waypoint_commands,
            params={"command_name": "waypoints"},
        )
        waypoint_distance = ObsTerm(
            func=mdp.waypoint_distance,
            params={"command_name": "waypoints", "asset_cfg": SceneEntityCfg("robot")},
        )
        waypoint_heading = ObsTerm(
            func=mdp.waypoint_heading_error,
            params={"command_name": "waypoints", "asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Primary task rewards: waypoint navigation
    waypoint_reached = RewTerm(
        func=mdp.waypoint_reached_bonus,
        weight=15.0,
        params={"command_name": "waypoints", "bonus": 1.0},
    )
    waypoint_distance = RewTerm(
        func=mdp.waypoint_distance_reward,
        weight=3.0,
        params={"command_name": "waypoints", "std": 2.0},
    )
    waypoint_heading = RewTerm(
        func=mdp.waypoint_heading_reward,
        weight=2.0,
        params={"command_name": "waypoints", "std": 0.5},
    )
    progress = RewTerm(
        func=mdp.progress_reward,
        weight=2.0,
        params={"command_name": "waypoints", "asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalty for flipping over
    flipped_penalty = RewTerm(
        func=is_terminated_term,
        weight=-50.0,
        params={"term_keys": ["flipped_over"]},
    )

    # -- Regularization penalties for smooth driving
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.5,
    )
    base_motion = RewTerm(
        func=mdp.base_motion_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    steering = RewTerm(
        func=mdp.steering_penalty,
        weight=-0.1,
        params={"steering_action_index": 1},  # Assuming [throttle, steering] action order
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Success termination: all waypoints reached
    all_waypoints_reached = DoneTerm(
        func=mdp.all_waypoints_reached,
        params={"command_name": "waypoints"},
    )

    # Failure terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flipped_over = DoneTerm(
        func=mdp.flipped_over,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class WaypointNavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Leatherback waypoint navigation environment."""

    # Scene settings
    scene: WaypointNavSceneCfg = WaypointNavSceneCfg(num_envs=4096, env_spacing=5.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.0, 10.0, 5.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4  # 125 Hz control (faster for wheeled robots)
        self.episode_length_s = 30.0  # Allow enough time to reach all waypoints
        
        # Simulation settings
        self.sim.dt = 0.002  # 500 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class WaypointNavEnvCfg_PLAY(WaypointNavEnvCfg):
    """Configuration for playing/evaluating the trained policy."""

    def __post_init__(self) -> None:
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

        # Disable randomization for play
        self.observations.policy.enable_corruption = False

        # Longer episode for demo
        self.episode_length_s = 60.0

