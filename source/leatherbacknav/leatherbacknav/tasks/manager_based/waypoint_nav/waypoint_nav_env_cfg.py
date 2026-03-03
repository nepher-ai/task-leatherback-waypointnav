# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Leatherback waypoint navigation environment.

This environment trains a policy for a wheeled robot to navigate through
a sequence of 2D waypoints. The robot receives observations about its current state
and the waypoints, and must output throttle and steering commands to navigate.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from leatherbacknav.robots import LEATHERBACK_CFG  # isort: skip


##
# Terrain configuration
##

# NOTE: Using simple plane terrain (not mesh generator) to match original Leatherback behavior
# Mesh-based terrain can cause slight height mismatches leading to initial bouncing

##
# Scene definition
##


@configclass
class WaypointNavSceneCfg(InteractiveSceneCfg):
    """Configuration for the waypoint navigation scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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


# Waypoint ranges (~±143 degrees, slightly restricted for wheeled vehicles)
_WAYPOINT_RANGES = mdp.WaypointCommandCfg.Ranges()
_WAYPOINT_RANGES.angle_range = (-2.5, 2.5)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    waypoints = mdp.WaypointCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),  # Never auto-resample
        num_waypoints=5,
        waypoint_spacing=(0.5, 3.0),
        initial_waypoint_distance=(0.5, 2.0),
        waypoint_reach_threshold=0.10,
        num_lookahead_waypoints=1,
        debug_vis=True,
        ranges=_WAYPOINT_RANGES,
    )


@configclass
class ActionsCfg:
    """Action specifications: throttle (wheel velocities) and steering (front wheel angles)."""

    throttle = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["Wheel.*"],
        scale=10.0,
    )
    steering = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Knuckle__Upright__Front.*"],
        scale=0.75,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]},
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
    """Reward terms: progress (1.0), heading (0.05), waypoint bonus (10.0)."""

    waypoint_reached = RewTerm(
        func=mdp.waypoint_reached_bonus,
        weight=10.0,
        params={"command_name": "waypoints", "bonus": 1.0},
    )
    progress = RewTerm(
        func=mdp.progress_reward,
        weight=1.0,
        params={"command_name": "waypoints", "asset_cfg": SceneEntityCfg("robot")},
    )
    waypoint_heading = RewTerm(
        func=mdp.waypoint_heading_reward,
        weight=0.05,
        params={"command_name": "waypoints", "std": 0.25},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    all_waypoints_reached = DoneTerm(
        func=mdp.all_waypoints_reached,
        params={"command_name": "waypoints"},
        time_out=True,
    )
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

    scene: WaypointNavSceneCfg = WaypointNavSceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    viewer = ViewerCfg(eye=(10.0, 10.0, 5.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class WaypointNavEnvCfg_PLAY(WaypointNavEnvCfg):
    """Configuration for playing/evaluating the trained policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 60.0

