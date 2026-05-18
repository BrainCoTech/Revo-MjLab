"""Standalone Revo3 right hand in-hand cylinder environment configuration."""

from __future__ import annotations

import math

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

try:
    from mjlab.terrains import TerrainImporterCfg
except ImportError:
    from mjlab.terrains import TerrainEntityCfg as TerrainImporterCfg

from mjlab.viewer import ViewerConfig

from brainco_mjlab_tasks.assets import get_revo3_right_hand_cfg
from brainco_mjlab_tasks.assets.revo3_right_constants import (
    REVO3_RIGHT_HOME_KEYFRAME,
)
from brainco_mjlab_tasks.inhand import mdp as inhand_mdp
from brainco_mjlab_tasks.inhand.mdp.actions import JointPositionDeltaActionCfg


REVO3_RIGHT_GRASP_INIT_JOINT_POS = dict(REVO3_RIGHT_HOME_KEYFRAME.joint_pos)
REVO3_RIGHT_HAND_SPAWN_POS = (0.0, 0.0, 0.5)
REVO3_RIGHT_HAND_SPAWN_ROT_WXYZ = (0.5, 0.5, -0.5, 0.5)
REVO3_RIGHT_OBJECT_SPAWN_POS = (0.0, -0.13, 0.56)
REVO3_RIGHT_OBJECT_SPAWN_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)
REVO3_RIGHT_CYLINDER_RADIUS = 0.035
REVO3_RIGHT_CYLINDER_HEIGHT = 0.07
REVO3_RIGHT_CYLINDER_MASS = (
    400.0 * math.pi * REVO3_RIGHT_CYLINDER_RADIUS**2 * REVO3_RIGHT_CYLINDER_HEIGHT
)
REVO3_RIGHT_GOAL_ROT_WXYZ = (
    math.cos(math.pi / 16.0),
    0.0,
    0.0,
    math.sin(math.pi / 16.0),
)
REVO3_RIGHT_GOAL_YAW_OFFSET = math.pi / 8.0
REVO3_RIGHT_RESET_DOF_POS_NOISE = 0.0
REVO3_RIGHT_RESET_DOF_VEL_NOISE = 0.0
REVO3_RIGHT_DIST_REWARD_SCALE = 1.0
REVO3_RIGHT_ROT_REWARD_SCALE = 3.0
REVO3_RIGHT_ROT_EPS = 0.1
REVO3_RIGHT_ACTION_PENALTY_SCALE = -0.0002
REVO3_RIGHT_REACH_GOAL_BONUS = 250.0
REVO3_RIGHT_FALL_PENALTY_SCALE = -20.0
REVO3_RIGHT_FALL_DIST = 0.24
REVO3_RIGHT_VEL_OBS_SCALE = 0.2
REVO3_RIGHT_SUCCESS_TOLERANCE = 0.2
REVO3_RIGHT_FINGERTIP_BODY_NAMES = (
    "right_little_DIP_Link",
    "right_ring_DIP_Link",
    "right_middle_DIP_Link",
    "right_index_DIP_Link",
    "right_thumb_DIP_Link",
)

REVO3_RIGHT_PALM_BODY_NAME = "right_palm"
REVO3_RIGHT_PALM_CENTER_GEOM_EXPR = "right_palm_collision_.*"


def _get_cylinder_spec(
    radius: float = REVO3_RIGHT_CYLINDER_RADIUS,
    height: float = REVO3_RIGHT_CYLINDER_HEIGHT,
    mass: float = REVO3_RIGHT_CYLINDER_MASS,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(radius, 0.5 * height, 0.0),
        mass=mass,
        rgba=(0.7, 0.7, 0.7, 1.0),
    )
    return spec


def _brainco_reorient_full_obs_terms() -> dict[str, ObservationTermCfg]:
    joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    fingertip_cfg = SceneEntityCfg(
        "robot",
        body_names=REVO3_RIGHT_FINGERTIP_BODY_NAMES,
        preserve_order=True,
    )
    return {
        "joint_pos": ObservationTermCfg(
            func=inhand_mdp.joint_pos_unscaled,
            params={"asset_cfg": joint_cfg},
        ),
        "joint_vel": ObservationTermCfg(
            func=inhand_mdp.joint_vel_scaled,
            params={"asset_cfg": joint_cfg, "scale": REVO3_RIGHT_VEL_OBS_SCALE},
        ),
        "object_pos": ObservationTermCfg(
            func=inhand_mdp.object_pos_env,
            params={"object_name": "cube"},
        ),
        "object_rot": ObservationTermCfg(
            func=inhand_mdp.object_quat_wxyz,
            params={"object_name": "cube"},
        ),
        "object_linvel": ObservationTermCfg(
            func=inhand_mdp.object_lin_vel_w,
            params={"object_name": "cube"},
        ),
        "object_angvel": ObservationTermCfg(
            func=inhand_mdp.object_ang_vel_w_scaled,
            params={"object_name": "cube", "scale": REVO3_RIGHT_VEL_OBS_SCALE},
        ),
        "in_hand_pos": ObservationTermCfg(
            func=inhand_mdp.constant_vector,
            params={"value": REVO3_RIGHT_OBJECT_SPAWN_POS},
        ),
        "goal_rot": ObservationTermCfg(
            func=inhand_mdp.brainco_goal_rot_wxyz,
            params={"initial_goal_quat_wxyz": REVO3_RIGHT_GOAL_ROT_WXYZ},
        ),
        "object_goal_rot_diff": ObservationTermCfg(
            func=inhand_mdp.brainco_object_goal_quat_diff_wxyz,
            params={
                "object_name": "cube",
                "initial_goal_quat_wxyz": REVO3_RIGHT_GOAL_ROT_WXYZ,
            },
        ),
        "fingertip_pos": ObservationTermCfg(
            func=inhand_mdp.body_pos_env_flat,
            params={"asset_cfg": fingertip_cfg},
        ),
        "fingertip_rot": ObservationTermCfg(
            func=inhand_mdp.body_quat_wxyz_flat,
            params={"asset_cfg": fingertip_cfg},
        ),
        "fingertip_velocities": ObservationTermCfg(
            func=inhand_mdp.body_vel_w_flat,
            params={"asset_cfg": fingertip_cfg},
        ),
        "actions": ObservationTermCfg(func=inhand_mdp.action_raw),
    }


def revo3_right_inhand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create a complete Revo3 right hand in-hand cylinder environment config."""

    observations = {
        "actor": ObservationGroupCfg(
            _brainco_reorient_full_obs_terms(),
            enable_corruption=False,
            history_length=1,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            _brainco_reorient_full_obs_terms(),
            enable_corruption=False,
            history_length=1,
            flatten_history_dim=True,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionDeltaActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clip_to_joint_limits=True,
            use_soft_joint_pos_limits=True,
            delta_min=-(1 / 12),
            delta_max=(1 / 12),
            interpolate_decimation=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "frame_viz": inhand_mdp.HandCubeFrameVizCommandCfg(
            hand_name="robot",
            object_name="cube",
            palm_body_name=REVO3_RIGHT_PALM_BODY_NAME,
            palm_center_geom_expr=REVO3_RIGHT_PALM_CENTER_GEOM_EXPR,
            target_pos=REVO3_RIGHT_OBJECT_SPAWN_POS,
            resampling_time_range=(1e9, 1e9),
            debug_vis=True,
        ),
    }

    events = {
        "reset_robot_pose": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (
                    -REVO3_RIGHT_RESET_DOF_POS_NOISE,
                    REVO3_RIGHT_RESET_DOF_POS_NOISE,
                ),
                "velocity_range": (
                    -REVO3_RIGHT_RESET_DOF_VEL_NOISE,
                    REVO3_RIGHT_RESET_DOF_VEL_NOISE,
                ),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_cube_pose": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube"),
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
    }

    rewards = {
        "dist": RewardTermCfg(
            func=inhand_mdp.brainco_object_goal_distance,
            weight=REVO3_RIGHT_DIST_REWARD_SCALE,
            params={
                "object_name": "cube",
                "target_pos": REVO3_RIGHT_OBJECT_SPAWN_POS,
            },
        ),
        "rot": RewardTermCfg(
            func=inhand_mdp.brainco_rotation_reward,
            weight=REVO3_RIGHT_ROT_REWARD_SCALE,
            params={
                "object_name": "cube",
                "initial_goal_quat_wxyz": REVO3_RIGHT_GOAL_ROT_WXYZ,
                "rot_eps": REVO3_RIGHT_ROT_EPS,
            },
        ),
        "action_penalty": RewardTermCfg(
            func=inhand_mdp.brainco_action_l2,
            weight=REVO3_RIGHT_ACTION_PENALTY_SCALE,
        ),
        "fall_penalty": RewardTermCfg(
            func=inhand_mdp.brainco_fall_penalty,
            weight=REVO3_RIGHT_FALL_PENALTY_SCALE,
            params={
                "object_name": "cube",
                "target_pos": REVO3_RIGHT_OBJECT_SPAWN_POS,
                "fall_dist": REVO3_RIGHT_FALL_DIST,
            },
        ),
        "reach_goal_bonus": RewardTermCfg(
            func=inhand_mdp.brainco_reach_goal_bonus,
            weight=REVO3_RIGHT_REACH_GOAL_BONUS,
            params={
                "object_name": "cube",
                "initial_goal_quat_wxyz": REVO3_RIGHT_GOAL_ROT_WXYZ,
                "success_tolerance": REVO3_RIGHT_SUCCESS_TOLERANCE,
                "yaw_offset": REVO3_RIGHT_GOAL_YAW_OFFSET,
                "target_pos": REVO3_RIGHT_OBJECT_SPAWN_POS,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "out_of_reach": TerminationTermCfg(
            func=inhand_mdp.object_distance_from_target_above,
            params={
                "max_distance": REVO3_RIGHT_FALL_DIST,
                "target_pos": REVO3_RIGHT_OBJECT_SPAWN_POS,
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    metrics = {
        "goal_dist": MetricsTermCfg(
            func=inhand_mdp.brainco_object_goal_distance,
            params={
                "object_name": "cube",
                "target_pos": REVO3_RIGHT_OBJECT_SPAWN_POS,
            },
        ),
        "rot_dist": MetricsTermCfg(
            func=inhand_mdp.brainco_rotation_distance,
            params={
                "object_name": "cube",
                "initial_goal_quat_wxyz": REVO3_RIGHT_GOAL_ROT_WXYZ,
            },
        ),
    }

    curriculum = {}

    robot_cfg = get_revo3_right_hand_cfg()
    robot_cfg.init_state.pos = REVO3_RIGHT_HAND_SPAWN_POS
    robot_cfg.init_state.rot = REVO3_RIGHT_HAND_SPAWN_ROT_WXYZ
    robot_cfg.init_state.joint_pos = REVO3_RIGHT_GRASP_INIT_JOINT_POS

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={
                "robot": robot_cfg,
                "cube": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(
                        pos=REVO3_RIGHT_OBJECT_SPAWN_POS,
                        rot=REVO3_RIGHT_OBJECT_SPAWN_ROT_WXYZ,
                        lin_vel=(0.0, 0.0, 0.0),
                        ang_vel=(0.0, 0.0, 0.0),
                    ),
                    spec_fn=_get_cylinder_spec,
                ),
            },
            num_envs=1,
            env_spacing=0.6,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=REVO3_RIGHT_PALM_BODY_NAME,
            distance=0.45,
            elevation=-25,
            azimuth=110,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=800,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=10,
        episode_length_s=10.0,
        scale_rewards_by_dt=False,
    )

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg
