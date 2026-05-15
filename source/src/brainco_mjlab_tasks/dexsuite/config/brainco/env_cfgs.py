from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
  ActionTermCfg,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from brainco_mjlab_tasks.assets.robots import URDF_ROOT, get_revo3_arm_hand_cfg
from brainco_mjlab_tasks.dexsuite import mdp

_DEXSUITE_NJMAX = 600 #256
_DEXSUITE_NCONMAX = 100#55
_TABLE_SIZE = (1.2, 1.6, 0.76)
_TABLE_POS = (0.0, 0.0, 0.38)
_TABLE_TOP_Z = _TABLE_POS[2] + _TABLE_SIZE[2] / 2
_ARM_ON_TABLE_POS = (-0.35, 0.0, _TABLE_TOP_Z)
_OBJECT_HALF_EXTENTS = (0.04, 0.04, 0.04)
_OBJECT_MASS = 0.05 # 0.2
_OBJECT_DROP_Z = _TABLE_TOP_Z - _OBJECT_HALF_EXTENTS[2] * 2
_OBJECT_SUPPORT_Z = _TABLE_TOP_Z + _OBJECT_HALF_EXTENTS[2]
_CONTACT_TARGET_FORCE = 7.5
_CONTACT_FORCE_K = 1.0
# MuJoCo collapses the URDF fixed palm link into Link7_R. This offset points to the
# original palm-center frame in Link7_R coordinates.
_PALM_BODY_NAME = "Link7_R"
_PALM_BODY_OFFSET = (-0.007,-0.19,0.05)
_REVO3_RIGHT_URDF_PATH = URDF_ROOT / "revo3_right_april27.urdf"
_OBJECT_RESET_START_POSE_RANGE = {
  "x": (0.0, 0.2),
  "y": (-0.15, 0.15),
  "z": (0.81, 0.82),
  "roll": (-3.14, 3.14),
  "pitch": (-3.14, 3.14),
  "yaw": (-3.14, 3.14),
}
_TABLE_OBJECT_CENTER_X_RANGE = (
  _TABLE_POS[0] - _TABLE_SIZE[0] / 2 + _OBJECT_HALF_EXTENTS[0],
  _TABLE_POS[0] + _TABLE_SIZE[0] / 2 - _OBJECT_HALF_EXTENTS[0],
)
_TABLE_OBJECT_CENTER_Y_RANGE = (
  _TABLE_POS[1] - _TABLE_SIZE[1] / 2 + _OBJECT_HALF_EXTENTS[1],
  _TABLE_POS[1] + _TABLE_SIZE[1] / 2 - _OBJECT_HALF_EXTENTS[1],
)
_OBJECT_RESET_Z_FINAL_SPAN_SCALE = 4.0
_OBJECT_RESET_FINAL_POSE_RANGE = {
  "x": _TABLE_OBJECT_CENTER_X_RANGE,
  "y": _TABLE_OBJECT_CENTER_Y_RANGE,
  "z": (
    _OBJECT_RESET_START_POSE_RANGE["z"][0],
    _OBJECT_RESET_START_POSE_RANGE["z"][0]
    + (
      _OBJECT_RESET_START_POSE_RANGE["z"][1]
      - _OBJECT_RESET_START_POSE_RANGE["z"][0]
    )
    * _OBJECT_RESET_Z_FINAL_SPAN_SCALE,
  ),
  "roll": _OBJECT_RESET_START_POSE_RANGE["roll"],
  "pitch": _OBJECT_RESET_START_POSE_RANGE["pitch"],
  "yaw": _OBJECT_RESET_START_POSE_RANGE["yaw"],
}
_TARGET_START_RANGES = {
  # "pos_x": (0.32, 0.52),
  # "pos_y": (-0.12, 0.12),
  # "pos_z": (0.18, 0.28),
  "pos_x": (0.32, 0.82),
  "pos_y": (-0.42, 0.42),
  "pos_z": (0.38, 0.68),
}
_TARGET_FINAL_RANGES = {
  "pos_x": (
    _TABLE_OBJECT_CENTER_X_RANGE[0] - _ARM_ON_TABLE_POS[0],
    _TABLE_OBJECT_CENTER_X_RANGE[1] - _ARM_ON_TABLE_POS[0],
  ),
  "pos_y": (
    _TABLE_OBJECT_CENTER_Y_RANGE[0] - _ARM_ON_TABLE_POS[1],
    _TABLE_OBJECT_CENTER_Y_RANGE[1] - _ARM_ON_TABLE_POS[1],
  ),
  "pos_z": (_TARGET_START_RANGES["pos_z"][0], 0.68),
}
_WORKSPACE_CURRICULUM_START_STEP = 1000 * 32
_WORKSPACE_CURRICULUM_END_STEP = 10000 * 32


@lru_cache(maxsize=1)
def _read_urdf_joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
  root = ET.parse(urdf_path).getroot()
  joint_limits: dict[str, tuple[float, float]] = {}
  for joint in root.findall("joint"):
    if joint.attrib.get("type") == "fixed":
      continue

    limit = joint.find("limit")
    if limit is None:
      continue

    lower = limit.attrib.get("lower")
    upper = limit.attrib.get("upper")
    if lower is None or upper is None:
      continue

    joint_limits[joint.attrib["name"]] = (float(lower), float(upper))
  return joint_limits


def _make_table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table", mocap=True)
  body.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=tuple(length / 2 for length in _TABLE_SIZE),
    pos=_TABLE_POS,
    mass=1.0,
    rgba=(0.98, 0.92, 0.95, 1.0),
  )
  return spec


def _robot_on_table(robot_cfg: EntityCfg) -> EntityCfg:
  return replace(
    robot_cfg,
    init_state=replace(
      robot_cfg.init_state,
      pos=_ARM_ON_TABLE_POS,
    ),
  )


def _make_box_spec(size=_OBJECT_HALF_EXTENTS, mass=_OBJECT_MASS) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_joint")
  body.add_geom(
    name="object_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=size,
    mass=mass,
    friction=(1.2, 0.01, 0.001),
    rgba=(0.2, 0.58, 0.88, 1.0),
  )
  return spec


def _make_actor_terms() -> dict[str, ObservationTermCfg]:
  return {
    "object_quat_b": ObservationTermCfg(
      func=mdp.object_quat_b,
      params={
        "robot_cfg": SceneEntityCfg("robot"),
        "object_cfg": SceneEntityCfg("object"),
      },
      noise=Unoise(n_min=-1e-6, n_max=1e-6),
    ),
    "target_object_pose_b": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "object_pose"}
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }


def _make_proprio_terms(contact_sensor_names: list[str], body_names: tuple[str, ...]) -> dict[str, ObservationTermCfg]:
  return {
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-1e-6, n_max=1e-6),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1e-6, n_max=1e-6),
      clip=(-40.0, 40.0),
      scale=1.0 / 20.0,
    ),
    "hand_tips_state_b": ObservationTermCfg(
      func=mdp.body_state_b,
      params={
        "body_asset_cfg": SceneEntityCfg("robot", body_names=body_names),
        "base_asset_cfg": SceneEntityCfg("robot"),
      },
      noise=Unoise(n_min=-1e-6, n_max=1e-6),
      clip=(-2.0, 2.0),
    ),
    "contact": ObservationTermCfg(
      func=mdp.fingers_contact_force_b,
      params={"contact_sensor_names": contact_sensor_names},
      clip=(-60.0, 60.0),
      # scale=1.0 / 30.0,
    ),
  }


def _make_observations(contact_sensor_names: list[str], body_names: tuple[str, ...]) -> dict[str, ObservationGroupCfg]:
  return {
    "policy": ObservationGroupCfg(
      terms=_make_actor_terms(),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "proprio": ObservationGroupCfg(
      terms=_make_proprio_terms(contact_sensor_names, body_names),
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
      flatten_history_dim=True,
    ),
    "perception": ObservationGroupCfg(
      terms={
        "object_point_cloud": ObservationTermCfg(
          func=mdp.object_point_cloud_b,
          params={"num_points": 64, "flatten": True},
          clip=(-2.0, 2.0),
        ),
      },
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
      flatten_history_dim=True,
    ),
  }


def _make_actions() -> dict[str, ActionTermCfg]:
  return {
    "joint_pos": mdp.TanhJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      clip=_read_urdf_joint_limits(_REVO3_RIGHT_URDF_PATH),
      use_default_offset=True,
      debug_print_actions=False,
    ),
  }


def _make_commands(position_only: bool) -> dict[str, CommandTermCfg]:
  return {
    "object_pose": mdp.ObjectUniformPoseCommandCfg(
      asset_name="robot",
      object_name="object",
      resampling_time_range=(10.0, 10.0),
      debug_vis=False,
      position_only=position_only,
      ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
        pos_x=_TARGET_START_RANGES["pos_x"],
        pos_y=_TARGET_START_RANGES["pos_y"],
        pos_z=_TARGET_START_RANGES["pos_z"],
        roll=(-3.14, 3.14),
        pitch=(-3.14, 3.14),
        yaw=(0.0, 0.0),
      ),
    )
  }


def _make_events() -> dict[str, EventTermCfg]:
  return {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("table"),
        "pose_range": {},  # Empty = use default pose + env_origins
        "velocity_range": {},
      },
    ),
    "reset_robot": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    ),
    "reset_object": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "pose_range": dict(_OBJECT_RESET_START_POSE_RANGE),
        "velocity_range": {},
      },
    ),
  }


def _make_rewards(robot_body_names: tuple[str, ...], position_only: bool) -> dict[str, RewardTermCfg]:
  rewards = {
    "object_drop_penalty": RewardTermCfg(
      func=mdp.object_drop_penalty,
      weight=-10.0,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "min_height": _OBJECT_DROP_Z,
      },
    ),
    "action_l2": RewardTermCfg(func=mdp.action_l2_clamped, weight=-0.005),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2_clamped, weight=-0.005),
    "fingers_to_object": RewardTermCfg(
      func=mdp.object_ee_distance,
      weight=1.0,
      params={
        "std": 0.35,
        "object_cfg": SceneEntityCfg("object"),
        "asset_cfg": SceneEntityCfg("robot", body_names=robot_body_names),
      },
    ),
    "palm_to_object_surface": RewardTermCfg(
      func=mdp.palm_to_object_surface_distance,
      weight=1.0,
      params={
        "std": 0.20,
        "object_cfg": SceneEntityCfg("object"),
        "asset_cfg": SceneEntityCfg("robot", body_names=(_PALM_BODY_NAME,)),
        "half_extents": _OBJECT_HALF_EXTENTS,
        "penetration_weight": 3.0,
        "body_offset": _PALM_BODY_OFFSET,
      },
    ),
    "staged_position": RewardTermCfg(
      func=mdp.dexterous_staged_position_reward,
      weight=3.0,
      params={
        "command_name": "object_pose",
        "object_cfg": SceneEntityCfg("object"),
        "asset_cfg": SceneEntityCfg("robot", body_names=robot_body_names),
        "reach_std": 0.35,
        "bring_std": 0.30,
        "ungated_bring_weight": 0.15,
      },
    ),
    "any_finger_contact": RewardTermCfg(
      func=mdp.any_finger_contact,
      weight=0.2,
      params={"threshold": 1.0},
    ),
    "good_finger_contact": RewardTermCfg(
      func=mdp.good_finger_contact,
      weight=2.0,
      params={"threshold": _CONTACT_TARGET_FORCE, "k": _CONTACT_FORCE_K},
    ),
    "object_lift_progress": RewardTermCfg(
      func=mdp.object_lift_progress_reward,
      weight=4.0,#2.0
      params={
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "reference_height": _OBJECT_SUPPORT_Z,
      },
    ),
    "grasped_lift_progress": RewardTermCfg(
      func=mdp.grasped_object_lift_progress_reward,
      weight=8.0, #5.0
      params={
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "reference_height": _OBJECT_SUPPORT_Z,
        "contact_target_force": _CONTACT_TARGET_FORCE,
        "contact_k": _CONTACT_FORCE_K,
        "contact_floor": 0.25,
      },
    ),
    "object_height_tracking": RewardTermCfg(
      func=mdp.object_height_tracking_reward,
      weight=5.0,#3.0,
      params={
        "std": 0.12,
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "contact_target_force": _CONTACT_TARGET_FORCE,
        "contact_k": _CONTACT_FORCE_K,
        "contact_floor": 0.20,
      },
    ),
    "object_upward_velocity": RewardTermCfg(
      func=mdp.object_upward_velocity_reward,
      weight=0.5,
      params={
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "reference_height": _OBJECT_SUPPORT_Z,
        "target_speed": 0.20,
        "contact_target_force": _CONTACT_TARGET_FORCE,
        "contact_k": _CONTACT_FORCE_K,
      },
    ),
    "object_stability": RewardTermCfg(
      func=mdp.stable_object_lift_reward,
      weight=0.2,
      params={
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "reference_height": _OBJECT_SUPPORT_Z,
        "min_progress": 0.65,
        "lin_vel_std": 0.20,
        "ang_vel_std": 1.50,
        "contact_target_force": _CONTACT_TARGET_FORCE,
        "contact_k": _CONTACT_FORCE_K,
        "target_hold_steps": 1,
      },
    ),
    "position_tracking": RewardTermCfg(
      func=mdp.position_command_error_tanh,
      weight=1.0,
      params={
        "std": 0.30,
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "contact_target_force": _CONTACT_TARGET_FORCE,
        "contact_k": _CONTACT_FORCE_K,
      },
    ),
    "success": RewardTermCfg(
      func=mdp.success_reward,
      weight=12.0,
      params={
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "pos_std": 0.1,
        "rot_std": None if position_only else 0.5,
        "contact_target_force": _CONTACT_TARGET_FORCE,
        "contact_k": _CONTACT_FORCE_K,
      },
    ),
    "joint_vel_hinge": RewardTermCfg(
      func=mdp.joint_velocity_hinge_penalty,
      weight=-0.001,
      params={
        "max_vel": 8.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }
  if not position_only:
    rewards["orientation_tracking"] = RewardTermCfg(
      func=mdp.orientation_command_error_tanh,
      weight=4.0,
      params={
        "std": 1.5,
        "command_name": "object_pose",
        "asset_cfg": SceneEntityCfg("robot"),
        "align_asset_cfg": SceneEntityCfg("object"),
        "contact_target_force": _CONTACT_TARGET_FORCE,
        "contact_k": _CONTACT_FORCE_K,
      },
    )
  return rewards


def _make_curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "joint_vel_hinge_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "joint_vel_hinge",
        "weight_stages": [
          {"step": 0, "weight": -0.001},
          {"step": 500 * 32, "weight": -0.005},
          {"step": 1500 * 32, "weight": -0.02},
          {"step": 3000 * 32, "weight": -0.05},
        ],
      },
    ),
    "object_stability_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "object_stability",
        "weight_stages": [
          {"step": 0, "weight": 0.2},
          {"step": 1000 * 32, "weight": 0.8},
          {"step": 2500 * 32, "weight": 1.5},
          {"step": 5000 * 32, "weight": 2.0},
        ],
      },
    ),
    "object_stability_hold_steps": CurriculumTermCfg(
      func=mdp.reward_param,
      params={
        "reward_name": "object_stability",
        "param_name": "target_hold_steps",
        "param_stages": [
          {"step": 0, "value": 1},
          {"step": 1000 * 32, "value": 8},
          {"step": 2500 * 32, "value": 20},
          {"step": 5000 * 32, "value": 35},
        ],
      },
    ),
    "object_reset_target_workspace_range": CurriculumTermCfg(
      func=mdp.object_reset_and_target_workspace_range,
      params={
        "command_name": "object_pose",
        "reset_event_name": "reset_object",
        "start_step": _WORKSPACE_CURRICULUM_START_STEP,
        "end_step": _WORKSPACE_CURRICULUM_END_STEP,
        "object_start_pose_range": _OBJECT_RESET_START_POSE_RANGE,
        "object_final_pose_range": _OBJECT_RESET_FINAL_POSE_RANGE,
        "target_start_ranges": _TARGET_START_RANGES,
        "target_final_ranges": _TARGET_FINAL_RANGES,
      },
    ),
  }


def _make_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "object_out_of_bound": TerminationTermCfg(
      func=mdp.out_of_bound,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "in_bound_range": {"x": (-1.5, 1.0), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
      },
    ),
    "object_dropped": TerminationTermCfg(
      func=mdp.object_dropped,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "min_height": _OBJECT_DROP_Z,
      },
    ),
    "abnormal_robot": TerminationTermCfg(func=mdp.abnormal_robot_state),
  }


def _contact_sensor(name: str, body_name: str) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="body", entity="robot", pattern=body_name),
    secondary=ContactMatch(mode="body", entity="object", pattern="object"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def _base_cfg(robot_cfg, contact_pairs: list[tuple[str, str]], body_names: tuple[str, ...], reward_body_names: tuple[str, ...], position_only: bool, play: bool = False) -> ManagerBasedRlEnvCfg:
  contact_sensor_names = [name for name, _ in contact_pairs]
  sensors = tuple(_contact_sensor(name, body) for name, body in contact_pairs)
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": robot_cfg,
        "object": EntityCfg(spec_fn=lambda: _make_box_spec()),
        "table": EntityCfg(spec_fn=_make_table_spec),
      },
      sensors=sensors,
      num_envs=1 if not play else 64,
      env_spacing=3.0,
    ),
    observations=_make_observations(contact_sensor_names, body_names),
    actions=_make_actions(),
    commands=_make_commands(position_only=position_only),
    events=_make_events(),
    rewards=_make_rewards(reward_body_names, position_only=position_only),
    curriculum=_make_curriculum(),
    terminations=_make_terminations(),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 1.0),
      distance=3.5,
      elevation=-20.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=_DEXSUITE_NCONMAX,
      njmax=_DEXSUITE_NJMAX,
      mujoco=MujocoCfg(
        timestep=1 / 200, #120,
        ccd_iterations=500, #500,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4, #2,
    episode_length_s=20 if play else 20.0,
    is_finite_horizon=True,
  )
  if play:
    cfg.observations["policy"].enable_corruption = False
    cfg.observations["proprio"].enable_corruption = False
    cfg.observations["perception"].enable_corruption = False
    cfg.commands["object_pose"].resampling_time_range = (2.0, 3.0)
    cfg.commands["object_pose"].debug_vis = True
  return cfg


def brainco_arm_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  contact_pairs = [
    ("right_thumb_DIP_Link_object_s", "right_thumb_DIP_Link"),
    ("right_index_DIP_Link_object_s", "right_index_DIP_Link"),
    ("right_middle_DIP_Link_object_s", "right_middle_DIP_Link"),
    ("right_ring_DIP_Link_object_s", "right_ring_DIP_Link"),
    ("right_little_DIP_Link_object_s", "right_little_DIP_Link"),
  ]
  body_names = tuple(body for _, body in contact_pairs)
  return _base_cfg(
    robot_cfg=_robot_on_table(get_revo3_arm_hand_cfg()),
    contact_pairs=contact_pairs,
    body_names=body_names,
    reward_body_names=body_names,
    position_only=True,
    play=play,
  )


def brainco_arm_reorient_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  contact_pairs = [
    ("right_thumb_DIP_Link_object_s", "right_thumb_DIP_Link"),
    ("right_index_DIP_Link_object_s", "right_index_DIP_Link"),
    ("right_middle_DIP_Link_object_s", "right_middle_DIP_Link"),
    ("right_ring_DIP_Link_object_s", "right_ring_DIP_Link"),
    ("right_little_DIP_Link_object_s", "right_little_DIP_Link"),
  ]
  body_names = tuple(body for _, body in contact_pairs)
  return _base_cfg(
    robot_cfg=_robot_on_table(get_revo3_arm_hand_cfg()),
    contact_pairs=contact_pairs,
    body_names=body_names,
    reward_body_names=body_names,
    position_only=False,
    play=play,
  )


def brainco_revo2_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  contact_pairs = [
    ("right_thumb_touch_link_object_s", "right_thumb_touch_link"),
    ("right_thumb_distal_link_object_s", "right_thumb_distal_link"),
    ("right_index_touch_link_object_s", "right_index_touch_link"),
    ("right_index_distal_link_object_s", "right_index_distal_link"),
    ("right_middle_touch_link_object_s", "right_middle_touch_link"),
    ("right_middle_distal_link_object_s", "right_middle_distal_link"),
    ("right_ring_touch_link_object_s", "right_ring_touch_link"),
    ("right_ring_distal_link_object_s", "right_ring_distal_link"),
    ("right_pinky_touch_link_object_s", "right_pinky_touch_link"),
    ("right_pinky_distal_link_object_s", "right_pinky_distal_link"),
  ]
  body_names = tuple(body for _, body in contact_pairs)
  reward_body_names = (
    "right_thumb_distal_link",
    "right_index_distal_link",
    "right_middle_distal_link",
    "right_ring_distal_link",
    "right_pinky_distal_link",
  )
  return _base_cfg(
    robot_cfg=_robot_on_table(get_revo2_arm_hand_cfg()),
    contact_pairs=contact_pairs,
    body_names=body_names,
    reward_body_names=reward_body_names,
    position_only=True,
    play=play,
  )
