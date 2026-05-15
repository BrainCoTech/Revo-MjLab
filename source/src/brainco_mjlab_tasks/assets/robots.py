from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ASSETS_ROOT = Path(__file__).resolve().parents[3] / "assets" 
URDF_ROOT = ASSETS_ROOT / "BrainCo-Revo3-URDF-April27"


def _compile_urdf_to_xml(urdf_name: str, xml_name: str) -> Path:
  urdf_path = URDF_ROOT / urdf_name
  xml_path = URDF_ROOT / xml_name
  if xml_path.exists():
    return xml_path

  model = mujoco.MjModel.from_xml_path(str(urdf_path))
  mujoco.mj_saveLastXML(str(xml_path), model)
  return xml_path


def _spec_from_generated_xml(xml_name: str):
  xml_path = URDF_ROOT / xml_name

  def _get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(xml_path))

  return _get_spec


def _make_arm_hand_articulation(hand_regex: tuple[str, ...]) -> EntityArticulationInfoCfg:
  return EntityArticulationInfoCfg(
    actuators=(
      BuiltinPositionActuatorCfg(
        target_names_expr=(r"Joint[1-2]_R",),
        stiffness=220.0,
        damping=35.0,
        # effort_limit=108.0,
        frictionloss=0.8,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=(r"Joint[3-4]_R",),
        stiffness=180.0,
        damping=30.0,
        # effort_limit=66.0,
        frictionloss=0.8,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=(r"Joint[5-7]_R",),
        stiffness=80.0,
        damping=12.0,
        # effort_limit=18.0,
        frictionloss=0.5,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=hand_regex,
        stiffness=6.0,
        damping=0.35,
        # effort_limit=0.35,
        frictionloss=0.2,
      ),
    ),
    soft_joint_pos_limit_factor=1.0,
  )


def get_revo3_arm_hand_cfg() -> EntityCfg:
  _compile_urdf_to_xml("revo3_right_april27.urdf", "revo3_right_april27.xml")
  return EntityCfg(
    spec_fn=_spec_from_generated_xml("revo3_right_april27.xml"),
    articulation=_make_arm_hand_articulation((r"right_.*_joint",)),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={
        "Joint1_R": 0.0,
        "Joint2_R": 0.0,
        "Joint3_R": 0.0,
        "Joint4_R": -0.1,
        "Joint5_R": 0.0,
        "Joint6_R": 0.0,
        "Joint7_R": 0.0,
        "right_thumb_CMP_joint": 1.0,
        "right_thumb_CMR_joint": 1.57,
        "right_thumb_MCP_joint": 0.42,
        "right_thumb_PIP_joint": 0.72,
        "right_thumb_DIP_joint": 0.72,
        "right_index_MPR_joint": 0.0,
        "right_index_MCP_joint": 0.5,
        "right_index_PIP_joint": 0.55,
        "right_index_DIP_joint": 0.31,
        "right_middle_MPR_joint": 0.0,
        "right_middle_MCP_joint": 0.5,
        "right_middle_PIP_joint": 0.55,
        "right_middle_DIP_joint": 0.31,
        "right_ring_MPR_joint": 0.0,
        "right_ring_MCP_joint": 0.5,
        "right_ring_PIP_joint": 0.55,
        "right_ring_DIP_joint": 0.31,
        "right_little_MPR_joint": 0.0,
        "right_little_MCP_joint": 0.5,
        "right_little_PIP_joint": 0.55,
        "right_little_DIP_joint": 0.31,
      },
      joint_vel={".*": 0.0},
    ),
  )


def get_revo2_arm_hand_cfg() -> EntityCfg:
  # _compile_urdf_to_xml("revoarm_revo2_right.urdf", "revoarm_revo2_right.xml")
  # return EntityCfg(
  #   spec_fn=_spec_from_generated_xml("revoarm_revo2_right.xml"),
  #   articulation=_make_arm_hand_articulation((r"right_.*_joint",)),
  #   init_state=EntityCfg.InitialStateCfg(
  #     pos=(0.0, 0.0, 0.0),
  #     rot=(1.0, 0.0, 0.0, 0.0),
  #     joint_pos={
  #       "arm_joint1": 0.0,
  #       "arm_joint2": 0.0,
  #       "arm_joint3": 0.0,
  #       "arm_joint4": 1.2,
  #       "arm_joint5": 0.0,
  #       "arm_joint6": 0.0,
  #       "arm_joint7": 0.0,
  #       "right_thumb_metacarpal_joint": 0.4,
  #       "right_thumb_proximal_joint": 0.4,
  #       "right_thumb_distal_joint": 0.4,
  #       "right_index_proximal_joint": 0.3,
  #       "right_index_distal_joint": 0.3,
  #       "right_middle_proximal_joint": 0.3,
  #       "right_middle_distal_joint": 0.3,
  #       "right_ring_proximal_joint": 0.3,
  #       "right_ring_distal_joint": 0.3,
  #       "right_pinky_proximal_joint": 0.3,
  #       "right_pinky_distal_joint": 0.3,
  #     },
  #     joint_vel={".*": 0.0},
  #   ),
  # )
  pass
