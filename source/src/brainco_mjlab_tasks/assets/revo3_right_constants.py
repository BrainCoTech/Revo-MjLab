"""Revo3 right hand constants and EntityCfg for MjLab.

This module only describes the robot body itself:
- MuJoCo XML path
- joint order
- initial joint pose
- MjLab ideal-PD actuator configuration
- collision configuration

It does not create a task, scene, reward, observation, or training environment.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from brainco_mjlab_tasks import BRAINCO_MJLAB_TASKS_SOURCE_PATH
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


REVO3_RIGHT_XML: Path = BRAINCO_MJLAB_TASKS_SOURCE_PATH / "assets" / "revo3_right.xml"
assert REVO3_RIGHT_XML.exists(), f"Missing MJCF: {REVO3_RIGHT_XML}"


REVO3_RIGHT_JOINT_ORDER = (
    "right_thumb_CMP_joint",
    "right_thumb_CMR_joint",
    "right_thumb_MCP_joint",
    "right_thumb_PIP_joint",
    "right_thumb_DIP_joint",
    "right_index_MPR_joint",
    "right_index_MCP_joint",
    "right_index_PIP_joint",
    "right_index_DIP_joint",
    "right_middle_MPR_joint",
    "right_middle_MCP_joint",
    "right_middle_PIP_joint",
    "right_middle_DIP_joint",
    "right_ring_MPR_joint",
    "right_ring_MCP_joint",
    "right_ring_PIP_joint",
    "right_ring_DIP_joint",
    "right_little_MPR_joint",
    "right_little_MCP_joint",
    "right_little_PIP_joint",
    "right_little_DIP_joint",
)


# Initial robot pose for standalone inspection and task insertion.
REVO3_RIGHT_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.1),
    joint_pos={
        # Thumb
        "right_thumb_CMP_joint": 0.43,
        "right_thumb_CMR_joint": 0,
        "right_thumb_MCP_joint": 0.4,
        "right_thumb_PIP_joint": 0.5,
        "right_thumb_DIP_joint": 0.18,
        # Index
        "right_index_MPR_joint": 0.0,
        "right_index_MCP_joint": 0.5,
        "right_index_PIP_joint": 0.15,
        "right_index_DIP_joint": 0.11,
        # Middle
        "right_middle_MPR_joint": 0.0,
        "right_middle_MCP_joint": 0.15,
        "right_middle_PIP_joint": 0.15,
        "right_middle_DIP_joint": 0.11,
        # Ring
        "right_ring_MPR_joint": 0.0,
        "right_ring_MCP_joint": 0.15,
        "right_ring_PIP_joint": 0.15,
        "right_ring_DIP_joint": 0.11,
        # Little
        "right_little_MPR_joint": 0.0,
        "right_little_MCP_joint": 0.15,
        "right_little_PIP_joint": 0.15,
        "right_little_DIP_joint": 0.11,
    },
    joint_vel={".*": 0.0},
)


# Imported from the IsaacLab BrainCo actuator config:
# ImplicitActuatorCfg(effort_limit_sim=0.5, stiffness=3.0, damping=0.1, friction=0.01).
REVO3_RIGHT_ACTUATOR_EFFORT_LIMIT_NM = 0.5
REVO3_RIGHT_PD_STIFFNESS_NM_PER_RAD = 0.5
REVO3_RIGHT_PD_DAMPING_NM_PER_RAD_S = 0.001
REVO3_RIGHT_FRICTION_LOSS_NM = 0.01


def _make_joint_actuator_cfg(joint_name: str) -> IdealPdActuatorCfg:
    return IdealPdActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=REVO3_RIGHT_PD_STIFFNESS_NM_PER_RAD,
        damping=REVO3_RIGHT_PD_DAMPING_NM_PER_RAD_S,
        effort_limit=REVO3_RIGHT_ACTUATOR_EFFORT_LIMIT_NM,
        frictionloss=REVO3_RIGHT_FRICTION_LOSS_NM,
    )


REVO3_RIGHT_ACTUATORS = tuple(
    _make_joint_actuator_cfg(joint_name) for joint_name in REVO3_RIGHT_JOINT_ORDER
)


REVO3_RIGHT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=REVO3_RIGHT_ACTUATORS,
    soft_joint_pos_limit_factor=1.0,
)


REVO3_RIGHT_ADJACENT_BODY_EXCLUDES = (
    ("right_hand_base_link", "right_palm"),
    ("right_hand_base_link", "right_thumb_CMP_Link"),
    ("right_thumb_CMP_Link", "right_thumb_CMR_Link"),
    ("right_thumb_CMR_Link", "right_thumb_MCP_Link"),
    ("right_thumb_MCP_Link", "right_thumb_PIP_Link"),
    ("right_thumb_PIP_Link", "right_thumb_DIP_Link"),
    ("right_thumb_DIP_Link", "right_thumb_tip_Link"),
    ("right_hand_base_link", "right_index_MPR_Link"),
    ("right_index_MPR_Link", "right_index_MCP_Link"),
    ("right_index_MCP_Link", "right_index_PIP_Link"),
    ("right_index_PIP_Link", "right_index_DIP_Link"),
    ("right_index_DIP_Link", "right_index_tip_Link"),
    ("right_hand_base_link", "right_middle_MPR_Link"),
    ("right_middle_MPR_Link", "right_middle_MCP_Link"),
    ("right_middle_MCP_Link", "right_middle_PIP_Link"),
    ("right_middle_PIP_Link", "right_middle_DIP_Link"),
    ("right_middle_DIP_Link", "right_middle_tip_Link"),
    ("right_hand_base_link", "right_ring_MPR_Link"),
    ("right_ring_MPR_Link", "right_ring_MCP_Link"),
    ("right_ring_MCP_Link", "right_ring_PIP_Link"),
    ("right_ring_PIP_Link", "right_ring_DIP_Link"),
    ("right_ring_DIP_Link", "right_ring_tip_Link"),
    ("right_hand_base_link", "right_little_MPR_Link"),
    ("right_little_MPR_Link", "right_little_MCP_Link"),
    ("right_little_MCP_Link", "right_little_PIP_Link"),
    ("right_little_PIP_Link", "right_little_DIP_Link"),
    ("right_little_DIP_Link", "right_little_tip_Link"),
)


# The converted XML currently has generic mesh geoms and no palm/fingertip
# naming convention. This broad collision config is enough for model-level
# integration, but the HandCube task should later use task-specific palm/tip
# names or a cleaned XML.
REVO3_RIGHT_COLLISION = CollisionCfg(
    geom_names_expr=(".*",),
    # Enable hand self-collision. Directly adjacent parent-child body pairs are
    # filtered through MuJoCo contact excludes in get_revo3_right_spec().
    contype=1,
    conaffinity=1,
    condim={".*": 3},
    friction={".*": (0.8, 5e-3, 1e-4)},
    solref={".*": (0.02, 1.0)},
    priority={".*": 0},
    disable_other_geoms=False,
)


def get_revo3_right_spec() -> mujoco.MjSpec:
    """Create the Revo3 right hand MuJoCo spec."""
    spec = mujoco.MjSpec.from_file(str(REVO3_RIGHT_XML))
    for body in spec.bodies:
        body.gravcomp = 1.0
    # The converted Revo3 XML has many unnamed collision geoms, so regex-based
    # CollisionCfg cannot reliably reach all of them. Set every hand geom here.
    for geom in spec.geoms:
        geom.contype = 1
        geom.conaffinity = 1
    for body1, body2 in REVO3_RIGHT_ADJACENT_BODY_EXCLUDES:
        spec.add_exclude(
            name=f"{body1}__{body2}",
            bodyname1=body1,
            bodyname2=body2,
        )
    return spec


def get_revo3_right_hand_cfg() -> EntityCfg:
    """Return an MjLab EntityCfg for the Revo3 right hand."""
    return EntityCfg(
        init_state=REVO3_RIGHT_HOME_KEYFRAME,
        collisions=(REVO3_RIGHT_COLLISION,),
        spec_fn=get_revo3_right_spec,
        articulation=REVO3_RIGHT_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer
    from mjlab.entity.entity import Entity

    hand = Entity(get_revo3_right_hand_cfg())
    mujoco.viewer.launch(hand.spec.compile())
