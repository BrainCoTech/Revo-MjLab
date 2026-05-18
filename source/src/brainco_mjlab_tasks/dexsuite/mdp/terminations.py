from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg

_URDF_ROOT = Path(__file__).resolve().parents[4] / "assets" / "BrainCo-Revo3-URDF"


@lru_cache(maxsize=None)
def _joint_velocity_limits_from_urdf(urdf_name: str) -> dict[str, float]:
  root = ET.parse(_URDF_ROOT / urdf_name).getroot()
  limits: dict[str, float] = {}
  for joint in root.findall("joint"):
    joint_name = joint.attrib.get("name")
    limit = joint.find("limit")
    if joint_name is None or limit is None:
      continue
    velocity = limit.attrib.get("velocity")
    if velocity is None:
      continue
    limits[joint_name] = float(velocity)
  return limits


def _robot_joint_velocity_limits(robot: Entity) -> torch.Tensor:
  if "right_thumb_CMP_joint" in robot.joint_names:
    velocity_limits = _joint_velocity_limits_from_urdf("revo3_right_april27.urdf")
  else:
    # velocity_limits = _joint_velocity_limits_from_urdf("revoarm_revo2_right.urdf")
    pass
  return torch.tensor(
    [velocity_limits[name] for name in robot.joint_names],
    device=robot.data.joint_vel.device,
    dtype=robot.data.joint_vel.dtype,
  ).unsqueeze(0)


def out_of_bound(env, asset_cfg: SceneEntityCfg, in_bound_range: dict[str, tuple[float, float]]) -> torch.Tensor:
  obj: Entity = env.scene[asset_cfg.name]
  pos = obj.data.root_link_pos_w - env.scene.env_origins
  ranges = torch.tensor(
    [in_bound_range["x"], in_bound_range["y"], in_bound_range["z"]],
    device=env.device,
    dtype=pos.dtype,
  )
  return ((pos < ranges[:, 0]) | (pos > ranges[:, 1])).any(dim=1)


def object_dropped(
  env,
  min_height: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  obj: Entity = env.scene[asset_cfg.name]
  object_pos = obj.data.root_link_pos_w - env.scene.env_origins
  object_height = object_pos[:, 2]
  res = (~torch.isfinite(object_height)) | (object_height < min_height)
  # if(res.any()):
  #   print("object_dropped:", res)
  return res


def abnormal_robot_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  non_finite_state = (
    ~torch.isfinite(robot.data.root_link_pos_w).all(dim=1)
    | ~torch.isfinite(robot.data.root_link_quat_w).all(dim=1)
    | ~torch.isfinite(robot.data.joint_pos).all(dim=1)
    | ~torch.isfinite(robot.data.joint_vel).all(dim=1)
    | ~torch.isfinite(robot.data.body_link_pos_w.reshape(env.num_envs, -1)).all(dim=1)
    | ~torch.isfinite(robot.data.body_link_quat_w.reshape(env.num_envs, -1)).all(dim=1)
    | ~torch.isfinite(robot.data.body_link_lin_vel_w.reshape(env.num_envs, -1)).all(dim=1)
    | ~torch.isfinite(robot.data.body_link_ang_vel_w.reshape(env.num_envs, -1)).all(dim=1)
  )
  if(non_finite_state.any()):
    print("abnormal_robot_state: non_finite_state", non_finite_state)
  joint_vel_limits = _robot_joint_velocity_limits(robot)
  excessive_joint_vel = (robot.data.joint_vel.abs() > (joint_vel_limits * 2)).any(dim=1)
  # print((robot.data.joint_vel.abs() > (joint_vel_limits * 2))[0])
  return non_finite_state #| excessive_joint_vel
