from __future__ import annotations

import math

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
)


def object_quat_b(env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  return quat_mul(quat_inv(robot.data.root_link_quat_w), obj.data.root_link_quat_w)


def body_state_b(env, body_asset_cfg: SceneEntityCfg, base_asset_cfg: SceneEntityCfg) -> torch.Tensor:
  body_asset: Entity = env.scene[body_asset_cfg.name]
  base_asset: Entity = env.scene[base_asset_cfg.name]
  body_pos_w = body_asset.data.body_link_pos_w[:, body_asset_cfg.body_ids].reshape(-1, 3)
  body_quat_w = body_asset.data.body_link_quat_w[:, body_asset_cfg.body_ids].reshape(-1, 4)
  body_lin_vel_w = body_asset.data.body_link_lin_vel_w[:, body_asset_cfg.body_ids].reshape(-1, 3)
  body_ang_vel_w = body_asset.data.body_link_ang_vel_w[:, body_asset_cfg.body_ids].reshape(-1, 3)
  num_bodies = len(body_asset_cfg.body_ids)
  root_pos_w = (
    base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 3)
  )
  root_quat_w = (
    base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 4)
  )
  body_pos_b, body_quat_b = subtract_frame_transforms(
    root_pos_w, root_quat_w, body_pos_w, body_quat_w
  )
  body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
  body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
  res = torch.cat(
    (body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=-1
  ).reshape(env.num_envs, -1)
  return res


def _box_surface_points(num_points: int, device: str, extents=(0.04, 0.04, 0.04)) -> torch.Tensor:
  n = max(1, math.ceil(num_points / 6))
  lin = torch.linspace(-1.0, 1.0, n, device=device)
  gx, gy = torch.meshgrid(lin, lin, indexing="ij")
  base = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
  faces = []
  for axis in range(3):
    for sign in (-1.0, 1.0):
      pts = torch.zeros(base.shape[0], 3, device=device)
      other = [i for i in range(3) if i != axis]
      pts[:, other[0]] = base[:, 0]
      pts[:, other[1]] = base[:, 1]
      pts[:, axis] = sign
      faces.append(pts)
  pts = torch.cat(faces, dim=0)[:num_points]
  return pts * torch.tensor(extents, device=device).unsqueeze(0) * 0.5


def object_point_cloud_b(
  env,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  num_points: int = 64,
  flatten: bool = True,
) -> torch.Tensor:
  obj: Entity = env.scene[object_cfg.name]
  ref: Entity = env.scene[ref_asset_cfg.name]
  local = _box_surface_points(num_points, env.device).unsqueeze(0).repeat(env.num_envs, 1, 1)
  object_quat_w = obj.data.root_link_quat_w.unsqueeze(1).repeat(1, num_points, 1)
  object_pos_w = obj.data.root_link_pos_w.unsqueeze(1).repeat(1, num_points, 1)
  world = quat_apply(object_quat_w, local) + object_pos_w
  ref_pos = ref.data.root_link_pos_w.unsqueeze(1).repeat(1, num_points, 1)
  ref_quat = ref.data.root_link_quat_w.unsqueeze(1).repeat(1, num_points, 1)
  pts_b, _ = subtract_frame_transforms(ref_pos, ref_quat, world, None)
  return pts_b.reshape(env.num_envs, -1) if flatten else pts_b


def fingers_contact_force_b(
  env,
  contact_sensor_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  forces = [
    env.scene.sensors[name].data.force.reshape(env.num_envs, -1, 3)[:, 0, :]
    for name in contact_sensor_names
  ]
  force_w = torch.stack(forces, dim=1)
  # print(torch.norm(force_w, dim=-1))
  robot: Entity = env.scene[asset_cfg.name]
  root_quat_w = robot.data.root_link_quat_w
  force_b = quat_apply_inverse(
    root_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1),
    force_w,
  )
  finite_mask = torch.isfinite(force_b)
  if not torch.all(finite_mask):
    force_w_finite = torch.isfinite(force_w).all(dim=-1)
    root_quat_finite = torch.isfinite(root_quat_w).all(dim=-1)
    force_b_finite = finite_mask.all(dim=-1)
    bad_env_ids = torch.where(~force_b_finite.any(dim=1) | ~force_b_finite.all(dim=1))[0]
    bad_sensor_ids = torch.where(~force_b_finite.any(dim=0) | ~force_b_finite.all(dim=0))[0]
    force_w_norm = torch.linalg.norm(torch.nan_to_num(force_w), dim=-1)
    force_b_norm = torch.linalg.norm(torch.nan_to_num(force_b), dim=-1)
    print(
      "[DexSuiteContactDebug] Non-finite force_b detected. "
      f"bad_env_ids={bad_env_ids[:10].detach().cpu().tolist()}, "
      f"bad_sensor_names={[contact_sensor_names[i] for i in bad_sensor_ids[:10].detach().cpu().tolist()]}, "
      f"force_w_all_finite={bool(force_w_finite.all().item())}, "
      f"root_quat_all_finite={bool(root_quat_finite.all().item())}, "
      f"force_b_all_finite={bool(force_b_finite.all().item())}, "
      f"force_w_norm_max={float(force_w_norm.max().item())}, "
      f"force_b_norm_max={float(force_b_norm.max().item())}"
    )
    force_b = torch.where(finite_mask, force_b, torch.ones_like(force_b)*60)
  return force_b.reshape(env.num_envs, -1)
