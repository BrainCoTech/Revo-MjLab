from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import combine_frame_transforms, compute_pose_error, quat_apply, quat_apply_inverse


_THUMB_CONTACT_SENSOR_NAMES = (
  "right_thumb_DIP_Link_object_s",
  "right_thumb_touch_link_object_s",
  "right_thumb_distal_link_object_s",
)
_OTHER_CONTACT_SENSOR_NAMES = (
  "right_index_DIP_Link_object_s",
  "right_middle_DIP_Link_object_s",
  "right_ring_DIP_Link_object_s",
  "right_little_DIP_Link_object_s",
  "right_index_touch_link_object_s",
  "right_middle_touch_link_object_s",
  "right_ring_touch_link_object_s",
  "right_pinky_touch_link_object_s",
  "right_index_distal_link_object_s",
  "right_middle_distal_link_object_s",
  "right_ring_distal_link_object_s",
  "right_pinky_distal_link_object_s",
)


def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  return torch.linalg.norm(torch.nan_to_num(x), dim=dim)


def _target_pose_w(env, command_name: str, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  return combine_frame_transforms(
    asset.data.root_link_pos_w,
    asset.data.root_link_quat_w,
    command[:, :3],
    command[:, 3:7],
  )


def _contact_force_stack(env, sensor_names: tuple[str, ...]) -> torch.Tensor:
  forces = []
  for name in sensor_names:
    if name not in env.scene.sensors:
      continue
    force = env.scene.sensors[name].data.force.reshape(env.num_envs, -1, 3)[:, 0, :]
    forces.append(_safe_norm(force, dim=-1))
  if not forces:
    return torch.zeros(env.num_envs, 0, device=env.device)
  return torch.stack(forces, dim=1)


def _force_target_score(force_norm: torch.Tensor, target_force: float, k: float) -> torch.Tensor:
  distance = torch.abs(force_norm - target_force)
  raw = k / (k + distance + 1.0e-6)
  no_contact = k / (k + target_force + 1.0e-6)
  return ((raw - no_contact) / (1.0 - no_contact)).clamp(0.0, 1.0)


def _grasp_contact_quality(env, target_force: float, k: float) -> torch.Tensor:
  thumb = _contact_force_stack(env, _THUMB_CONTACT_SENSOR_NAMES)
  other = _contact_force_stack(env, _OTHER_CONTACT_SENSOR_NAMES)
  if thumb.shape[1] == 0 or other.shape[1] == 0:
    return torch.zeros(env.num_envs, device=env.device)
  thumb_score = _force_target_score(thumb, target_force, k).max(dim=1).values
  other_score = _force_target_score(other, target_force, k).max(dim=1).values
  return torch.sqrt(torch.clamp(thumb_score * other_score, min=0.0, max=1.0))


def _box_surface_error(
  points_w: torch.Tensor,
  object_pos_w: torch.Tensor,
  object_quat_w: torch.Tensor,
  half_extents: tuple[float, float, float],
  penetration_weight: float,
) -> torch.Tensor:
  half_extents_t = torch.tensor(half_extents, device=points_w.device, dtype=points_w.dtype)
  object_quat = object_quat_w.unsqueeze(1).expand(-1, points_w.shape[1], -1)
  points_o = quat_apply_inverse(object_quat, points_w - object_pos_w.unsqueeze(1))
  q = points_o.abs() - half_extents_t
  outside_distance = torch.linalg.norm(q.clamp_min(0.0), dim=-1)
  signed_distance = outside_distance + q.max(dim=-1).values.clamp_max(0.0)
  penetration = (-signed_distance).clamp_min(0.0)
  return signed_distance.abs() + penetration_weight * penetration


def action_rate_l2_clamped(env) -> torch.Tensor:
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  ).clamp(max=1000.0)


def action_l2_clamped(env) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(max=1000.0)


def object_drop_penalty(
  env,
  min_height: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  obj: Entity = env.scene[asset_cfg.name]
  object_pos = obj.data.root_link_pos_w - env.scene.env_origins
  object_height = object_pos[:, 2]
  dropped = (~torch.isfinite(object_height)) | (object_height < min_height)
  return dropped.float()


def object_ee_distance(
  env,
  std: float,
  object_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward selected finger bodies approaching the object center."""
  asset: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  ee_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  obj_pos = obj.data.root_link_pos_w
  distance = torch.norm(ee_pos - obj_pos[:, None, :], dim=-1).mean(dim=-1)
  return 1.0 - torch.tanh(distance / std)


def palm_to_object_surface_distance(
  env,
  std: float,
  object_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
  half_extents: tuple[float, float, float] = (0.04, 0.04, 0.04),
  penetration_weight: float = 3.0,
  body_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
  """Reward the palm body approaching the object surface."""
  asset: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  palm_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  if body_offset is not None:
    offset = torch.tensor(body_offset, device=palm_pos.device, dtype=palm_pos.dtype)
    body_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    offset = offset.reshape(1, 1, 3).expand_as(palm_pos)
    palm_pos = palm_pos + quat_apply(body_quat, offset)
  surface_error = _box_surface_error(
    palm_pos,
    obj.data.root_link_pos_w,
    obj.data.root_link_quat_w,
    half_extents,
    penetration_weight,
  )
  return torch.mean(1.0 - torch.tanh(surface_error / std), dim=-1)


def any_finger_contact(env, threshold: float) -> torch.Tensor:
  sensor_names = [
    "right_thumb_DIP_Link_object_s",
    "right_index_DIP_Link_object_s",
    "right_middle_DIP_Link_object_s",
    "right_ring_DIP_Link_object_s",
    "right_little_DIP_Link_object_s",
    "right_thumb_touch_link_object_s",
    "right_index_touch_link_object_s",
    "right_middle_touch_link_object_s",
    "right_ring_touch_link_object_s",
    "right_pinky_touch_link_object_s",
  ]
  found = []
  for name in sensor_names:
    if name in env.scene.sensors:
      force = env.scene.sensors[name].data.force.reshape(env.num_envs, -1, 3)[:, 0, :]
      found.append(torch.norm(force, dim=-1))
  if not found:
    return torch.zeros(env.num_envs, device=env.device)
  return (torch.stack(found, dim=1) > threshold).any(dim=1).float()


def good_finger_contact(env, threshold: float, k: float = 0.5) -> torch.Tensor:
  """Reward thumb/opposing-finger forces close to a target instead of just large."""
  return _grasp_contact_quality(env, target_force=threshold, k=k)


def position_command_error_tanh(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  align_asset_cfg: SceneEntityCfg,
  contact_target_force: float = 3.0,
  contact_k: float = 0.5,
) -> torch.Tensor:
  obj: Entity = env.scene[align_asset_cfg.name]
  des_pos_w, _ = _target_pose_w(env, command_name, asset_cfg)
  distance = _safe_norm(obj.data.root_link_pos_w - des_pos_w, dim=1)
  return (1.0 - torch.tanh(distance / std)) * _grasp_contact_quality(env, contact_target_force, contact_k)


def orientation_command_error_tanh(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  align_asset_cfg: SceneEntityCfg,
  contact_target_force: float = 3.0,
  contact_k: float = 0.5,
) -> torch.Tensor:
  obj: Entity = env.scene[align_asset_cfg.name]
  _, des_quat_w = _target_pose_w(env, command_name, asset_cfg)
  _, rot_err = compute_pose_error(
    obj.data.root_link_pos_w,
    obj.data.root_link_quat_w,
    obj.data.root_link_pos_w,
    des_quat_w,
  )
  return (1.0 - torch.tanh(_safe_norm(rot_err, dim=1) / std)) * _grasp_contact_quality(
    env, contact_target_force, contact_k
  )


def success_reward(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  align_asset_cfg: SceneEntityCfg,
  pos_std: float,
  rot_std: float | None = None,
  contact_target_force: float = 3.0,
  contact_k: float = 0.5,
) -> torch.Tensor:
  obj: Entity = env.scene[align_asset_cfg.name]
  des_pos_w, des_quat_w = _target_pose_w(env, command_name, asset_cfg)
  pos_err, rot_err = compute_pose_error(
    des_pos_w,
    des_quat_w,
    obj.data.root_link_pos_w,
    obj.data.root_link_quat_w,
  )
  contact_quality = _grasp_contact_quality(env, contact_target_force, contact_k)
  pos_dist = _safe_norm(pos_err, dim=1)
  if rot_std is None:
    return ((1.0 - torch.tanh(pos_dist / pos_std)) ** 2) * contact_quality
  rot_dist = _safe_norm(rot_err, dim=1)
  return (
    (1.0 - torch.tanh(pos_dist / pos_std))
    * (1.0 - torch.tanh(rot_dist / rot_std))
    * contact_quality
  )


def dexterous_staged_position_reward(
  env,
  command_name: str,
  object_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
  reach_std: float,
  bring_std: float,
  ungated_bring_weight: float = 0.15,
) -> torch.Tensor:
  """Stage reward: approach the surface first, then move the object to target."""
  obj: Entity = env.scene[object_cfg.name]
  reach = object_ee_distance(
    env,
    std=reach_std,
    object_cfg=object_cfg,
    asset_cfg=asset_cfg,
  )
  des_pos_w, _ = _target_pose_w(env, command_name, asset_cfg)
  position_error = _safe_norm(des_pos_w - obj.data.root_link_pos_w, dim=-1)
  bring = 1.0 - torch.tanh(position_error / bring_std)
  return reach * (1.0 + bring) + ungated_bring_weight * bring


def object_lift_progress_reward(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  align_asset_cfg: SceneEntityCfg,
  reference_height: float,
) -> torch.Tensor:
  obj: Entity = env.scene[align_asset_cfg.name]
  des_pos_w, _ = _target_pose_w(env, command_name, asset_cfg)
  denom = (des_pos_w[:, 2] - reference_height).clamp_min(1.0e-3)
  return ((obj.data.root_link_pos_w[:, 2] - reference_height) / denom).clamp(0.0, 1.0)


def grasped_object_lift_progress_reward(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  align_asset_cfg: SceneEntityCfg,
  reference_height: float,
  contact_target_force: float,
  contact_k: float,
  contact_floor: float = 0.25,
) -> torch.Tensor:
  progress = object_lift_progress_reward(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
    align_asset_cfg=align_asset_cfg,
    reference_height=reference_height,
  )
  contact_quality = _grasp_contact_quality(env, contact_target_force, contact_k)
  lift_gate = contact_floor + (1.0 - contact_floor) * contact_quality
  return progress * lift_gate


def object_height_tracking_reward(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  align_asset_cfg: SceneEntityCfg,
  std: float,
  contact_target_force: float,
  contact_k: float,
  contact_floor: float = 0.20,
) -> torch.Tensor:
  obj: Entity = env.scene[align_asset_cfg.name]
  des_pos_w, _ = _target_pose_w(env, command_name, asset_cfg)
  height_error = torch.abs(des_pos_w[:, 2] - obj.data.root_link_pos_w[:, 2])
  contact_quality = _grasp_contact_quality(env, contact_target_force, contact_k)
  height_gate = contact_floor + (1.0 - contact_floor) * contact_quality
  return (1.0 - torch.tanh(height_error / std)) * height_gate


def object_upward_velocity_reward(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  align_asset_cfg: SceneEntityCfg,
  reference_height: float,
  target_speed: float,
  contact_target_force: float,
  contact_k: float,
) -> torch.Tensor:
  obj: Entity = env.scene[align_asset_cfg.name]
  progress = object_lift_progress_reward(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
    align_asset_cfg=align_asset_cfg,
    reference_height=reference_height,
  )
  upward_speed = torch.nan_to_num(obj.data.root_link_lin_vel_w[:, 2]).clamp_min(0.0)
  speed_score = (upward_speed / target_speed).clamp(0.0, 1.0)
  return speed_score * (1.0 - progress).clamp(0.0, 1.0) * _grasp_contact_quality(
    env, contact_target_force, contact_k
  )


def joint_velocity_hinge_penalty(
  env,
  max_vel: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  excess = (torch.nan_to_num(robot.data.joint_vel[:, asset_cfg.joint_ids]).abs() - max_vel).clamp_min(0.0)
  return torch.sum(excess.square(), dim=-1)


class stable_object_lift_reward:
  def __init__(self, cfg: RewardTermCfg, env):
    del cfg
    self._stable_steps = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    reference_height: float,
    min_progress: float,
    lin_vel_std: float,
    ang_vel_std: float,
    contact_target_force: float,
    contact_k: float,
    target_hold_steps: int,
  ) -> torch.Tensor:
    obj: Entity = env.scene[align_asset_cfg.name]
    progress = object_lift_progress_reward(
      env,
      command_name=command_name,
      asset_cfg=asset_cfg,
      align_asset_cfg=align_asset_cfg,
      reference_height=reference_height,
    )
    contact_quality = _grasp_contact_quality(env, contact_target_force, contact_k)
    lin_speed = _safe_norm(obj.data.root_link_lin_vel_w, dim=-1)
    ang_speed = _safe_norm(obj.data.root_link_ang_vel_w, dim=-1)
    vel_score = torch.exp(-torch.square(lin_speed / lin_vel_std) - torch.square(ang_speed / ang_vel_std))
    stable_now = (progress > min_progress) & (contact_quality > 0.25)
    self._stable_steps = torch.where(
      stable_now,
      self._stable_steps + 1.0,
      torch.zeros_like(self._stable_steps),
    )
    hold_steps = max(int(target_hold_steps), 1)
    hold_score = (self._stable_steps / hold_steps).clamp(0.0, 1.0)
    return progress * contact_quality * vel_score * hold_score

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._stable_steps[env_ids] = 0.0
