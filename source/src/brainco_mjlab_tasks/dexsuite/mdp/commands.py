from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  combine_frame_transforms,
  compute_pose_error,
  quat_from_euler_xyz,
  quat_unique,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class ObjectUniformPoseCommand(CommandTerm):
  cfg: "ObjectUniformPoseCommandCfg"

  def __init__(self, cfg: "ObjectUniformPoseCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    self.object: Entity = env.scene[cfg.object_name]
    self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
    self.pose_command_b[:, 3] = 1.0
    self.pose_command_w = torch.zeros_like(self.pose_command_b)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.pose_command_b

  def _update_metrics(self) -> None:
    self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
      self.robot.data.root_link_pos_w,
      self.robot.data.root_link_quat_w,
      self.pose_command_b[:, :3],
      self.pose_command_b[:, 3:],
    )
    pos_error, rot_error = compute_pose_error(
      self.pose_command_w[:, :3],
      self.pose_command_w[:, 3:],
      self.object.data.root_link_pos_w,
      self.object.data.root_link_quat_w,
    )
    self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
    self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = self.cfg.ranges
    self.pose_command_b[env_ids, 0] = sample_uniform(
      r.pos_x[0], r.pos_x[1], (len(env_ids),), device=self.device
    )
    self.pose_command_b[env_ids, 1] = sample_uniform(
      r.pos_y[0], r.pos_y[1], (len(env_ids),), device=self.device
    )
    self.pose_command_b[env_ids, 2] = sample_uniform(
      r.pos_z[0], r.pos_z[1], (len(env_ids),), device=self.device
    )
    if self.cfg.position_only:
      self.pose_command_b[env_ids, 3:] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0), device=self.device, dtype=self.pose_command_b.dtype
      )
    else:
      roll = sample_uniform(r.roll[0], r.roll[1], (len(env_ids),), device=self.device)
      pitch = sample_uniform(
        r.pitch[0], r.pitch[1], (len(env_ids),), device=self.device
      )
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (len(env_ids),), device=self.device)
      quat = quat_from_euler_xyz(roll, pitch, yaw)
      self.pose_command_b[env_ids, 3:] = (
        quat_unique(quat) if self.cfg.make_quat_unique else quat
      )

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    for batch in env_indices:
      visualizer.add_sphere(
        center=self.pose_command_w[batch, :3].cpu().numpy(),
        radius=0.03,
        color=(1.0, 0.4, 0.1, 0.4),
        label=f"brainco_target_{batch}",
      )


@dataclass(kw_only=True)
class ObjectUniformPoseCommandCfg(CommandTermCfg):
  asset_name: str
  object_name: str
  make_quat_unique: bool = False
  position_only: bool = False

  @dataclass
  class Ranges:
    pos_x: tuple[float, float] = (0.4, 0.5)
    pos_y: tuple[float, float] = (-0.15, 0.15)
    pos_z: tuple[float, float] = (0.35, 0.5)
    roll: tuple[float, float] = (-3.14, 3.14)
    pitch: tuple[float, float] = (-3.14, 3.14)
    yaw: tuple[float, float] = (0.0, 0.0)

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: "ManagerBasedRlEnv") -> ObjectUniformPoseCommand:
    return ObjectUniformPoseCommand(self, env)
