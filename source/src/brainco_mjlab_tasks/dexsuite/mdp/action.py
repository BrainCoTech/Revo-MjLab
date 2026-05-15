from __future__ import annotations

import re
from dataclasses import dataclass, field
import math
import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


_DEXSUITE_JOINT_SCALE = {
  r"Joint1_R": 1.0,
  r"Joint2_R": 1.2,
  r"Joint3_R": 1.0,
  r"Joint4_R": 1.0,
  r"Joint5_R": 0.8,
  r"Joint6_R": 0.45,
  r"Joint7_R": 0.6,
  r"right_thumb_CMP_joint": 0.89,
  r"right_thumb_CMR_joint": 0.9,
  r"right_thumb_MCP_joint": 0.4,
  r"right_thumb_PIP_joint": 0.7,
  r"right_thumb_DIP_joint": 0.7,
  r"right_(index|middle|ring|little)_MPR_joint": 0.22,
  r"right_(index|middle|ring|little)_MCP_joint": 0.69,
  r"right_(index|middle|ring|little)_PIP_joint": 0.69,
  r"right_(index|middle|ring|little)_DIP_joint": 0.69,
}



@dataclass(kw_only=True)
class TanhJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action with an extra tanh layer on policy actions.

  ``joint_scale`` lives here instead of in the task config so the policy-output
  processing path can be inspected and adjusted in one place.
  """
  sine_period_s: float = 6.0
  sine_amplitude: float = 1.0
  sine_phase_span: float = 2.0 * math.pi

  joint_scale: float | dict[str, float] = field(
    default_factory=lambda: dict(_DEXSUITE_JOINT_SCALE)
  )
  scale_from_clip: bool = True
  debug_print_actions: bool = False
  debug_print_interval: int = 50
  debug_print_first_n: int = 8

  def __post_init__(self):
    self.scale = self._make_joint_scale()
    super().__post_init__()

  def build(self, env) -> "TanhJointPositionAction":
    return TanhJointPositionAction(self, env)

  def _make_joint_scale(self) -> float | dict[str, float]:
    if not self.scale_from_clip or self.clip is None:
      return self.joint_scale

    scale = {}
    for pattern, bounds in self.clip.items():
      lower, upper = bounds
      scale[pattern] = 0.5 * (float(upper) - float(lower))
    return scale


class TanhJointPositionAction(JointPositionAction):
  cfg: TanhJointPositionActionCfg

  def __init__(self, cfg: TanhJointPositionActionCfg, env):
    super().__init__(cfg=cfg, env=env)

    model_limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    limits = self._resolve_joint_limits_from_cfg(model_limits)
    finite = torch.isfinite(limits).all(dim=-1)
    self._lower = torch.where(finite, limits[..., 0], model_limits[..., 0])
    self._upper = torch.where(finite, limits[..., 1], model_limits[..., 1])
    self._tanh_actions = torch.zeros_like(self._raw_actions)
    self._debug_print_counter = 0
    self._phase_offsets = torch.linspace(
      0.0,
      cfg.sine_phase_span,
      self.action_dim,
      device=self.device,
      dtype=self._raw_actions.dtype,
    ).unsqueeze(0)
  
  def _episode_time(self) -> torch.Tensor:
    return self._env.episode_length_buf.to(self._raw_actions.dtype).unsqueeze(1) * self._env.step_dt

  def _make_sine_raw_actions(self) -> torch.Tensor:
    period = max(self.cfg.sine_period_s, self._env.step_dt)
    phase = 2.0 * math.pi * self._episode_time() / period + self._phase_offsets
    return self.cfg.sine_amplitude * torch.sin(phase)

  @property
  def tanh_action(self) -> torch.Tensor:
    return self._tanh_actions

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._tanh_actions[:] = torch.tanh(actions)
    # self._tanh_actions[:,:] = self._make_sine_raw_actions()[:,:]
    # self._tanh_actions[:,0:7] = 0.0  # Disable sine for arm joints, only use it for fingers
    target = self._tanh_actions * self._scale + self._offset
    self._processed_actions[:] = torch.clamp(target, self._lower, self._upper)
    # self._maybe_print_actions()
    # print("RAW ACTIONS:", actions)
    # print("TANH ACTIONS:", self._tanh_actions)
    # print("lower:", self._lower)
    # print("upper:", self._upper)

  def _resolve_joint_limits_from_cfg(self, fallback: torch.Tensor) -> torch.Tensor:
    limits = fallback.clone()
    if self.cfg.clip is None:
      return limits

    for pattern, bounds in self.cfg.clip.items():
      for index, joint_name in enumerate(self._target_names):
        if re.fullmatch(pattern, joint_name) is None:
          continue
        limits[:, index, 0] = bounds[0]
        limits[:, index, 1] = bounds[1]
    return limits

  def _maybe_print_actions(self) -> None:
    if not self.cfg.debug_print_actions:
      return

    interval = max(int(self.cfg.debug_print_interval), 1)
    if self._debug_print_counter % interval != 0:
      self._debug_print_counter += 1
      return

    count = max(int(self.cfg.debug_print_first_n), 1)
    env_id = 0
    raw = self._raw_actions[env_id].detach().cpu()
    tanh_action = self._tanh_actions[env_id].detach().cpu()
    target = self._processed_actions[env_id].detach().cpu()
    print(
      "[DexSuiteAction] "
      f"step={self._debug_print_counter} "
      f"raw_min={raw.min().item():+.3f} raw_max={raw.max().item():+.3f} "
      f"tanh_min={tanh_action.min().item():+.3f} tanh_max={tanh_action.max().item():+.3f} "
      f"raw[:{count}]={raw[:count].tolist()} "
      f"tanh[:{count}]={tanh_action[:count].tolist()} "
      f"target[:{count}]={target[:count].tolist()}"
    )
    self._debug_print_counter += 1
