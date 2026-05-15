from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RewardWeightStage(TypedDict):
  step: int
  weight: float


class RewardParamStage(TypedDict):
  step: int
  value: Any


Range = tuple[float, float]


def _lerp_range(start: Range, end: Range, alpha: float) -> Range:
  return (
    start[0] + (end[0] - start[0]) * alpha,
    start[1] + (end[1] - start[1]) * alpha,
  )


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  del env_ids
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter >= stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight], device=env.device)


def reward_param(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_name: str,
  param_stages: list[RewardParamStage],
) -> torch.Tensor:
  del env_ids
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  value = reward_term_cfg.params[param_name]
  for stage in param_stages:
    if env.common_step_counter >= stage["step"]:
      value = stage["value"]
  reward_term_cfg.params[param_name] = value
  scalar = float(value) if isinstance(value, (int, float)) else 0.0
  return torch.tensor([scalar], device=env.device)


def object_reset_and_target_workspace_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  reset_event_name: str,
  start_step: int,
  end_step: int,
  object_start_pose_range: dict[str, Range],
  object_final_pose_range: dict[str, Range],
  target_start_ranges: dict[str, Range],
  target_final_ranges: dict[str, Range],
) -> dict[str, torch.Tensor]:
  """Gradually expand object reset and target sampling ranges."""
  del env_ids
  span = max(end_step - start_step, 1)
  progress = min(max((env.common_step_counter - start_step) / span, 0.0), 1.0)
  alpha = progress * progress * (3.0 - 2.0 * progress)

  event_cfg = env.event_manager.get_term_cfg(reset_event_name)
  pose_range = dict(event_cfg.params["pose_range"])
  for key, start_range in object_start_pose_range.items():
    final_range = object_final_pose_range.get(key, start_range)
    pose_range[key] = _lerp_range(start_range, final_range, alpha)
  event_cfg.params["pose_range"] = pose_range

  command_cfg = env.command_manager.get_term_cfg(command_name)
  target_ranges = command_cfg.ranges
  for key, start_range in target_start_ranges.items():
    final_range = target_final_ranges.get(key, start_range)
    setattr(target_ranges, key, _lerp_range(start_range, final_range, alpha))

  return {
    "progress": torch.tensor(progress, device=env.device),
    "object_reset_x_span": torch.tensor(
      pose_range["x"][1] - pose_range["x"][0], device=env.device
    ),
    "object_reset_y_span": torch.tensor(
      pose_range["y"][1] - pose_range["y"][0], device=env.device
    ),
    "object_reset_z_max": torch.tensor(pose_range["z"][1], device=env.device),
    "target_x_span": torch.tensor(
      target_ranges.pos_x[1] - target_ranges.pos_x[0], device=env.device
    ),
    "target_y_span": torch.tensor(
      target_ranges.pos_y[1] - target_ranges.pos_y[0], device=env.device
    ),
    "target_z_max": torch.tensor(target_ranges.pos_z[1], device=env.device),
  }
