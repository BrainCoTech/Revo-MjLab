from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_from_euler_xyz


_GOAL_ROT_ATTR = "_brainco_reorient_goal_rot_wxyz"


def _resolve_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        start, stop, step = env_ids.indices(env.num_envs)
        return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _initial_goal(
    env,
    initial_goal_quat_wxyz: tuple[float, float, float, float],
) -> torch.Tensor:
    return torch.tensor(
        initial_goal_quat_wxyz,
        dtype=torch.float32,
        device=env.device,
    ).repeat(env.num_envs, 1)


def get_goal_rot_wxyz(
    env,
    initial_goal_quat_wxyz: tuple[float, float, float, float],
) -> torch.Tensor:
    goal = getattr(env, _GOAL_ROT_ATTR, None)
    if goal is None or goal.shape != (env.num_envs, 4):
        goal = _initial_goal(env, initial_goal_quat_wxyz)
        setattr(env, _GOAL_ROT_ATTR, goal)
    return goal


def reset_goal_rot_wxyz(
    env,
    env_ids: torch.Tensor | slice | None,
    initial_goal_quat_wxyz: tuple[float, float, float, float],
) -> None:
    goal = get_goal_rot_wxyz(env, initial_goal_quat_wxyz)
    ids = _resolve_env_ids(env, env_ids)
    goal[ids] = _initial_goal(env, initial_goal_quat_wxyz)[ids]


def advance_goal_yaw_wxyz(
    env,
    env_ids: torch.Tensor,
    yaw_offset: float,
    initial_goal_quat_wxyz: tuple[float, float, float, float],
) -> None:
    goal = get_goal_rot_wxyz(env, initial_goal_quat_wxyz)
    if env_ids.numel() == 0:
        return
    _, _, current_yaw = euler_xyz_from_quat(goal[env_ids])
    zeros = torch.zeros_like(current_yaw)
    goal[env_ids] = quat_from_euler_xyz(zeros, zeros, current_yaw + yaw_offset)
