"""Revo3 right hand in-hand rotation environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg

from brainco_mjlab_tasks.inhand.inhand_env_cfg import revo3_right_inhand_env_cfg


def revo3_right_inhand_rotate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Revo3 right hand in-hand cube rotation environment configuration."""
  return revo3_right_inhand_env_cfg(play=play)
