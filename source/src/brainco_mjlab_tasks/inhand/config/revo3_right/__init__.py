from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import revo3_right_inhand_rotate_env_cfg
from .rl_cfg import revo3_right_inhand_rotate_ppo_cfg


register_mjlab_task(
  task_id="BrainCo-Revo3-Right-Inhand-Rotate-v0",
  env_cfg=revo3_right_inhand_rotate_env_cfg(),
  play_env_cfg=revo3_right_inhand_rotate_env_cfg(play=True),
  rl_cfg=revo3_right_inhand_rotate_ppo_cfg(),
)
