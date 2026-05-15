from mjlab.tasks.registry import register_mjlab_task

from brainco_mjlab_tasks.dexsuite.config.brainco.env_cfgs import (
  brainco_arm_lift_env_cfg,
  brainco_arm_reorient_env_cfg,
  # brainco_revo2_lift_env_cfg,
)
from brainco_mjlab_tasks.dexsuite.config.brainco.rl_cfg import (
  brainco_dexsuite_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="BrainCo-Dexsuite-Arm-BrainCo-Lift-v0",
  env_cfg=brainco_arm_lift_env_cfg(),
  play_env_cfg=brainco_arm_lift_env_cfg(play=True),
  rl_cfg=brainco_dexsuite_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="BrainCo-Dexsuite-Arm-BrainCo-Reorient-v0",
  env_cfg=brainco_arm_reorient_env_cfg(),
  play_env_cfg=brainco_arm_reorient_env_cfg(play=True),
  rl_cfg=brainco_dexsuite_ppo_runner_cfg(),
)

# register_mjlab_task(
#   task_id="BrainCo-Dexsuite-Revo2-Lift-v0",
#   env_cfg=brainco_revo2_lift_env_cfg(),
#   play_env_cfg=brainco_revo2_lift_env_cfg(play=True),
#   rl_cfg=brainco_dexsuite_ppo_runner_cfg(),
# )

