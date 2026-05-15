# Revo-MjLab
This repo contains tasks for Revo3 base on MjLab, Standalone BrainCo task package for `mjlab`.

Current migration status:

- Manager-based Dexsuite tasks are migrated into a native `mjlab.tasks` package.
- Registered tasks:
  - `BrainCo-Dexsuite-Arm-BrainCo-Lift-v0`

- Direct BrainCo in-hand tasks are not yet ported in this package.

## 安装

```bash
conda activate mjlab
git clone https://github.com/BrainCoTech/Revo-MjLab.git
cd Revo-MjLab/srouce/
pip install -e .
```


## 任务总览

### Revo3

| Task Description | Task Name | Weight | Demo |
| --- | --- | --- | --- |
| Grasp (Lift) | `BrainCo-Dexsuite-Arm-BrainCo-Lift-v0` | `-` | <img src="image/BrainCo-Dexsuite-Arm-BrainCo-Lift-v0.gif" width="320"/> |


## 配置文件

### Revo3

#### 1) Grasp (Lift)
- Env config: `source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/arm_brainco/dexsuite_arm_brain_env_cfg_grasp.py` (`DexsuiteArmBrainCoLiftEnvCfg`)
- Base config used: `source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/dexsuite_env_cfg_grasp.py`

## 训练与演示

#### Train command

Training does not provide an IsaacLab-style live interactive scene viewer.
```python
train BrainCo-Dexsuite-Arm-BrainCo-Lift-v0 \
    --video True  \
    --env.scene.num-envs  4096\
    --gpu-ids '[0]' \
```
#### Play command

render by local machine

```python
play BrainCo-Dexsuite-Arm-BrainCo-Lift-v0 \
    --wandb-run-path <user>/<project_id>/<task_id>  \
    --num-envs 1 \
```
render by the website
```
play BrainCo-Dexsuite-Arm-BrainCo-Lift-v0 \
    --wandb-run-path <user>/<project_id>/<task_id>  \
    --num-envs 1 \
    --viewer viser \
```