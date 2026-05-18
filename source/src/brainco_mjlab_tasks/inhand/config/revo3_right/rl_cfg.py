from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def revo3_right_inhand_rotate_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 512, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.016,
      max_grad_norm=1.0,
    ),
    experiment_name="revo3_right_inhand_rotate",
    logger="tensorboard",
    upload_model=False,
    save_interval=250,
    num_steps_per_env=32,
    max_iterations=10_000,
    clip_actions=1.0,
  )


revo3_right_inhand_rotate_rl_cfg = revo3_right_inhand_rotate_ppo_cfg
