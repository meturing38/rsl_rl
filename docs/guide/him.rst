Hybrid Internal Model Policies
==============================

RSL-RL includes a TensorDict-native implementation of the Hybrid Internal Model (HIM) policy for blind locomotion.
It combines privileged-state estimation, temporal convolutions, cross-attention, and a soft mixture-of-experts actor.
The implementation uses the standard :class:`~rsl_rl.runners.on_policy_runner.OnPolicyRunner` and a standard MLP
critic.

Observation Contract
--------------------

The environment must return these entries in its observation ``TensorDict``:

``policy_history``
  A flattened fixed-length history with shape ``[num_envs, temporal_steps * num_one_step_obs]``. Every frame must use
  the same feature order. Both oldest-first and newest-first histories are supported.

``base_lin_vel``
  The privileged true base linear velocity with shape ``[num_envs, 3]``. It is used only as the next-state estimator
  target and does not need to be part of the actor observation set.

``privileged``
  The critic observation. Its contents are task-specific and may include terrain scans and domain-randomization state.

Isaac Lab's observation manager should create and flatten the history before returning it through the RSL-RL wrapper.
The policy is therefore feed-forward from RSL-RL's perspective and does not maintain hidden state across rollouts.
Samples ending an episode are excluded from estimator learning because most vectorized environments return the reset
observation instead of the terminal observation.

Example Configuration
---------------------

The following dictionary uses a six-frame history with 53 values per frame and predicts only three-dimensional base
linear velocity:

.. code-block:: yaml

  num_steps_per_env: 24
  save_interval: 100
  obs_groups:
    actor: [policy_history]
    critic: [privileged]
  algorithm:
    class_name: HIMPPO
    num_learning_epochs: 5
    num_mini_batches: 4
    learning_rate: 0.001
    estimator_learning_rate: 0.001
    estimator_num_learning_epochs: 1
    estimator_num_mini_batches: 4
    estimator_max_grad_norm: 1.0
  actor:
    class_name: HIMActorModel
    history_obs_group: policy_history
    estimator_target_group: base_lin_vel
    temporal_steps: 6
    num_one_step_obs: 53
    num_command_obs: 3
    history_order: oldest_first
    hidden_dims: [512, 256, 128]
    estimator_cfg:
      num_latent: 32
      num_prototypes: 32
      encoder_hidden_dims: [128, 64]
      target_hidden_dims: [128, 64]
    history_encoder_cfg:
      hidden_dims: [64, 128]
    attention_cfg:
      model_dim: 128
      num_heads: 8
    moe_cfg:
      num_experts: 4
      gate_hidden_dims: [32]
    distribution_cfg:
      class_name: GaussianDistribution
      init_std: 1.0
      std_type: scalar
  critic:
    class_name: MLPModel
    hidden_dims: [512, 256, 128]

``num_command_obs`` specifies the command prefix in each observation frame. The target encoder receives the next
current frame without this prefix, followed by ``base_lin_vel``. Set it to zero when commands are not at the beginning
of each frame.

Training and Deployment
-----------------------

:class:`~rsl_rl.algorithms.him_ppo.HIMPPO` keeps PPO and estimator optimization separate. PPO gradients do not update
the estimator; the estimator uses next-state regression and swapped prototype-assignment losses. Checkpoints retain the
standard actor, critic, and optimizer keys and add ``estimator_optimizer_state_dict``.

JIT and ONNX exports retain the standard single ``obs`` input and ``actions`` output. Deployment must maintain the same
history buffer, frame order, feature order, and flattening convention used during training. The privileged
``base_lin_vel`` entry is not required for exported-policy inference.
