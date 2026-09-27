# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for HIMPPO and HIM rollout storage."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import HIMPPO
from rsl_rl.models import HIMActorModel, MLPModel
from rsl_rl.storage import HIMRolloutStorage

NUM_ENVS = 4
NUM_STEPS = 4
TEMPORAL_STEPS = 3
ONE_STEP_DIM = 5
NUM_ACTIONS = 2


class _DummyEnv:
    """Minimal environment metadata required by construct_algorithm."""

    num_envs = NUM_ENVS
    num_actions = NUM_ACTIONS


def _make_obs() -> TensorDict:
    """Create observations satisfying the HIM TensorDict contract."""
    return TensorDict(
        {
            "policy_history": torch.randn(NUM_ENVS, TEMPORAL_STEPS * ONE_STEP_DIM),
            "base_lin_vel": torch.randn(NUM_ENVS, 3),
            "privileged": torch.randn(NUM_ENVS, 7),
        },
        batch_size=[NUM_ENVS],
    )


def _build_algorithm() -> tuple[HIMPPO, TensorDict]:
    """Build a small CPU HIMPPO instance."""
    obs = _make_obs()
    obs_groups = {"actor": ["policy_history"], "critic": ["privileged"]}
    actor = HIMActorModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        history_obs_group="policy_history",
        estimator_target_group="base_lin_vel",
        temporal_steps=TEMPORAL_STEPS,
        num_one_step_obs=ONE_STEP_DIM,
        hidden_dims=[16, 8],
        estimator_cfg={"encoder_hidden_dims": [16], "target_hidden_dims": [16], "num_latent": 4},
        history_encoder_cfg={"hidden_dims": [8]},
        attention_cfg={"model_dim": 8, "num_heads": 2},
        moe_cfg={"num_experts": 2, "gate_hidden_dims": [4]},
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[16, 8])
    storage = HIMRolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], actor.estimator_target_dim)
    algorithm = HIMPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        estimator_num_learning_epochs=1,
        estimator_num_mini_batches=2,
        schedule="fixed",
    )
    return algorithm, obs


def test_done_transitions_are_excluded_from_estimator_training() -> None:
    """Done transitions should not use reset observations as estimator labels."""
    algorithm, obs = _build_algorithm()
    algorithm.act(obs)
    dones = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    algorithm.process_env_step(_make_obs(), torch.ones(NUM_ENVS), dones, {})
    assert torch.equal(
        algorithm.storage.estimator_valid[0, :, 0],
        torch.tensor([True, False, True, False]),
    )


def test_full_rollout_updates_ppo_and_estimator() -> None:
    """A complete rollout should update both optimizers and report HIM metrics."""
    algorithm, obs = _build_algorithm()
    estimator_before = [parameter.detach().clone() for parameter in algorithm.actor.estimator.parameters()]
    for _ in range(NUM_STEPS):
        algorithm.act(obs)
        next_obs = _make_obs()
        algorithm.process_env_step(next_obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        obs = next_obs
    algorithm.compute_returns(obs)
    losses = algorithm.update()

    assert {"value", "surrogate", "entropy", "him_estimation", "him_swap"} <= losses.keys()
    assert algorithm.storage.step == 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(estimator_before, algorithm.actor.estimator.parameters())
    )
    assert all(parameter.grad is None for parameter in algorithm.actor.estimator.parameters())


def test_estimator_generator_honors_requested_batch_count() -> None:
    """Estimator batching should produce exactly epochs times mini-batches."""
    algorithm, _ = _build_algorithm()
    batches = list(algorithm.storage.estimator_mini_batch_generator(num_mini_batches=3, num_epochs=2))
    assert len(batches) == 6


def test_checkpoint_round_trip_includes_estimator_optimizer() -> None:
    """Checkpoint round trips should include actor and estimator optimizer state."""
    algorithm, _ = _build_algorithm()
    checkpoint = algorithm.save()
    assert "estimator_optimizer_state_dict" in checkpoint

    restored, _ = _build_algorithm()
    restored.load(copy.deepcopy(checkpoint), load_cfg=None, strict=True)
    for expected, actual in zip(algorithm.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(expected, actual)


def test_construct_algorithm_from_rsl_rl_v5_config() -> None:
    """The standard runner configuration shape should construct all HIM components."""
    config = {
        "num_steps_per_env": NUM_STEPS,
        "obs_groups": {"actor": ["policy_history"], "critic": ["privileged"]},
        "multi_gpu": None,
        "algorithm": {
            "class_name": "HIMPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "estimator_num_mini_batches": 2,
        },
        "actor": {
            "class_name": "HIMActorModel",
            "history_obs_group": "policy_history",
            "estimator_target_group": "base_lin_vel",
            "temporal_steps": TEMPORAL_STEPS,
            "num_one_step_obs": ONE_STEP_DIM,
            "hidden_dims": [16, 8],
            "estimator_cfg": {
                "encoder_hidden_dims": [16],
                "target_hidden_dims": [16],
                "num_latent": 4,
            },
            "history_encoder_cfg": {"hidden_dims": [8]},
            "attention_cfg": {"model_dim": 8, "num_heads": 2},
            "moe_cfg": {"num_experts": 2, "gate_hidden_dims": [4]},
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16, 8]},
    }
    algorithm = HIMPPO.construct_algorithm(_make_obs(), _DummyEnv(), copy.deepcopy(config), "cpu")
    assert isinstance(algorithm, HIMPPO)
    assert isinstance(algorithm.actor, HIMActorModel)
    assert isinstance(algorithm.storage, HIMRolloutStorage)
