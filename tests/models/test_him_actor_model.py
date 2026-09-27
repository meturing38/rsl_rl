# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for HIM network modules and actor model."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import HIMActorModel
from rsl_rl.modules import AttentionEncoder, HIMEstimator, HistoryEncoder, SoftMOE

NUM_ENVS = 4
TEMPORAL_STEPS = 4
ONE_STEP_DIM = 6
NUM_ACTIONS = 3


def _make_obs() -> TensorDict:
    """Create actor history, estimator target, and critic observations."""
    return TensorDict(
        {
            "policy_history": torch.randn(NUM_ENVS, TEMPORAL_STEPS * ONE_STEP_DIM),
            "base_lin_vel": torch.randn(NUM_ENVS, 3),
            "privileged": torch.randn(NUM_ENVS, 5),
        },
        batch_size=[NUM_ENVS],
    )


def _make_actor(history_order: str = "oldest_first") -> tuple[HIMActorModel, TensorDict]:
    """Create a compact HIM actor and matching observations."""
    obs = _make_obs()
    actor = HIMActorModel(
        obs,
        {"actor": ["policy_history"], "critic": ["privileged"]},
        "actor",
        NUM_ACTIONS,
        history_obs_group="policy_history",
        estimator_target_group="base_lin_vel",
        temporal_steps=TEMPORAL_STEPS,
        num_one_step_obs=ONE_STEP_DIM,
        num_command_obs=2,
        history_order=history_order,
        hidden_dims=[16, 8],
        estimator_cfg={"encoder_hidden_dims": [16], "target_hidden_dims": [16], "num_latent": 5},
        history_encoder_cfg={"hidden_dims": [8]},
        attention_cfg={"model_dim": 8, "num_heads": 2},
        moe_cfg={"num_experts": 2, "gate_hidden_dims": [4]},
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    return actor, obs


def test_him_modules_output_shapes() -> None:
    """HIM building blocks should preserve documented batch and feature shapes."""
    history = torch.randn(NUM_ENVS, TEMPORAL_STEPS * ONE_STEP_DIM)
    estimator = HIMEstimator(TEMPORAL_STEPS, ONE_STEP_DIM, target_input_dim=7, num_pred=3, num_latent=5)
    prediction, latent = estimator(history)
    assert prediction.shape == (NUM_ENVS, 3)
    assert latent.shape == (NUM_ENVS, 5)

    history_encoder = HistoryEncoder(ONE_STEP_DIM, TEMPORAL_STEPS, hidden_dims=[8])
    encoded = history_encoder(history)
    assert encoded.shape == (NUM_ENVS, TEMPORAL_STEPS, ONE_STEP_DIM + 8)

    attention = AttentionEncoder(ONE_STEP_DIM, ONE_STEP_DIM + 8, model_dim=8, num_heads=2)
    attended, weights = attention.forward_with_weights(torch.randn(NUM_ENVS, ONE_STEP_DIM), encoded)
    assert attended.shape == (NUM_ENVS, 8)
    assert weights.shape == (NUM_ENVS, 1, TEMPORAL_STEPS)

    moe = SoftMOE(8, [16], NUM_ACTIONS, num_experts=3)
    assert moe(torch.randn(NUM_ENVS, 8)).shape == (NUM_ENVS, NUM_ACTIONS)


def test_actor_uses_history_order_and_detaches_estimator_from_ppo() -> None:
    """Actor PPO gradients should not update the independently trained estimator."""
    actor, obs = _make_actor("oldest_first")
    sequence = torch.arange(TEMPORAL_STEPS * ONE_STEP_DIM, dtype=torch.float32).repeat(NUM_ENVS, 1)
    obs["policy_history"] = sequence
    assert torch.equal(actor._current_obs(sequence), sequence[:, -ONE_STEP_DIM:])

    actor(obs).sum().backward()
    assert all(parameter.grad is None for parameter in actor.estimator.parameters())
    assert any(parameter.grad is not None for parameter in actor.mlp.parameters())


def test_estimator_loss_trains_estimator_and_masks_done_samples() -> None:
    """Estimator loss should train valid samples and safely handle an empty mask."""
    actor, obs = _make_actor()
    target = actor.build_estimator_target(_make_obs())
    loss, metrics = actor.compute_estimator_loss(obs, target, torch.ones(NUM_ENVS, 1, dtype=torch.bool))
    loss.backward()
    assert loss.item() > 0.0
    assert metrics["estimation"].item() >= 0.0
    assert any(parameter.grad is not None for parameter in actor.estimator.parameters())

    empty_loss, empty_metrics = actor.compute_estimator_loss(obs, target, torch.zeros(NUM_ENVS, 1, dtype=torch.bool))
    assert empty_loss.item() == 0.0
    assert empty_metrics["swap"].item() == 0.0


def test_him_actor_jit_export_matches_deterministic_policy() -> None:
    """TorchScript inference should match the TensorDict actor."""
    actor, obs = _make_actor()
    actor.eval()
    expected = actor(obs)
    exported = torch.jit.script(actor.as_jit())
    actual = exported(obs["policy_history"])
    assert torch.allclose(expected, actual, atol=1e-5)


@pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
def test_him_actor_onnx_export_is_valid() -> None:
    """The exported HIM ONNX graph should pass the ONNX checker."""
    actor, _ = _make_actor()
    actor.eval()
    exported = actor.as_onnx(verbose=False)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
        torch.onnx.export(
            exported,
            exported.get_dummy_inputs(),
            file.name,
            opset_version=18,
            input_names=exported.input_names,
            output_names=exported.output_names,
        )
        onnx.checker.check_model(onnx.load(file.name))
