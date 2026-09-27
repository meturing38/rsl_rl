# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TensorDict-native actor model using Hybrid Internal Model features."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import AttentionEncoder, HiddenState, HIMEstimator, HistoryEncoder, SoftMOE


class HIMActorModel(MLPModel):
    """HIM actor compatible with the rsl_rl 5.x model and distribution APIs."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_obs_group: str,
        estimator_target_group: str,
        temporal_steps: int,
        num_one_step_obs: int,
        history_order: str = "oldest_first",
        num_command_obs: int = 0,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        distribution_cfg: dict | None = None,
        estimator_cfg: dict | None = None,
        history_encoder_cfg: dict | None = None,
        attention_cfg: dict | None = None,
        moe_cfg: dict | None = None,
        obs_normalization: bool = False,
    ) -> None:
        """Initialize a TensorDict-native HIM actor and standard output distribution."""
        if obs_set != "actor":
            raise ValueError("HIMActorModel can only be constructed for the actor observation set.")
        if history_order not in ("oldest_first", "newest_first"):
            raise ValueError("history_order must be 'oldest_first' or 'newest_first'.")
        if obs_normalization:
            raise ValueError("HIMActorModel uses per-module LayerNorm and does not support obs_normalization.")
        if obs_groups[obs_set] != [history_obs_group]:
            raise ValueError(
                "The actor observation set must contain only history_obs_group so exported input ordering stays stable."
            )
        if history_obs_group not in obs:
            raise KeyError(f"History observation group '{history_obs_group}' is missing.")
        if estimator_target_group not in obs:
            raise KeyError(f"Estimator target group '{estimator_target_group}' is missing.")
        history_width = obs[history_obs_group].shape[-1]
        expected_width = temporal_steps * num_one_step_obs
        if history_width != expected_width:
            raise ValueError(f"Expected history width {expected_width}, got {history_width}.")
        if len(obs[estimator_target_group].shape) != 2:
            raise ValueError("The estimator target observation must be a flat 2D TensorDict entry.")
        if not 0 <= num_command_obs < num_one_step_obs:
            raise ValueError("num_command_obs must be in [0, num_one_step_obs).")

        self.history_obs_group = history_obs_group
        self.estimator_target_group = estimator_target_group
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.history_order = history_order
        self.num_command_obs = num_command_obs
        self.num_pred = obs[estimator_target_group].shape[-1]

        estimator_cfg = {} if estimator_cfg is None else estimator_cfg.copy()
        history_encoder_cfg = {} if history_encoder_cfg is None else history_encoder_cfg.copy()
        attention_cfg = {} if attention_cfg is None else attention_cfg.copy()
        moe_cfg = {} if moe_cfg is None else moe_cfg.copy()

        history_hidden_dims = history_encoder_cfg.get("hidden_dims", (64, 128))
        attention_dim = attention_cfg.get("model_dim", 128)
        estimator_latent_dim = estimator_cfg.get("num_latent", 32)
        self.him_latent_dim = num_one_step_obs + self.num_pred + estimator_latent_dim + attention_dim

        # MLPModel sets up the standard rsl_rl distribution API. Its MLP is replaced by SoftMOE below.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=False,
            distribution_cfg=distribution_cfg,
        )

        target_input_dim = num_one_step_obs - num_command_obs + self.num_pred
        self.estimator = HIMEstimator(
            temporal_steps=temporal_steps,
            num_one_step_obs=num_one_step_obs,
            target_input_dim=target_input_dim,
            num_pred=self.num_pred,
            **estimator_cfg,
        )
        self.history_encoder = HistoryEncoder(
            num_one_step_obs=num_one_step_obs,
            temporal_steps=temporal_steps,
            **history_encoder_cfg,
        )
        self.attention_encoder = AttentionEncoder(
            num_one_step_obs=num_one_step_obs,
            history_feature_dim=num_one_step_obs + history_hidden_dims[-1],
            **attention_cfg,
        )
        self.current_obs_norm = nn.LayerNorm(num_one_step_obs)

        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        if not isinstance(mlp_output_dim, int):
            raise ValueError("HIMActorModel currently requires a distribution with a scalar input dimension.")
        self.mlp = SoftMOE(
            input_dim=self.him_latent_dim,
            hidden_dims=hidden_dims,
            output_dim=mlp_output_dim,
            activation=activation,
            **moe_cfg,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _get_latent_dim(self) -> int:
        return self.him_latent_dim

    def _current_obs(self, history: torch.Tensor) -> torch.Tensor:
        sequence = history.reshape(*history.shape[:-1], self.temporal_steps, self.num_one_step_obs)
        index = -1 if self.history_order == "oldest_first" else 0
        return sequence[..., index, :]

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Fuse current proprioception, HIM prediction/latent, and attended history."""
        del masks, hidden_state
        history = obs[self.history_obs_group]
        current_obs = self.current_obs_norm(self._current_obs(history))
        prediction, estimator_latent = self.estimator(history, detach=True)
        encoded_history = self.history_encoder(history)
        attended_history = self.attention_encoder(current_obs, encoded_history)
        return torch.cat((current_obs, prediction, estimator_latent, attended_history), dim=-1)

    @property
    def estimator_target_dim(self) -> int:
        """Width of the compact next-state target stored in rollouts."""
        return self.num_one_step_obs - self.num_command_obs + self.num_pred

    def build_estimator_target(self, next_obs: TensorDict) -> torch.Tensor:
        """Build target-encoder input from the next observation."""
        next_history = next_obs[self.history_obs_group]
        next_current = self._current_obs(next_history)[..., self.num_command_obs :]
        prediction_target = next_obs[self.estimator_target_group]
        return torch.cat((next_current, prediction_target), dim=-1)

    def compute_estimator_loss(
        self,
        obs: TensorDict,
        target_input: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the estimator auxiliary loss for a rollout mini-batch."""
        return self.estimator.compute_loss(obs[self.history_obs_group], target_input, valid_mask)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible deterministic HIM policy."""
        return _HIMPolicyInference(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible deterministic HIM policy."""
        return _OnnxHIMPolicyInference(self, verbose)


class _HIMPolicyInference(nn.Module):
    """Export-only deterministic policy over a pre-concatenated history tensor."""

    def __init__(self, model: HIMActorModel) -> None:
        super().__init__()
        self.temporal_steps = model.temporal_steps
        self.num_one_step_obs = model.num_one_step_obs
        self.current_index = model.temporal_steps - 1 if model.history_order == "oldest_first" else 0
        self.estimator_input_norm = copy.deepcopy(model.estimator.input_norm)
        self.estimator_encoder = copy.deepcopy(model.estimator.encoder)
        self.num_pred = model.estimator.num_pred
        self.history_input_norm = copy.deepcopy(model.history_encoder.input_norm)
        self.history_convolutions = copy.deepcopy(model.history_encoder.convolutions)
        self.history_output_norm = copy.deepcopy(model.history_encoder.output_norm)
        self.query = copy.deepcopy(model.attention_encoder.query)
        self.query_norm = copy.deepcopy(model.attention_encoder.query_norm)
        self.attention = copy.deepcopy(model.attention_encoder.attention)
        self.current_obs_norm = copy.deepcopy(model.current_obs_norm)
        self.moe = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from flattened observation history."""
        sequence = history.reshape(-1, self.temporal_steps, self.num_one_step_obs)
        current_obs = self.current_obs_norm(sequence[:, self.current_index, :])

        estimator_output = self.estimator_encoder(self.estimator_input_norm(history))
        prediction = estimator_output[:, : self.num_pred]
        estimator_latent = functional.normalize(estimator_output[:, self.num_pred :], dim=-1)

        normalized_sequence = self.history_input_norm(sequence)
        encoded = self.history_convolutions(normalized_sequence.transpose(1, 2)).transpose(1, 2)
        encoded_history = self.history_output_norm(torch.cat((normalized_sequence, encoded), dim=-1))
        query = self.query_norm(self.query(current_obs)).unsqueeze(1)
        attended_history, _ = self.attention(query, encoded_history, encoded_history, need_weights=False)

        features = torch.cat((current_obs, prediction, estimator_latent, attended_history.squeeze(1)), dim=-1)
        return self.deterministic_output(self.moe(features))

    @torch.jit.export
    def reset(self) -> None:
        """Reset policy state (no-op because history is supplied explicitly)."""
        pass


class _OnnxHIMPolicyInference(_HIMPolicyInference):
    """ONNX wrapper exposing the standard rsl_rl export metadata."""

    is_recurrent: bool = False

    def __init__(self, model: HIMActorModel, verbose: bool) -> None:
        """Create an ONNX wrapper from a trained HIM actor."""
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.temporal_steps * model.num_one_step_obs

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return a representative flattened-history input."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX graph input names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX graph output names."""
        return ["actions"]
