# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural-network building blocks for Hybrid Internal Model (HIM) policies."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from rsl_rl.utils import resolve_nn_activation


def _build_normalized_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    """Build an MLP with LayerNorm after every hidden linear layer."""
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            resolve_nn_activation(activation),
        ])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, epsilon: float = 0.05, num_iters: int = 6) -> torch.Tensor:
    """Compute balanced prototype assignments with Sinkhorn-Knopp normalization."""
    if scores.ndim != 2:
        raise ValueError(f"Sinkhorn expects a 2D tensor, got shape {tuple(scores.shape)}.")
    # Subtracting the row maximum leaves softmax-equivalent scores and avoids exp overflow.
    scores = scores - scores.max(dim=1, keepdim=True).values
    assignments = torch.exp(scores / epsilon).t()
    num_prototypes, batch_size = assignments.shape
    assignments /= assignments.sum().clamp_min(1e-12)
    for _ in range(num_iters):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
        assignments /= num_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
        assignments /= batch_size
    return (assignments * batch_size).t()


class HIMEstimator(nn.Module):
    """Infer privileged state and a contrastive latent from observation history."""

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        target_input_dim: int,
        num_pred: int = 3,
        num_latent: int = 32,
        num_prototypes: int = 32,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        target_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        temperature: float = 3.0,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 6,
    ) -> None:
        """Initialize estimator encoders and prototype assignments."""
        super().__init__()
        if temporal_steps < 1:
            raise ValueError("temporal_steps must be at least 1.")
        if target_input_dim < num_pred:
            raise ValueError("target_input_dim must contain the prediction target.")

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_pred = num_pred
        self.num_latent = num_latent
        self.target_input_dim = target_input_dim
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

        history_dim = temporal_steps * num_one_step_obs
        self.input_norm = nn.LayerNorm(history_dim)
        self.target_input_norm = nn.LayerNorm(target_input_dim)
        self.encoder = _build_normalized_mlp(history_dim, encoder_hidden_dims, num_pred + num_latent, activation)
        self.target = _build_normalized_mlp(target_input_dim, target_hidden_dims, num_latent, activation)
        self.prototypes = nn.Embedding(num_prototypes, num_latent)

    def forward(self, obs_history: torch.Tensor, detach: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the privileged prediction and normalized history latent."""
        if obs_history.shape[-1] != self.temporal_steps * self.num_one_step_obs:
            raise ValueError(
                f"Expected history width {self.temporal_steps * self.num_one_step_obs}, got {obs_history.shape[-1]}."
            )
        encoded = self.encoder(self.input_norm(obs_history))
        prediction = encoded[..., : self.num_pred]
        latent = functional.normalize(encoded[..., self.num_pred :], dim=-1)
        if detach:
            return prediction.detach(), latent.detach()
        return prediction, latent

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        target_input: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute privileged-state regression and swapped prototype losses."""
        if target_input.shape[-1] != self.target_input_dim:
            raise ValueError(f"Expected target width {self.target_input_dim}, got {target_input.shape[-1]}.")

        history = obs_history
        targets = target_input
        if valid_mask is not None:
            mask = valid_mask.reshape(-1).bool()
            history = history[mask]
            targets = targets[mask]

        if history.shape[0] == 0:
            zero = self.encoder[0].weight.sum() * 0.0
            return zero, {"estimation": zero.detach(), "swap": zero.detach(), "prototype_entropy": zero.detach()}

        prediction, student_latent = self.forward(history, detach=False)
        prediction_target = targets[..., -self.num_pred :].detach()
        target_latent = functional.normalize(self.target(self.target_input_norm(targets.detach())), dim=-1)

        with torch.no_grad():
            self.prototypes.weight.copy_(functional.normalize(self.prototypes.weight, dim=-1))

        student_scores = student_latent @ self.prototypes.weight.t()
        target_scores = target_latent @ self.prototypes.weight.t()
        student_assignments = sinkhorn(
            student_scores, epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iterations
        )
        target_assignments = sinkhorn(target_scores, epsilon=self.sinkhorn_epsilon, num_iters=self.sinkhorn_iterations)

        student_log_probs = functional.log_softmax(student_scores / self.temperature, dim=-1)
        target_log_probs = functional.log_softmax(target_scores / self.temperature, dim=-1)
        swap_loss = -0.5 * (student_assignments * target_log_probs + target_assignments * student_log_probs).mean()
        estimation_loss = functional.mse_loss(prediction, prediction_target)
        loss = estimation_loss + swap_loss

        with torch.no_grad():
            mean_assignments = student_assignments.mean(dim=0)
            prototype_entropy = -(mean_assignments * torch.log(mean_assignments + 1e-8)).sum()
        metrics = {
            "estimation": estimation_loss.detach(),
            "swap": swap_loss.detach(),
            "prototype_entropy": prototype_entropy,
        }
        return loss, metrics


class HistoryEncoder(nn.Module):
    """Encode a fixed observation history with temporal 1D convolutions."""

    def __init__(
        self,
        num_one_step_obs: int,
        temporal_steps: int,
        hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        activation: str = "elu",
    ) -> None:
        """Initialize temporal convolutions and feature normalization."""
        super().__init__()
        if not hidden_dims:
            raise ValueError("HistoryEncoder requires at least one hidden dimension.")
        self.num_one_step_obs = num_one_step_obs
        self.temporal_steps = temporal_steps
        self.output_dim = num_one_step_obs + hidden_dims[-1]
        self.input_norm = nn.LayerNorm(num_one_step_obs)
        layers: list[nn.Module] = []
        input_channels = num_one_step_obs
        for output_channels in hidden_dims:
            layers.extend([
                nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1),
                resolve_nn_activation(activation),
            ])
            input_channels = output_channels
        self.convolutions = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Return encoded history with shape ``[..., temporal_steps, output_dim]``."""
        expected_dim = self.temporal_steps * self.num_one_step_obs
        if obs_history.shape[-1] != expected_dim:
            raise ValueError(f"Expected history width {expected_dim}, got {obs_history.shape[-1]}.")
        sequence = obs_history.reshape(*obs_history.shape[:-1], self.temporal_steps, self.num_one_step_obs)
        sequence = self.input_norm(sequence)
        channel_first = sequence.transpose(-1, -2)
        encoded = self.convolutions(channel_first).transpose(-1, -2)
        return self.output_norm(torch.cat((sequence, encoded), dim=-1))


class AttentionEncoder(nn.Module):
    """Attend from the current observation to encoded historical features."""

    def __init__(
        self,
        num_one_step_obs: int,
        history_feature_dim: int,
        model_dim: int = 128,
        num_heads: int = 8,
    ) -> None:
        """Initialize current-state query and cross-attention projections."""
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.output_dim = model_dim
        self.query = nn.Linear(num_one_step_obs, model_dim)
        self.query_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, kdim=history_feature_dim, vdim=history_feature_dim, batch_first=True
        )

    def forward(
        self,
        current_obs: torch.Tensor,
        encoded_history: torch.Tensor,
    ) -> torch.Tensor:
        """Return one attended feature per batch item."""
        query = self.query_norm(self.query(current_obs)).unsqueeze(-2)
        output, _ = self.attention(query, encoded_history, encoded_history, need_weights=False)
        return output.squeeze(-2)

    def forward_with_weights(
        self, current_obs: torch.Tensor, encoded_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attended features and attention weights for diagnostics."""
        query = self.query_norm(self.query(current_obs)).unsqueeze(-2)
        output, weights = self.attention(query, encoded_history, encoded_history, need_weights=True)
        return output.squeeze(-2), weights


class SoftMOE(nn.Module):
    """Softly combine several expert MLP action heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        output_dim: int,
        num_experts: int = 4,
        gate_hidden_dims: tuple[int, ...] | list[int] = (32,),
        activation: str = "elu",
    ) -> None:
        """Initialize expert action heads and their soft gate."""
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be at least 1.")
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            _build_normalized_mlp(input_dim, hidden_dims, output_dim, activation) for _ in range(num_experts)
        ])
        self.gate = _build_normalized_mlp(input_dim, gate_hidden_dims, num_experts, activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the gate-weighted sum of expert outputs."""
        weights = functional.softmax(self.gate(inputs), dim=-1)
        outputs: list[torch.Tensor] = []
        for expert in self.experts:
            outputs.append(expert(inputs))  # noqa: PERF401 - explicit loop is TorchScript-compatible.
        expert_outputs = torch.stack(outputs, dim=-2)
        return torch.sum(expert_outputs * weights.unsqueeze(-1), dim=-2)
