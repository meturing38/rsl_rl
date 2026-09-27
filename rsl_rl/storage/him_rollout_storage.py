# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage extensions for HIM auxiliary targets."""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from .rollout_storage import RolloutStorage


class HIMRolloutStorage(RolloutStorage):
    """Store compact next-state supervision alongside standard PPO transitions."""

    class Transition(RolloutStorage.Transition):
        """Transition carrying compact next-state HIM supervision."""

        def __init__(self) -> None:
            """Initialize an empty HIM transition."""
            super().__init__()
            self.estimator_target: torch.Tensor | None = None
            self.estimator_valid: torch.Tensor | None = None

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        estimator_target_dim: int,
        device: str = "cpu",
    ) -> None:
        """Allocate PPO buffers plus compact HIM target and validity buffers."""
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.estimator_targets = torch.zeros(num_transitions_per_env, num_envs, estimator_target_dim, device=device)
        self.estimator_valid = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Add a transition and its HIM target."""
        if not isinstance(transition, HIMRolloutStorage.Transition):
            raise TypeError("HIMRolloutStorage requires HIMRolloutStorage.Transition instances.")
        step = self.step
        if transition.estimator_target is None or transition.estimator_valid is None:
            raise ValueError("HIM transition is missing estimator target data.")
        super().add_transition(transition)
        self.estimator_targets[step].copy_(transition.estimator_target)
        self.estimator_valid[step].copy_(transition.estimator_valid.reshape(-1, 1))

    def estimator_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int
    ) -> Generator[tuple[TensorDict, torch.Tensor, torch.Tensor], None, None]:
        """Yield shuffled feed-forward batches for estimator optimization."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError("num_mini_batches exceeds the number of rollout samples.")

        observations = self.observations.flatten(0, 1)
        targets = self.estimator_targets.flatten(0, 1)
        valid = self.estimator_valid.flatten(0, 1)
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)[: num_mini_batches * mini_batch_size]
            for batch_index in range(num_mini_batches):
                start = batch_index * mini_batch_size
                batch_indices = indices[start : start + mini_batch_size]
                yield observations[batch_indices], targets[batch_indices], valid[batch_indices]
