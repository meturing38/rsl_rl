# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an independently optimized Hybrid Internal Model estimator."""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import HIMActorModel, MLPModel
from rsl_rl.storage import HIMRolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class HIMPPO(PPO):
    """PPO variant that trains the actor's HIM estimator with next-state supervision."""

    actor: HIMActorModel
    storage: HIMRolloutStorage

    def __init__(
        self,
        actor: HIMActorModel,
        critic: MLPModel,
        storage: HIMRolloutStorage,
        estimator_learning_rate: float = 1e-3,
        estimator_num_learning_epochs: int = 1,
        estimator_num_mini_batches: int = 4,
        estimator_max_grad_norm: float = 1.0,
        estimator_optimizer: str = "adam",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        **kwargs: object,
    ) -> None:
        """Initialize separate PPO and estimator optimizers."""
        super().__init__(
            actor,
            critic,
            storage,
            optimizer=optimizer,
            learning_rate=learning_rate,
            **kwargs,
        )

        estimator_parameter_ids = {id(parameter) for parameter in self.actor.estimator.parameters()}
        policy_parameters = (
            parameter for parameter in self.actor.parameters() if id(parameter) not in estimator_parameter_ids
        )
        self.optimizer = resolve_optimizer(optimizer)(
            chain(policy_parameters, self.critic.parameters()), lr=learning_rate
        )
        self.estimator_optimizer = resolve_optimizer(estimator_optimizer)(
            self.actor.estimator.parameters(), lr=estimator_learning_rate
        )
        self.estimator_num_learning_epochs = estimator_num_learning_epochs
        self.estimator_num_mini_batches = estimator_num_mini_batches
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.transition = HIMRolloutStorage.Transition()

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record standard PPO data and compact valid next-state HIM targets."""
        self.transition.estimator_target = self.actor.build_estimator_target(obs).detach()
        # Isaac Lab usually returns reset observations for done environments. Excluding these samples
        # avoids requiring a non-standard terminal_observation field in the VecEnv API.
        self.transition.estimator_valid = (~dones.bool()).reshape(-1, 1)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Update the estimator first, then run the unchanged PPO update."""
        estimator_metrics = self._update_estimator()
        losses = super().update()
        losses.update(estimator_metrics)
        return losses

    def _update_estimator(self) -> dict[str, float]:
        metric_sums = {"him_estimation": 0.0, "him_swap": 0.0, "him_prototype_entropy": 0.0}
        num_updates = 0
        generator = self.storage.estimator_mini_batch_generator(
            self.estimator_num_mini_batches, self.estimator_num_learning_epochs
        )
        for observations, targets, valid in generator:
            loss, metrics = self.actor.compute_estimator_loss(observations, targets, valid)
            self.estimator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._reduce_estimator_gradients()
            nn.utils.clip_grad_norm_(self.actor.estimator.parameters(), self.estimator_max_grad_norm)
            self.estimator_optimizer.step()

            metric_sums["him_estimation"] += metrics["estimation"].item()
            metric_sums["him_swap"] += metrics["swap"].item()
            metric_sums["him_prototype_entropy"] += metrics["prototype_entropy"].item()
            num_updates += 1

        # Estimator parameters are not owned by the PPO optimizer. Clear their last
        # gradients so base PPO actor-wide clipping cannot include stale values.
        self.estimator_optimizer.zero_grad(set_to_none=True)
        if num_updates == 0:
            return metric_sums
        return {name: value / num_updates for name, value in metric_sums.items()}

    def _reduce_estimator_gradients(self) -> None:
        parameters = list(self.actor.estimator.parameters())
        if not parameters:
            return
        flattened = torch.cat([
            parameter.grad.view(-1) if parameter.grad is not None else torch.zeros_like(parameter).view(-1)
            for parameter in parameters
        ])
        torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
        flattened /= self.gpu_world_size
        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            reduced_gradient = flattened[offset : offset + numel].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = reduced_gradient.clone()
            else:
                parameter.grad.copy_(reduced_gradient)
            offset += numel

    def save(self) -> dict:
        """Return standard PPO state plus the estimator optimizer state."""
        saved_dict = super().save()
        saved_dict["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load standard PPO state and, when requested, the estimator optimizer."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        should_load_optimizer = load_cfg is None or load_cfg.get("optimizer", True)
        if should_load_optimizer and "estimator_optimizer_state_dict" in loaded_dict:
            self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> HIMPPO:
        """Construct a HIM actor, standard critic, HIM storage, and HIMPPO."""
        alg_class: type[HIMPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[HIMActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            raise ValueError("HIMActorModel does not support share_cnn_encoders.")

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        storage = HIMRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            actor.estimator_target_dim,
            device,
        )
        return alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
