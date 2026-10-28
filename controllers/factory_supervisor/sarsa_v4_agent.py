"""Masked dueling n-step Expected SARSA implemented with PyTorch."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from config import RL_ENVIRONMENT_VERSION
from sarsa_features import FEATURE_DIM as V3_FEATURE_DIM
from sarsa_v3_agent import SarsaV3Agent
from sarsa_v4_features import (
    ACTION_DIM, ACTION_FEATURE_DIM, CONTEXT_DIM, FEATURE_DIM, FEATURE_NAMES,
    FEATURE_VERSION, WAIT_ACTION)
from traffic_features import TRAFFIC_VERSION


MODEL_VERSION = "sarsa-v4-masked-dueling-expected-5step"
REWARD_VERSION = "sarsa-v3-event-time-v1"


@dataclass(frozen=True)
class SarsaV4Config:
    learning_rate: float = 1e-4
    gamma: float = 1.0
    epsilon_start: float = 0.20
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.985
    physical_epsilon_cap: float = 0.01
    n_step: int = 5
    batch_size: int = 32
    target_sync_transitions: int = 500
    gradient_norm_cap: float = 5.0
    huber_beta: float = 1.0
    wait_seconds: float = 2.0
    max_consecutive_waits: int = 2

    def __post_init__(self):
        numeric = [getattr(self, name) for name in self.__dataclass_fields__]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) for value in numeric):
            raise ValueError("invalid neural SARSA configuration")
        if (not 0.0 < self.learning_rate <= 0.01 or self.gamma != 1.0 or
                not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0 or
                not 0.0 < self.epsilon_decay <= 1.0 or
                not 0.0 <= self.physical_epsilon_cap <= 0.05 or
                not isinstance(self.n_step, int) or not 1 <= self.n_step <= 20 or
                not isinstance(self.batch_size, int) or not 1 <= self.batch_size <= 256 or
                not isinstance(self.target_sync_transitions, int) or
                self.target_sync_transitions < self.batch_size or
                not 0.0 < self.gradient_norm_cap <= 100.0 or
                not 0.0 < self.huber_beta <= 100.0 or
                not 0.0 < self.wait_seconds <= 10.0 or
                not isinstance(self.max_consecutive_waits, int) or
                not 0 <= self.max_consecutive_waits <= 5):
            raise ValueError("unsupported neural SARSA configuration")


@dataclass(frozen=True)
class NeuralDecision:
    state: np.ndarray
    actions: np.ndarray
    mask: np.ndarray

    @classmethod
    def from_snapshot(cls, snapshot):
        features = np.asarray(snapshot.features, dtype=np.float32)
        mask = np.asarray(snapshot.mask, dtype=bool)
        if (features.shape != (ACTION_DIM, FEATURE_DIM) or
                mask.shape != (ACTION_DIM,) or not mask.any() or
                not np.isfinite(features).all()):
            raise ValueError("invalid neural SARSA decision snapshot")
        legal = np.flatnonzero(mask)
        state = features[legal[0], :CONTEXT_DIM].copy()
        if not np.allclose(
                features[legal, :CONTEXT_DIM], state,
                rtol=0.0, atol=0.0):
            raise ValueError("legal actions do not share one state context")
        actions = features[:, CONTEXT_DIM:].copy()
        mask = mask.copy()
        for value in (state, actions, mask):
            value.setflags(write=False)
        return cls(state, actions, mask)


class MaskedDuelingNetwork(nn.Module):
    """Encode state once and score every feasible robot-task edge."""

    def __init__(self, objective):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM, 64), nn.SiLU())
        self.dispatch_advantage = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.wait_advantage = nn.Sequential(
            nn.Linear(64 + ACTION_FEATURE_DIM, 32), nn.SiLU(),
            nn.Linear(32, 1))
        self.state_value = nn.Sequential(
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        v3 = SarsaV3Agent(objective)
        prior = np.zeros(FEATURE_DIM, dtype=np.float32)
        prior[:V3_FEATURE_DIM] = (
            v3.weights / math.sqrt(V3_FEATURE_DIM)).astype(np.float32)
        self.register_buffer("linear_prior", torch.from_numpy(prior))
        for layer in (self.dispatch_advantage[-1], self.wait_advantage[-1],
                      self.state_value[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, actions, mask, *, mask_illegal=True):
        if (state.ndim != 2 or actions.ndim != 3 or mask.ndim != 2 or
                state.shape[0] != actions.shape[0] or
                mask.shape != actions.shape[:2] or
                state.shape[1] != CONTEXT_DIM or
                actions.shape[1:] != (ACTION_DIM, ACTION_FEATURE_DIM)):
            raise ValueError("invalid neural SARSA tensor shapes")
        encoded_state = self.state_encoder(state)
        encoded_actions = self.action_encoder(actions[:, :WAIT_ACTION])
        expanded_state = encoded_state.unsqueeze(1).expand(
            -1, WAIT_ACTION, -1)
        dispatch = self.dispatch_advantage(torch.cat(
            (expanded_state, encoded_actions), dim=-1)).squeeze(-1)
        wait = self.wait_advantage(torch.cat(
            (encoded_state, actions[:, WAIT_ACTION]), dim=-1))
        advantage = torch.cat((dispatch, wait), dim=1)
        legal = mask.to(advantage.dtype)
        legal_count = legal.sum(dim=1, keepdim=True).clamp_min(1.0)
        centered = advantage - (
            advantage * legal).sum(dim=1, keepdim=True) / legal_count
        value = self.state_value(encoded_state)
        prior_state = state @ self.linear_prior[:CONTEXT_DIM]
        prior_action = actions @ self.linear_prior[CONTEXT_DIM:]
        q_values = prior_state.unsqueeze(1) + prior_action + value + centered
        if mask_illegal:
            q_values = q_values.masked_fill(~mask, -torch.inf)
        return q_values


@dataclass(frozen=True)
class _Transition:
    decision: NeuralDecision
    action: int
    reward: float
    next_decision: NeuralDecision | None
    next_probabilities: np.ndarray | None
    done: bool


@dataclass(frozen=True)
class _TrainingSample:
    decision: NeuralDecision
    action: int
    reward_return: float
    bootstrap_decision: NeuralDecision | None
    bootstrap_probabilities: np.ndarray | None


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


class SarsaV4Agent:
    def __init__(self, objective, *, config=None, seed=42, device=None):
        if objective not in {"count", "utility_v2"}:
            raise ValueError("invalid neural SARSA objective")
        self.objective = objective
        self.config = config or SarsaV4Config()
        requested = (device or os.environ.get(
            "SMART_FACTORY_SARSA_V4_DEVICE", "cpu")).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested not in {"cpu", "cuda"}:
            raise ValueError("neural SARSA device must be cpu, cuda, or auto")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(requested)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        self.rng = np.random.default_rng(seed)
        self.online = MaskedDuelingNetwork(objective).to(self.device)
        self.target = MaskedDuelingNetwork(objective).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.config.learning_rate)
        self.epsilon = self.config.epsilon_start
        self.episode = 0
        self.optimizer_updates = 0
        self.transition_updates = 0
        self.target_syncs = 0
        self.last_loss = 0.0
        self.last_mean_absolute_td = 0.0
        self._next_target_sync = self.config.target_sync_transitions
        self._trajectory = deque()
        self._ready = deque()

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.online.parameters())

    def _tensors(self, decisions):
        states = torch.as_tensor(np.stack([
            item.state for item in decisions]), device=self.device)
        actions = torch.as_tensor(np.stack([
            item.actions for item in decisions]), device=self.device)
        masks = torch.as_tensor(np.stack([
            item.mask for item in decisions]), device=self.device)
        return states, actions, masks

    def values(self, decision):
        if not isinstance(decision, NeuralDecision):
            decision = NeuralDecision.from_snapshot(decision)
        self.online.eval()
        with torch.inference_mode():
            state, actions, mask = self._tensors([decision])
            values = self.online(state, actions, mask)[0].detach().cpu().numpy()
        return values

    def policy_probabilities(self, decision, *, training=True, physical=False):
        legal = np.flatnonzero(decision.mask)
        if not len(legal):
            raise ValueError("neural SARSA decision has no legal actions")
        values = self.values(decision)
        best = np.max(values[legal])
        tied = legal[np.isclose(values[legal], best, rtol=0.0, atol=1e-7)]
        epsilon = 0.0
        if training:
            epsilon = (min(self.epsilon, self.config.physical_epsilon_cap)
                       if physical else self.epsilon)
        probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
        probabilities[legal] = epsilon / len(legal)
        probabilities[tied] += (1.0 - epsilon) / len(tied)
        if (not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12) or
                np.any(probabilities[~decision.mask] != 0.0)):
            raise RuntimeError("invalid masked epsilon-greedy distribution")
        probabilities.setflags(write=False)
        return probabilities

    def select_action(self, snapshot, *, training=False, physical=False):
        decision = NeuralDecision.from_snapshot(snapshot)
        probabilities = self.policy_probabilities(
            decision, training=training, physical=physical)
        if training:
            action = int(self.rng.choice(ACTION_DIM, p=probabilities))
            return action, decision
        tied = np.flatnonzero(probabilities > 0.0)
        def key(action):
            if action == WAIT_ACTION:
                return (math.inf, math.inf, math.inf)
            row, column = divmod(int(action), 20)
            return (snapshot.route_costs[row, column],
                    snapshot.tasks[column].task_id,
                    snapshot.robot_ids[row])
        return int(min(tied, key=key)), decision

    def _make_sample(self, count):
        rows = list(self._trajectory)[:count]
        reward_return = 0.0
        discount = 1.0
        terminal = False
        for row in rows:
            reward_return += discount * row.reward
            discount *= self.config.gamma
            if row.done:
                terminal = True
                break
        last = rows[-1]
        bootstrap = None if terminal else last.next_decision
        probabilities = None if terminal else last.next_probabilities
        if not terminal and (bootstrap is None or probabilities is None):
            raise RuntimeError("nonterminal n-step return lacks a bootstrap")
        first = rows[0]
        return _TrainingSample(
            first.decision, first.action, reward_return,
            bootstrap, probabilities)

    def _mature(self, *, terminal):
        if terminal:
            while self._trajectory:
                count = min(self.config.n_step, len(self._trajectory))
                self._ready.append(self._make_sample(count))
                self._trajectory.popleft()
        elif len(self._trajectory) >= self.config.n_step:
            self._ready.append(self._make_sample(self.config.n_step))
            self._trajectory.popleft()

    def _optimize(self, samples):
        decisions = [sample.decision for sample in samples]
        states, actions, masks = self._tensors(decisions)
        selected = torch.as_tensor(
            [sample.action for sample in samples], dtype=torch.int64,
            device=self.device)
        returns = torch.as_tensor(
            [sample.reward_return for sample in samples], dtype=torch.float32,
            device=self.device)
        self.online.train()
        predicted = self.online(states, actions, masks).gather(
            1, selected.unsqueeze(1)).squeeze(1)
        targets = returns.clone()
        bootstrap_indices = [index for index, sample in enumerate(samples)
                             if sample.bootstrap_decision is not None]
        if bootstrap_indices:
            bootstrap_decisions = [
                samples[index].bootstrap_decision
                for index in bootstrap_indices]
            b_state, b_actions, b_mask = self._tensors(bootstrap_decisions)
            probabilities = torch.as_tensor(np.stack([
                samples[index].bootstrap_probabilities
                for index in bootstrap_indices]), dtype=torch.float32,
                device=self.device)
            with torch.no_grad():
                target_values = self.target(
                    b_state, b_actions, b_mask, mask_illegal=False)
                expected = (target_values * probabilities).sum(dim=1)
            targets[torch.as_tensor(
                bootstrap_indices, dtype=torch.int64,
                device=self.device)] += expected
        if not torch.isfinite(predicted).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("nonfinite neural SARSA values")
        td = targets - predicted
        loss = functional.smooth_l1_loss(
            predicted, targets, beta=self.config.huber_beta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_norm_cap,
            error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite neural SARSA gradient")
        self.optimizer.step()
        self.optimizer_updates += 1
        self.transition_updates += len(samples)
        self.last_loss = float(loss.detach().cpu())
        self.last_mean_absolute_td = float(td.detach().abs().mean().cpu())
        if self.transition_updates >= self._next_target_sync:
            self.target.load_state_dict(self.online.state_dict())
            self.target_syncs += 1
            while self._next_target_sync <= self.transition_updates:
                self._next_target_sync += self.config.target_sync_transitions
        return td.detach().cpu().numpy().tolist()

    def _train_ready(self, *, flush=False):
        errors = []
        while (len(self._ready) >= self.config.batch_size or
               (flush and self._ready)):
            count = min(self.config.batch_size, len(self._ready))
            samples = [self._ready.popleft() for _ in range(count)]
            errors.extend(self._optimize(samples))
        return errors

    def observe_transition(self, decision, action, reward,
                           next_decision=None, *, done=False, physical=False):
        if (not isinstance(decision, NeuralDecision) or
                isinstance(action, bool) or not isinstance(action, (int, np.integer)) or
                not 0 <= int(action) < ACTION_DIM or not decision.mask[int(action)] or
                isinstance(reward, bool) or not isinstance(
                    reward, (int, float, np.number)) or not math.isfinite(reward) or
                not isinstance(done, bool)):
            raise ValueError("invalid neural SARSA transition")
        if done:
            if next_decision is not None:
                raise ValueError("terminal neural SARSA transition has a next state")
            probabilities = None
        else:
            if not isinstance(next_decision, NeuralDecision):
                raise ValueError("nonterminal neural SARSA transition lacks next state")
            probabilities = self.policy_probabilities(
                next_decision, training=True, physical=physical)
        self._trajectory.append(_Transition(
            decision, int(action), float(reward), next_decision,
            probabilities, done))
        self._mature(terminal=done)
        return self._train_ready(flush=done)

    def start_episode(self, *, seed=None):
        if self._trajectory or self._ready:
            raise RuntimeError("cannot start with an unfinished SARSA trajectory")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def end_episode(self):
        if self._trajectory or self._ready:
            raise RuntimeError("cannot checkpoint an unfinished SARSA trajectory")
        self.episode += 1
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay)

    def set_learning_rate(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value <= self.config.learning_rate:
            raise ValueError("invalid fine-tune learning rate")
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _state_digest(self):
        digest = hashlib.sha256()
        for prefix, module in (("online", self.online), ("target", self.target)):
            for name, tensor in sorted(module.state_dict().items()):
                array = tensor.detach().cpu().contiguous().numpy()
                digest.update(f"{prefix}:{name}:{array.dtype}:{array.shape}".encode())
                digest.update(array.tobytes())
        return digest.hexdigest()

    def document(self):
        return {
            "algorithm": "ExpectedSARSA",
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "environment_version": RL_ENVIRONMENT_VERSION,
            "traffic_version": TRAFFIC_VERSION,
            "reward_version": REWARD_VERSION,
            "objective": self.objective,
            "action_dim": ACTION_DIM,
            "context_dim": CONTEXT_DIM,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "config": asdict(self.config),
            "epsilon": self.epsilon,
            "episode": self.episode,
            "optimizer_updates": self.optimizer_updates,
            "transition_updates": self.transition_updates,
            "target_syncs": self.target_syncs,
            "next_target_sync": self._next_target_sync,
            "last_loss": self.last_loss,
            "last_mean_absolute_td": self.last_mean_absolute_td,
            "parameter_count": self.parameter_count,
            "state_sha256": self._state_digest(),
            "rng_state": self.rng.bit_generator.state,
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
        }

    def save(self, path):
        if self._trajectory or self._ready:
            raise RuntimeError("neural SARSA checkpoint requires an episode boundary")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": json.dumps(
                self.document(), sort_keys=True, allow_nan=False),
            "online": _cpu_tree(self.online.state_dict()),
            "target": _cpu_tree(self.target.state_dict()),
            "optimizer": _cpu_tree(self.optimizer.state_dict()),
            "torch_rng": torch.get_rng_state().cpu(),
            "cuda_rng": ([value.cpu() for value in torch.cuda.get_rng_state_all()]
                         if torch.cuda.is_available() else []),
        }
        with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, *, objective=None, device=None):
        try:
            payload = torch.load(
                Path(path), map_location="cpu", weights_only=True)
            metadata = json.loads(payload["metadata"])
        except Exception as exc:
            raise ValueError("invalid neural SARSA checkpoint") from exc
        expected = {
            "algorithm": "ExpectedSARSA",
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "environment_version": RL_ENVIRONMENT_VERSION,
            "traffic_version": TRAFFIC_VERSION,
            "reward_version": REWARD_VERSION,
            "action_dim": ACTION_DIM,
            "context_dim": CONTEXT_DIM,
            "action_feature_dim": ACTION_FEATURE_DIM,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("neural SARSA checkpoint contract mismatch")
        if objective is not None and metadata.get("objective") != objective:
            raise ValueError("neural SARSA objective mismatch")
        agent = cls(
            metadata["objective"],
            config=SarsaV4Config(**metadata["config"]),
            seed=0, device=device)
        try:
            agent.online.load_state_dict(payload["online"], strict=True)
            agent.target.load_state_dict(payload["target"], strict=True)
            agent.optimizer.load_state_dict(payload["optimizer"])
            for state in agent.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(agent.device)
            agent.epsilon = float(metadata["epsilon"])
            agent.episode = int(metadata["episode"])
            agent.optimizer_updates = int(metadata["optimizer_updates"])
            agent.transition_updates = int(metadata["transition_updates"])
            agent.target_syncs = int(metadata["target_syncs"])
            agent._next_target_sync = int(metadata["next_target_sync"])
            agent.last_loss = float(metadata["last_loss"])
            agent.last_mean_absolute_td = float(
                metadata["last_mean_absolute_td"])
            agent.rng.bit_generator.state = metadata["rng_state"]
            torch.set_rng_state(payload["torch_rng"])
            if agent.device.type == "cuda" and payload.get("cuda_rng"):
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        except Exception as exc:
            raise ValueError("neural SARSA checkpoint state is invalid") from exc
        if agent.document()["state_sha256"] != metadata.get("state_sha256"):
            raise ValueError("neural SARSA checkpoint hash mismatch")
        return agent


__all__ = (
    "MODEL_VERSION", "REWARD_VERSION", "MaskedDuelingNetwork",
    "NeuralDecision", "SarsaV4Agent", "SarsaV4Config")
