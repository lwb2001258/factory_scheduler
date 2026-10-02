"""LinUCB meta-scheduler over existing deterministic assignment policies."""

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from advanced_ai_common import load_checkpoint, save_checkpoint
from config import (
    MAX_ROBOTS, RL_ENVIRONMENT_VERSION, RobotState, TaskStatus,
)
from schedulers import (
    AuctionScheduler, BaseScheduler, FCFSScheduler, GreedyScheduler,
    HungarianScheduler, ModelValidationError, NearestNeighbourScheduler,
    SchedulerResult, SchedulingContext, build_cost_matrix,
)
from training_scenarios import factory_scenario


LINUCB_ALGORITHM = "LinUCB"
LINUCB_CONTEXT_VERSION = "factory-bandit-context-v1"
DEFAULT_BANDIT_ARMS = (
    "FCFS", "NearestNeighbour", "Greedy", "Hungarian", "Auction",
)
BANDIT_CONTEXT_FEATURES = (
    "bias",
    "fleet_fraction",
    "idle_robot_fraction",
    "pending_task_fraction",
    "mean_battery_fraction",
    "minimum_battery_fraction",
    "mean_priority_fraction",
    "high_priority_fraction",
    "mean_wait_fraction",
    "maximum_wait_fraction",
    "mean_congestion",
    "feasible_edge_fraction",
    "minimum_cost_fraction",
    "mean_cost_fraction",
)


def _visible_tasks(pending_tasks, now):
    visible = [
        task for task in pending_tasks
        if task.status == TaskStatus.PENDING
        and float(task.arrival_time) <= float(now)+1e-9
    ]
    ids = [task.task_id for task in visible]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate visible task IDs")
    return visible


def bandit_context_vector(robot_states: Dict[int, dict], pending_tasks,
                          context: Optional[SchedulingContext] = None
                          ) -> np.ndarray:
    """Encode only information observable at the scheduling instant."""
    context = context or SchedulingContext()
    visible = _visible_tasks(pending_tasks, context.current_time)
    matrix = build_cost_matrix(visible, robot_states, context)
    robot_count = len(robot_states)
    idle_count = len(matrix.robot_ids)
    batteries = np.asarray([
        float(state.get("battery", 100.0))
        for state in robot_states.values()], dtype=np.float64)
    if batteries.size and not np.all(np.isfinite(batteries)):
        raise ValueError("robot battery context is non-finite")
    priorities = np.asarray([
        float(task.priority) for task in visible], dtype=np.float64)
    waits = np.asarray([
        max(0.0, float(context.current_time)-float(task.arrival_time))
        for task in visible], dtype=np.float64)
    congestion = context.congestion_map
    if congestion is None:
        mean_congestion = 0.0
    else:
        congestion_values = np.asarray(congestion, dtype=np.float64)
        if (congestion_values.size and
                not np.all(np.isfinite(congestion_values))):
            raise ValueError("congestion context is non-finite")
        mean_congestion = (
            float(np.mean(congestion_values))
            if congestion_values.size else 0.0)
    feasible_costs = matrix.values[matrix.feasible]
    edge_fraction = (
        float(np.mean(matrix.feasible)) if matrix.feasible.size else 0.0)
    values = np.asarray([
        1.0,
        min(robot_count/max(1, MAX_ROBOTS), 2.0),
        idle_count/max(1, robot_count),
        min(len(visible)/20.0, 2.0),
        float(np.mean(batteries))/100.0 if batteries.size else 0.0,
        float(np.min(batteries))/100.0 if batteries.size else 0.0,
        float(np.mean(priorities))/3.0 if priorities.size else 0.0,
        float(np.mean(priorities >= 3.0)) if priorities.size else 0.0,
        min(float(np.mean(waits))/120.0, 10.0) if waits.size else 0.0,
        min(float(np.max(waits))/120.0, 10.0) if waits.size else 0.0,
        max(0.0, mean_congestion),
        edge_fraction,
        min(float(np.min(feasible_costs))/30.0, 10.0)
        if feasible_costs.size else 0.0,
        min(float(np.mean(feasible_costs))/30.0, 10.0)
        if feasible_costs.size else 0.0,
    ], dtype=np.float64)
    if (values.shape != (len(BANDIT_CONTEXT_FEATURES),) or
            not np.all(np.isfinite(values))):
        raise ValueError("LinUCB context vector is invalid")
    return values


def _arm_factory(name: str):
    factories = {
        "FCFS": FCFSScheduler,
        "NearestNeighbour": NearestNeighbourScheduler,
        "Greedy": GreedyScheduler,
        "Hungarian": HungarianScheduler,
        "Auction": AuctionScheduler,
    }
    if name not in factories:
        raise ModelValidationError(f"unsupported LinUCB arm: {name}")
    return factories[name]()


@dataclass
class LinUCBModel:
    arms: Tuple[str, ...]
    alpha: float
    covariance: np.ndarray
    reward_sum: np.ndarray
    training_samples: int = 0

    @classmethod
    def create(cls, arms: Sequence[str] = DEFAULT_BANDIT_ARMS,
               alpha: float = 0.5):
        arms = tuple(str(arm) for arm in arms)
        if (not arms or len(arms) != len(set(arms)) or
                not math.isfinite(alpha) or alpha < 0):
            raise ValueError("invalid LinUCB model configuration")
        for arm in arms:
            _arm_factory(arm)
        dimension = len(BANDIT_CONTEXT_FEATURES)
        covariance = np.repeat(
            np.eye(dimension, dtype=np.float64)[None, :, :],
            len(arms), axis=0)
        reward_sum = np.zeros((len(arms), dimension), dtype=np.float64)
        return cls(arms, float(alpha), covariance, reward_sum, 0)

    def __post_init__(self):
        self.arms = tuple(self.arms)
        dimension = len(BANDIT_CONTEXT_FEATURES)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        self.reward_sum = np.asarray(self.reward_sum, dtype=np.float64)
        if (not self.arms or len(self.arms) != len(set(self.arms)) or
                self.covariance.shape != (len(self.arms), dimension, dimension) or
                self.reward_sum.shape != (len(self.arms), dimension) or
                not np.all(np.isfinite(self.covariance)) or
                not np.all(np.isfinite(self.reward_sum)) or
                not math.isfinite(self.alpha) or self.alpha < 0 or
                isinstance(self.training_samples, bool) or
                not isinstance(self.training_samples, (int, np.integer)) or
                int(self.training_samples) < 0):
            raise ValueError("invalid LinUCB parameters")
        for arm, matrix in zip(self.arms, self.covariance):
            _arm_factory(arm)
            if not np.allclose(matrix, matrix.T, atol=1e-10):
                raise ValueError("LinUCB covariance must be symmetric")
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "LinUCB covariance must be positive definite") from exc
        self.training_samples = int(self.training_samples)

    def scores(self, context_vector) -> np.ndarray:
        vector = np.asarray(context_vector, dtype=np.float64)
        if (vector.shape != (len(BANDIT_CONTEXT_FEATURES),) or
                not np.all(np.isfinite(vector))):
            raise ValueError("invalid LinUCB context")
        scores = []
        for matrix, rewards in zip(self.covariance, self.reward_sum):
            theta = np.linalg.solve(matrix, rewards)
            uncertainty_vector = np.linalg.solve(matrix, vector)
            uncertainty = math.sqrt(max(
                0.0, float(vector @ uncertainty_vector)))
            scores.append(float(theta @ vector)+self.alpha*uncertainty)
        result = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("LinUCB produced non-finite scores")
        return result

    def select(self, context_vector):
        scores = self.scores(context_vector)
        index = int(np.argmax(scores))
        return self.arms[index], scores

    def update(self, context_vector, arm: str, reward: float) -> None:
        vector = np.asarray(context_vector, dtype=np.float64)
        if (vector.shape != (len(BANDIT_CONTEXT_FEATURES),) or
                not np.all(np.isfinite(vector)) or
                not math.isfinite(reward)):
            raise ValueError("invalid LinUCB update")
        try:
            index = self.arms.index(arm)
        except ValueError as exc:
            raise ValueError(f"unknown LinUCB arm: {arm}") from exc
        self.covariance[index] += np.outer(vector, vector)
        self.reward_sum[index] += float(reward)*vector
        self.training_samples += 1

    def save(self, path) -> None:
        save_checkpoint(path, {
            "algorithm": LINUCB_ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": len(BANDIT_CONTEXT_FEATURES),
            "action_dim": len(self.arms),
            "context_version": LINUCB_CONTEXT_VERSION,
            "context_features": list(BANDIT_CONTEXT_FEATURES),
            "arms": list(self.arms),
            "alpha": self.alpha,
            "training_samples": self.training_samples,
        }, {
            "covariance": self.covariance,
            "reward_sum": self.reward_sum,
        })

    @classmethod
    def load(cls, path):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=LINUCB_ALGORITHM,
            expected_state_dim=len(BANDIT_CONTEXT_FEATURES))
        if (metadata.get("context_version") != LINUCB_CONTEXT_VERSION or
                tuple(metadata.get("context_features", ())) !=
                BANDIT_CONTEXT_FEATURES):
            raise ModelValidationError("LinUCB context contract mismatch")
        try:
            arms = tuple(metadata["arms"])
            if metadata["action_dim"] != len(arms):
                raise ValueError("arm count does not match action dimension")
            raw_alpha = metadata["alpha"]
            raw_samples = metadata["training_samples"]
            if (isinstance(raw_alpha, bool) or
                    not isinstance(raw_alpha, (int, float)) or
                    not math.isfinite(float(raw_alpha)) or raw_alpha < 0 or
                    isinstance(raw_samples, bool) or
                    not isinstance(raw_samples, int) or raw_samples < 0):
                raise ValueError("invalid alpha or training sample count")
            return cls(
                arms, float(raw_alpha), arrays["covariance"],
                arrays["reward_sum"], raw_samples)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(f"invalid LinUCB checkpoint: {exc}") from exc


class LinUCBScheduler(BaseScheduler):
    def __init__(self, model_path):
        super().__init__(LINUCB_ALGORITHM)
        self.model = LinUCBModel.load(model_path)
        self.arms = {name: _arm_factory(name) for name in self.model.arms}
        self._last_arm = None

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        assignment = result.assignments[0]
        return assignment.robot_id, assignment.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        try:
            vector = bandit_context_vector(
                robot_states, pending_tasks, context)
            arm_name, scores = self.model.select(vector)
            result = self.arms[arm_name].assign(
                _visible_tasks(pending_tasks, context.current_time),
                robot_states, context)
            self._last_arm = arm_name
            result.algorithm_name = self.name
            result.computation_time = time.perf_counter()-started
            result.diagnostics.update({
                "selected_arm": arm_name,
                "arm_scores": {
                    name: float(score)
                    for name, score in zip(self.model.arms, scores)},
                "bandit_context_version": LINUCB_CONTEXT_VERSION,
            })
            return result
        except Exception as exc:
            self._last_arm = None
            return SchedulerResult(
                [], None, time.perf_counter()-started, False, self.name,
                {"reason": f"linucb_inference_error:{type(exc).__name__}"})

    def on_assignment_committed(self, assignment):
        if self._last_arm is not None:
            self.arms[self._last_arm].on_assignment_committed(assignment)
        super().on_assignment_committed(assignment)

    def on_assignment_rejected(self, assignment, reason):
        if self._last_arm is not None:
            self.arms[self._last_arm].on_assignment_rejected(
                assignment, reason)

    def reset(self):
        super().reset()
        for arm in self.arms.values():
            arm.reset()
        self._last_arm = None


def _strict_seeds(values: Iterable[int]):
    seeds = tuple(values)
    if (not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            for seed in seeds)):
        raise ValueError("LinUCB seeds must be integers")
    return tuple(int(seed) for seed in seeds)


def _snapshot_arm_rewards(robot_states, tasks, context, arms):
    visible = _visible_tasks(tasks, context.current_time)
    matrix = build_cost_matrix(visible, robot_states, context)
    target_count = min(len(matrix.robot_ids), len(matrix.tasks))
    if target_count == 0 or not np.any(matrix.feasible):
        return None
    finite_costs = matrix.values[matrix.feasible]
    missing_penalty = max(1.0, float(np.max(finite_costs))*2.0)
    row_by_robot = {
        robot_id: row for row, robot_id in enumerate(matrix.robot_ids)}
    column_by_task = {
        task.task_id: column for column, task in enumerate(matrix.tasks)}
    rewards = {}
    for arm_name in arms:
        scheduler = _arm_factory(arm_name)
        try:
            result = scheduler.assign(visible, robot_states, context)
            selected_cost = 0.0
            selected_rows = set()
            selected_columns = set()
            for assignment in result.assignments if result.is_feasible else ():
                pair = (row_by_robot[assignment.robot_id],
                        column_by_task[assignment.task.task_id])
                if (pair[0] in selected_rows or
                        pair[1] in selected_columns or
                        not matrix.feasible[pair]):
                    raise ValueError("arm returned an invalid matching")
                selected_rows.add(pair[0])
                selected_columns.add(pair[1])
                selected_cost += float(matrix.values[pair])
            missing = target_count-len(selected_rows)
            if missing < 0:
                raise ValueError("arm matching exceeds target cardinality")
            normalised_cost = (
                selected_cost+missing*missing_penalty) / target_count
            rewards[arm_name] = -normalised_cost/30.0
        except Exception:
            rewards[arm_name] = -(missing_penalty*2.0)/30.0
    return rewards


def train_linucb(snapshot_seeds: Iterable[int], *, alpha: float = 0.5,
                 arms: Sequence[str] = DEFAULT_BANDIT_ARMS):
    seeds = _strict_seeds(snapshot_seeds)
    model = LinUCBModel.create(arms, alpha)
    reward_history = {arm: [] for arm in model.arms}
    used_seeds = []
    for seed in seeds:
        robot_states, tasks, context = factory_scenario(seed)
        rewards = _snapshot_arm_rewards(
            robot_states, tasks, context, model.arms)
        if rewards is None:
            continue
        vector = bandit_context_vector(robot_states, tasks, context)
        for arm, reward in rewards.items():
            model.update(vector, arm, reward)
            reward_history[arm].append(reward)
        used_seeds.append(seed)
    if not used_seeds:
        raise ValueError("LinUCB training found no schedulable snapshots")
    report = {
        "snapshot_seeds": used_seeds,
        "snapshots": len(used_seeds),
        "updates": model.training_samples,
        "mean_reward_by_arm": {
            arm: float(np.mean(values))
            for arm, values in reward_history.items()},
    }
    return model, report
