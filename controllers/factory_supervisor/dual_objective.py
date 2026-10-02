"""Versioned Count and Utility V2 evaluation for fixed-horizon runs."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Dict, Iterable, Mapping, Optional
from types import MappingProxyType

from config import PRIORITY_MAX_COMPLETION_SECONDS, TaskStatus
from task_generator import TransportTask


UTILITY_SCORE_VERSION = "utility_deadline_v2_horizon_adjusted"
UTILITY_PRIORITY_WEIGHTS = {1: 1.0, 2: 2.0, 3: 4.0}
UTILITY_COMPONENT_WEIGHTS = {
    "priority_completion": 0.50,
    "timeliness": 0.30,
    "queue_extra": 0.10,
    "route_efficiency": 0.10,
}


def _finite(value, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _priority_map(values, name: str, *, allow_none: bool = False) -> dict:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a priority mapping")
    canonical = {}
    for key, value in values.items():
        if isinstance(key, bool):
            raise ValueError(f"{name} contains an invalid priority")
        try:
            priority = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains an invalid priority") from exc
        if str(priority) != str(key) and key != priority:
            raise ValueError(f"{name} contains an invalid priority")
        if priority in canonical or priority not in (1, 2, 3):
            raise ValueError(f"{name} priorities must be exactly 1, 2, 3")
        if value is None and allow_none:
            canonical[priority] = None
        else:
            canonical[priority] = _finite(value, f"{name}[{priority}]",
                                           positive=True)
    if set(canonical) != {1, 2, 3}:
        raise ValueError(f"{name} priorities must be exactly 1, 2, 3")
    return canonical


def _deadline_map(values) -> dict:
    if not isinstance(values, Mapping):
        raise ValueError("deadline_seconds must be a priority mapping")
    canonical = {}
    for key, value in values.items():
        if isinstance(key, bool):
            raise ValueError("deadline_seconds contains an invalid priority")
        try:
            priority = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "deadline_seconds contains an invalid priority") from exc
        if str(priority) != str(key) and key != priority:
            raise ValueError("deadline_seconds contains an invalid priority")
        if priority in canonical or priority not in (1, 2, 3):
            raise ValueError("deadline_seconds priorities must be exactly 1, 2, 3")
        if priority == 1:
            if value is not None:
                raise ValueError("priority 1 deadline must be null")
            canonical[priority] = None
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("priority 2/3 deadlines must be integers")
            lower, upper = ((400, 450) if priority == 2 else (250, 300))
            if not lower <= value <= upper:
                raise ValueError(
                    "deadline_seconds lies outside the registered range")
            canonical[priority] = value
    if set(canonical) != {1, 2, 3}:
        raise ValueError("deadline_seconds priorities must be exactly 1, 2, 3")
    if canonical[3] >= canonical[2]:
        raise ValueError("priority 3 deadline must be less than priority 2")
    return canonical


def _sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UtilityScoreConfig:
    """Frozen, hashable-by-content configuration for formal Utility V2."""

    tau_by_priority: Mapping[int, float]
    kappa_by_priority: Mapping[int, float]
    horizon_seconds: float = 1800.0
    max_deadline_penalty_points: float = 15.0
    priority_weights: Mapping[int, float] = field(
        default_factory=lambda: dict(UTILITY_PRIORITY_WEIGHTS))
    component_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(UTILITY_COMPONENT_WEIGHTS))
    deadline_seconds: Mapping[int, Optional[int]] = field(
        default_factory=lambda: dict(PRIORITY_MAX_COMPLETION_SECONDS))
    version: str = UTILITY_SCORE_VERSION

    def __post_init__(self):
        tau = _priority_map(self.tau_by_priority, "tau_by_priority")
        kappa = _priority_map(self.kappa_by_priority, "kappa_by_priority")
        weights = _priority_map(self.priority_weights, "priority_weights")
        if weights != UTILITY_PRIORITY_WEIGHTS:
            raise ValueError("Utility V2 priority weights must be 1/2/4")
        if not isinstance(self.component_weights, Mapping):
            raise ValueError("component_weights must be a mapping")
        components = {
            str(key): _finite(value, f"component_weights[{key}]",
                              nonnegative=True)
            for key, value in self.component_weights.items()
        }
        if (set(components) != set(UTILITY_COMPONENT_WEIGHTS) or
                any(not math.isclose(
                    components[key], UTILITY_COMPONENT_WEIGHTS[key],
                    rel_tol=0.0, abs_tol=1e-12)
                    for key in UTILITY_COMPONENT_WEIGHTS) or
                not math.isclose(sum(components.values()), 1.0,
                                 rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("Utility V2 component weights must be 0.50/0.30/0.10/0.10")
        deadlines = _deadline_map(self.deadline_seconds)
        horizon = _finite(
            self.horizon_seconds, "horizon_seconds", positive=True)
        penalty = _finite(
            self.max_deadline_penalty_points,
            "max_deadline_penalty_points", nonnegative=True)
        if self.version != UTILITY_SCORE_VERSION:
            raise ValueError("unsupported Utility score version")
        object.__setattr__(self, "tau_by_priority", MappingProxyType(tau))
        object.__setattr__(self, "kappa_by_priority", MappingProxyType(kappa))
        object.__setattr__(self, "priority_weights", MappingProxyType(weights))
        object.__setattr__(self, "component_weights",
                           MappingProxyType(components))
        object.__setattr__(self, "deadline_seconds",
                           MappingProxyType(deadlines))
        object.__setattr__(self, "horizon_seconds", horizon)
        object.__setattr__(self, "max_deadline_penalty_points", penalty)

    def canonical(self) -> dict:
        return {
            "version": self.version,
            "horizon_seconds": self.horizon_seconds,
            "priority_weights": {
                str(key): self.priority_weights[key] for key in (1, 2, 3)},
            "component_weights": {
                key: self.component_weights[key]
                for key in sorted(self.component_weights)},
            "tau_by_priority": {
                str(key): self.tau_by_priority[key] for key in (1, 2, 3)},
            "kappa_by_priority": {
                str(key): self.kappa_by_priority[key]
                for key in (1, 2, 3)},
            "deadline_seconds": {
                str(key): self.deadline_seconds[key] for key in (1, 2, 3)},
            "max_deadline_penalty_points": (
                self.max_deadline_penalty_points),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical())


def _task_snapshot(
        tasks: Iterable[TransportTask], horizon: float,
        deadline_seconds: Optional[Mapping[int, Optional[int]]] = None) -> list:
    try:
        rows = list(tasks)
    except TypeError as exc:
        raise ValueError("tasks must be an iterable of TransportTask objects") from exc
    if any(not isinstance(task, TransportTask) for task in rows):
        raise ValueError("evaluation tasks must be TransportTask objects")
    identifiers = [task.task_id for task in rows]
    if (any(isinstance(value, bool) or not isinstance(value, int)
            for value in identifiers) or len(set(identifiers)) != len(rows)):
        raise ValueError("evaluation tasks must have unique integer IDs")
    allowed = {
        TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS,
        TaskStatus.COMPLETED, TaskStatus.FAILED,
    }
    result = []
    for task in rows:
        if task.status not in allowed:
            raise ValueError(f"task {task.task_id} has unsupported status")
        arrival = _finite(task.arrival_time, "arrival_time", nonnegative=True)
        if not arrival < horizon:
            continue
        priority = int(task.priority)
        if (deadline_seconds is not None and
                task.max_completion_time_seconds != deadline_seconds[priority]):
            raise ValueError("task deadline config differs from score config")
        complete = task.status == TaskStatus.COMPLETED
        completion = None
        assignment = None
        if task.completion_time is not None:
            completion = _finite(task.completion_time, "completion_time",
                                 nonnegative=True)
        if complete:
            if completion is None or completion < arrival or completion > horizon:
                raise ValueError("completed task has invalid completion_time")
        elif completion is not None:
            raise ValueError("non-completed task cannot have completion_time")
        if task.assignment_time is not None:
            assignment = _finite(task.assignment_time, "assignment_time",
                                 nonnegative=True)
            if assignment < arrival or assignment > horizon:
                raise ValueError("task has invalid assignment_time")
        if complete and (assignment is None or assignment > completion):
            raise ValueError("completed task has invalid assignment_time")
        result.append({
            "task": task,
            "task_id": task.task_id,
            "priority": priority,
            "arrival": arrival,
            "assignment": assignment,
            "completion": completion,
            "complete": complete,
        })
    return result


def count_evaluation(tasks: Iterable[TransportTask], *,
                     horizon_seconds: float = 1800.0) -> dict:
    """Evaluate pure completed-task count over ``[0, H)``."""
    horizon = _finite(horizon_seconds, "horizon_seconds", positive=True)
    rows = _task_snapshot(tasks, horizon)
    completed = sum(row["complete"] for row in rows)
    return {
        "objective": "count",
        "horizon_seconds": horizon,
        "total_tasks_arrived": len(rows),
        "total_tasks_completed": completed,
        "throughput_per_minute": completed / (horizon / 60.0),
        "status": "ok" if rows else "no_evaluable_tasks",
    }


def paired_count_delta(candidate_count: int, baseline_count: int) -> dict:
    """Return paired absolute/relative Count changes without fake divisors."""
    if (isinstance(candidate_count, bool) or not isinstance(candidate_count, int)
            or candidate_count < 0 or isinstance(baseline_count, bool)
            or not isinstance(baseline_count, int) or baseline_count < 0):
        raise ValueError("paired counts must be nonnegative integers")
    absolute = candidate_count - baseline_count
    return {
        "candidate_count": candidate_count,
        "baseline_count": baseline_count,
        "absolute_delta": absolute,
        "relative_delta": (
            None if baseline_count == 0 else absolute / baseline_count),
        "status": "baseline_zero" if baseline_count == 0 else "ok",
    }


REWARD_COMPONENT_NAMES = (
    "reward_dispatch", "reward_completion", "reward_terminal_loss",
    "reward_route_excess", "reward_deadline_base",
    "reward_deadline_severity", "reward_invalid", "reward_defer",
    "reward_collision", "reward_proximity",
)


def _reward_task_index(tasks: Iterable[TransportTask]) -> dict:
    rows = list(tasks)
    if any(not isinstance(task, TransportTask) for task in rows):
        raise ValueError("reward tasks must be TransportTask objects")
    result = {task.task_id: task for task in rows}
    if len(result) != len(rows):
        raise ValueError("reward tasks contain duplicate IDs")
    return result


def _reward_events(events) -> list:
    try:
        result = list(events)
    except TypeError as exc:
        raise ValueError("reward events must be iterable") from exc
    if any(not isinstance(event, dict) or not isinstance(event.get("type"), str)
           for event in result):
        raise ValueError("reward event does not match the event contract")
    return result


@dataclass(frozen=True)
class CountRewardProfile:
    """Pure finite-horizon completed-task reward."""

    version: str = "count_reward_v1"
    gamma: float = 1.0
    horizon_seconds: float = 1800.0

    def __post_init__(self):
        gamma = _finite(self.gamma, "gamma", positive=True)
        horizon = _finite(
            self.horizon_seconds, "horizon_seconds", positive=True)
        if self.version != "count_reward_v1" or gamma != 1.0:
            raise ValueError("Count reward requires version v1 and gamma=1")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "horizon_seconds", horizon)

    def canonical(self) -> dict:
        return {
            "version": self.version,
            "objective": "count",
            "gamma": self.gamma,
            "horizon_seconds": self.horizon_seconds,
            "reward_clipping": None,
            "return_normalization": None,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical())

    def transition(self, tasks: Iterable[TransportTask], events, *,
                   horizon_seconds: float = 1800.0, **_ignored) -> dict:
        task_by_id = _reward_task_index(tasks)
        horizon = _finite(
            horizon_seconds, "horizon_seconds", positive=True)
        if horizon != self.horizon_seconds:
            raise ValueError("Count reward horizon differs from profile")
        completion = 0.0
        for event in _reward_events(events):
            if event["type"] != "task_completed":
                continue
            task = task_by_id.get(event.get("task_id"))
            if task is None or task.status != TaskStatus.COMPLETED:
                raise ValueError("completion event references invalid task state")
            if (task.completion_time is None or
                    not task.arrival_time <= task.completion_time <= horizon):
                raise ValueError("completion event lies outside the horizon")
            if (task.candidate_reward_eligible and
                    not task.completion_reward_emitted):
                task.completion_reward_emitted = True
                completion += 1.0
        return {
            "reward_profile_version": self.version,
            "reward_profile_hash": self.sha256,
            "reward_completion": completion,
            "reward_total": completion,
        }


@dataclass(frozen=True)
class UtilityV2RewardProfile:
    """Initial Utility V2 reward profile with explicit deadline scale."""

    lambda_deadline: float
    b_valid: float = 0.5
    b_priority: float = 0.5
    b_age: float = 0.5
    age_scale_seconds: float = 120.0
    max_age_bonus_units: float = 2.0
    b_complete: float = 10.0
    lambda_wait: float = 0.03
    lambda_execution: float = 0.01
    wait_cap_seconds: float = 300.0
    execution_cap_seconds: float = 300.0
    lambda_excess_distance: float = 0.08
    b_unfinished: float = 10.0
    lambda_unfinished_age: float = 0.03
    unfinished_age_cap_seconds: float = 300.0
    b_invalid: float = 5.0
    b_defer: float = 2.0
    b_collision: float = 100.0
    b_proximity: float = 20.0
    safety_episode_rearm_seconds: float = 1.0
    safety_observation: str = "unobserved"
    proximity_enabled: bool = False
    gamma: float = 1.0
    horizon_seconds: float = 1800.0
    version: str = "utility_reward_v2_initial"

    def __post_init__(self):
        positive = (
            "lambda_deadline", "age_scale_seconds", "max_age_bonus_units",
            "wait_cap_seconds", "execution_cap_seconds",
            "unfinished_age_cap_seconds", "safety_episode_rearm_seconds",
            "horizon_seconds",
        )
        nonnegative = (
            "b_valid", "b_priority", "b_age", "b_complete",
            "lambda_wait", "lambda_execution", "lambda_excess_distance",
            "b_unfinished", "lambda_unfinished_age", "b_invalid",
            "b_defer", "b_collision", "b_proximity",
        )
        for name in positive:
            object.__setattr__(
                self, name,
                _finite(getattr(self, name), name, positive=True))
        for name in nonnegative:
            object.__setattr__(
                self, name,
                _finite(getattr(self, name), name, nonnegative=True))
        if not isinstance(self.proximity_enabled, bool):
            raise ValueError("proximity_enabled must be boolean")
        if self.safety_observation not in {"unobserved", "webots_physical"}:
            raise ValueError("unsupported safety observation mode")
        gamma = _finite(self.gamma, "gamma", positive=True)
        if gamma != 1.0:
            raise ValueError("Utility V2 requires gamma=1")
        object.__setattr__(self, "gamma", gamma)
        if self.version != "utility_reward_v2_initial":
            raise ValueError("unsupported Utility reward version")

    def canonical(self) -> dict:
        names = (
            "b_valid", "b_priority", "b_age", "age_scale_seconds",
            "max_age_bonus_units", "b_complete", "lambda_wait",
            "lambda_execution", "wait_cap_seconds",
            "execution_cap_seconds", "lambda_excess_distance",
            "b_unfinished", "lambda_unfinished_age",
            "unfinished_age_cap_seconds", "b_invalid", "b_defer",
            "b_collision", "b_proximity", "safety_episode_rearm_seconds",
            "lambda_deadline", "safety_observation", "proximity_enabled",
            "gamma",
            "horizon_seconds",
        )
        result = {
            "version": self.version,
            "objective": "utility_v2",
            "priority_weights": {
                str(key): value for key, value in UTILITY_PRIORITY_WEIGHTS.items()},
            "reward_clipping": None,
            "return_normalization": None,
        }
        result.update({name: getattr(self, name) for name in names})
        return result

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical())

    def transition(
            self, tasks: Iterable[TransportTask], events, *,
            dispatch_task: Optional[TransportTask] = None,
            invalid_action: bool = False, defer: bool = False,
            horizon_seconds: float = 1800.0,
            runtime_mode: str = "headless_webots_logic") -> dict:
        if not isinstance(invalid_action, bool) or not isinstance(defer, bool):
            raise ValueError("reward action flags must be boolean")
        if invalid_action and defer:
            raise ValueError("a transition cannot be both invalid and deferred")
        horizon = _finite(
            horizon_seconds, "horizon_seconds", positive=True)
        if horizon != self.horizon_seconds:
            raise ValueError("Utility reward horizon differs from profile")
        task_by_id = _reward_task_index(tasks)
        event_rows = _reward_events(events)
        components = {
            name: 0.0 for name in REWARD_COMPONENT_NAMES
        }
        if self.safety_observation == "unobserved":
            components["reward_collision"] = None
            components["reward_proximity"] = None
        elif runtime_mode != "webots":
            raise ValueError(
                "physical safety reward requires Webots runtime")

        # Validate the full batch before mutating any idempotency cursor.
        for event in event_rows:
            kind = event["type"]
            task = None
            if "task_id" in event:
                task = task_by_id.get(event["task_id"])
                if task is None:
                    raise ValueError("reward event references an unknown task")
            if kind == "task_completed":
                if (task is None or task.status != TaskStatus.COMPLETED or
                        task.assignment_time is None or
                        task.completion_time is None or
                        task.completion_time > horizon):
                    raise ValueError("completion event has invalid task state")
            elif kind == "task_deadline_base":
                if (task is None or not task.deadline_penalty_emitted or
                        task.priority not in (2, 3) or
                        event.get("priority") != task.priority):
                    raise ValueError("deadline base event was not committed")
            elif kind == "task_deadline_severity":
                if (task is None or not task.deadline_severity_emitted or
                        task.deadline_tardiness_seconds is None):
                    raise ValueError("deadline severity event was not committed")
                severity = _finite(
                    event.get("severity"), "deadline severity",
                    nonnegative=True)
                expected_severity = min(
                    task.deadline_tardiness_seconds
                    / task.max_completion_time_seconds, 1.0)
                if (severity > 1.0 or not math.isclose(
                        severity, expected_severity,
                        rel_tol=0.0, abs_tol=1e-12)):
                    raise ValueError("deadline severity event was tampered")
            elif kind == "episode_horizon":
                event_time = _finite(
                    event.get("current_time"), "episode horizon event time",
                    nonnegative=True)
                if not math.isclose(
                        event_time, horizon, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("episode horizon event occurred before H")

        if dispatch_task is not None:
            canonical = task_by_id.get(dispatch_task.task_id)
            if canonical is not dispatch_task:
                raise ValueError("dispatch reward task is not canonical")
            if dispatch_task.candidate_reward_eligible:
                if dispatch_task.ideal_distance is None:
                    raise ValueError(
                        "Utility V2 dispatch requires ideal task distance")
                if not dispatch_task.dispatch_reward_emitted:
                    if dispatch_task.assignment_time is None:
                        raise ValueError("dispatched task is missing assignment_time")
                    waiting = _finite(
                        dispatch_task.assignment_time - dispatch_task.arrival_time,
                        "dispatch waiting time", nonnegative=True)
                    if dispatch_task.assignment_time > horizon:
                        raise ValueError("dispatch occurred after horizon")
                    weight = UTILITY_PRIORITY_WEIGHTS[dispatch_task.priority]
                    components["reward_dispatch"] += (
                        self.b_valid + self.b_priority * weight
                        + self.b_age * weight * min(
                            waiting / self.age_scale_seconds,
                            self.max_age_bonus_units))
                    dispatch_task.dispatch_reward_emitted = True

        def terminal_loss(task):
            if (not task.candidate_reward_eligible or
                    task.terminal_loss_emitted or
                    task.status == TaskStatus.COMPLETED or
                    not 0 <= task.arrival_time < horizon):
                return 0.0
            effective = task.effective_terminal_priority(horizon)
            weight = UTILITY_PRIORITY_WEIGHTS[effective]
            exposure = min(
                horizon - task.arrival_time,
                self.unfinished_age_cap_seconds)
            task.terminal_loss_emitted = True
            return -weight * (
                self.b_unfinished + self.lambda_unfinished_age * exposure)

        for event in event_rows:
            kind = event["type"]
            task = None
            if "task_id" in event:
                task = task_by_id.get(event["task_id"])
                if task is None:
                    raise ValueError("reward event references an unknown task")
            if kind == "task_completed":
                if task.status != TaskStatus.COMPLETED:
                    raise ValueError("completion event has invalid task state")
                if (task.candidate_reward_eligible and
                        not task.completion_reward_emitted):
                    if task.assignment_time is None or task.completion_time is None:
                        raise ValueError("completed task is missing timestamps")
                    waiting = _finite(
                        task.assignment_time - task.arrival_time,
                        "completion waiting time", nonnegative=True)
                    execution = _finite(
                        task.completion_time - task.assignment_time,
                        "completion execution time", nonnegative=True)
                    if task.completion_time > horizon:
                        raise ValueError("completion occurred after horizon")
                    weight = UTILITY_PRIORITY_WEIGHTS[task.priority]
                    components["reward_completion"] += weight * (
                        self.b_complete
                        - self.lambda_wait * min(waiting, self.wait_cap_seconds)
                        - self.lambda_execution * min(
                            execution, self.execution_cap_seconds))
                    task.completion_reward_emitted = True
            elif kind == "task_failed_battery":
                components["reward_terminal_loss"] += terminal_loss(task)
            elif kind == "task_route_excess":
                # Compatibility marker only. Cursor settlement below is the
                # authoritative source and coalesces any number of 16 ms ticks.
                pass
            elif kind == "task_deadline_base":
                if (task.candidate_reward_eligible and
                        not task.deadline_penalty_rewarded):
                    if (not task.deadline_penalty_emitted or
                            task.priority not in (2, 3) or
                            event.get("priority") != task.priority):
                        raise ValueError("deadline base event was not committed")
                    components["reward_deadline_base"] -= (
                        0.5 * self.lambda_deadline
                        * UTILITY_PRIORITY_WEIGHTS[task.priority])
                    task.deadline_penalty_rewarded = True
            elif kind == "task_deadline_severity":
                if (task.candidate_reward_eligible and
                        not task.deadline_severity_rewarded):
                    if not task.deadline_severity_emitted:
                        raise ValueError("deadline severity event was not committed")
                    severity = _finite(
                        event.get("severity"), "deadline severity",
                        nonnegative=True)
                    if severity > 1.0:
                        raise ValueError("deadline severity exceeds one")
                    if task.deadline_tardiness_seconds is None:
                        raise ValueError("deadline severity lacks tardiness")
                    expected_severity = min(
                        task.deadline_tardiness_seconds
                        / task.max_completion_time_seconds, 1.0)
                    if not math.isclose(
                            severity, expected_severity,
                            rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError("deadline severity event was tampered")
                    components["reward_deadline_severity"] -= (
                        0.5 * self.lambda_deadline
                        * UTILITY_PRIORITY_WEIGHTS[task.priority] * severity)
                    task.deadline_severity_rewarded = True
            elif kind == "episode_horizon":
                event_time = _finite(
                    event.get("current_time"), "episode horizon event time",
                    nonnegative=True)
                if not math.isclose(
                        event_time, horizon, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("episode horizon event occurred before H")
                for final_task in task_by_id.values():
                    components["reward_terminal_loss"] += terminal_loss(
                        final_task)
            elif (kind == "physical_collision" and
                  self.safety_observation == "webots_physical"):
                components["reward_collision"] -= self.b_collision
            elif (kind == "pair_distance_violation" and
                  self.safety_observation == "webots_physical" and
                  self.proximity_enabled):
                components["reward_proximity"] -= self.b_proximity

        if invalid_action:
            components["reward_invalid"] = -self.b_invalid
        if defer:
            components["reward_defer"] = -self.b_defer
        for task in task_by_id.values():
            if task.ideal_distance is None:
                if (task.excess_distance_cursor != 0 or
                        task.rewarded_excess_distance_cursor != 0):
                    raise ValueError("route reward state lacks ideal distance")
            elif task.excess_distance_cursor > max(
                    0.0, task.actual_distance - task.ideal_distance) + 1e-9:
                raise ValueError("route excess exceeds attributable distance")
            unrewarded = (task.excess_distance_cursor
                          - task.rewarded_excess_distance_cursor)
            if unrewarded < -1e-9:
                raise ValueError("rewarded route cursor moved backwards")
            if unrewarded > 0:
                if task.candidate_reward_eligible:
                    components["reward_route_excess"] -= (
                        self.lambda_excess_distance * unrewarded)
                task.rewarded_excess_distance_cursor = (
                    task.excess_distance_cursor)
        numeric = [value for value in components.values() if value is not None]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("reward component is non-finite")
        total = float(sum(numeric))
        return {
            "reward_profile_version": self.version,
            "reward_profile_hash": self.sha256,
            **components,
            "reward_total": total,
        }


def _percentile95(values) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = 0.95 * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return (ordered[lower] * (upper - index)
            + ordered[upper] * (index - lower))


def utility_v2_evaluation(
        tasks: Iterable[TransportTask], config: UtilityScoreConfig, *,
        route_distances: Optional[Mapping[int, Mapping[str, float]]] = None
        ) -> dict:
    """Compute every formal Utility V2 component from a horizon snapshot."""
    if not isinstance(config, UtilityScoreConfig):
        raise ValueError("config must be UtilityScoreConfig")
    rows = _task_snapshot(
        tasks, config.horizon_seconds, config.deadline_seconds)
    try:
        routes = {} if route_distances is None else dict(route_distances)
    except (TypeError, ValueError) as exc:
        raise ValueError("route_distances must be a mapping") from exc
    if route_distances is not None and not isinstance(route_distances, Mapping):
        raise ValueError("route_distances must be a mapping")
    empty = {
        "priority_completion": None,
        "timeliness": None,
        "queue_extra": None,
        "route_efficiency": None,
        "utility_score_base_v2": None,
        "deadline_loss": None,
        "deadline_adjustment": None,
        "utility_score_deadline_v2": None,
    }
    if not rows:
        if routes:
            raise ValueError("route distances cannot reference unevaluable tasks")
        return {
            "objective": "utility_v2",
            "score_version": config.version,
            "utility_score_config": config.canonical(),
            "utility_score_config_hash": config.sha256,
            "status": "no_evaluable_tasks",
            "task_count": 0,
            "completed_task_count": 0,
            "route_efficiency_score_value": 0.0,
            "route_ideal_distance_sum": 0.0,
            "route_actual_distance_sum": 0.0,
            "horizon_limited_task_count": 0,
            "horizon_limited_unfinished_count": 0,
            "effective_priority_weight_sum": 0.0,
            "original_priority_weight_sum": 0.0,
            "shadow_priority_completion": None,
            "shadow_timeliness": None,
            "shadow_queue_extra": None,
            "deadline_by_priority": {
                str(priority): {
                    "eligible_count": 0,
                    "on_time_count": 0,
                    "miss_count": 0,
                    "on_time_rate": None,
                    "deadline_miss_rate": None,
                    "tardiness_p95": None,
                } for priority in (2, 3)
            },
            **empty,
        }

    row_by_id = {row["task_id"]: row for row in rows}
    if any(isinstance(key, bool) or not isinstance(key, int)
           for key in routes):
        raise ValueError("route distance keys must be integer task IDs")
    assigned_ids = {
        row["task_id"] for row in rows if row["assignment"] is not None}
    if set(routes) != assigned_ids:
        raise ValueError("route distances must cover exactly the assigned tasks")

    route_ideal = 0.0
    route_actual = 0.0
    for task_id, values in routes.items():
        if not isinstance(values, Mapping):
            raise ValueError("route distance entry must be a mapping")
        if set(values) != {"ideal_distance", "actual_distance"}:
            raise ValueError("route distance fields do not match V2 contract")
        ideal = _finite(values["ideal_distance"], "ideal_distance",
                        nonnegative=True)
        actual = _finite(values["actual_distance"], "actual_distance",
                         nonnegative=True)
        route_ideal += ideal
        route_actual += actual
        row_by_id[task_id]["ideal_distance"] = ideal
        row_by_id[task_id]["actual_distance"] = actual
    if not routes:
        route_efficiency = None
    elif route_ideal == 0.0 and route_actual == 0.0:
        route_efficiency = 1.0
    else:
        route_efficiency = route_ideal / max(route_ideal, route_actual)

    effective_weight_sum = 0.0
    original_weight_sum = 0.0
    completion_sum = 0.0
    time_sum = 0.0
    queue_sum = 0.0
    shadow_completion_sum = 0.0
    shadow_time_sum = 0.0
    shadow_queue_sum = 0.0
    horizon_limited_count = 0
    horizon_limited_unfinished = 0
    for row in rows:
        task = row["task"]
        original = config.priority_weights[row["priority"]]
        limited = task.is_horizon_limited(config.horizon_seconds)
        horizon_limited_count += int(limited)
        horizon_limited_unfinished += int(limited and not row["complete"])
        effective_priority = task.effective_terminal_priority(
            config.horizon_seconds)
        effective = config.priority_weights[effective_priority]
        duration_credit = 0.0
        queue_credit = 0.0
        if row["complete"]:
            duration = row["completion"] - row["arrival"]
            waiting = row["assignment"] - row["arrival"]
            duration_credit = math.exp(
                -duration / config.tau_by_priority[row["priority"]])
            queue_credit = math.exp(
                -waiting / config.kappa_by_priority[row["priority"]])
        effective_weight_sum += effective
        original_weight_sum += original
        completion_sum += effective * int(row["complete"])
        time_sum += effective * duration_credit
        queue_sum += effective * queue_credit
        shadow_completion_sum += original * int(row["complete"])
        shadow_time_sum += original * duration_credit
        shadow_queue_sum += original * queue_credit

    priority_completion = completion_sum / effective_weight_sum
    timeliness = time_sum / effective_weight_sum
    queue_extra = queue_sum / effective_weight_sum
    route_for_score = 0.0 if route_efficiency is None else route_efficiency
    base = 100.0 * (
        config.component_weights["priority_completion"] * priority_completion
        + config.component_weights["timeliness"] * timeliness
        + config.component_weights["queue_extra"] * queue_extra
        + config.component_weights["route_efficiency"] * route_for_score)

    deadline_weight = 0.0
    deadline_loss_sum = 0.0
    deadline_stats: Dict[int, dict] = {
        2: {"eligible": 0, "on_time": 0, "missed": 0, "tardiness": []},
        3: {"eligible": 0, "on_time": 0, "missed": 0, "tardiness": []},
    }
    for row in rows:
        priority = row["priority"]
        if priority not in (2, 3):
            continue
        task = row["task"]
        if task.deadline_time > config.horizon_seconds:
            continue
        stats = deadline_stats[priority]
        stats["eligible"] += 1
        on_time = bool(
            row["complete"] and row["completion"] <= task.deadline_time)
        if on_time:
            stats["on_time"] += 1
            tardiness = 0.0
            severity = 0.0
        else:
            stats["missed"] += 1
            tardiness = ((row["completion"] - task.deadline_time)
                          if row["complete"] else
                          (config.horizon_seconds - task.deadline_time))
            if tardiness < 0:
                raise ValueError("deadline tardiness cannot be negative")
            severity = 0.5 + 0.5 * min(
                tardiness / task.max_completion_time_seconds, 1.0)
        stats["tardiness"].append(tardiness)
        weight = config.priority_weights[priority]
        deadline_weight += weight
        deadline_loss_sum += weight * severity

    deadline_loss = (None if deadline_weight == 0 else
                     deadline_loss_sum / deadline_weight)
    adjustment = (0.0 if deadline_loss is None else
                  config.max_deadline_penalty_points * deadline_loss)
    final_score = min(100.0, max(0.0, base - adjustment))
    deadline_report = {}
    for priority, stats in deadline_stats.items():
        eligible = stats["eligible"]
        deadline_report[str(priority)] = {
            "eligible_count": eligible,
            "on_time_count": stats["on_time"],
            "miss_count": stats["missed"],
            "on_time_rate": (
                None if eligible == 0 else stats["on_time"] / eligible),
            "deadline_miss_rate": (
                None if eligible == 0 else stats["missed"] / eligible),
            "tardiness_p95": _percentile95(stats["tardiness"]),
        }

    return {
        "objective": "utility_v2",
        "score_version": config.version,
        "utility_score_config": config.canonical(),
        "utility_score_config_hash": config.sha256,
        "status": "ok",
        "task_count": len(rows),
        "completed_task_count": sum(row["complete"] for row in rows),
        "priority_completion": priority_completion,
        "timeliness": timeliness,
        "queue_extra": queue_extra,
        "route_efficiency": route_efficiency,
        "route_efficiency_score_value": route_for_score,
        "route_ideal_distance_sum": route_ideal,
        "route_actual_distance_sum": route_actual,
        "utility_score_base_v2": base,
        "horizon_limited_task_count": horizon_limited_count,
        "horizon_limited_unfinished_count": horizon_limited_unfinished,
        "effective_priority_weight_sum": effective_weight_sum,
        "original_priority_weight_sum": original_weight_sum,
        "shadow_priority_completion": (
            shadow_completion_sum / original_weight_sum),
        "shadow_timeliness": shadow_time_sum / original_weight_sum,
        "shadow_queue_extra": shadow_queue_sum / original_weight_sum,
        "deadline_by_priority": deadline_report,
        "deadline_loss": deadline_loss,
        "deadline_adjustment": adjustment,
        "utility_score_deadline_v2": final_score,
    }
