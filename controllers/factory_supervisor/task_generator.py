"""
Task Generation Module for Smart Factory Simulation.
Generates transport tasks using a Poisson process with configurable arrival rate.
Each task specifies a pickup location and delivery location within the factory.
"""

import random
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from config import (
    ALL_LOCATIONS, WORKSTATIONS, STORAGE_AREAS,
    PRIORITY_MAX_COMPLETION_SECONDS, TaskStatus,
    validate_priority_max_completion_seconds,
)


TASK_GENERATION_PARAMETER_FIELDS = (
    "task_id",
    "sequence_index",
    "pickup_location",
    "delivery_location",
    "pickup_position",
    "delivery_position",
    "arrival_time",
    "priority",
    "max_completion_time_seconds",
    "deadline_time",
)

TASK_MANIFEST_VERSION = "canonical-task-manifest-v1"
TASK_GENERATOR_VERSION = "webots-poisson-tick-v1"


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _strict_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def canonical_task_manifest(entries) -> List[dict]:
    """Validate and canonicalize task parameters for paired evaluation."""
    canonical = []
    expected_task_id = 1
    for entry in entries:
        missing = [field for field in TASK_GENERATION_PARAMETER_FIELDS
                   if field not in entry]
        if missing:
            raise ValueError(
                f"task manifest entry is missing fields: {missing}")
        task_id = _strict_integer(entry["task_id"], "task_id")
        if task_id != expected_task_id:
            raise ValueError(
                "task manifest IDs must be contiguous and ordered from 1")
        expected_task_id += 1
        sequence_index = _strict_integer(
            entry["sequence_index"], "sequence_index")
        if sequence_index != task_id - 1:
            raise ValueError(
                "task manifest sequence_index must equal task_id minus one")
        priority = _strict_integer(entry["priority"], "task priority")
        arrival_time = _strict_float(entry["arrival_time"], "arrival_time")
        if arrival_time < 0:
            raise ValueError(
                "task manifest arrival_time must be finite and nonnegative")
        limit = validate_priority_max_completion_seconds(
            priority, entry["max_completion_time_seconds"])
        deadline = entry["deadline_time"]
        if limit is None:
            if deadline is not None:
                raise ValueError(
                    "priority 1 task manifest deadline must be null")
            canonical_deadline = None
        else:
            canonical_deadline = _strict_float(deadline, "deadline_time")
            if (not math.isfinite(canonical_deadline) or
                    not math.isclose(
                        canonical_deadline, arrival_time + limit,
                        rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError(
                    "task manifest deadline must equal arrival plus limit")
        pickup_location = str(entry["pickup_location"])
        delivery_location = str(entry["delivery_location"])
        if (pickup_location not in ALL_LOCATIONS or
                delivery_location not in ALL_LOCATIONS or
                pickup_location == delivery_location):
            raise ValueError("task manifest contains invalid endpoints")
        pickup_position = [_strict_float(value, "pickup position")
                           for value in entry["pickup_position"]]
        delivery_position = [_strict_float(value, "delivery position")
                             for value in entry["delivery_position"]]
        if len(pickup_position) != 2 or len(delivery_position) != 2:
            raise ValueError(
                "task manifest positions must contain two coordinates")
        if not all(math.isfinite(value) for value in
                   pickup_position + delivery_position):
            raise ValueError("task manifest positions must be finite")
        expected_pickup = [float(value)
                           for value in ALL_LOCATIONS[pickup_location]]
        expected_delivery = [float(value)
                             for value in ALL_LOCATIONS[delivery_location]]
        if (pickup_position != expected_pickup or
                delivery_position != expected_delivery):
            raise ValueError(
                "task manifest positions do not match endpoint names")
        canonical.append({
            "task_id": task_id,
            "sequence_index": sequence_index,
            "pickup_location": pickup_location,
            "delivery_location": delivery_location,
            "pickup_position": pickup_position,
            "delivery_position": delivery_position,
            "arrival_time": arrival_time,
            "priority": priority,
            "max_completion_time_seconds": limit,
            "deadline_time": canonical_deadline,
        })
    return canonical


def task_manifest_sha256(entries) -> str:
    """Return a stable hash of all scheduler-independent task parameters."""
    return _canonical_json_sha256(canonical_task_manifest(entries))


def _canonical_deadline_config(values) -> dict:
    raw = dict(values)
    unknown = set(raw) - {1, 2, 3, "1", "2", "3"}
    if unknown:
        raise ValueError(
            f"deadline config contains unsupported priorities: {unknown}")
    result = {}
    for priority in (1, 2, 3):
        value = raw.get(priority, raw.get(str(priority)))
        result[str(priority)] = validate_priority_max_completion_seconds(
            priority, value)
    if result["3"] >= result["2"]:
        raise ValueError("priority 3 deadline must be less than priority 2")
    return result


def canonical_task_manifest_document(document: dict) -> dict:
    """Validate a complete manifest and return its canonical representation."""
    if not isinstance(document, dict):
        raise ValueError("task manifest document must be an object")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("task manifest metadata must be an object")
    if metadata.get("manifest_version") != TASK_MANIFEST_VERSION:
        raise ValueError("unsupported task manifest version")
    if metadata.get("task_generator_version") != TASK_GENERATOR_VERSION:
        raise ValueError("unsupported task generator version")
    scenario_id = str(metadata.get("scenario_id", ""))
    if not scenario_id:
        raise ValueError("scenario_id must be non-empty")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest seed must be an integer")
    duration = _strict_float(
        metadata.get("duration_seconds"), "manifest duration")
    timestep_ms = metadata.get("timestep_ms")
    if (not math.isfinite(duration) or duration <= 0 or
            isinstance(timestep_ms, bool) or not isinstance(timestep_ms, int)
            or timestep_ms <= 0):
        raise ValueError("manifest duration and timestep must be positive")
    initial = metadata.get("initial_task_immediately")
    if not isinstance(initial, bool):
        raise ValueError("initial_task_immediately must be boolean")
    scenario_config = metadata.get("scenario_config")
    if not isinstance(scenario_config, dict) or not scenario_config:
        raise ValueError("scenario_config must be a non-empty object")
    deadline_config = _canonical_deadline_config(
        metadata.get("deadline_config", {}))
    scenario_hash = _canonical_json_sha256(scenario_config)
    deadline_hash = _canonical_json_sha256(deadline_config)
    if (metadata.get("scenario_config_hash") not in (None, scenario_hash) or
            metadata.get("deadline_config_hash") not in (None, deadline_hash)):
        raise ValueError("manifest configuration hash mismatch")
    raw_tasks = document.get("tasks", [])
    if (not isinstance(raw_tasks, list) or
            any(not isinstance(entry, dict) for entry in raw_tasks)):
        raise ValueError("manifest tasks must be an array")
    expected_task_fields = set(TASK_GENERATION_PARAMETER_FIELDS)
    if any(set(entry) != expected_task_fields for entry in raw_tasks
           if isinstance(entry, dict)):
        raise ValueError("manifest task fields do not match version contract")
    tasks = canonical_task_manifest(raw_tasks)
    if any(not 0.0 <= task["arrival_time"] < duration for task in tasks):
        raise ValueError("task arrival lies outside the manifest duration")
    tick_seconds = timestep_ms / 1000.0
    for task in tasks:
        tick = round(task["arrival_time"] / tick_seconds)
        if not math.isclose(
                task["arrival_time"], tick * tick_seconds,
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("task arrival_time is not on a simulation tick")
        configured = deadline_config[str(task["priority"])]
        if task["max_completion_time_seconds"] != configured:
            raise ValueError("task deadline does not match manifest config")
    task_hash = task_manifest_sha256(tasks)
    if document.get("task_manifest_sha256") not in (None, task_hash):
        raise ValueError("task manifest hash mismatch")
    canonical_metadata = {
        "manifest_version": TASK_MANIFEST_VERSION,
        "task_generator_version": TASK_GENERATOR_VERSION,
        "scenario_id": scenario_id,
        "seed": seed,
        "duration_seconds": duration,
        "timestep_ms": timestep_ms,
        "initial_task_immediately": initial,
        "scenario_config": scenario_config,
        "scenario_config_hash": scenario_hash,
        "deadline_config": deadline_config,
        "deadline_config_hash": deadline_hash,
    }
    consistency_key = {
        "scenario_id": scenario_id,
        "seed": seed,
        "scenario_config_hash": scenario_hash,
        "task_generator_version": TASK_GENERATOR_VERSION,
        "deadline_config_hash": deadline_hash,
        "simulation_timestep_ms": timestep_ms,
        "initial_task_immediately": initial,
        "duration_seconds": duration,
    }
    if document.get("consistency_key") not in (None, consistency_key):
        raise ValueError("manifest consistency key mismatch")
    payload = {
        "metadata": canonical_metadata,
        "consistency_key": consistency_key,
        "tasks": tasks,
        "task_manifest_sha256": task_hash,
    }
    manifest_hash = _canonical_json_sha256(payload)
    if document.get("manifest_sha256") not in (None, manifest_hash):
        raise ValueError("manifest document hash mismatch")
    payload["manifest_sha256"] = manifest_hash
    return payload


def generate_task_manifest(scenario_id: str, scenario_config: dict, seed: int,
                           duration_seconds: float = 1800.0,
                           timestep_ms: int = 16,
                           priority_max_completion_seconds=None) -> dict:
    """Generate the authoritative Webots-tick task manifest for one run."""
    if not isinstance(scenario_config, dict):
        raise ValueError("scenario_config must be an object")
    seed = _strict_integer(seed, "seed")
    mean_interval = _strict_float(
        scenario_config.get("task_interval"), "task_interval")
    initial = scenario_config.get("initial_task_immediately", False)
    if not isinstance(initial, bool):
        raise ValueError("initial_task_immediately must be boolean")
    if (isinstance(timestep_ms, bool) or not isinstance(timestep_ms, int) or
            timestep_ms <= 0):
        raise ValueError("timestep_ms must be a positive integer")
    duration = _strict_float(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    deadline_config = _canonical_deadline_config(
        priority_max_completion_seconds or PRIORITY_MAX_COMPLETION_SECONDS)
    generator = TaskGenerator(
        mean_interval=mean_interval, seed=seed,
        initial_task_immediately=initial,
        priority_max_completion_seconds=deadline_config)
    tick_seconds = timestep_ms / 1000.0
    tick_count = int(math.ceil(duration / tick_seconds))
    tasks = []
    for tick in range(tick_count):
        current_time = tick * tick_seconds
        if current_time >= duration:
            break
        task = generator.update(current_time)
        if task is not None:
            tasks.append(task.generation_parameters())
    document = {
        "metadata": {
            "manifest_version": TASK_MANIFEST_VERSION,
            "task_generator_version": TASK_GENERATOR_VERSION,
            "scenario_id": str(scenario_id),
            "seed": seed,
            "duration_seconds": duration,
            "timestep_ms": timestep_ms,
            "initial_task_immediately": initial,
            "scenario_config": dict(scenario_config),
            "deadline_config": deadline_config,
        },
        "tasks": tasks,
    }
    return canonical_task_manifest_document(document)


def write_task_manifest(path, document: dict) -> dict:
    """Validate and atomically write a canonical task manifest."""
    canonical = canonical_task_manifest_document(document)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(canonical, indent=2, sort_keys=True,
                       ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return canonical


def load_task_manifest(path) -> dict:
    """Load and strictly validate a task manifest from disk."""
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")))
    return canonical_task_manifest_document(value)


def tasks_from_manifest(document_or_entries) -> List["TransportTask"]:
    """Build fresh task objects without sharing mutable run state."""
    entries = (canonical_task_manifest_document(document_or_entries)["tasks"]
               if isinstance(document_or_entries, dict)
               else canonical_task_manifest(document_or_entries))
    return [TransportTask(
        task_id=entry["task_id"],
        pickup_location=entry["pickup_location"],
        delivery_location=entry["delivery_location"],
        pickup_position=tuple(entry["pickup_position"]),
        delivery_position=tuple(entry["delivery_position"]),
        arrival_time=entry["arrival_time"],
        priority=entry["priority"],
        max_completion_time_seconds=entry["max_completion_time_seconds"],
        deadline_time=entry["deadline_time"],
    ) for entry in entries]


@dataclass
class TransportTask:
    """Represents a single transport task in the factory."""
    task_id: int
    pickup_location: str          # Location name (e.g., "S1", "WS2")
    delivery_location: str        # Location name (e.g., "WS3", "S5")
    pickup_position: Tuple[float, float]   # (x, y) on floor plane (Webots ENU)
    delivery_position: Tuple[float, float] # (x, y) on floor plane (Webots ENU)
    arrival_time: float           # Simulation time when task appeared
    status: str = TaskStatus.PENDING
    assigned_robot: Optional[int] = None
    assignment_time: Optional[float] = None
    pickup_time: Optional[float] = None
    completion_time: Optional[float] = None
    priority: float = 1.0         # Higher = more urgent
    max_completion_time_seconds: Optional[int] = None
    deadline_time: Optional[float] = None
    deadline_missed: Optional[bool] = None
    deadline_missed_at: Optional[float] = None
    deadline_penalty_emitted: bool = False
    deadline_severity_emitted: bool = False
    deadline_penalty_rewarded: bool = False
    deadline_severity_rewarded: bool = False
    deadline_tardiness_seconds: Optional[float] = None
    completed_on_time: Optional[bool] = None
    dispatch_reward_emitted: bool = False
    completion_reward_emitted: bool = False
    terminal_loss_emitted: bool = False
    candidate_reward_eligible: bool = True
    ideal_distance: Optional[float] = None
    actual_distance: float = 0.0
    excess_distance_cursor: float = 0.0
    rewarded_excess_distance_cursor: float = 0.0
    learning_trace: Optional[dict] = field(
        default=None, repr=False, compare=False)

    def __post_init__(self):
        """Attach and validate scheduler-independent SLA metadata."""
        self.arrival_time = _strict_float(self.arrival_time, "arrival_time")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be nonnegative")
        if isinstance(self.priority, bool):
            raise ValueError("task priority must be 1, 2, or 3")
        numeric_priority = float(self.priority)
        if (not math.isfinite(numeric_priority) or
                not numeric_priority.is_integer()):
            raise ValueError("task priority must be 1, 2, or 3")
        priority = int(numeric_priority)
        self.priority = priority
        configured = self.max_completion_time_seconds
        if priority in (2, 3) and configured is None:
            configured = PRIORITY_MAX_COMPLETION_SECONDS[priority]
        configured = validate_priority_max_completion_seconds(
            priority, configured)
        self.max_completion_time_seconds = configured
        expected_deadline = (None if configured is None else
                             float(self.arrival_time) + configured)
        if self.deadline_time is None:
            self.deadline_time = expected_deadline
        elif expected_deadline is None or not math.isclose(
                float(self.deadline_time), expected_deadline,
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "deadline_time must equal arrival_time plus the configured "
                "maximum completion time")
        elif self.deadline_time is not None:
            self.deadline_time = float(self.deadline_time)
        for name in ("deadline_penalty_emitted",
                     "deadline_severity_emitted",
                     "deadline_penalty_rewarded",
                     "deadline_severity_rewarded",
                     "dispatch_reward_emitted",
                     "completion_reward_emitted",
                     "terminal_loss_emitted",
                     "candidate_reward_eligible"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.ideal_distance is not None:
            self.ideal_distance = _strict_float(
                self.ideal_distance, "ideal_distance")
            if self.ideal_distance < 0:
                raise ValueError("ideal_distance must be nonnegative")
        self.actual_distance = _strict_float(
            self.actual_distance, "actual_distance")
        self.excess_distance_cursor = _strict_float(
            self.excess_distance_cursor, "excess_distance_cursor")
        self.rewarded_excess_distance_cursor = _strict_float(
            self.rewarded_excess_distance_cursor,
            "rewarded_excess_distance_cursor")
        if (self.actual_distance < 0 or self.excess_distance_cursor < 0 or
                self.rewarded_excess_distance_cursor < 0):
            raise ValueError("task distance state must be nonnegative")
        if self.ideal_distance is None:
            if (self.actual_distance != 0 or self.excess_distance_cursor != 0
                    or self.rewarded_excess_distance_cursor != 0):
                raise ValueError("task distance state requires ideal_distance")
        elif self.excess_distance_cursor > max(
                0.0, self.actual_distance - self.ideal_distance) + 1e-9:
            raise ValueError("excess distance cursor exceeds actual excess")
        if (self.rewarded_excess_distance_cursor >
                self.excess_distance_cursor + 1e-9):
            raise ValueError("rewarded excess cursor exceeds emitted excess")
        if (self.deadline_missed is not None and
                not isinstance(self.deadline_missed, bool)):
            raise ValueError("deadline_missed must be boolean or null")
        if (self.completed_on_time is not None and
                not isinstance(self.completed_on_time, bool)):
            raise ValueError("completed_on_time must be boolean or null")
        if self.deadline_missed_at is not None:
            self.deadline_missed_at = _strict_float(
                self.deadline_missed_at, "deadline_missed_at")
            if self.deadline_missed_at < self.arrival_time:
                raise ValueError("deadline_missed_at precedes arrival_time")
        if self.deadline_tardiness_seconds is not None:
            self.deadline_tardiness_seconds = _strict_float(
                self.deadline_tardiness_seconds,
                "deadline_tardiness_seconds")
            if self.deadline_tardiness_seconds < 0:
                raise ValueError("deadline tardiness must be nonnegative")
        if priority == 1:
            if any((self.deadline_missed is not None,
                    self.deadline_missed_at is not None,
                    self.deadline_penalty_emitted,
                    self.deadline_severity_emitted,
                    self.deadline_penalty_rewarded,
                    self.deadline_severity_rewarded,
                    self.deadline_tardiness_seconds is not None,
                    self.completed_on_time is not None)):
                raise ValueError("priority 1 tasks cannot have deadline state")
        elif self.deadline_missed is None:
            self.deadline_missed = False
        if priority in (2, 3):
            if self.deadline_severity_emitted and not self.deadline_penalty_emitted:
                raise ValueError("deadline severity requires the base event")
            if (self.deadline_penalty_rewarded and
                    not self.deadline_penalty_emitted):
                raise ValueError("deadline reward requires the base event")
            if (self.deadline_severity_rewarded and
                    not self.deadline_severity_emitted):
                raise ValueError("deadline reward requires the severity event")
            if bool(self.deadline_missed) != self.deadline_penalty_emitted:
                raise ValueError("deadline miss state conflicts with base event")
            if ((self.deadline_missed_at is not None)
                    != self.deadline_penalty_emitted):
                raise ValueError("deadline_missed_at conflicts with base event")
            if (self.deadline_severity_emitted and
                    self.deadline_tardiness_seconds is None):
                raise ValueError("deadline tardiness conflicts with severity event")
            if (self.deadline_tardiness_seconds is not None and
                    not self.deadline_severity_emitted and
                    self.completed_on_time is not True):
                raise ValueError("deadline tardiness conflicts with severity event")
            if self.completed_on_time is True and self.deadline_missed:
                raise ValueError("on-time completion conflicts with deadline miss")

    def is_horizon_limited(self, horizon_seconds: float) -> bool:
        """Return whether H cuts off the task's full P2/P3 SLA window."""
        horizon = _strict_float(horizon_seconds, "episode horizon")
        if horizon <= 0:
            raise ValueError("episode horizon must be positive")
        return bool(
            self.priority in (2, 3)
            and self.arrival_time < horizon
            and horizon - self.arrival_time
            < float(self.max_completion_time_seconds))

    def effective_terminal_priority(self, horizon_seconds: float) -> int:
        """Return the priority used by unfinished loss at the horizon."""
        if self.status not in {
                TaskStatus.PENDING, TaskStatus.ASSIGNED,
                TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED,
                TaskStatus.FAILED}:
            raise ValueError("task has an unsupported status")
        if (self.status != TaskStatus.COMPLETED and
                self.is_horizon_limited(horizon_seconds)):
            return 1
        return int(self.priority)

    def deadline_events(self, current_time: float, *,
                        horizon_seconds: Optional[float] = None,
                        final: bool = False) -> List[dict]:
        """Advance deadline state and emit each deadline event exactly once.

        Events contain dimensionless facts.  Reward profiles apply their own
        versioned coefficients later instead of coupling them to task state.
        """
        now = _strict_float(current_time, "current_time")
        if now < 0:
            raise ValueError("current_time must be nonnegative")
        horizon = None
        if horizon_seconds is not None:
            horizon = _strict_float(horizon_seconds, "episode horizon")
            if horizon <= 0:
                raise ValueError("episode horizon must be positive")
        if final:
            if horizon is None:
                raise ValueError("final deadline settlement requires a horizon")
            if not math.isclose(now, horizon, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("final deadline settlement must occur at horizon")
        if self.status not in {
                TaskStatus.PENDING, TaskStatus.ASSIGNED,
                TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED,
                TaskStatus.FAILED}:
            raise ValueError("task has an unsupported status")
        if now + 1e-9 < self.arrival_time:
            return []
        if self.priority == 1:
            return []
        if self.deadline_time is None or self.max_completion_time_seconds is None:
            raise ValueError("priority 2/3 task is missing deadline metadata")

        completed = self.status == TaskStatus.COMPLETED
        completion_time = None
        if completed:
            if self.completion_time is None:
                raise ValueError("completed task is missing completion_time")
            completion_time = _strict_float(
                self.completion_time, "completion_time")
            if completion_time < self.arrival_time:
                raise ValueError("completion_time precedes arrival_time")
            if horizon is not None and completion_time > horizon:
                raise ValueError("completion_time exceeds episode horizon")
            on_time = completion_time <= self.deadline_time
            if (self.deadline_penalty_emitted and on_time) or (
                    self.deadline_missed is True and on_time):
                raise ValueError("deadline state conflicts with completion_time")
            self.completed_on_time = on_time
            self.deadline_missed = not on_time
            self.deadline_tardiness_seconds = max(
                0.0, completion_time - self.deadline_time)
            event_time = completion_time
            should_miss = not on_time
            should_settle_severity = not on_time
        else:
            self.completed_on_time = False if final else None
            event_time = now
            should_miss = now > self.deadline_time
            should_settle_severity = False
            if final:
                in_scoring_window = self.arrival_time < horizon
                should_miss = bool(
                    in_scoring_window and self.deadline_time <= horizon)
                should_settle_severity = should_miss
                if should_miss:
                    self.deadline_tardiness_seconds = max(
                        0.0, horizon - self.deadline_time)

        if not should_miss:
            if completed:
                self.deadline_tardiness_seconds = 0.0
            return []

        events = []
        if not self.deadline_penalty_emitted:
            self.deadline_missed = True
            self.deadline_missed_at = event_time
            self.deadline_penalty_emitted = True
            events.append({
                "type": "task_deadline_base",
                "task_id": int(self.task_id),
                "priority": int(self.priority),
                "deadline_time": float(self.deadline_time),
                "observed_at": float(event_time),
            })
        if should_settle_severity and not self.deadline_severity_emitted:
            tardiness = (max(0.0, completion_time - self.deadline_time)
                         if completed else
                         max(0.0, horizon - self.deadline_time))
            severity = min(
                tardiness / float(self.max_completion_time_seconds), 1.0)
            self.deadline_tardiness_seconds = tardiness
            self.deadline_severity_emitted = True
            events.append({
                "type": "task_deadline_severity",
                "task_id": int(self.task_id),
                "priority": int(self.priority),
                "deadline_time": float(self.deadline_time),
                "tardiness_seconds": float(tardiness),
                "severity": float(severity),
            })
        return events

    def generation_parameters(self) -> dict:
        """Return scheduler-independent parameters used for paired runs."""
        return {
            "task_id": int(self.task_id),
            "sequence_index": int(self.task_id) - 1,
            "pickup_location": str(self.pickup_location),
            "delivery_location": str(self.delivery_location),
            "pickup_position": [float(value)
                                for value in self.pickup_position],
            "delivery_position": [float(value)
                                  for value in self.delivery_position],
            "arrival_time": float(self.arrival_time),
            "priority": int(self.priority),
            "max_completion_time_seconds": (
                int(self.max_completion_time_seconds)
                if self.max_completion_time_seconds is not None else None),
            "deadline_time": (float(self.deadline_time)
                              if self.deadline_time is not None else None),
        }

    @property
    def waiting_time(self) -> Optional[float]:
        """Time from arrival to assignment."""
        if self.assignment_time is not None:
            return self.assignment_time - self.arrival_time
        return None

    @property
    def completion_duration(self) -> Optional[float]:
        """Total time from arrival to completion."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None

    @property
    def execution_time(self) -> Optional[float]:
        """Time from assignment to completion."""
        if self.completion_time is not None and self.assignment_time is not None:
            return self.completion_time - self.assignment_time
        return None


class TaskGenerator:
    """
    Generates transport tasks using a Poisson process.
    
    Tasks represent material transport between workstations and storage areas.
    The arrival rate follows a Poisson distribution with configurable mean
    inter-arrival time.
    """

    def __init__(self, mean_interval: float, seed: int = 42,
                 initial_task_immediately: bool = False,
                 priority_max_completion_seconds=None,
                 manifest_entries=None):
        """
        Args:
            mean_interval: Mean time between task arrivals in seconds (lambda^-1).
            seed: Random seed for reproducibility.
            priority_max_completion_seconds: Optional per-priority override.
        """
        if not math.isfinite(mean_interval) or mean_interval <= 0:
            raise ValueError("mean_interval must be finite and greater than zero")
        self.mean_interval = mean_interval
        self.seed = int(seed)
        self.initial_task_immediately = bool(initial_task_immediately)
        self.rng = random.Random(seed)
        configured_deadlines = dict(PRIORITY_MAX_COMPLETION_SECONDS)
        if priority_max_completion_seconds is not None:
            overrides = {}
            for priority, value in dict(
                    priority_max_completion_seconds).items():
                canonical_priority = _strict_integer(
                    priority, "deadline priority")
                if canonical_priority in overrides:
                    raise ValueError("duplicate canonical deadline priority")
                overrides[canonical_priority] = value
            unknown_priorities = set(overrides) - {1, 2, 3}
            if unknown_priorities:
                raise ValueError(
                    "deadline overrides contain unsupported priorities: "
                    f"{sorted(unknown_priorities, key=str)}")
            configured_deadlines.update(overrides)
        self.priority_max_completion_seconds = {
            priority: validate_priority_max_completion_seconds(
                priority, configured_deadlines.get(priority))
            for priority in (1, 2, 3)
        }
        self.task_counter = 0
        self.next_arrival_time = 0.0
        self.tasks_generated: List[TransportTask] = []
        self._manifest_entries = (
            None if manifest_entries is None else
            canonical_task_manifest(manifest_entries))
        self._manifest_index = 0
        if self._manifest_entries is not None:
            for entry in self._manifest_entries:
                expected = self.priority_max_completion_seconds[
                    entry["priority"]]
                if entry["max_completion_time_seconds"] != expected:
                    raise ValueError(
                        "manifest deadline does not match TaskGenerator config")
            self.next_arrival_time = (
                self._manifest_entries[0]["arrival_time"]
                if self._manifest_entries else math.inf)
        elif initial_task_immediately:
            self.next_arrival_time = 0.0
        else:
            self._schedule_next_arrival(0.0)

    def _schedule_next_arrival(self, current_time: float):
        """Schedule the next task arrival using exponential distribution (Poisson process)."""
        # Exponential inter-arrival time = Poisson process
        inter_arrival = self.rng.expovariate(1.0 / self.mean_interval)
        self.next_arrival_time = current_time + inter_arrival

    def _generate_task_pair(self) -> Tuple[str, str]:
        """
        Generate a valid pickup-delivery location pair.
        
        Task types:
        1. Storage -> Workstation (material delivery to production line)
        2. Workstation -> Storage (finished goods to storage)
        3. Workstation -> Workstation (inter-line transfer)
        """
        task_type = self.rng.random()
        
        if task_type < 0.4:
            # Storage to Workstation (40% of tasks)
            pickup = self.rng.choice(list(STORAGE_AREAS.keys()))
            delivery = self.rng.choice(list(WORKSTATIONS.keys()))
        elif task_type < 0.75:
            # Workstation to Storage (35% of tasks)
            pickup = self.rng.choice(list(WORKSTATIONS.keys()))
            delivery = self.rng.choice(list(STORAGE_AREAS.keys()))
        else:
            # Workstation to Workstation (25% of tasks)
            ws_list = list(WORKSTATIONS.keys())
            pickup = self.rng.choice(ws_list)
            delivery = self.rng.choice([ws for ws in ws_list if ws != pickup])

        return pickup, delivery

    def update(self, current_time: float) -> Optional[TransportTask]:
        """
        Check if a new task should be generated at the current simulation time.
        
        Args:
            current_time: Current simulation time in seconds.
            
        Returns:
            A new TransportTask if one is generated, None otherwise.
        """
        if self._manifest_entries is not None:
            if (self._manifest_index >= len(self._manifest_entries) or
                    current_time + 1e-9 < self.next_arrival_time):
                return None
            entry = self._manifest_entries[self._manifest_index]
            task = TransportTask(
                task_id=entry["task_id"],
                pickup_location=entry["pickup_location"],
                delivery_location=entry["delivery_location"],
                pickup_position=tuple(entry["pickup_position"]),
                delivery_position=tuple(entry["delivery_position"]),
                arrival_time=entry["arrival_time"],
                priority=entry["priority"],
                max_completion_time_seconds=(
                    entry["max_completion_time_seconds"]),
                deadline_time=entry["deadline_time"],
            )
            self._manifest_index += 1
            self.task_counter = task.task_id
            self.tasks_generated.append(task)
            self.next_arrival_time = (
                self._manifest_entries[self._manifest_index]["arrival_time"]
                if self._manifest_index < len(self._manifest_entries)
                else math.inf)
            return task

        if current_time >= self.next_arrival_time:
            # Generate new task
            pickup_loc, delivery_loc = self._generate_task_pair()
            priority_roll = self.rng.random()
            
            self.task_counter += 1
            priority = (3 if priority_roll < 0.05 else
                        2 if priority_roll < 0.25 else 1)
            task = TransportTask(
                task_id=self.task_counter,
                pickup_location=pickup_loc,
                delivery_location=delivery_loc,
                pickup_position=ALL_LOCATIONS[pickup_loc],
                delivery_position=ALL_LOCATIONS[delivery_loc],
                arrival_time=current_time,
                priority=priority,
                max_completion_time_seconds=(
                    self.priority_max_completion_seconds[priority])
            )
            
            self.tasks_generated.append(task)
            self._schedule_next_arrival(current_time)
            return task
        
        return None

    def get_pending_tasks(self) -> List[TransportTask]:
        """Return all tasks that are still pending (not yet assigned)."""
        return [t for t in self.tasks_generated if t.status == TaskStatus.PENDING]

    def get_active_tasks(self) -> List[TransportTask]:
        """Return all tasks currently being executed."""
        return [t for t in self.tasks_generated
                if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)]

    def get_completed_tasks(self) -> List[TransportTask]:
        """Return all completed tasks."""
        return [t for t in self.tasks_generated if t.status == TaskStatus.COMPLETED]

    def get_statistics(self) -> dict:
        """Compute summary statistics of task generation and completion."""
        completed = self.get_completed_tasks()
        pending = self.get_pending_tasks()
        active = self.get_active_tasks()
        
        stats = {
            "total_generated": len(self.tasks_generated),
            "pending": len(pending),
            "active": len(active),
            "completed": len(completed),
            "failed": len([t for t in self.tasks_generated if t.status == TaskStatus.FAILED]),
        }
        
        if completed:
            completion_times = [t.completion_duration for t in completed if t.completion_duration]
            waiting_times = [t.waiting_time for t in completed if t.waiting_time is not None]
            
            stats["avg_completion_time"] = sum(completion_times) / len(completion_times) if completion_times else 0
            stats["max_completion_time"] = max(completion_times) if completion_times else 0
            stats["avg_waiting_time"] = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            stats["throughput"] = len(completed)  # tasks completed in sim duration
        else:
            stats["avg_completion_time"] = 0
            stats["max_completion_time"] = 0
            stats["avg_waiting_time"] = 0
            stats["throughput"] = 0
            
        return stats

    def reset(self, seed: Optional[int] = None):
        """Reset the task generator for a new experiment run."""
        if seed is not None:
            self.seed = int(seed)
            self.rng = random.Random(seed)
        self.task_counter = 0
        self.next_arrival_time = 0.0
        self.tasks_generated = []
        self._manifest_index = 0
        if self._manifest_entries is not None:
            self.next_arrival_time = (
                self._manifest_entries[0]["arrival_time"]
                if self._manifest_entries else math.inf)
        elif self.initial_task_immediately:
            self.next_arrival_time = 0.0
        else:
            self._schedule_next_arrival(0.0)
