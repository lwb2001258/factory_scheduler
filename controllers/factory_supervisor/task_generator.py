"""
Task Generation Module for Smart Factory Simulation.
Generates transport tasks using a Poisson process with configurable arrival rate.
Each task specifies a pickup location and delivery location within the factory.
"""

import random
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from config import (
    ALL_LOCATIONS, WORKSTATIONS, STORAGE_AREAS,
    PRIORITY_MAX_COMPLETION_SECONDS, TaskStatus,
    validate_priority_max_completion_seconds,
)


TASK_GENERATION_PARAMETER_FIELDS = (
    "task_id",
    "pickup_location",
    "delivery_location",
    "pickup_position",
    "delivery_position",
    "arrival_time",
    "priority",
    "max_completion_time_seconds",
    "deadline_time",
)


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
        task_id = int(entry["task_id"])
        if task_id != expected_task_id:
            raise ValueError(
                "task manifest IDs must be contiguous and ordered from 1")
        expected_task_id += 1
        priority_number = float(entry["priority"])
        if (not math.isfinite(priority_number) or
                not priority_number.is_integer()):
            raise ValueError("task manifest priority must be 1, 2, or 3")
        priority = int(priority_number)
        arrival_time = float(entry["arrival_time"])
        if not math.isfinite(arrival_time) or arrival_time < 0:
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
            canonical_deadline = float(deadline)
            if (not math.isfinite(canonical_deadline) or
                    not math.isclose(
                        canonical_deadline, arrival_time + limit,
                        rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError(
                    "task manifest deadline must equal arrival plus limit")
        pickup_position = [float(value)
                           for value in entry["pickup_position"]]
        delivery_position = [float(value)
                             for value in entry["delivery_position"]]
        if len(pickup_position) != 2 or len(delivery_position) != 2:
            raise ValueError(
                "task manifest positions must contain two coordinates")
        if not all(math.isfinite(value) for value in
                   pickup_position + delivery_position):
            raise ValueError("task manifest positions must be finite")
        canonical.append({
            "task_id": task_id,
            "pickup_location": str(entry["pickup_location"]),
            "delivery_location": str(entry["delivery_location"]),
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
    payload = json.dumps(
        canonical_task_manifest(entries), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    learning_trace: Optional[dict] = field(
        default=None, repr=False, compare=False)

    def __post_init__(self):
        """Attach and validate scheduler-independent SLA metadata."""
        numeric_priority = float(self.priority)
        if (not math.isfinite(numeric_priority) or
                not numeric_priority.is_integer()):
            raise ValueError("task priority must be 1, 2, or 3")
        priority = int(numeric_priority)
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

    def generation_parameters(self) -> dict:
        """Return scheduler-independent parameters used for paired runs."""
        return {
            "task_id": int(self.task_id),
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
                 priority_max_completion_seconds=None):
        """
        Args:
            mean_interval: Mean time between task arrivals in seconds (lambda^-1).
            seed: Random seed for reproducibility.
            priority_max_completion_seconds: Optional per-priority override.
        """
        if not math.isfinite(mean_interval) or mean_interval <= 0:
            raise ValueError("mean_interval must be finite and greater than zero")
        self.mean_interval = mean_interval
        self.rng = random.Random(seed)
        configured_deadlines = dict(PRIORITY_MAX_COMPLETION_SECONDS)
        if priority_max_completion_seconds is not None:
            overrides = dict(priority_max_completion_seconds)
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
        if initial_task_immediately:
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
            self.rng = random.Random(seed)
        self.task_counter = 0
        self.next_arrival_time = 0.0
        self.tasks_generated = []
        self._schedule_next_arrival(0.0)
