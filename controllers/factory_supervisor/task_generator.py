"""
Task Generation Module for Smart Factory Simulation.
Generates transport tasks using a Poisson process with configurable arrival rate.
Each task specifies a pickup location and delivery location within the factory.
"""

import random
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from config import (
    ALL_LOCATIONS, WORKSTATIONS, STORAGE_AREAS,
    TaskStatus, SCENARIOS
)


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
                 initial_task_immediately: bool = False):
        """
        Args:
            mean_interval: Mean time between task arrivals in seconds (lambda^-1).
            seed: Random seed for reproducibility.
        """
        self.mean_interval = mean_interval
        self.rng = random.Random(seed)
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
            
            self.task_counter += 1
            task = TransportTask(
                task_id=self.task_counter,
                pickup_location=pickup_loc,
                delivery_location=delivery_loc,
                pickup_position=ALL_LOCATIONS[pickup_loc],
                delivery_position=ALL_LOCATIONS[delivery_loc],
                arrival_time=current_time,
                priority=1.0 + self.rng.random() * 0.5  # slight priority variation
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
