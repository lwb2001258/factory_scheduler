"""
Task Schedulers for Multi-Robot Factory Coordination.

Implements:
1. FCFS (First-Come-First-Served) - baseline
2. Nearest-Neighbour - baseline
3. Round-Robin - baseline
4. Deterministic Greedy and seeded Random baselines
5. Hungarian exact linear assignment
6. Centralized epsilon-Auction
7. Time-bounded Genetic Algorithm
8. Time-bounded Simulated Annealing
9. PPO, DQN and SARSA RL schedulers - require validated checkpoints

All schedulers implement the same interface for fair comparison.
"""

import math
import os
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from task_generator import TransportTask
from config import (
    TaskStatus, RobotState, RL_CONFIG,
    REWARD_TASK_COMPLETE, REWARD_IDLE_PENALTY,
    REWARD_CONGESTION_PENALTY, REWARD_DISTANCE_PENALTY,
    REWARD_BALANCE_BONUS, ALL_LOCATIONS,
    MAX_ROBOTS, GRID_WIDTH, GRID_HEIGHT, MIN_TASK_BATTERY,
    RL_ENVIRONMENT_VERSION,
)


class ModelValidationError(RuntimeError):
    """Raised when an RL checkpoint is absent or incompatible."""


@dataclass(frozen=True)
class SchedulingContext:
    """Algorithm-independent snapshot used to validate a proposed assignment."""

    current_time: float = 0.0
    congestion_map: Optional[List[List[float]]] = None
    path_cost_provider: Optional[Callable[[int, TransportTask], float]] = None
    failed_pairs: frozenset = frozenset()
    map_context: Any = None
    traffic_context: Any = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    random_generator: Any = None


@dataclass(frozen=True)
class Assignment:
    robot_id: int
    task: TransportTask
    estimated_cost: Optional[float] = None


@dataclass
class SchedulerResult:
    assignments: List[Assignment] = field(default_factory=list)
    objective_value: Optional[float] = None
    computation_time: float = 0.0
    is_feasible: bool = False
    algorithm_name: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostMatrix:
    robot_ids: Tuple[int, ...]
    tasks: Tuple[TransportTask, ...]
    values: np.ndarray
    feasible: np.ndarray


def _greedy_permutation_seed(matrix: CostMatrix) -> Tuple[Tuple[int, ...],
                                                           Tuple[int, ...]]:
    """Build a deterministic maximum-cardinality, cost-biased seed."""
    row_count, column_count = len(matrix.robot_ids), len(matrix.tasks)
    column_to_row = {}

    def augment(row: int, seen: set) -> bool:
        columns = sorted(
            (column for column in range(column_count)
             if matrix.feasible[row, column]),
            key=lambda column: (matrix.values[row, column], column))
        for column in columns:
            if column in seen:
                continue
            seen.add(column)
            previous = column_to_row.get(column)
            if previous is None or augment(previous, seen):
                column_to_row[column] = row
                return True
        return False

    rows = sorted(range(row_count), key=lambda row: (
        int(np.count_nonzero(matrix.feasible[row])), row))
    for row in rows:
        augment(row, set())
    chosen = [(row, column) for column, row in column_to_row.items()]
    chosen.sort(key=lambda pair: (matrix.values[pair[0], pair[1]], pair))
    unused_rows = set(range(row_count)) - {row for row, _ in chosen}
    unused_columns = set(range(column_count)) - {
        column for _, column in chosen}
    robot_order = tuple([row for row, _ in chosen]
                        + sorted(unused_rows))
    task_order = tuple([column for _, column in chosen]
                       + sorted(unused_columns))
    return robot_order, task_order


def _metaheuristic_score(matrix: CostMatrix, individual,
                         length: int) -> Tuple[int, float]:
    """Lexicographic objective: maximise cardinality, then minimise cost."""
    robot_order, task_order = individual
    feasible_costs = [
        float(matrix.values[robot_order[index], task_order[index]])
        for index in range(length)
        if matrix.feasible[robot_order[index], task_order[index]]]
    return -len(feasible_costs), float(sum(feasible_costs))


def _feasible_pairs(matrix: CostMatrix, individual,
                    length: int) -> List[Tuple[int, int]]:
    robot_order, task_order = individual
    return [(robot_order[index], task_order[index])
            for index in range(length)
            if matrix.feasible[robot_order[index], task_order[index]]]


def _order_crossover(first: Tuple[int, ...], second: Tuple[int, ...],
                     rng: random.Random) -> Tuple[int, ...]:
    """Order crossover preserving every element of a permutation once."""
    size = len(first)
    if size < 2:
        return first
    left, right = sorted(rng.sample(range(size), 2))
    child = [None] * size
    child[left:right + 1] = first[left:right + 1]
    remaining = [value for value in second if value not in child]
    cursor = 0
    for index in list(range(right + 1, size)) + list(range(0, left)):
        child[index] = remaining[cursor]
        cursor += 1
    return tuple(child)


def validate_assignment(assignment: Optional[Assignment],
                        pending_tasks: List[TransportTask],
                        robot_states: Dict[int, dict],
                        context: Optional[SchedulingContext] = None
                        ) -> Tuple[bool, str]:
    """Validate one mutually-exclusive robot/task assignment without side effects."""
    if assignment is None:
        return False, "empty_assignment"
    if assignment.robot_id not in robot_states:
        return False, "unknown_robot"
    task_by_id = {task.task_id: task for task in pending_tasks}
    if len(task_by_id) != len(pending_tasks):
        return False, "duplicate_task_id"
    if assignment.task.task_id not in task_by_id:
        return False, "unknown_task"
    canonical_task = task_by_id[assignment.task.task_id]
    if canonical_task is not assignment.task:
        return False, "noncanonical_task_object"
    if assignment.task.status != TaskStatus.PENDING:
        return False, "task_not_pending"
    robot = robot_states[assignment.robot_id]
    if robot.get("state") != RobotState.IDLE:
        return False, "robot_not_idle"
    if robot.get("current_task") is not None:
        return False, "robot_has_task"
    if robot.get("faulted", False) or robot.get("failed", False):
        return False, "robot_faulted"
    if robot.get("battery", 100.0) <= MIN_TASK_BATTERY:
        return False, "battery_too_low"
    context = context or SchedulingContext()
    if (assignment.robot_id, assignment.task.task_id) in context.failed_pairs:
        return False, "pair_temporarily_blocked"
    if assignment.estimated_cost is not None and not math.isfinite(assignment.estimated_cost):
        return False, "nonfinite_cost"
    if context.path_cost_provider is not None:
        try:
            cost = float(context.path_cost_provider(assignment.robot_id, assignment.task))
        except Exception as exc:
            return False, f"path_cost_error:{type(exc).__name__}"
        if not math.isfinite(cost) or cost < 0:
            return False, "unreachable"
    return True, "ok"


def validate_assignments(assignments: List[Assignment],
                         pending_tasks: List[TransportTask],
                         robot_states: Dict[int, dict],
                         context: Optional[SchedulingContext] = None
                         ) -> Tuple[bool, str]:
    """Validate a complete matching, including cross-assignment uniqueness."""
    if not assignments:
        return False, "empty_assignments"
    robot_ids = [item.robot_id for item in assignments]
    task_ids = [item.task.task_id for item in assignments]
    if len(robot_ids) != len(set(robot_ids)):
        return False, "duplicate_robot_assignment"
    if len(task_ids) != len(set(task_ids)):
        return False, "duplicate_task_assignment"
    for assignment in assignments:
        valid, reason = validate_assignment(
            assignment, pending_tasks, robot_states, context)
        if not valid:
            return False, reason
    return True, "ok"


# ================================================================
# ABSTRACT BASE SCHEDULER
# ================================================================

class BaseScheduler(ABC):
    """Abstract base class for all task schedulers."""
    
    def __init__(self, name: str):
        self.name = name
        self.assignments_made = 0
    
    @abstractmethod
    def assign_task(self, 
                    pending_tasks: List[TransportTask],
                    robot_states: Dict[int, dict],
                    congestion_map: Optional[List[List[float]]] = None
                    ) -> Optional[Tuple[int, TransportTask]]:
        """
        Select a task and assign it to a robot.
        
        Args:
            pending_tasks: List of unassigned tasks.
            robot_states: Dict of robot_id -> {
                'position': (x, z),
                'state': RobotState,
                'battery': float,
                'current_task': Optional[TransportTask],
            }
            congestion_map: Optional congestion grid.
            
        Returns:
            Tuple of (robot_id, task) or None if no assignment possible.
        """
        pass
    
    def get_idle_robots(self, robot_states: Dict[int, dict]) -> List[int]:
        """Get list of robot IDs that are idle and available for tasks.

        A robot is "available" iff:
          - state is IDLE  (NOT charging, NOT returning to charge, etc.)
          - battery > MIN_TASK_BATTERY (15%)  — prevent assigning a task
            the robot can't physically finish.
        """
        return [rid for rid, state in robot_states.items()
                if state['state'] == RobotState.IDLE
                and state.get('battery', 100) > MIN_TASK_BATTERY]
    
    def reset(self):
        """Reset scheduler state for new experiment."""
        self.assignments_made = 0

    def assign(self, pending_tasks: List[TransportTask],
               robot_states: Dict[int, dict],
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        """Structured, validated adapter around the legacy assign_task interface."""
        context = context or SchedulingContext()
        started = time.perf_counter()
        try:
            pair = self.assign_task(pending_tasks, robot_states,
                                    context.congestion_map)
            assignment = Assignment(pair[0], pair[1]) if pair else None
            feasible, reason = validate_assignment(
                assignment, pending_tasks, robot_states, context)
        except Exception as exc:
            assignment = None
            feasible = False
            reason = f"scheduler_exception:{type(exc).__name__}"
        elapsed = time.perf_counter() - started
        return SchedulerResult(
            assignments=[assignment] if feasible and assignment else [],
            objective_value=assignment.estimated_cost if assignment else None,
            computation_time=elapsed,
            is_feasible=feasible,
            algorithm_name=self.name,
            diagnostics={"reason": reason},
        )

    def on_assignment_committed(self, assignment: Assignment) -> None:
        """Record side effects only after Supervisor commits a feasible path."""
        self.assignments_made += 1

    def on_assignment_rejected(self, assignment: Optional[Assignment],
                               reason: str) -> None:
        """Hook for algorithm diagnostics; intentionally side-effect free by default."""
        return None


# ================================================================
# FIRST-COME-FIRST-SERVED (FCFS) SCHEDULER
# ================================================================

class FCFSScheduler(BaseScheduler):
    """
    First-Come-First-Served scheduler.
    Assigns the oldest pending task to the first available idle robot.
    Simple, fast, but ignores robot locations and workload balance.
    """
    
    def __init__(self):
        super().__init__("FCFS")
    
    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        if not pending_tasks:
            return None
        
        idle_robots = self.get_idle_robots(robot_states)
        if not idle_robots:
            return None
        
        # Sort tasks by arrival time (oldest first)
        sorted_tasks = sorted(pending_tasks, key=lambda t: t.arrival_time)
        
        # Assign the oldest task to the first idle robot
        task = sorted_tasks[0]
        robot_id = idle_robots[0]
        
        return (robot_id, task)


# ================================================================
# NEAREST-NEIGHBOUR SCHEDULER
# ================================================================

class NearestNeighbourScheduler(BaseScheduler):
    """
    Nearest-Neighbour scheduler.
    For each pending task, assigns it to the closest idle robot.
    Reduces empty travel distance but may lead to unbalanced workloads.
    """
    
    def __init__(self):
        super().__init__("NearestNeighbour")
    
    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        if not pending_tasks:
            return None
        
        idle_robots = self.get_idle_robots(robot_states)
        if not idle_robots:
            return None
        
        # Find the best (task, robot) pair by minimum distance
        best_pair = None
        best_distance = float('inf')
        
        for task in pending_tasks:
            pickup_pos = task.pickup_position
            for rid in idle_robots:
                robot_pos = robot_states[rid]['position']
                dist = math.sqrt(
                    (robot_pos[0] - pickup_pos[0])**2 + 
                    (robot_pos[1] - pickup_pos[1])**2
                )
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (rid, task)
        
        return best_pair

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        """Select the cheapest non-blocked pair using the shared cost contract."""
        context = context or SchedulingContext()
        started = time.perf_counter()
        try:
            matrix = build_cost_matrix(pending_tasks, robot_states, context)
        except ValueError as exc:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": str(exc)})
        best = None
        diagnostics = {
            "reason": "no_feasible_pair",
            "cost_source": ("path" if context.path_cost_provider else "euclidean"),
        }
        for row, robot_id in enumerate(matrix.robot_ids):
            for column, task in enumerate(matrix.tasks):
                if not matrix.feasible[row, column]:
                    continue
                cost = float(matrix.values[row, column])
                candidate = Assignment(robot_id, task, cost)
                feasible, reason = validate_assignment(
                    candidate, pending_tasks, robot_states, context)
                if feasible and (best is None or cost < best.estimated_cost):
                    best = candidate
                elif not feasible:
                    diagnostics["last_rejection"] = reason
        elapsed = time.perf_counter() - started
        if best is None:
            return SchedulerResult(
                computation_time=elapsed, is_feasible=False,
                algorithm_name=self.name, diagnostics=diagnostics)
        return SchedulerResult(
            assignments=[best], objective_value=best.estimated_cost,
            computation_time=elapsed, is_feasible=True,
            algorithm_name=self.name,
            diagnostics={
                "reason": "ok",
                "cost_source": (
                    "path" if context.path_cost_provider else "euclidean"),
            },
        )


# ================================================================
# ROUND-ROBIN SCHEDULER
# ================================================================

class RoundRobinScheduler(BaseScheduler):
    """
    Round-Robin scheduler.
    Distributes tasks cyclically among robots to balance workload.
    Ignores spatial information and congestion.
    """
    
    def __init__(self):
        super().__init__("RoundRobin")
        self.last_assigned_index = -1
    
    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        if not pending_tasks:
            return None
        
        idle_robots = self.get_idle_robots(robot_states)
        if not idle_robots:
            return None
        
        # Sort idle robots by ID for consistent ordering
        idle_robots.sort()
        
        # Find next robot in round-robin order
        # Start from the robot after the last assigned
        robot_id = None
        all_robot_ids = sorted(robot_states.keys())
        
        for i in range(len(all_robot_ids)):
            idx = (self.last_assigned_index + 1 + i) % len(all_robot_ids)
            candidate = all_robot_ids[idx]
            if candidate in idle_robots:
                robot_id = candidate
                self.last_assigned_index = idx
                break
        
        if robot_id is None:
            return None
        
        # Assign the oldest pending task
        sorted_tasks = sorted(pending_tasks, key=lambda t: t.arrival_time)
        task = sorted_tasks[0]
        
        return (robot_id, task)
    
    def reset(self):
        super().reset()
        self.last_assigned_index = -1


def _pair_cost(robot_id: int, task: TransportTask,
               robot_states: Dict[int, dict],
               context: SchedulingContext) -> float:
    if context.path_cost_provider is not None:
        try:
            travel = float(context.path_cost_provider(robot_id, task))
        except Exception:
            return float("inf")
        if not math.isfinite(travel) or travel < 0:
            return float("inf")
    else:
        robot_pos = robot_states[robot_id]["position"]
        pickup = task.pickup_position
        delivery = task.delivery_position
        empty = math.hypot(robot_pos[0] - pickup[0], robot_pos[1] - pickup[1])
        loaded = math.hypot(pickup[0] - delivery[0], pickup[1] - delivery[1])
        travel = empty + loaded
    weights = {
        "travel": 1.0,
        "priority": 1.0,
        "waiting": 0.01,
        "congestion": 0.05,
    }
    weights.update(context.configuration.get("cost_weights", {}))
    waiting = max(0.0, context.current_time - float(task.arrival_time))
    congestion = 0.0
    grid = context.congestion_map
    if grid and grid[0]:
        height, width = len(grid), len(grid[0])
        resolution = float(context.configuration.get(
            "congestion_grid_resolution", 1.0))
        half_w, half_h = width * resolution / 2.0, height * resolution / 2.0
        samples = []
        for x, y in (task.pickup_position, task.delivery_position):
            column = int((x + half_w) / resolution)
            row = int((y + half_h) / resolution)
            if 0 <= row < height and 0 <= column < width:
                samples.append(float(grid[row][column]))
        congestion = sum(samples) / len(samples) if samples else 0.0
    cost = (
        weights["travel"] * travel
        - weights["priority"] * max(0.0, float(task.priority) - 1.0)
        - weights["waiting"] * waiting
        + weights["congestion"] * congestion
    )
    return max(0.0, cost)


def build_cost_matrix(pending_tasks: List[TransportTask],
                      robot_states: Dict[int, dict],
                      context: Optional[SchedulingContext] = None
                      ) -> CostMatrix:
    """Build the single shared robot-row/task-column cost matrix."""
    context = context or SchedulingContext()
    task_ids = [task.task_id for task in pending_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs are not allowed")
    robots = tuple(sorted(
        rid for rid, state in robot_states.items()
        if state.get("state") == RobotState.IDLE
        and state.get("current_task") is None
        and state.get("battery", 100.0) > MIN_TASK_BATTERY
        and not state.get("faulted", False)
        and not state.get("failed", False)))
    tasks = tuple(sorted(
        (task for task in pending_tasks if task.status == TaskStatus.PENDING),
        key=lambda task: task.task_id))
    values = np.full((len(robots), len(tasks)), np.inf, dtype=float)
    feasible = np.zeros_like(values, dtype=bool)
    for row, robot_id in enumerate(robots):
        for column, task in enumerate(tasks):
            if (robot_id, task.task_id) in context.failed_pairs:
                continue
            cost = _pair_cost(robot_id, task, robot_states, context)
            if math.isfinite(cost) and cost >= 0:
                values[row, column] = cost
                feasible[row, column] = True
    return CostMatrix(robots, tasks, values, feasible)


def _result_from_matching(name: str, matrix: CostMatrix,
                          pairs: List[Tuple[int, int]],
                          pending_tasks: List[TransportTask],
                          robot_states: Dict[int, dict],
                          context: SchedulingContext,
                          started: float,
                          diagnostics: Optional[Dict[str, Any]] = None
                          ) -> SchedulerResult:
    assignments = []
    for row, column in pairs:
        if (row < 0 or column < 0 or
                row >= len(matrix.robot_ids) or column >= len(matrix.tasks) or
                not matrix.feasible[row, column]):
            continue
        candidate = Assignment(
            matrix.robot_ids[row], matrix.tasks[column],
            float(matrix.values[row, column]))
        valid, _ = validate_assignment(
            candidate, pending_tasks, robot_states, context)
        if valid:
            assignments.append(candidate)
    matching_valid, matching_reason = validate_assignments(
        assignments, pending_tasks, robot_states, context)
    if not matching_valid:
        assignments = []
    objective = (sum(item.estimated_cost for item in assignments)
                 if assignments else None)
    info = dict(diagnostics or {})
    info.setdefault(
        "reason", "ok" if assignments else matching_reason)
    info["matching_size"] = len(assignments)
    return SchedulerResult(
        assignments=assignments,
        objective_value=objective,
        computation_time=time.perf_counter() - started,
        is_feasible=bool(assignments),
        algorithm_name=name,
        diagnostics=info,
    )


class GreedyScheduler(BaseScheduler):
    """Deterministic minimum-cost matching baseline."""

    def __init__(self):
        super().__init__("Greedy")

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        item = result.assignments[0]
        return item.robot_id, item.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        try:
            matrix = build_cost_matrix(pending_tasks, robot_states, context)
        except ValueError as exc:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": str(exc)})
        candidates = [
            (float(matrix.values[row, column]),
             -float(matrix.tasks[column].priority),
             float(matrix.tasks[column].arrival_time),
             matrix.robot_ids[row], matrix.tasks[column].task_id,
             row, column)
            for row in range(len(matrix.robot_ids))
            for column in range(len(matrix.tasks))
            if matrix.feasible[row, column]
        ]
        used_rows, used_columns, pairs = set(), set(), []
        for *_tie, row, column in sorted(candidates):
            if row not in used_rows and column not in used_columns:
                pairs.append((row, column))
                used_rows.add(row)
                used_columns.add(column)
        return _result_from_matching(
            self.name, matrix, pairs, pending_tasks, robot_states,
            context, started, {"tie_break": "cost,priority,arrival,robot,task"})


class RandomScheduler(BaseScheduler):
    """Seeded random legal matching; experiment-only baseline."""

    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = random.Random(seed)

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        item = result.assignments[0]
        return item.robot_id, item.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        matrix = build_cost_matrix(pending_tasks, robot_states, context)
        candidates = [
            (row, column)
            for row in range(len(matrix.robot_ids))
            for column in range(len(matrix.tasks))
            if matrix.feasible[row, column]
        ]
        self.rng.shuffle(candidates)
        used_rows, used_columns, pairs = set(), set(), []
        for row, column in candidates:
            if row not in used_rows and column not in used_columns:
                pairs.append((row, column))
                used_rows.add(row)
                used_columns.add(column)
        return _result_from_matching(
            self.name, matrix, pairs, pending_tasks, robot_states,
            context, started, {"baseline_only": True})


class HungarianScheduler(BaseScheduler):
    """Exact rectangular linear assignment over the shared cost matrix."""

    def __init__(self):
        super().__init__("Hungarian")

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        item = result.assignments[0]
        return item.robot_id, item.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        from scipy.optimize import linear_sum_assignment
        context = context or SchedulingContext()
        started = time.perf_counter()
        matrix = build_cost_matrix(pending_tasks, robot_states, context)
        if matrix.values.size == 0 or not np.any(matrix.feasible):
            return _result_from_matching(
                self.name, matrix, [], pending_tasks, robot_states,
                context, started)
        finite = matrix.values[matrix.feasible]
        penalty = max(1.0, float(np.max(finite))) * (
            max(matrix.values.shape, default=1) + 1) * 1_000.0
        solver_values = np.where(matrix.feasible, matrix.values, penalty)
        # Stable infinitesimal tie break without changing meaningful costs.
        rows, columns = solver_values.shape
        tie = (np.arange(rows)[:, None] * max(columns, 1) +
               np.arange(columns)[None, :]) * np.finfo(float).eps
        row_idx, column_idx = linear_sum_assignment(solver_values + tie)
        pairs = [
            (int(row), int(column))
            for row, column in zip(row_idx, column_idx)
            if matrix.feasible[row, column]
        ]
        return _result_from_matching(
            self.name, matrix, pairs, pending_tasks, robot_states,
            context, started, {"solver": "scipy.linear_sum_assignment",
                               "dummy_used": False})


class AuctionScheduler(BaseScheduler):
    """Centralized epsilon-auction with iteration and wall-clock limits."""

    def __init__(self, epsilon: float = 1e-3,
                 max_iterations: int = 10_000,
                 time_budget_ms: float = 5.0):
        super().__init__("Auction")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and > 0")
        self.epsilon = epsilon
        self.max_iterations = max(1, max_iterations)
        self.time_budget_ms = max(0.1, time_budget_ms)

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        item = result.assignments[0]
        return item.robot_id, item.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        deadline = started + self.time_budget_ms / 1000.0
        matrix = build_cost_matrix(pending_tasks, robot_states, context)
        if matrix.values.size == 0 or not np.any(matrix.feasible):
            return _result_from_matching(
                self.name, matrix, [], pending_tasks, robot_states,
                context, started)

        # Auction requires bidders <= objects. Transpose when robots > tasks.
        transposed = len(matrix.robot_ids) > len(matrix.tasks)
        costs = matrix.values.T if transposed else matrix.values
        feasible = matrix.feasible.T if transposed else matrix.feasible
        bidder_count, object_count = costs.shape
        prices = np.zeros(object_count, dtype=float)
        owner = np.full(object_count, -1, dtype=int)
        bidder_object = np.full(bidder_count, -1, dtype=int)
        queue = [
            bidder for bidder in range(bidder_count)
            if np.any(feasible[bidder])
        ]
        iterations = 0
        while (queue and iterations < self.max_iterations and
               time.perf_counter() < deadline):
            bidder = queue.pop(0)
            valid_objects = np.flatnonzero(feasible[bidder])
            if valid_objects.size == 0:
                iterations += 1
                continue
            utilities = -costs[bidder, valid_objects] - prices[valid_objects]
            order = np.argsort(-utilities, kind="stable")
            best_object = int(valid_objects[order[0]])
            best_value = float(utilities[order[0]])
            second_value = (float(utilities[order[1]])
                            if len(order) > 1
                            else best_value - self.epsilon)
            increment = max(self.epsilon, best_value - second_value + self.epsilon)
            prices[best_object] += increment
            previous = int(owner[best_object])
            owner[best_object] = bidder
            bidder_object[bidder] = best_object
            if previous >= 0 and previous != bidder:
                bidder_object[previous] = -1
                queue.append(previous)
            iterations += 1
        pairs = []
        for bidder, obj in enumerate(bidder_object):
            if obj < 0:
                continue
            pairs.append((int(obj), bidder) if transposed
                         else (bidder, int(obj)))
        stop_reason = (
            "complete" if not queue else
            "time_budget" if time.perf_counter() >= deadline
            else "max_iterations")
        return _result_from_matching(
            self.name, matrix, pairs, pending_tasks, robot_states,
            context, started, {
                "epsilon": self.epsilon,
                "iterations": iterations,
                "stop_reason": stop_reason,
                "max_price": float(np.max(prices, initial=0.0)),
                "centralized": True,
            })


class GeneticScheduler(BaseScheduler):
    """Time-bounded GA for a one-to-one robot/task matching permutation."""

    def __init__(self, seed: int = 42, population_size: int = 32,
                 max_generations: int = 50, time_budget_ms: float = 5.0):
        super().__init__("GA")
        self.rng = random.Random(seed)
        self.population_size = max(2, population_size)
        self.max_generations = max(1, max_generations)
        self.time_budget_ms = max(0.1, time_budget_ms)

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        chosen = result.assignments[0]
        return chosen.robot_id, chosen.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        matrix = build_cost_matrix(pending_tasks, robot_states, context)
        matrix_seconds = time.perf_counter() - started
        search_started = time.perf_counter()
        deadline = search_started + self.time_budget_ms / 1000.0
        rows, columns = len(matrix.robot_ids), len(matrix.tasks)
        if not rows or not columns:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": "no_candidates"})
        chromosome_length = min(rows, columns)

        def random_individual():
            robot_order = list(range(rows))
            task_order = list(range(columns))
            self.rng.shuffle(robot_order)
            self.rng.shuffle(task_order)
            return tuple(robot_order), tuple(task_order)

        def mutate(individual):
            robot_order, task_order = map(list, individual)
            target = robot_order if self.rng.random() < 0.5 else task_order
            if len(target) > 1:
                left, right = self.rng.sample(range(len(target)), 2)
                target[left], target[right] = target[right], target[left]
            return tuple(robot_order), tuple(task_order)

        population = [_greedy_permutation_seed(matrix)]
        while (len(population) < self.population_size and
               time.perf_counter() < deadline):
            candidate = random_individual()
            if candidate not in population:
                population.append(candidate)
            elif (math.factorial(min(rows, 8))
                  * math.factorial(min(columns, 8)) <= len(population)):
                break
        best = min(population, key=lambda item: _metaheuristic_score(
            matrix, item, chromosome_length))
        best_score = _metaheuristic_score(matrix, best, chromosome_length)
        generations = 0
        while generations < self.max_generations and time.perf_counter() < deadline:
            ranked = sorted(population, key=lambda item: _metaheuristic_score(
                matrix, item, chromosome_length))
            candidate_score = _metaheuristic_score(
                matrix, ranked[0], chromosome_length)
            if candidate_score < best_score:
                best, best_score = ranked[0], candidate_score
            elites = ranked[:max(1, len(ranked) // 4)]
            next_population = list(elites)
            while len(next_population) < self.population_size:
                first = self.rng.choice(elites)
                second = self.rng.choice(elites)
                child = (
                    _order_crossover(first[0], second[0], self.rng),
                    _order_crossover(first[1], second[1], self.rng),
                )
                next_population.append(mutate(child))
                if time.perf_counter() >= deadline:
                    break
            population = next_population
            generations += 1
        pairs = _feasible_pairs(matrix, best, chromosome_length)
        if not pairs:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": "no_feasible_pair",
                             "generations": generations})
        # Supervisor commits one result then rebuilds the snapshot. Put the
        # cheapest member of the optimized matching first so that incremental
        # commit preserves the metaheuristic's useful choice.
        pairs.sort(key=lambda pair: matrix.values[pair[0], pair[1]])
        return _result_from_matching(
            self.name, matrix, pairs, pending_tasks, robot_states,
            context, started, {
                "generations": generations,
                "population_size": len(population),
                "matching_cardinality": len(pairs),
                "matrix_time_ms": matrix_seconds * 1000.0,
                "search_time_ms": (
                    time.perf_counter() - search_started) * 1000.0,
                "stop_reason": ("time_budget" if time.perf_counter() >= deadline
                                else "max_generations"),
            })


class SimulatedAnnealingScheduler(BaseScheduler):
    """Time-bounded SA over the same one-to-one matching permutation."""

    def __init__(self, seed: int = 42, initial_temperature: float = 10.0,
                 cooling_rate: float = 0.95, minimum_temperature: float = 0.01,
                 max_iterations: int = 1000, time_budget_ms: float = 5.0):
        super().__init__("SA")
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be > 0")
        if not 0 < cooling_rate < 1:
            raise ValueError("cooling_rate must be in (0, 1)")
        self.rng = random.Random(seed)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.minimum_temperature = max(0.0, minimum_temperature)
        self.max_iterations = max(1, max_iterations)
        self.time_budget_ms = max(0.1, time_budget_ms)

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        chosen = result.assignments[0]
        return chosen.robot_id, chosen.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        matrix = build_cost_matrix(pending_tasks, robot_states, context)
        matrix_seconds = time.perf_counter() - started
        search_started = time.perf_counter()
        deadline = search_started + self.time_budget_ms / 1000.0
        rows, columns = len(matrix.robot_ids), len(matrix.tasks)
        if not rows or not columns:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": "no_candidates"})
        length = min(rows, columns)

        finite_values = matrix.values[matrix.feasible]
        cardinality_penalty = (
            float(np.max(finite_values)) * (length + 1) + 1.0
            if finite_values.size else 1e6)

        def score(individual):
            return _metaheuristic_score(matrix, individual, length)

        def energy(individual):
            negative_cardinality, total = score(individual)
            missing = length + negative_cardinality
            return missing * cardinality_penalty + total

        current = _greedy_permutation_seed(matrix)
        current_energy = energy(current)
        best, best_score = current, score(current)
        temperature = self.initial_temperature
        accepted_worse = 0
        rejected = 0
        iterations = 0
        while (iterations < self.max_iterations and
               temperature > self.minimum_temperature and
               time.perf_counter() < deadline):
            robot_order, task_order = map(list, current)
            target = robot_order if self.rng.random() < 0.5 else task_order
            if len(target) > 1:
                left, right = self.rng.sample(range(len(target)), 2)
                target[left], target[right] = target[right], target[left]
            neighbor = (tuple(robot_order), tuple(task_order))
            neighbor_energy = energy(neighbor)
            delta = neighbor_energy - current_energy
            accept = delta <= 0
            if not accept and math.isfinite(delta):
                accept = self.rng.random() < math.exp(
                    -min(delta / max(temperature, 1e-12), 700.0))
            if accept:
                if delta > 0:
                    accepted_worse += 1
                current, current_energy = neighbor, neighbor_energy
                current_score = score(current)
                if current_score < best_score:
                    best, best_score = current, current_score
            else:
                rejected += 1
            temperature *= self.cooling_rate
            iterations += 1
        pairs = _feasible_pairs(matrix, best, length)
        if not pairs:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": "no_feasible_pair"})
        pairs.sort(key=lambda pair: matrix.values[pair[0], pair[1]])
        return _result_from_matching(
            self.name, matrix, pairs, pending_tasks, robot_states,
            context, started, {
                "iterations": iterations,
                "final_temperature": temperature,
                "accepted_worse": accepted_worse, "rejected": rejected,
                "matching_cardinality": len(pairs),
                "matrix_time_ms": matrix_seconds * 1000.0,
                "search_time_ms": (
                    time.perf_counter() - search_started) * 1000.0,
                "stop_reason": ("time_budget" if time.perf_counter() >= deadline
                                else "temperature_or_iterations"),
            })


# ================================================================
# PPO-BASED DEEP RL SCHEDULER
# ================================================================

class PPONetwork:
    """
    Neural network for PPO policy and value function.
    Implemented with NumPy for portability (no PyTorch dependency in Webots).
    Training uses the standalone NumPy PPO trainer.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Initialize weights with HALF Xavier scale.
        # Half-scale gives well-bounded random outputs in the
        # un-trained state, which we evaluate immediately at
        # supervisor init (before any training has occurred).
        scale1 = 0.5 * math.sqrt(2.0 / (state_dim + hidden_size))
        scale2 = 0.5 * math.sqrt(2.0 / (hidden_size + hidden_size))
        scale3 = 0.5 * math.sqrt(2.0 / (hidden_size + action_dim))
        
        self.rng = np.random.RandomState(42)
        
        # Policy network weights
        self.W1 = self.rng.randn(state_dim, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.W2 = self.rng.randn(hidden_size, hidden_size) * scale2
        self.b2 = np.zeros(hidden_size)
        self.W_policy = self.rng.randn(hidden_size, action_dim) * scale3
        self.b_policy = np.zeros(action_dim)
        
        # Value network weights (shares first layers)
        self.W_value = self.rng.randn(hidden_size, 1) * math.sqrt(2.0 / (hidden_size + 1))
        self.b_value = np.zeros(1)
    
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass through the network.
        
        Numerical-safety guarantees:
          - Replace any NaN/Inf in input with zero (defensive — should
            never happen in a healthy state encoder).
          - Cap activations after each ReLU to prevent runaway growth
            from cascaded multiplications with un-trained random weights.
          - Subtract per-row max before softmax (already present).
          - Final action_probs are renormalised and uniform-fallback if
            they degenerate to NaN/all-zero.
        
        Returns:
            action_probs: Probability distribution over actions (sum=1).
            value: Estimated state value (clipped to a sane range).
        """
        # Sanitise input
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Wrap matmul ops in errstate — macOS Accelerate BLAS sometimes
        # raises spurious divide-by-zero warnings during sparse matmul
        # (result is correct but the kernel touches NaN intermediates).
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            h1 = np.maximum(0, state @ self.W1 + self.b1)
            h1 = np.clip(h1, 0.0, 50.0)
            h2 = np.maximum(0, h1 @ self.W2 + self.b2)
            h2 = np.clip(h2, 0.0, 50.0)
            logits = h2 @ self.W_policy + self.b_policy
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            value_raw_inner = (h2 @ self.W_value + self.b_value)[0]
        
        # ↓↓↓ DUPLICATE LINE TO PRESERVE STRUCTURE — the original code
        # below assumes logits is fresh; we keep its logic identical.
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = np.clip(logits, -30.0, 30.0)        # avoid exp() overflow
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        denom = np.sum(exp_logits)
        if denom < 1e-8 or not np.isfinite(denom):
            # Degenerate — fall back to uniform distribution
            action_probs = np.ones_like(logits) / len(logits)
        else:
            action_probs = exp_logits / denom
        
        # Re-validate (paranoid): replace any remaining NaN with uniform
        if not np.all(np.isfinite(action_probs)) or np.sum(action_probs) <= 0:
            action_probs = np.ones_like(logits) / len(logits)
        
        # Value head — clip to a sane range (computed inside errstate above)
        value = float(np.clip(np.nan_to_num(value_raw_inner, nan=0.0,
                                            posinf=100.0, neginf=-100.0),
                              -100.0, 100.0))
        
        return action_probs, value
    
    def save(self, filepath: str):
        """Save network weights to file."""
        action_semantics = (
            "robot_task_pair_plus_noop"
            if self.action_dim > MAX_ROBOTS else "legacy_robot_only")
        observation_schema = (
            "pairwise_v1" if self.action_dim > MAX_ROBOTS
            else "legacy_robot_state_v1")
        np.savez(filepath,
                 schema_version=np.array([3], dtype=np.int64),
                 environment_version=np.array([RL_ENVIRONMENT_VERSION]),
                 action_semantics=np.array([action_semantics]),
                 observation_schema=np.array([observation_schema]),
                 state_dim=np.array([self.state_dim], dtype=np.int64),
                 action_dim=np.array([self.action_dim], dtype=np.int64),
                 hidden_size=np.array([self.hidden_size], dtype=np.int64),
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W_policy=self.W_policy, b_policy=self.b_policy,
                 W_value=self.W_value, b_value=self.b_value)
    
    def load(self, filepath: str):
        """Load and validate versioned network weights."""
        if not filepath or not os.path.isfile(filepath):
            raise ModelValidationError(f"checkpoint not found: {filepath}")
        required = {
            "schema_version", "environment_version", "state_dim",
            "action_dim", "hidden_size",
            "W1", "b1", "W2", "b2", "W_policy", "b_policy",
            "W_value", "b_value",
        }
        with np.load(filepath, allow_pickle=False) as data:
            missing = required.difference(data.files)
            if missing:
                raise ModelValidationError(
                    f"checkpoint metadata/weights missing: {sorted(missing)}")
            schema_version = int(data["schema_version"][0])
            environment_version = str(data["environment_version"][0])
            if schema_version not in {2, 3}:
                raise ModelValidationError(
                    f"checkpoint schema {schema_version} not in supported {{2, 3}}")
            if schema_version == 3:
                semantic_fields = {"action_semantics", "observation_schema"}
                semantic_missing = semantic_fields.difference(data.files)
                if semantic_missing:
                    raise ModelValidationError(
                        "checkpoint semantic metadata missing: "
                        f"{sorted(semantic_missing)}")
                expected_action_semantics = (
                    "robot_task_pair_plus_noop"
                    if self.action_dim > MAX_ROBOTS else "legacy_robot_only")
                action_semantics = str(data["action_semantics"][0])
                if action_semantics != expected_action_semantics:
                    raise ModelValidationError(
                        "PPO action semantics mismatch: "
                        f"{action_semantics!r} != {expected_action_semantics!r}")
            if environment_version != RL_ENVIRONMENT_VERSION:
                raise ModelValidationError(
                    "PPO environment version mismatch: "
                    f"{environment_version!r} != {RL_ENVIRONMENT_VERSION!r}")
            metadata = (
                int(data["state_dim"][0]),
                int(data["action_dim"][0]),
                int(data["hidden_size"][0]),
            )
            expected = (self.state_dim, self.action_dim, self.hidden_size)
            if metadata != expected:
                raise ModelValidationError(
                    f"checkpoint dimensions {metadata} != expected {expected}")
            expected_shapes = {
                "W1": self.W1.shape, "b1": self.b1.shape,
                "W2": self.W2.shape, "b2": self.b2.shape,
                "W_policy": self.W_policy.shape, "b_policy": self.b_policy.shape,
                "W_value": self.W_value.shape, "b_value": self.b_value.shape,
            }
            loaded = {}
            for key, shape in expected_shapes.items():
                value = np.asarray(data[key], dtype=float)
                if value.shape != shape:
                    raise ModelValidationError(
                        f"{key} shape {value.shape} != expected {shape}")
                if not np.all(np.isfinite(value)):
                    raise ModelValidationError(f"{key} contains NaN/Inf")
                loaded[key] = value.copy()
        for key, value in loaded.items():
            setattr(self, key, value)


class RLScheduler(BaseScheduler):
    """
    Deep RL-based task scheduler using PPO.
    
    State space:
    - Per robot (8 robots max): x, z, heading, state_one_hot(8), battery, has_task = 13 dims
    - Task queue info: num_pending, avg_distance_to_nearest_robot = 2 dims
    - Flattened congestion map (downsampled): 5x4 = 20 dims
    Total state dim = 8 * 13 + 2 + 20 = 126
    
    Action space:
    - Discrete: (robot_id, task_index) combinations
    - Simplified to: which idle robot to assign the highest-priority task to
    - Action dim = MAX_ROBOTS (select which robot gets the next task)
    
    Reward function:
    - +10 for task completion
    - -0.01 per timestep per idle robot
    - -0.5 for congestion events
    - -0.1 per metre of empty travel
    - +1.0 for balanced workload
    """
    
    OBSERVATION_DIM = MAX_ROBOTS * 13 + 2 + 20

    def __init__(self, model_path: Optional[str] = None, *,
                 training: bool = False, seed: int = 42,
                 deterministic: bool = True):
        super().__init__("PPO_RL")
        
        self.state_dim = self.OBSERVATION_DIM
        self.action_dim = MAX_ROBOTS  # 8
        self.training = training
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        
        self.network = PPONetwork(self.state_dim, self.action_dim,
                                  RL_CONFIG['hidden_size'])
        
        if model_path:
            self.network.load(model_path)
            print(f"[RL Scheduler] Loaded validated model from {model_path}")
        elif not training:
            raise ModelValidationError(
                "PPO_RL requires a validated MODEL_PATH for inference")
        
        # Experience is collected only in training mode and committed only
        # after path feasibility has been confirmed by the caller.
        self.experience_buffer = []
        self._pending_experience: Dict[Tuple[int, int], dict] = {}
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # Task completion tracking
        self.tasks_per_robot: Dict[int, int] = {}
        self.total_distance_per_robot: Dict[int, float] = {}
    
    def encode_state(self, robot_states: Dict[int, dict],
                     pending_tasks: List[TransportTask],
                     congestion_map: Optional[List[List[float]]] = None
                     ) -> np.ndarray:
        """
        Encode the current factory state into a feature vector.
        """
        state = np.zeros(self.state_dim)
        idx = 0
        
        # Encode each robot's state — 13 dims:
        #   0      x position (normalized)
        #   1      y position (normalized)
        #   2      heading (normalized)
        #   3-10   state one-hot (8 distinct states)
        #   11     battery (normalized 0-1)
        #   12     has_task flag (0/1)
        for rid in range(1, MAX_ROBOTS + 1):
            if rid in robot_states:
                rs = robot_states[rid]
                pos = rs['position']
                state[idx] = pos[0] / 10.0      # ① x
                state[idx + 1] = pos[1] / 8.0   # ② y
                state[idx + 2] = rs.get('heading', 0) / math.pi  # ③ heading
                
                # ④-⑪ One-hot encode robot state (8 states).
                # The PPO needs to differentiate IDLE (truly idle, can
                # accept tasks) from RETURNING_HOME (post-delivery, will
                # be idle soon) and RETURNING_TO_CHARGE (low-battery,
                # do NOT assign tasks). Adding these two as their own
                # one-hot dims gives the policy net direct signal.
                state_map = {
                    RobotState.IDLE:                0,
                    RobotState.EN_ROUTE_PICKUP:     1,
                    RobotState.CARRYING:            2,
                    RobotState.EN_ROUTE_DELIVERY:   3,
                    RobotState.RETURNING_HOME:      4,  # NEW
                    RobotState.RETURNING_TO_CHARGE: 5,  # NEW
                    RobotState.CHARGING:            6,
                    RobotState.WAITING:             7,
                }
                s_idx = state_map.get(rs['state'], 0)
                state[idx + 3 + s_idx] = 1.0
                
                state[idx + 11] = rs.get('battery', 100) / 100.0  # ⑫ battery
                state[idx + 12] = 1.0 if rs.get('current_task') else 0.0  # ⑬ has_task
            idx += 13
        
        # Task queue info
        state[idx] = min(len(pending_tasks) / 10.0, 1.0)  # normalized pending count
        
        if pending_tasks and robot_states:
            # Average distance from pending tasks to nearest robot
            avg_dist = 0.0
            for task in pending_tasks:
                min_d = float('inf')
                for rid, rs in robot_states.items():
                    d = math.sqrt(
                        (rs['position'][0] - task.pickup_position[0])**2 +
                        (rs['position'][1] - task.pickup_position[1])**2
                    )
                    min_d = min(min_d, d)
                avg_dist += min_d
            avg_dist /= len(pending_tasks)
            state[idx + 1] = min(avg_dist / 20.0, 1.0)
        idx += 2
        
        # Congestion map (downsample to 5x4 = 20 values)
        if congestion_map:
            ds_h = max(1, len(congestion_map) // 4)
            ds_w = max(1, len(congestion_map[0]) // 5) if congestion_map else 1
            for gz in range(4):
                for gx in range(5):
                    src_z = min(gz * ds_h, len(congestion_map) - 1)
                    src_x = min(gx * ds_w, len(congestion_map[0]) - 1) if congestion_map[0] else 0
                    state[idx] = min(congestion_map[src_z][src_x] / 5.0, 1.0)
                    idx += 1
        
        return state
    
    def compute_reward(self, robot_states: Dict[int, dict],
                       tasks_completed_this_step: int,
                       congestion_events: int,
                       empty_travel_distance: float) -> float:
        """
        Compute the reward for the current step.
        
        Reward = task_completion_bonus 
               + idle_penalty 
               + congestion_penalty 
               + distance_penalty 
               + balance_bonus
        """
        reward = 0.0
        
        # Task completion bonus
        reward += tasks_completed_this_step * REWARD_TASK_COMPLETE
        
        # Idle penalty
        idle_count = sum(1 for rs in robot_states.values() 
                        if rs['state'] == RobotState.IDLE)
        reward += idle_count * REWARD_IDLE_PENALTY
        
        # Congestion penalty
        reward += congestion_events * REWARD_CONGESTION_PENALTY
        
        # Distance penalty
        reward += empty_travel_distance * REWARD_DISTANCE_PENALTY
        
        # Workload balance bonus
        if self.tasks_per_robot:
            counts = list(self.tasks_per_robot.values())
            if len(counts) > 1 and max(counts) > 0:
                balance = 1.0 - (max(counts) - min(counts)) / max(max(counts), 1)
                reward += balance * REWARD_BALANCE_BONUS
        
        return reward

    def assign(self, pending_tasks: List[TransportTask],
               robot_states: Dict[int, dict],
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        """Route-aware PPO deployment adapter.

        The legacy PPO checkpoint has an eight-action policy that selects only
        a robot ID.  Its observation contains aggregate queue information but
        no features for the concrete task subsequently selected by
        ``assign_task``.  A policy can therefore prefer a robot without knowing
        where that task's pickup/delivery points are.  DQN/SARSA do not have
        this deployment blind spot because their actions encode robot-task
        pairs.

        Preserve the learned robot prior, but combine it with the same runtime
        route cost used by the deterministic schedulers.  This is a deployment
        safety/reranking layer, not a claim that the old checkpoint learned
        pairwise assignment.  Set ``PPO_ROUTE_AWARE=0`` for a clean ablation.
        """
        enabled = os.environ.get(
            "PPO_ROUTE_AWARE", "1").strip().lower() in {
                "1", "true", "yes", "on"}
        if not enabled or self.training:
            return super().assign(pending_tasks, robot_states, context)

        context = context or SchedulingContext()
        started = time.perf_counter()
        try:
            idle_robots = self.get_idle_robots(robot_states)
            tasks = [task for task in pending_tasks
                     if task.status == TaskStatus.PENDING]
            if not idle_robots or not tasks:
                return SchedulerResult(
                    [], None, time.perf_counter() - started, False,
                    self.name, {"reason": "no_candidates",
                                "route_aware": True})

            state = self.encode_state(
                robot_states, tasks, context.congestion_map)
            action_probs, _ = self.network.forward(state)
            robot_probs = {
                rid: max(float(action_probs[rid - 1]), 1e-12)
                for rid in idle_robots if 1 <= rid <= self.action_dim
            }
            if not robot_probs:
                raise ValueError("no PPO action corresponds to an idle robot")
            probability_total = sum(robot_probs.values())
            robot_probs = {
                rid: value / probability_total
                for rid, value in robot_probs.items()
            }

            # Bound route-oracle work under a long queue while retaining urgent
            # and old requests. _pair_cost applies priority/waiting/congestion
            # terms again using the live SchedulingContext.
            task_limit = max(1, int(os.environ.get(
                "PPO_TASK_CANDIDATES", "8")))
            ranked_tasks = sorted(
                tasks,
                key=lambda task: (-float(task.priority), task.arrival_time,
                                  task.task_id))[:task_limit]
            candidates = []
            for task in ranked_tasks:
                for rid in idle_robots:
                    if rid not in robot_probs:
                        continue
                    if (rid, task.task_id) in context.failed_pairs:
                        continue
                    route_cost = _pair_cost(rid, task, robot_states, context)
                    if math.isfinite(route_cost):
                        candidates.append((rid, task, float(route_cost),
                                           -math.log(robot_probs[rid])))
            if not candidates:
                return SchedulerResult(
                    [], None, time.perf_counter() - started, False,
                    self.name, {"reason": "no_feasible_pair",
                                "route_aware": True})

            route_values = [item[2] for item in candidates]
            policy_values = [item[3] for item in candidates]

            def normalise(value, values):
                lower, upper = min(values), max(values)
                if upper - lower < 1e-12:
                    return 0.0
                return (value - lower) / (upper - lower)

            route_weight = float(os.environ.get(
                "PPO_ROUTE_WEIGHT", "0.50"))
            route_weight = min(1.0, max(0.0, route_weight))
            scored = []
            for rid, task, route_cost, policy_cost in candidates:
                score = (
                    route_weight * normalise(route_cost, route_values)
                    + (1.0 - route_weight)
                    * normalise(policy_cost, policy_values))
                scored.append((score, route_cost, task.arrival_time,
                               rid, task, policy_cost))
            score, route_cost, _, rid, task, policy_cost = min(scored)
            assignment = Assignment(rid, task, route_cost)
            valid, reason = validate_assignment(
                assignment, pending_tasks, robot_states, context)
            elapsed = time.perf_counter() - started
            return SchedulerResult(
                [assignment] if valid else [],
                route_cost if valid else None,
                elapsed, valid, self.name,
                {
                    "reason": reason,
                    "route_aware": True,
                    "route_weight": route_weight,
                    "candidate_pairs": len(candidates),
                    "selected_task_id": task.task_id,
                    "selected_robot_id": rid,
                    "route_cost": route_cost,
                    "policy_cost": policy_cost,
                    "hybrid_score": score,
                })
        except Exception as exc:
            return SchedulerResult(
                [], None, time.perf_counter() - started, False,
                self.name,
                {"reason": f"ppo_route_rerank_error:{type(exc).__name__}",
                 "route_aware": True})
    
    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        if not pending_tasks:
            return None
        
        idle_robots = self.get_idle_robots(robot_states)
        if not idle_robots:
            return None
        
        # Encode state
        state = self.encode_state(robot_states, pending_tasks, congestion_map)
        
        # Get action probabilities from policy network
        action_probs, value = self.network.forward(state)
        
        # Mask out non-idle robots
        mask = np.zeros(self.action_dim)
        for rid in idle_robots:
            if 1 <= rid <= self.action_dim:
                mask[rid - 1] = 1.0
        
        masked_probs = action_probs * mask
        prob_sum = np.sum(masked_probs)
        
        if prob_sum < 1e-8:
            # Fallback to uniform over idle robots
            masked_probs = mask / max(np.sum(mask), 1)
        else:
            masked_probs = masked_probs / prob_sum
        
        # Evaluation is deterministic; stochastic sampling is training-only.
        if self.deterministic:
            action = int(np.argmax(masked_probs))
        else:
            try:
                action = int(self.rng.choice(self.action_dim, p=masked_probs))
            except ValueError:
                action = idle_robots[0] - 1
        
        robot_id = action + 1  # Convert 0-indexed to 1-indexed
        
        # Verify the selected robot is actually idle
        if robot_id not in idle_robots:
            robot_id = idle_robots[0]
        
        # Select the highest-priority pending task
        sorted_tasks = sorted(pending_tasks, 
                              key=lambda t: (-t.priority, t.arrival_time))
        task = sorted_tasks[0]
        
        if self.training:
            self._pending_experience[(robot_id, task.task_id)] = {
                'state': state,
                'action': action,
                'value': value,
                'log_prob': math.log(max(masked_probs[action], 1e-8)),
            }
        return (robot_id, task)

    def on_assignment_committed(self, assignment: Assignment) -> None:
        super().on_assignment_committed(assignment)
        robot_id = assignment.robot_id
        self.tasks_per_robot[robot_id] = self.tasks_per_robot.get(robot_id, 0) + 1
        experience = self._pending_experience.pop(
            (robot_id, assignment.task.task_id), None)
        if experience is not None:
            self.experience_buffer.append(experience)

    def on_assignment_rejected(self, assignment: Optional[Assignment],
                               reason: str) -> None:
        if assignment is not None:
            self._pending_experience.pop(
                (assignment.robot_id, assignment.task.task_id), None)
    
    def reset(self):
        super().reset()
        self.tasks_per_robot.clear()
        self.total_distance_per_robot.clear()
        self.current_episode_reward = 0.0
        self.experience_buffer.clear()
        self._pending_experience.clear()


# ================================================================
# SCHEDULER FACTORY
# ================================================================

def create_scheduler(scheduler_type: str, model_path: Optional[str] = None,
                     *, seed: int = 42,
                     allow_safe_fallback: bool = True) -> BaseScheduler:
    """
    Factory function to create scheduler instances.
    
    Args:
        scheduler_type: One of "FCFS", "NearestNeighbour", "RoundRobin",
            "Greedy", "Random", "Hungarian", "Auction", "GA", "SA",
            "PPO_RL", "DQN", "SARSA"
        model_path: Path to a validated checkpoint for an RL scheduler.
        
    Returns:
        Scheduler instance.
    """
    schedulers = {
        "FCFS": FCFSScheduler,
        "NearestNeighbour": NearestNeighbourScheduler,
        "RoundRobin": RoundRobinScheduler,
        "Greedy": GreedyScheduler,
        "Random": lambda: RandomScheduler(seed=seed),
        "Hungarian": HungarianScheduler,
        "Auction": AuctionScheduler,
        "GA": lambda: GeneticScheduler(
            seed=seed,
            population_size=int(os.environ.get("GA_POPULATION", "48")),
            max_generations=int(os.environ.get("GA_GENERATIONS", "50")),
            time_budget_ms=float(os.environ.get("GA_TIME_BUDGET_MS", "10"))),
        "SA": lambda: SimulatedAnnealingScheduler(
            seed=seed,
            initial_temperature=float(os.environ.get("SA_INITIAL_TEMPERATURE", "10")),
            cooling_rate=float(os.environ.get("SA_COOLING_RATE", "0.985")),
            minimum_temperature=float(os.environ.get("SA_MIN_TEMPERATURE", "0.01")),
            max_iterations=int(os.environ.get("SA_MAX_ITERATIONS", "2000")),
            time_budget_ms=float(os.environ.get("SA_TIME_BUDGET_MS", "5"))),
    }
    
    if scheduler_type in {"DQN", "SARSA"}:
        try:
            from rl_schedulers import (
                DQNScheduler, RLSchedulerSafetyWrapper, SarsaScheduler)
            scheduler = (
                DQNScheduler(model_path, seed=seed)
                if scheduler_type == "DQN"
                else SarsaScheduler(model_path, seed=seed))
            return RLSchedulerSafetyWrapper(scheduler)
        except ModelValidationError as exc:
            if not allow_safe_fallback:
                raise
            fallback = HungarianScheduler()
            fallback.name = f"{scheduler_type}_FALLBACK_HUNGARIAN"
            fallback.fallback_reason = str(exc)
            print(f"[Scheduler] {scheduler_type} unavailable; "
                  f"using Hungarian: {exc}")
            return fallback

    if scheduler_type == "PPO_RL":
        try:
            # New checkpoints use the same robot-task pair action contract as
            # DQN/SARSA. If dimensions identify an older eight-action model,
            # retain the legacy route-aware adapter for reproducibility.
            try:
                from rl_schedulers import (PairwisePPOScheduler,
                                           RLSchedulerSafetyWrapper)
                return RLSchedulerSafetyWrapper(
                    PairwisePPOScheduler(model_path, seed=seed))
            except ModelValidationError as pairwise_error:
                legacy = RLScheduler(
                    model_path, seed=seed, deterministic=True)
                legacy.legacy_checkpoint_reason = str(pairwise_error)
                print("[Scheduler] Loaded legacy 8-action PPO checkpoint; "
                      "using route-aware compatibility adapter")
                return legacy
        except ModelValidationError as exc:
            if not allow_safe_fallback:
                raise
            fallback = HungarianScheduler()
            fallback.name = "PPO_RL_FALLBACK_HUNGARIAN"
            fallback.fallback_reason = str(exc)
            print(f"[Scheduler] PPO_RL unavailable; using Hungarian: {exc}")
            return fallback
    
    if scheduler_type in schedulers:
        return schedulers[scheduler_type]()
    
    raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                     f"Available: {list(schedulers.keys()) + ['PPO_RL', 'DQN', 'SARSA']}")
