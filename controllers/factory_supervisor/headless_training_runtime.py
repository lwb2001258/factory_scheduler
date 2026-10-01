"""Headless execution runtime mirroring Webots Supervisor business semantics.

This module deliberately does not emulate Webots physics, sensors, radio
delivery, or the robot controller's local DWA.  It advances the project's
authoritative task/battery state machine at the Webots basic timestep and uses
the unchanged MotionCoordinator for routes and reservations.  AI policies only
submit robot/task assignments; navigation and safety remain outside the policy.
"""

from dataclasses import dataclass, replace
import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import (
    ALL_LOCATIONS, BATTERY_CAPACITY, BATTERY_DRAIN_RATE, CHARGING_STATIONS,
    FULL_BATTERY_THRESHOLD, GOAL_TOLERANCE, LOW_BATTERY_THRESHOLD,
    TASK_ABORT_BATTERY_THRESHOLD, TIMESTEP, RobotState, TaskStatus,
)
from motion_coordinator import MotionCoordinator
from schedulers import Assignment, SchedulingContext, validate_assignment
from task_generator import TransportTask


HEADLESS_DYNAMICS_VERSION = "headless-webots-business-v1"


@dataclass(frozen=True)
class HeadlessRuntimeConfig:
    """Execution constants that are observable in the current Webots runtime."""

    timestep_seconds: float = TIMESTEP / 1000.0
    linear_speed: float = 0.22
    waypoint_tolerance: float = GOAL_TOLERANCE
    battery_swap_seconds: float = 5.0
    assignment_failure_ttl: float = 5.0
    lifelong_tick_seconds: float = 1.5
    deadlock_scan_seconds: float = 0.5
    retry_route_seconds: float = 1.0
    max_advance_seconds: float = 1800.0

    def __post_init__(self):
        finite_positive = (
            self.timestep_seconds, self.linear_speed, self.waypoint_tolerance,
            self.battery_swap_seconds, self.assignment_failure_ttl,
            self.lifelong_tick_seconds, self.deadlock_scan_seconds,
            self.retry_route_seconds, self.max_advance_seconds,
        )
        if any(not math.isfinite(value) or value <= 0
               for value in finite_positive):
            raise ValueError("headless runtime constants must be finite and positive")


def _polyline_length(start: Tuple[float, float],
                     points: Iterable[Tuple[float, float]]) -> float:
    total = 0.0
    previous = tuple(start)
    for point in points:
        current = (float(point[0]), float(point[1]))
        total += math.hypot(current[0] - previous[0],
                            current[1] - previous[1])
        previous = current
    return total


def summarize_runtime_telemetry(rows: Iterable[dict]) -> dict:
    """Aggregate per-episode runtime counters without hiding fidelity."""
    rows = tuple(rows)
    integer_keys = (
        "ticks", "assignments_committed", "assignments_rejected",
        "tasks_completed", "tasks_failed_battery", "charge_swaps",
        "route_retries", "deadlock_breaks",
    )
    result = {
        "runtime_mode": "headless_webots_logic",
        "dynamics_version": HEADLESS_DYNAMICS_VERSION,
        "physics_fidelity": "business_logic_only",
        "episodes": len(rows),
    }
    for key in integer_keys:
        result[key] = sum(int(row.get(key, 0)) for row in rows)
    result["distance_travelled"] = float(sum(
        float(row.get("distance_travelled", 0.0)) for row in rows))
    result["simulated_seconds"] = float(sum(
        float(row.get("current_time", 0.0)) for row in rows))
    if not all(math.isfinite(result[key])
               for key in ("distance_travelled", "simulated_seconds")):
        raise ValueError("runtime telemetry contains non-finite totals")
    return result


class HeadlessFactoryRuntime:
    """Time-stepped factory execution used only by offline training.

    The caller owns ``robot_states`` and ``tasks``.  SchedulingEnvironment
    supplies private deep copies, while this runtime mutates those copies in
    the same order as the Supervisor's business state machine.
    """

    def __init__(self, robot_states: Dict[int, dict],
                 tasks: List[TransportTask], context: SchedulingContext,
                 *, seed: int = 0,
                 config: Optional[HeadlessRuntimeConfig] = None):
        if not robot_states:
            raise ValueError("headless runtime requires at least one robot")
        if len(set(robot_states)) != len(robot_states):
            raise ValueError("duplicate robot IDs")
        task_ids = [task.task_id for task in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate task IDs")
        if not math.isfinite(float(context.current_time)):
            raise ValueError("non-finite initial time")

        self.config = config or HeadlessRuntimeConfig()
        self.robots = robot_states
        self.tasks = tasks
        self._base_context = context
        self.current_time = float(context.current_time)
        self.rng = np.random.default_rng(int(seed))
        self.coordinator = MotionCoordinator(
            num_active_robots=max(1, len(robot_states)))
        self.coordinator.set_priorities(sorted(robot_states))
        self.coordinator.lifelong_reset()
        self.coordinator.init_deadlock_monitor()
        self.coordinator.set_sim_time(self.current_time)

        self._waypoints = {rid: [] for rid in robot_states}
        self._waypoint_index = {rid: 0 for rid in robot_states}
        self._dispatch_not_before = {rid: self.current_time
                                     for rid in robot_states}
        self._failed_pairs = {
            tuple(pair): self.current_time + self.config.assignment_failure_ttl
            for pair in context.failed_pairs
        }
        self._next_lifelong_tick = (
            self.current_time + self.config.lifelong_tick_seconds)
        self._next_deadlock_scan = (
            self.current_time + self.config.deadlock_scan_seconds)
        self._next_retry = {
            rid: self.current_time for rid in robot_states}
        self._visible_task_ids = {
            task.task_id for task in tasks
            if task.status == TaskStatus.PENDING
            and float(task.arrival_time) <= self.current_time + 1e-9
        }
        self._completed_ids = {
            task.task_id for task in tasks
            if task.status == TaskStatus.COMPLETED
        }
        self.telemetry = {
            "ticks": 0,
            "assignments_committed": 0,
            "assignments_rejected": 0,
            "tasks_completed": len(self._completed_ids),
            "tasks_failed_battery": 0,
            "charge_swaps": 0,
            "route_retries": 0,
            "deadlock_breaks": 0,
            "distance_travelled": 0.0,
        }

        for rid, state in self.robots.items():
            position = state.get("position")
            if (not isinstance(position, (tuple, list)) or len(position) < 2 or
                    not all(math.isfinite(float(value))
                            for value in position[:2])):
                raise ValueError(f"robot {rid} has invalid position")
            state["position"] = (float(position[0]), float(position[1]))
            state.setdefault("heading", 0.0)
            state.setdefault("state", RobotState.IDLE)
            state.setdefault("battery", BATTERY_CAPACITY)
            state.setdefault("current_task", None)
            state.setdefault("has_task", state["current_task"] is not None)
            state.setdefault("goal_location", None)
            state.setdefault("tasks_completed", 0)
            state.setdefault("total_distance", 0.0)
            state.setdefault("speed_scale", 1.0)
            state["_headless_swap_ready_at"] = None
            if not math.isfinite(float(state["battery"])):
                raise ValueError(f"robot {rid} has invalid battery")
            state["battery"] = float(np.clip(
                state["battery"], 0.0, BATTERY_CAPACITY))

        provider = context.path_cost_provider
        bind = getattr(provider, "bind_robot_states", None)
        if callable(bind):
            bind(self.robots)

        # Match the Supervisor's dispatch-side battery guard.
        for rid, state in self.robots.items():
            if (state["state"] == RobotState.IDLE and
                    state["current_task"] is None and
                    state["battery"] < LOW_BATTERY_THRESHOLD):
                self._send_to_charging(rid)

    @property
    def completed_ids(self):
        return frozenset(self._completed_ids)

    @property
    def context(self) -> SchedulingContext:
        self._expire_failed_pairs()
        provider = self._base_context.path_cost_provider
        bind = getattr(provider, "bind_robot_states", None)
        if callable(bind):
            bind(self.robots)
        configuration = dict(self._base_context.configuration or {})
        configuration.update({
            "runtime_mode": "headless_webots_logic",
            "dynamics_version": HEADLESS_DYNAMICS_VERSION,
            "physics_fidelity": "business_logic_only",
        })
        return replace(
            self._base_context,
            current_time=self.current_time,
            congestion_map=self.coordinator.get_congestion_map(),
            failed_pairs=frozenset(self._failed_pairs),
            configuration=configuration,
        )

    def arrived_pending_tasks(self) -> List[TransportTask]:
        return [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING
            and float(task.arrival_time) <= self.current_time + 1e-9
        ]

    def has_future_arrivals(self) -> bool:
        return any(
            task.status == TaskStatus.PENDING
            and float(task.arrival_time) > self.current_time + 1e-9
            for task in self.tasks)

    def has_active_execution(self) -> bool:
        active = {
            RobotState.EN_ROUTE_PICKUP, RobotState.CARRYING,
            RobotState.EN_ROUTE_DELIVERY, RobotState.RETURNING_HOME,
            RobotState.RETURNING_TO_CHARGE, RobotState.CHARGING,
            RobotState.WAITING,
        }
        return any(state.get("state") in active for state in self.robots.values())

    def is_terminal(self) -> bool:
        pending = any(task.status == TaskStatus.PENDING for task in self.tasks)
        assigned = any(task.status in (TaskStatus.ASSIGNED,
                                       TaskStatus.IN_PROGRESS)
                       for task in self.tasks)
        return not pending and not assigned and not self.has_active_execution()

    def _expire_failed_pairs(self) -> None:
        self._failed_pairs = {
            pair: expiry for pair, expiry in self._failed_pairs.items()
            if expiry > self.current_time
        }

    def _plan(self, rid: int, goal) -> Optional[List[Tuple[float, float]]]:
        state = self.robots[rid]
        path = self.coordinator.plan_grid_lifelong(
            rid, state["position"], goal)
        if path is None:
            priority = getattr(state.get("current_task"), "priority", 0.0)
            self.coordinator.robot_priorities[rid] = -float(priority or 0.0)
            path = self.coordinator.plan_path_for_robot(
                rid, state["position"], goal)
        if path is None:
            return None
        clean = []
        for point in path:
            xy = (float(point[0]), float(point[1]))
            if not all(math.isfinite(value) for value in xy):
                self.coordinator.rollback_robot_plan(rid)
                return None
            clean.append(xy)
        return clean or None

    def _install_path(self, rid: int,
                      path: List[Tuple[float, float]]) -> None:
        self._waypoints[rid] = list(path)
        self._waypoint_index[rid] = 0
        delay = max(0.0, float(self.coordinator.get_dispatch_delay(rid)))
        self._dispatch_not_before[rid] = self.current_time + delay

    def dispatch(self, assignment: Assignment) -> dict:
        """Plan and commit one assignment after dynamic route validation."""
        pending = self.arrived_pending_tasks()
        valid, reason = validate_assignment(
            assignment, pending, self.robots, self.context)
        if not valid:
            self.telemetry["assignments_rejected"] += 1
            return {"type": "assignment_rejected", "reason": reason,
                    "robot_id": getattr(assignment, "robot_id", None),
                    "task_id": getattr(getattr(assignment, "task", None),
                                       "task_id", None)}

        rid, task = assignment.robot_id, assignment.task
        path = self._plan(rid, task.pickup_location)
        if not path:
            self.coordinator.rollback_robot_plan(rid)
            self._failed_pairs[(rid, task.task_id)] = (
                self.current_time + self.config.assignment_failure_ttl)
            self.telemetry["assignments_rejected"] += 1
            return {"type": "assignment_rejected",
                    "reason": "pickup_path_unreachable",
                    "robot_id": rid, "task_id": task.task_id}

        state = self.robots[rid]
        self.coordinator.release_home(rid)
        task.status = TaskStatus.ASSIGNED
        task.assigned_robot = rid
        task.assignment_time = self.current_time
        state["current_task"] = task
        state["has_task"] = True
        state["state"] = RobotState.EN_ROUTE_PICKUP
        state["goal_location"] = task.pickup_location
        self._install_path(rid, path)
        self._failed_pairs.pop((rid, task.task_id), None)
        self.telemetry["assignments_committed"] += 1
        return {
            "type": "assignment_committed", "reason": "ok",
            "robot_id": rid, "task_id": task.task_id,
            "planned_pickup_distance": _polyline_length(
                state["position"], path),
        }

    def _clear_path(self, rid: int, *, release: bool = True) -> None:
        self._waypoints[rid] = []
        self._waypoint_index[rid] = 0
        self.coordinator.clear_robot_path(rid)
        if release:
            self.coordinator.release_lifelong(rid)
            self.coordinator.release_robot_grid(rid)

    def _goal_position(self, goal) -> Optional[Tuple[float, float]]:
        if isinstance(goal, (tuple, list)) and len(goal) >= 2:
            return float(goal[0]), float(goal[1])
        if goal in ALL_LOCATIONS:
            return tuple(ALL_LOCATIONS[goal])
        if goal in CHARGING_STATIONS:
            return tuple(CHARGING_STATIONS[goal])
        return None

    def _send_to_charging(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        if self.current_time + 1e-9 < self._next_retry[rid]:
            return []
        self._clear_path(rid)

        def distance(station):
            point = CHARGING_STATIONS[station]
            return math.hypot(state["position"][0] - point[0],
                              state["position"][1] - point[1])

        nearest_first = sorted(CHARGING_STATIONS, key=distance)
        queue_lengths = {name: 0 for name in CHARGING_STATIONS}
        for other_id, other in self.robots.items():
            if other_id != rid and other.get("state") in {
                    RobotState.RETURNING_TO_CHARGE, RobotState.CHARGING}:
                station = other.get("goal_location")
                if station in queue_lengths:
                    queue_lengths[station] += 1

        ordered = nearest_first[:1] + sorted(
            nearest_first[1:], key=lambda name: (queue_lengths[name],
                                                  distance(name)))
        for station in ordered:
            path = self._plan(rid, station)
            if path:
                state["state"] = RobotState.RETURNING_TO_CHARGE
                state["goal_location"] = station
                self._install_path(rid, path)
                self._next_retry[rid] = self.current_time
                return [{"type": "charge_return_started", "robot_id": rid,
                         "station": station}]
            self.coordinator.rollback_robot_plan(rid)

        if nearest_first and distance(nearest_first[0]) <= 0.5:
            station = nearest_first[0]
            state["state"] = RobotState.CHARGING
            state["goal_location"] = station
            state["_headless_swap_ready_at"] = (
                self.current_time + self.config.battery_swap_seconds)
            return [{"type": "charging_started", "robot_id": rid,
                     "station": station}]

        state["state"] = RobotState.WAITING
        state["goal_location"] = None
        self._next_retry[rid] = self.current_time + self.config.retry_route_seconds
        return [{"type": "charge_route_wait", "robot_id": rid}]

    def _fail_task_for_battery(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        task = state.get("current_task")
        events = []
        if task is not None:
            task.status = TaskStatus.FAILED
            task.assigned_robot = None
            task.assignment_time = None
            events.append({"type": "task_failed_battery", "robot_id": rid,
                           "task_id": task.task_id})
            self.telemetry["tasks_failed_battery"] += 1
        state["current_task"] = None
        state["has_task"] = False
        state["goal_location"] = None
        state["state"] = RobotState.IDLE
        self._clear_path(rid)
        events.extend(self._send_to_charging(rid))
        return events

    def _update_battery(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        events = []
        swap_at = state.get("_headless_swap_ready_at")
        if (swap_at is not None and self.current_time + 1e-9 >= float(swap_at)
                and state.get("state") == RobotState.CHARGING):
            state["battery"] = float(self.rng.uniform(
                FULL_BATTERY_THRESHOLD, BATTERY_CAPACITY))
            state["_headless_swap_ready_at"] = None
            state["state"] = RobotState.IDLE
            state["goal_location"] = None
            self.telemetry["charge_swaps"] += 1
            return [{"type": "charge_swap_completed", "robot_id": rid}]

        if state.get("state") in {
                RobotState.EN_ROUTE_PICKUP, RobotState.CARRYING,
                RobotState.EN_ROUTE_DELIVERY, RobotState.RETURNING_HOME}:
            state["battery"] = max(
                0.0, float(state["battery"])
                - BATTERY_DRAIN_RATE * self.config.timestep_seconds)

        if (state["battery"] < LOW_BATTERY_THRESHOLD and
                state.get("state") not in {
                    RobotState.RETURNING_TO_CHARGE, RobotState.CHARGING}):
            if (state["battery"] < TASK_ABORT_BATTERY_THRESHOLD and
                    state.get("current_task") is not None):
                return self._fail_task_for_battery(rid)
            if state.get("current_task") is None:
                events.extend(self._send_to_charging(rid))
        return events

    def _plan_delivery(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        task = state.get("current_task")
        if task is None:
            return []
        path = self._plan(rid, task.delivery_location)
        if not path:
            self.coordinator.rollback_robot_plan(rid)
            self._clear_path(rid, release=False)
            self._next_retry[rid] = self.current_time + self.config.retry_route_seconds
            return [{"type": "delivery_route_wait", "robot_id": rid,
                     "task_id": task.task_id}]
        task.status = TaskStatus.IN_PROGRESS
        task.pickup_time = self.current_time
        state["state"] = RobotState.EN_ROUTE_DELIVERY
        state["goal_location"] = task.delivery_location
        self._install_path(rid, path)
        return [{"type": "task_picked_up", "robot_id": rid,
                 "task_id": task.task_id}]

    def _complete_task(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        task = state.get("current_task")
        if task is None:
            return []
        task.status = TaskStatus.COMPLETED
        task.completion_time = self.current_time
        self._completed_ids.add(task.task_id)
        state["tasks_completed"] = int(state.get("tasks_completed", 0)) + 1
        state["current_task"] = None
        state["has_task"] = False
        state["goal_location"] = None
        state["state"] = RobotState.IDLE
        self._clear_path(rid)
        self.telemetry["tasks_completed"] += 1
        events = [{"type": "task_completed", "robot_id": rid,
                   "task_id": task.task_id}]
        if state["battery"] < LOW_BATTERY_THRESHOLD:
            events.extend(self._send_to_charging(rid))
        return events

    def _handle_goal_reached(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        goal = self._goal_position(state.get("goal_location"))
        if goal is not None:
            distance = math.hypot(state["position"][0] - goal[0],
                                  state["position"][1] - goal[1])
            if distance > GOAL_TOLERANCE * 2.5:
                return []
        if state["state"] == RobotState.EN_ROUTE_PICKUP:
            return self._plan_delivery(rid)
        if state["state"] == RobotState.EN_ROUTE_DELIVERY:
            return self._complete_task(rid)
        if state["state"] == RobotState.RETURNING_TO_CHARGE:
            state["state"] = RobotState.CHARGING
            state["_headless_swap_ready_at"] = (
                self.current_time + self.config.battery_swap_seconds)
            self._clear_path(rid)
            return [{"type": "charging_started", "robot_id": rid,
                     "station": state.get("goal_location")}]
        if state["state"] == RobotState.RETURNING_HOME:
            state["state"] = RobotState.IDLE
            state["goal_location"] = None
            self._clear_path(rid)
            return [{"type": "robot_idle", "robot_id": rid}]
        return []

    def _move_robot(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        waypoints = self._waypoints[rid]
        index = self._waypoint_index[rid]
        if (not waypoints or index >= len(waypoints) or
                self.current_time + 1e-9 < self._dispatch_not_before[rid]):
            return []
        target = waypoints[index]
        x, y = state["position"]
        dx, dy = target[0] - x, target[1] - y
        distance = math.hypot(dx, dy)
        max_step = (self.config.linear_speed
                    * float(np.clip(state.get("speed_scale", 1.0), 0.0, 1.0))
                    * self.config.timestep_seconds)
        if distance <= max(self.config.waypoint_tolerance, max_step):
            # The real controller reports a waypoint reached while physically
            # inside its tolerance; it does not teleport to the coordinate.
            moved = 0.0
            self._waypoint_index[rid] += 1
            self.coordinator.advance_robot(rid)
        elif max_step > 0.0:
            moved = max_step
            state["position"] = (x + dx / distance * moved,
                                 y + dy / distance * moved)
            state["heading"] = math.atan2(dy, dx)
        else:
            moved = 0.0
        state["total_distance"] = float(
            state.get("total_distance", 0.0)) + moved
        self.telemetry["distance_travelled"] += moved
        if self._waypoint_index[rid] >= len(waypoints):
            return self._handle_goal_reached(rid)
        return []

    def _retry_routes(self) -> List[dict]:
        events = []
        for rid, state in self.robots.items():
            if (self._waypoints[rid] or
                    self.current_time + 1e-9 < self._next_retry[rid]):
                continue
            if (state["state"] == RobotState.WAITING and
                    state.get("current_task") is None and
                    state["battery"] < LOW_BATTERY_THRESHOLD):
                events.extend(self._send_to_charging(rid))
                self.telemetry["route_retries"] += 1
                continue
            task = state.get("current_task")
            if task is None:
                continue
            if state["state"] == RobotState.EN_ROUTE_PICKUP:
                pickup_distance = math.hypot(
                    state["position"][0] - task.pickup_position[0],
                    state["position"][1] - task.pickup_position[1])
                if pickup_distance <= GOAL_TOLERANCE * 2.5:
                    self.telemetry["route_retries"] += 1
                    events.extend(self._plan_delivery(rid))
                    continue
            goal = (task.pickup_location
                    if state["state"] in {RobotState.EN_ROUTE_PICKUP,
                                          RobotState.CARRYING}
                    else task.delivery_location)
            path = self._plan(rid, goal)
            self.telemetry["route_retries"] += 1
            if path:
                if state["state"] == RobotState.CARRYING:
                    state["state"] = RobotState.EN_ROUTE_DELIVERY
                self._install_path(rid, path)
                events.append({"type": "route_retry_succeeded",
                               "robot_id": rid, "task_id": task.task_id})
            else:
                self.coordinator.rollback_robot_plan(rid)
                self._next_retry[rid] = (
                    self.current_time + self.config.retry_route_seconds)
        return events

    def _scan_deadlocks(self) -> List[dict]:
        states = {
            rid: {
                "position": state["position"],
                "state": state["state"],
                "goal_location": state.get("goal_location"),
                "speed_scale": state.get("speed_scale", 1.0),
                "task_priority": float(getattr(
                    state.get("current_task"), "priority", 0.0) or 0.0),
                "wait_age": 0.0,
                "entered_zone": False,
                "conflict_distance": 0.0,
            }
            for rid, state in self.robots.items()
        }
        stuck = self.coordinator.update_deadlock_monitor(states)
        if not stuck:
            return []
        broken = self.coordinator.break_deadlock(stuck, states)
        events = []
        for rid in broken:
            self._waypoints[rid] = []
            self._waypoint_index[rid] = 0
            self._next_retry[rid] = self.current_time
            events.append({"type": "deadlock_replan", "robot_id": rid})
        self.telemetry["deadlock_breaks"] += len(broken)
        return events

    def tick(self) -> List[dict]:
        """Advance exactly one Webots basic timestep."""
        self.current_time += self.config.timestep_seconds
        self.telemetry["ticks"] += 1
        self.coordinator.set_sim_time(self.current_time)
        self._expire_failed_pairs()
        events = []

        while self.current_time + 1e-9 >= self._next_lifelong_tick:
            self.coordinator.lifelong_tick(1)
            self._next_lifelong_tick += self.config.lifelong_tick_seconds

        for rid in sorted(self.robots):
            events.extend(self._update_battery(rid))
            events.extend(self._move_robot(rid))

        events.extend(self._retry_routes())
        while self.current_time + 1e-9 >= self._next_deadlock_scan:
            events.extend(self._scan_deadlocks())
            self._next_deadlock_scan += self.config.deadlock_scan_seconds

        arrived = {
            task.task_id for task in self.tasks
            if task.status == TaskStatus.PENDING
            and float(task.arrival_time) <= self.current_time + 1e-9
        }
        for task_id in sorted(arrived - self._visible_task_ids):
            events.append({"type": "task_arrived", "task_id": task_id})
        self._visible_task_ids.update(arrived)
        return events

    def advance_until_event(self, *, max_seconds: Optional[float] = None
                            ) -> List[dict]:
        """Advance until a policy-relevant event or a bounded stall."""
        limit = (self.config.max_advance_seconds if max_seconds is None
                 else float(max_seconds))
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("max_seconds must be finite and positive")
        deadline = self.current_time + limit
        relevant = {
            "task_arrived", "task_completed", "task_failed_battery",
            "charge_swap_completed", "assignment_rejected",
            "deadlock_replan", "robot_idle",
        }
        collected = []
        while self.current_time + 1e-9 < deadline:
            events = self.tick()
            collected.extend(events)
            if any(event.get("type") in relevant for event in events):
                break
            if self.is_terminal():
                break
            if not self.has_active_execution() and not self.has_future_arrivals():
                break
        if self.current_time + 1e-9 >= deadline and not collected:
            collected.append({"type": "advance_timeout"})
        return collected

    def telemetry_snapshot(self) -> dict:
        result = dict(self.telemetry)
        result.update({
            "current_time": self.current_time,
            "completed_ids": sorted(self._completed_ids),
            "failed_pairs": sorted(list(self.context.failed_pairs)),
            "dynamics_version": HEADLESS_DYNAMICS_VERSION,
        })
        return result
