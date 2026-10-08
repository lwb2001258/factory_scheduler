"""Headless execution runtime mirroring Webots business-motion semantics.

The offline runtime uses the production MotionCoordinator, rolling joint
space-time planner, duplicate-goal staging, timed waypoint activation, idle
reservations and the joint predictive speed shield.  It also mirrors the
robot controller's obstacle-free differential-drive steering law.  It cannot
emulate Webots rigid-body contacts, LiDAR/DWA reactions or radio transaction
failures, so its declared physics fidelity remains ``business_logic_only``.
AI policies only submit robot/task assignments; navigation and safety remain
outside the policy.
"""

from dataclasses import dataclass, replace
import itertools
import math
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import (
    ALL_LOCATIONS, BATTERY_CAPACITY, BATTERY_DRAIN_RATE, CHARGING_STATIONS,
    FULL_BATTERY_THRESHOLD, GOAL_TOLERANCE, LOW_BATTERY_THRESHOLD,
    INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX,
    JOINT_STALL_RECOVERY_COOLDOWN, JOINT_STALL_RELOCATION_TIMEOUT,
    JOINT_WATCHDOG_INTERVAL, MIN_NAV_DISPLACEMENT, PARKING_SPOTS,
    RELOCATION_PEER_CLEARANCE, RESERVATION_REFRESH_INTERVAL, REST_NODES,
    STALL_PROGRESS_DISTANCE, TASK_ABORT_BATTERY_THRESHOLD, TIMESTEP,
    WAYPOINTS, YIELD_RESUME_MIN_CLEARANCE, RobotState, TaskStatus,
)
from motion_coordinator import MotionCoordinator
from schedulers import Assignment, SchedulingContext, validate_assignment
from task_generator import TransportTask


HEADLESS_DYNAMICS_VERSION = "headless-webots-business-v25-watchdog-timer-parity"


@dataclass(frozen=True)
class HeadlessRuntimeConfig:
    """Execution constants that are observable in the current Webots runtime."""

    timestep_seconds: float = TIMESTEP / 1000.0
    # RobotNavigator's effective MAX_LINEAR_SPEED is the later 0.22 m/s
    # definition used by its joint-coordinated branch; the motor boundary then
    # applies its validated 1.09 calibration.
    # The Supervisor predicts trajectories at an independent nominal 0.24 m/s.
    linear_speed: float = 0.22 * 0.85 * 1.09
    prediction_linear_speed: float = 0.24
    max_angular_speed: float = 2.84 * 0.70 * 1.09
    # The joint-coordinated controller returns before the legacy MAX_SPEED
    # clamp.  The actual RotationalMotor nodes in smart_factory.wbt therefore
    # enforce their 26 rad/s maxVelocity (the largest joint command is only
    # about 9.4 rad/s, so it remains unsaturated).
    max_wheel_speed: float = 26.0
    wheel_radius: float = 0.033
    wheel_base: float = 0.287
    heading_tolerance: float = 0.30
    heading_rotate_gain: float = 2.5 * 1.09
    heading_drive_gain: float = 2.0 * 1.09
    distance_speed_gain: float = 1.5 * 1.09
    # _activate_scheduled_joint_plan overrides the normal 0.35 m controller
    # threshold with 0.22 m for every committed joint-grid epoch.
    waypoint_tolerance: float = 0.22
    direct_waypoint_tolerance: float = 0.35
    battery_swap_seconds: float = 5.0
    assignment_failure_ttl: float = 5.0
    lifelong_tick_seconds: float = 1.5
    deadlock_scan_seconds: float = 0.5
    relocation_scan_seconds: float = 0.5
    joint_watchdog_seconds: float = JOINT_WATCHDOG_INTERVAL
    reservation_refresh_seconds: float = RESERVATION_REFRESH_INTERVAL
    joint_stall_relocation_seconds: float = JOINT_STALL_RELOCATION_TIMEOUT
    joint_stall_recovery_cooldown_seconds: float = JOINT_STALL_RECOVERY_COOLDOWN
    stall_progress_distance: float = STALL_PROGRESS_DISTANCE
    minimum_navigation_displacement: float = MIN_NAV_DISPLACEMENT
    relocation_peer_clearance: float = RELOCATION_PEER_CLEARANCE
    joint_first_plan_seconds: float = 2.0
    joint_replan_seconds: float = 3.0
    joint_activation_delay: float = 0.5
    joint_candidate_budget_seconds: float = 0.25
    joint_candidate_max_budget_seconds: float = 1.00
    avoidance_scan_seconds: float = 0.5
    avoidance_horizon_seconds: float = 6.0
    avoidance_sample_seconds: float = 0.25
    avoidance_minimum_distance: float = 0.85
    emergency_stop_distance: float = 0.70
    controller_peer_stop_distance: float = 0.55
    hard_minimum_distance: float = 0.50
    # Four-robot Webots conflict components contain at most 5**4 profiles.
    # Standalone evaluates that deterministic set without an 80 ms wall-clock
    # cutoff, so training data cannot change with host CPU load.
    avoidance_profile_limit: int = 625
    retry_route_seconds: float = 1.0
    max_advance_seconds: float = 1800.0
    episode_end_time: float = 1800.0
    fixed_horizon: bool = False
    joint_runtime: bool = True

    def __post_init__(self):
        finite_positive = (
            self.timestep_seconds, self.linear_speed,
            self.prediction_linear_speed, self.max_angular_speed,
            self.max_wheel_speed, self.wheel_radius, self.wheel_base,
            self.heading_tolerance, self.heading_rotate_gain,
            self.heading_drive_gain, self.distance_speed_gain,
            self.waypoint_tolerance, self.direct_waypoint_tolerance,
            self.battery_swap_seconds, self.assignment_failure_ttl,
            self.lifelong_tick_seconds, self.deadlock_scan_seconds,
            self.relocation_scan_seconds, self.joint_watchdog_seconds,
            self.reservation_refresh_seconds,
            self.joint_stall_relocation_seconds,
            self.joint_stall_recovery_cooldown_seconds,
            self.stall_progress_distance,
            self.minimum_navigation_displacement,
            self.relocation_peer_clearance,
            self.joint_first_plan_seconds, self.joint_replan_seconds,
            self.joint_activation_delay, self.joint_candidate_budget_seconds,
            self.joint_candidate_max_budget_seconds,
            self.avoidance_scan_seconds,
            self.avoidance_horizon_seconds, self.avoidance_sample_seconds,
            self.avoidance_minimum_distance, self.emergency_stop_distance,
            self.controller_peer_stop_distance, self.hard_minimum_distance,
            self.retry_route_seconds, self.max_advance_seconds,
            self.episode_end_time,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0
               for value in finite_positive):
            raise ValueError("headless runtime constants must be finite and positive")
        if not isinstance(self.fixed_horizon, bool):
            raise ValueError("fixed_horizon must be boolean")
        if not isinstance(self.joint_runtime, bool):
            raise ValueError("joint_runtime must be boolean")
        if (isinstance(self.avoidance_profile_limit, bool) or
                not isinstance(self.avoidance_profile_limit, int) or
                self.avoidance_profile_limit <= 0):
            raise ValueError("avoidance profile limit must be positive integer")
        if self.emergency_stop_distance >= self.avoidance_minimum_distance:
            raise ValueError("emergency distance must be below avoidance distance")
        if self.hard_minimum_distance >= self.emergency_stop_distance:
            raise ValueError("hard distance must be below emergency distance")
        if self.controller_peer_stop_distance <= self.hard_minimum_distance:
            raise ValueError("controller peer stop must exceed hard distance")
        if (self.joint_candidate_budget_seconds >
                self.joint_candidate_max_budget_seconds):
            raise ValueError("initial joint budget exceeds maximum")


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
        "route_retries", "deadlock_breaks", "joint_plans_attempted",
        "joint_plans_prepared", "joint_plans_activated",
        "joint_plans_failed", "joint_transactions_aborted",
        "joint_fallbacks_prepared", "joint_plan_cache_hits",
        "joint_plan_cache_misses", "joint_plan_negative_cache_hits",
        "predicted_peer_conflicts", "emergency_peer_stops",
        "controller_peer_stop_entries", "hard_distance_violations",
        "joint_stall_recoveries", "joint_escape_recoveries",
        "joint_escape_cache_hits", "joint_direct_yield_replans",
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
    observed_distances = [
        float(row["minimum_peer_distance"])
        for row in rows if row.get("minimum_peer_distance") is not None]
    result["minimum_peer_distance"] = (
        min(observed_distances) if observed_distances else None)
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
        if (isinstance(context.current_time, bool) or
                not isinstance(context.current_time, (int, float)) or
                not math.isfinite(float(context.current_time))):
            raise ValueError("non-finite initial time")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("headless runtime seed must be an integer")

        self.config = config or HeadlessRuntimeConfig()
        self.robots = robot_states
        self.tasks = tasks
        self._base_context = context
        self.current_time = float(context.current_time)
        if self.current_time < 0 or self.current_time >= self.config.episode_end_time:
            raise ValueError("initial time must lie inside the episode horizon")
        self._tick_seconds = self.config.timestep_seconds
        self._horizon_finalized = False
        self.rng = np.random.default_rng(int(seed))
        # FactorySupervisor uses one Python RNG for both its initial battery
        # snapshots and every later 95-100% station swap.  The snapshots have
        # already been materialized by factory_scenario(), but consume the
        # same draws here so subsequent swaps reproduce Webots exactly.
        self._battery_rng = random.Random(int(seed))
        for _rid in sorted(robot_states):
            self._battery_rng.uniform(
                INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX)
        self.coordinator = MotionCoordinator(
            num_active_robots=max(1, len(robot_states)))
        self.coordinator.set_priorities(sorted(robot_states))
        self.coordinator.lifelong_reset()
        self.coordinator.init_deadlock_monitor()
        self.coordinator.set_sim_time(self.current_time)

        self._waypoints = {rid: [] for rid in robot_states}
        self._waypoint_index = {rid: 0 for rid in robot_states}
        self._waypoint_not_before = {rid: [] for rid in robot_states}
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
        # Match FactorySupervisor.run(): first lazy-relocation scan at T=1.0,
        # then every 0.5 simulated seconds.
        self._next_relocation_scan = self.current_time + 1.0
        self._next_joint_watchdog = self.current_time
        self._next_reservation_refresh = self.current_time
        self._next_joint_plan = (
            self.current_time + self.config.joint_first_plan_seconds)
        self._next_avoidance_scan = self.current_time
        self._joint_candidate_failures = 0
        self._joint_liveness_needed = False
        self._force_joint_replan = False
        self._joint_replan_cooldown_until = 0.0
        self._pending_joint_transaction = None
        self._joint_epoch = 0
        self._active_joint_members = set()
        self._last_joint_activation_time = float("-inf")
        # A joint candidate depends only on quantized grid endpoints, the
        # priority order and its deterministic search budget. Webots must plan
        # again because measured poses and radio state are live; standalone can
        # safely reuse an already validated candidate while those inputs are
        # identical. This removes repeated CPU work without changing a route.
        self._joint_candidate_cache = {}
        self._joint_candidate_cache_limit = 4096
        self._joint_escape_cache = {}
        self._joint_escape_cache_limit = 1024
        self._joint_planning_wall_seconds = 0.0
        self._next_retry = {
            rid: self.current_time for rid in robot_states}
        self._visible_task_ids = {
            task.task_id for task in tasks
            if task.status == TaskStatus.PENDING
            and float(task.arrival_time) <= self.current_time + 1e-9
            and float(task.arrival_time) < self.config.episode_end_time
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
            "joint_plans_attempted": 0,
            "joint_plans_prepared": 0,
            "joint_plans_activated": 0,
            "joint_plans_failed": 0,
            "joint_transactions_aborted": 0,
            "joint_fallbacks_prepared": 0,
            "joint_plan_cache_hits": 0,
            "joint_plan_cache_misses": 0,
            "joint_plan_negative_cache_hits": 0,
            "predicted_peer_conflicts": 0,
            "emergency_peer_stops": 0,
            "controller_peer_stop_entries": 0,
            "joint_stall_recoveries": 0,
            "joint_escape_recoveries": 0,
            "joint_escape_cache_hits": 0,
            "joint_direct_yield_replans": 0,
            "hard_distance_violations": 0,
            "distance_travelled": 0.0,
            "minimum_peer_distance": None,
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
            state["_headless_joint_route_active"] = False
            state["_headless_joint_plan_partial"] = False
            state["_headless_plan_source"] = None
            state["_headless_waypoint_tolerance"] = (
                self.config.direct_waypoint_tolerance)
            state["_headless_shield_until"] = 0.0
            state["_headless_hold_until"] = 0.0
            state["_headless_peer_stop_latched"] = False
            state["_headless_stall_watch_pos"] = tuple(state["position"])
            state["_headless_stall_since"] = None
            state["_headless_joint_watch_pos"] = tuple(state["position"])
            state["_headless_joint_watch_since"] = None
            state["_headless_hard_stall_watch_pos"] = tuple(state["position"])
            state["_headless_hard_stall_since"] = None
            state["_headless_any_wait_since"] = None
            state["_headless_joint_wait_since"] = None
            state["_headless_route_less_since"] = None
            state["_headless_emergency_since"] = None
            state["_headless_stall_recovery_until"] = 0.0
            state["_headless_escape_until"] = 0.0
            state["_headless_recovery_active"] = False
            state["_headless_dock_clearance_origin"] = None
            # Webots initially schedules from the Supervisor snapshot.  Once
            # a robot becomes active, its first controller status packet is
            # authoritative and replaces that value with Random(robot_id)'s
            # battery.  Preserve supplied active snapshots, which already
            # represent a running controller, but reproduce the hand-off for
            # the normal IDLE episode start.
            state["_headless_controller_battery"] = random.Random(
                int(rid)).uniform(INITIAL_BATTERY_MIN, BATTERY_CAPACITY)
            state["_headless_controller_battery_synced"] = (
                state["state"] != RobotState.IDLE)
            if not math.isfinite(float(state["battery"])):
                raise ValueError(f"robot {rid} has invalid battery")
            state["battery"] = float(np.clip(
                state["battery"], 0.0, BATTERY_CAPACITY))

        # Webots registers every initial parking position before its first
        # scheduler tick.  Without the same static reservations, headless A*
        # can route through a robot that is physically occupying its home.
        for rid in self.robots:
            home = PARKING_SPOTS.get(rid)
            if home is not None:
                self.coordinator.reserve_home(rid, home)

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
            "joint_runtime": self.config.joint_runtime,
            "route_planner": (
                "rolling_joint_grid" if self.config.joint_runtime else
                "per_robot_lifelong"),
            "peer_avoidance": (
                "analytic_joint_predictive_shield" if self.config.joint_runtime
                else "legacy_deadlock_monitor"),
            "joint_candidate_budget_seconds": (
                self.config.joint_candidate_budget_seconds),
            "joint_candidate_max_budget_seconds": (
                self.config.joint_candidate_max_budget_seconds),
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
            and float(task.arrival_time) < self.config.episode_end_time
        ]

    def has_future_arrivals(self) -> bool:
        return any(
            task.status == TaskStatus.PENDING
            and float(task.arrival_time) > self.current_time + 1e-9
            and float(task.arrival_time) < self.config.episode_end_time
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
        if self.current_time + 1e-9 >= self.config.episode_end_time:
            return True
        if self.config.fixed_horizon:
            return False
        pending = any(
            task.status == TaskStatus.PENDING
            and task.arrival_time < self.config.episode_end_time
            for task in self.tasks)
        assigned = any(task.status in (TaskStatus.ASSIGNED,
                                       TaskStatus.IN_PROGRESS)
                       for task in self.tasks)
        return not pending and not assigned and not self.has_active_execution()

    def _expire_failed_pairs(self) -> None:
        self._failed_pairs = {
            pair: expiry for pair, expiry in self._failed_pairs.items()
            if expiry > self.current_time
        }

    def _plan(self, rid: int, goal, *,
              fallback_priority: Optional[float] = None,
              release_grid_before_fallback: bool = False
              ) -> Optional[List[Tuple[float, float]]]:
        state = self.robots[rid]
        path = self.coordinator.plan_grid_lifelong(
            rid, state["position"], goal)
        if path is None:
            if release_grid_before_fallback:
                self.coordinator.release_robot_grid(rid)
            priority = (getattr(state.get("current_task"), "priority", 0.0)
                        if fallback_priority is None else fallback_priority)
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
                      path: List[Tuple[float, float]], *,
                      activate_joint: bool = False,
                      waypoint_offsets: Optional[List[float]] = None,
                      activation_time: Optional[float] = None,
                      preserve_pending: bool = False,
                      partial: bool = False,
                      recovery: bool = False,
                      source: Optional[str] = None) -> None:
        if not activate_joint and not preserve_pending:
            self._abort_pending_joint_transaction(rid)
            # A business-leg successor retires the fleet epoch in Webots;
            # controllers retain their old safe prefix until replacement.
            self._active_joint_members.clear()
        self._waypoints[rid] = list(path)
        self._waypoint_index[rid] = 0
        self.robots[rid]["_headless_joint_plan_partial"] = bool(
            activate_joint and partial)
        self.robots[rid]["_headless_recovery_active"] = bool(recovery)
        self.robots[rid]["_headless_plan_source"] = (
            source if source is not None else
            "joint_grid_transaction" if activate_joint else
            "business_intent")
        self.robots[rid]["_headless_waypoint_tolerance"] = (
            self.config.waypoint_tolerance
            if self.robots[rid]["_headless_plan_source"] ==
            "joint_grid_transaction"
            else self.config.direct_waypoint_tolerance)
        self.robots[rid]["_headless_any_wait_since"] = None
        # RobotNavigator.set_waypoints() resets its reactive emergency latch.
        self.robots[rid]["_headless_peer_stop_latched"] = False
        if activate_joint:
            activation = (self.current_time + self.config.joint_activation_delay
                          if activation_time is None else
                          float(activation_time))
            offsets = list(waypoint_offsets or [0.0] * len(path))
            if len(offsets) != len(path):
                raise ValueError("joint waypoint offsets do not match path")
            self._waypoint_not_before[rid] = [
                activation + max(0.0, float(offset)) for offset in offsets]
            self._dispatch_not_before[rid] = activation
            self.robots[rid]["_headless_joint_route_active"] = True
        else:
            self._waypoint_not_before[rid] = [self.current_time] * len(path)
            delay = max(0.0, float(self.coordinator.get_dispatch_delay(rid)))
            self._dispatch_not_before[rid] = self.current_time + delay
            self.robots[rid]["_headless_joint_route_active"] = bool(
                not self.config.joint_runtime)
            if self.config.joint_runtime:
                # Webots treats ordinary task/home/charge routes as business
                # intent only and pulls the all-active planner forward to the
                # current tick.  No independent route may start in joint mode.
                self._next_joint_plan = min(
                    self._next_joint_plan, self.current_time)

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
        state = self.robots[rid]
        pickup_distance = math.hypot(
            state["position"][0] - task.pickup_position[0],
            state["position"][1] - task.pickup_position[1])
        pickup_already_reached = pickup_distance <= GOAL_TOLERANCE * 2.5
        route_goal = (task.delivery_location if pickup_already_reached
                      else task.pickup_location)
        path = self._plan(rid, route_goal)
        if not path:
            self.coordinator.rollback_robot_plan(rid)
            self._failed_pairs[(rid, task.task_id)] = (
                self.current_time + self.config.assignment_failure_ttl)
            self.telemetry["assignments_rejected"] += 1
            return {"type": "assignment_rejected",
                    "reason": ("delivery_path_unreachable_at_pickup"
                               if pickup_already_reached else
                               "pickup_path_unreachable"),
                    "robot_id": rid, "task_id": task.task_id}

        provider = self.context.path_cost_provider
        ideal_distance = None
        if provider is not None:
            ideal_distance = float(provider(rid, task))
            if not math.isfinite(ideal_distance) or ideal_distance < 0:
                self.coordinator.rollback_robot_plan(rid)
                self.telemetry["assignments_rejected"] += 1
                return {"type": "assignment_rejected",
                        "reason": "invalid_ideal_distance",
                        "robot_id": rid, "task_id": task.task_id}
        elif assignment.estimated_cost is not None:
            ideal_distance = float(assignment.estimated_cost)
        self.coordinator.release_home(rid)
        task.status = (TaskStatus.IN_PROGRESS if pickup_already_reached
                       else TaskStatus.ASSIGNED)
        task.assigned_robot = rid
        # Arrival ticks are generated by multiplication while runtime ticks
        # are accumulated.  Canonicalize sub-nanosecond drift so a task made
        # available by the tolerance can never appear assigned before arrival.
        task.assignment_time = max(
            float(self.current_time), float(task.arrival_time))
        if pickup_already_reached:
            task.pickup_time = self.current_time
        task.ideal_distance = ideal_distance
        task.actual_distance = 0.0
        task.excess_distance_cursor = 0.0
        task.rewarded_excess_distance_cursor = 0.0
        state["current_task"] = task
        state["has_task"] = True
        state["state"] = (RobotState.EN_ROUTE_DELIVERY
                          if pickup_already_reached else
                          RobotState.EN_ROUTE_PICKUP)
        state["goal_location"] = route_goal
        self._install_path(rid, path)
        self._failed_pairs.pop((rid, task.task_id), None)
        self.telemetry["assignments_committed"] += 1
        return {
            "type": "assignment_committed", "reason": "ok",
            "robot_id": rid, "task_id": task.task_id,
            "ideal_distance": ideal_distance,
            "planned_pickup_distance": (
                0.0 if pickup_already_reached else
                _polyline_length(state["position"], path)),
            "pickup_already_reached": pickup_already_reached,
            "pickup_proximity_distance": pickup_distance,
        }

    def _clear_path(self, rid: int, *, release: bool = True) -> None:
        self._abort_pending_joint_transaction(rid)
        self._waypoints[rid] = []
        self._waypoint_index[rid] = 0
        self._waypoint_not_before[rid] = []
        self.robots[rid]["_headless_joint_plan_partial"] = False
        self.robots[rid]["_headless_recovery_active"] = False
        self.robots[rid]["_headless_plan_source"] = None
        self.robots[rid]["_headless_waypoint_tolerance"] = (
            self.config.direct_waypoint_tolerance)
        self.robots[rid]["_headless_any_wait_since"] = None
        self.robots[rid]["_headless_joint_route_active"] = False
        self._active_joint_members.discard(rid)
        self.coordinator.clear_robot_path(rid)
        if release:
            self.coordinator.release_lifelong(rid)
            self.coordinator.release_robot_grid(rid)
        self.robots[rid]["speed_scale"] = 1.0
        self.robots[rid]["_headless_shield_until"] = 0.0
        self.robots[rid]["_headless_peer_stop_latched"] = False

    def _abort_pending_joint_transaction(self, rid: Optional[int] = None
                                         ) -> bool:
        pending = getattr(self, "_pending_joint_transaction", None)
        if pending is None:
            return False
        if rid is not None and rid not in pending["plans"]:
            return False
        self._pending_joint_transaction = None
        self.telemetry["joint_transactions_aborted"] += 1
        self._next_joint_plan = min(self._next_joint_plan, self.current_time)
        return True

    def _activate_pending_joint_transaction(self) -> List[dict]:
        """Atomically swap prepared routes after Webots' 0.5 s barrier."""
        pending = self._pending_joint_transaction
        if (pending is None or self.current_time + 1e-9 <
                pending["activate_at"]):
            return []
        for rid, signature in pending["goals"].items():
            state = self.robots.get(rid)
            current_goal = (self._goal_position(state.get("goal_location"))
                            if state is not None else None)
            if (state is None or
                    not self._is_moving_state(state.get("state")) or
                    current_goal != signature):
                epoch = pending["epoch"]
                self._abort_pending_joint_transaction()
                return [{"type": "joint_plan_aborted", "epoch": epoch,
                         "reason": "stale_business_goal"}]
        self._pending_joint_transaction = None
        for rid in sorted(pending["plans"]):
            self._install_path(
                rid, pending["plans"][rid], activate_joint=True,
                waypoint_offsets=pending["offsets"][rid],
                activation_time=self.current_time, preserve_pending=True,
                partial=pending["partial"].get(rid, False))
        self._active_joint_members = set(pending["plans"])
        self._last_joint_activation_time = self.current_time
        self.telemetry["joint_plans_activated"] += 1
        return [{
            "type": "joint_plan_activated",
            "epoch": pending["epoch"],
            "robot_ids": sorted(pending["plans"]),
            "time_slot_seconds": pending["time_slot_seconds"],
        }]

    def _prepare_joint_transaction(self, plans: Dict[int, List[tuple]],
                                   offsets: Dict[int, List[float]], *,
                                   time_slot_seconds: float,
                                   fallback: bool = False,
                                   partial: Optional[Dict[int, bool]] = None
                                   ) -> List[dict]:
        if not plans or self._pending_joint_transaction is not None:
            return []
        self._joint_epoch += 1
        self._pending_joint_transaction = {
            "epoch": self._joint_epoch,
            "activate_at": (
                self.current_time + self.config.joint_activation_delay),
            "plans": plans,
            "offsets": offsets,
            "partial": dict(partial or {}),
            "goals": {
                rid: self._goal_position(
                    self.robots[rid].get("goal_location"))
                for rid in plans},
            "time_slot_seconds": float(time_slot_seconds),
        }
        self.telemetry["joint_plans_prepared"] += 1
        if fallback:
            self.telemetry["joint_fallbacks_prepared"] += 1
        return [{
            "type": ("joint_fallback_prepared" if fallback else
                     "joint_plan_prepared"),
            "epoch": self._joint_epoch,
            "robot_ids": sorted(plans),
            "activate_at": self._pending_joint_transaction["activate_at"],
            "time_slot_seconds": float(time_slot_seconds),
        }]

    def _prepare_joint_fallback(self, agents) -> List[dict]:
        """Atomically activate safe per-robot routes as a liveness backstop."""
        plans = {}
        offsets = {}
        partial = {}
        for rid in sorted(agents, key=self._priority_keep_key,
                          reverse=True):
            start, planning_goal = agents[rid]
            path = self.coordinator.plan_grid_lifelong(
                rid, start, planning_goal)
            if not path:
                continue
            points = [tuple(point) for point in path]
            if (not points or math.hypot(
                    points[-1][0] - start[0],
                    points[-1][1] - start[1]) <
                    self.config.minimum_navigation_displacement):
                continue
            if math.hypot(points[0][0] - start[0],
                          points[0][1] - start[1]) > 0.18:
                points.insert(0, tuple(start))
            plans[rid] = points
            offsets[rid] = [0.0] * len(points)
            true_goal = self._goal_position(
                self.robots[rid].get("goal_location"))
            partial[rid] = bool(
                true_goal is None or math.hypot(
                    points[-1][0] - true_goal[0],
                    points[-1][1] - true_goal[1]) > GOAL_TOLERANCE * 2.0)
        return self._prepare_joint_transaction(
            plans, offsets, time_slot_seconds=0.0, fallback=True,
            partial=partial)

    def _reserve_idle_position(self, rid: int) -> None:
        """Mirror Webots' static reservation for a stopped idle robot."""
        node = self.coordinator.graph.get_nearest_node(
            self.robots[rid]["position"])
        if node:
            self.coordinator.lifelong.reserve_static(rid, node)

    def _goal_position(self, goal) -> Optional[Tuple[float, float]]:
        if isinstance(goal, (tuple, list)) and len(goal) >= 2:
            return float(goal[0]), float(goal[1])
        if goal in ALL_LOCATIONS:
            return tuple(ALL_LOCATIONS[goal])
        if goal in CHARGING_STATIONS:
            return tuple(CHARGING_STATIONS[goal])
        if goal in WAYPOINTS:
            return tuple(WAYPOINTS[goal])
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
            # FactorySupervisor assigns charging fallback routes the explicit
            # lowest scheduling priority (-1000 in coordinator storage).
            path = self._plan(rid, station, fallback_priority=1000.0)
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
        self._reserve_idle_position(rid)
        events.extend(self._send_to_charging(rid))
        return events

    def _update_battery(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        events = []
        swap_at = state.get("_headless_swap_ready_at")
        if (swap_at is not None and self.current_time + 1e-9 >= float(swap_at)
                and state.get("state") == RobotState.CHARGING):
            state["battery"] = float(self._battery_rng.uniform(
                FULL_BATTERY_THRESHOLD, BATTERY_CAPACITY))
            state["_headless_controller_battery"] = state["battery"]
            state["_headless_controller_battery_synced"] = True
            state["_headless_swap_ready_at"] = None
            state["state"] = RobotState.IDLE
            state["goal_location"] = None
            self.telemetry["charge_swaps"] += 1
            return [{"type": "charge_swap_completed", "robot_id": rid}]

        if (not state.get("_headless_controller_battery_synced", True) and
                state.get("state") in {
                    RobotState.EN_ROUTE_PICKUP, RobotState.CARRYING,
                    RobotState.EN_ROUTE_DELIVERY, RobotState.RETURNING_HOME}):
            state["battery"] = float(
                state["_headless_controller_battery"])
            state["_headless_controller_battery_synced"] = True

        if state.get("state") in {
                RobotState.EN_ROUTE_PICKUP, RobotState.CARRYING,
                RobotState.EN_ROUTE_DELIVERY, RobotState.RETURNING_HOME}:
            state["battery"] = max(
                0.0, float(state["battery"])
                - BATTERY_DRAIN_RATE * self._tick_seconds)
            state["_headless_controller_battery"] = state["battery"]

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
        # A completed pickup leg must release its retained grid reservation
        # before the legacy A* fallback.  This is the exact Webots transition;
        # otherwise standalone plans around a route that no longer exists.
        path = self._plan(
            rid, task.delivery_location,
            release_grid_before_fallback=True)
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
        completed_location = task.delivery_location
        task.status = TaskStatus.COMPLETED
        task.completion_time = self.current_time
        self._completed_ids.add(task.task_id)
        state["tasks_completed"] = int(state.get("tasks_completed", 0)) + 1
        state["current_task"] = None
        state["has_task"] = False
        state["goal_location"] = None
        state["state"] = RobotState.IDLE
        self._clear_path(rid)
        self._reserve_idle_position(rid)
        self.telemetry["tasks_completed"] += 1
        events = [{"type": "task_completed", "robot_id": rid,
                   "task_id": task.task_id}]
        events.extend(task.deadline_events(self.current_time))
        if state["battery"] < LOW_BATTERY_THRESHOLD:
            events.extend(self._send_to_charging(rid))
        elif self._dock_clearance_required(rid, completed_location):
            state["_headless_dock_clearance_origin"] = tuple(
                ALL_LOCATIONS[completed_location])
            relocation = self._relocate_idle_robot(rid)
            events.extend(relocation)
            if not relocation:
                state["_headless_dock_clearance_origin"] = None
        return events

    def _handle_goal_reached(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        goal = self._goal_position(state.get("goal_location"))
        if goal is not None:
            distance = math.hypot(state["position"][0] - goal[0],
                                  state["position"][1] - goal[1])
            multiplier = 2.0 if self.config.joint_runtime else 2.5
            if distance > GOAL_TOLERANCE * multiplier:
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
            state["_headless_dock_clearance_origin"] = None
            self._clear_path(rid)
            self._reserve_idle_position(rid)
            return [{"type": "robot_idle", "robot_id": rid}]
        return []

    def _move_robot(self, rid: int) -> List[dict]:
        state = self.robots[rid]
        waypoints = self._waypoints[rid]
        index = self._waypoint_index[rid]
        if (not waypoints or index >= len(waypoints) or
                self.current_time + 1e-9 < self._dispatch_not_before[rid] or
                self.current_time + 1e-9 <
                    float(state.get("_headless_hold_until", 0.0)) or
                (self.config.joint_runtime and
                 not state.get("_headless_joint_route_active", False))):
            return []
        not_before = (self._waypoint_not_before[rid][index]
                      if index < len(self._waypoint_not_before[rid])
                      else self._dispatch_not_before[rid])
        target = waypoints[index]
        x, y = state["position"]
        dx, dy = target[0] - x, target[1] - y
        distance = math.hypot(dx, dy)
        # RobotNavigator's coordinated-mode peer guard is directional: a peer
        # inside 0.55 m stops this robot only when it lies ahead of the current
        # waypoint vector.  The Webots controller then raises one replan
        # request; pulling the fleet tick forward is the headless equivalent.
        dangerous_radial_peer = any(
            math.hypot(peer["position"][0] - x,
                       peer["position"][1] - y) <
                self.config.controller_peer_stop_distance
            and dx * (peer["position"][0] - x) +
                dy * (peer["position"][1] - y) >= 0.0
            for peer_id, peer in self.robots.items()
            if peer_id != rid)
        if dangerous_radial_peer:
            # Webots raises _replan_requested only on entry into emergency
            # stop. Keeping this edge-triggered is essential: a level-triggered
            # request replans every 16 ms while the same peer remains present.
            if not state.get("_headless_peer_stop_latched", False):
                state["_headless_peer_stop_latched"] = True
                self.telemetry["controller_peer_stop_entries"] += 1
                self._next_joint_plan = min(
                    self._next_joint_plan, self.current_time)
                self._joint_liveness_needed = True
            return []
        state["_headless_peer_stop_latched"] = False
        speed_scale = float(np.clip(
            state.get("speed_scale", 1.0), 0.4, 1.0))
        max_step = self.config.linear_speed * speed_scale * self._tick_seconds
        waypoint_tolerance = float(state.get(
            "_headless_waypoint_tolerance",
            self.config.direct_waypoint_tolerance))
        if distance <= max(waypoint_tolerance, max_step):
            # RobotNavigator's slot time constrains waypoint *completion*, not
            # departure toward that waypoint. It drives during the slot and
            # waits only after reaching the cell early.
            if self.current_time + 1e-9 < not_before:
                return []
            if (self.config.joint_runtime and
                    state.get("_headless_joint_plan_partial", False) and
                    index == len(waypoints) - 1):
                # RobotNavigator holds a rolling-window endpoint for the next
                # joint epoch; it is not a business-goal arrival.
                return []
            # The real controller reports a waypoint reached while physically
            # inside its tolerance; it does not teleport to the coordinate.
            moved = 0.0
            self._waypoint_index[rid] += 1
            if not self.config.joint_runtime:
                self.coordinator.advance_robot(rid)
        elif max_step > 0.0:
            desired_heading = math.atan2(dy, dx)
            heading = float(state.get("heading", 0.0))
            heading_error = (desired_heading - heading + math.pi) % (
                2.0 * math.pi) - math.pi
            if abs(heading_error) > self.config.heading_tolerance:
                # Match RobotNavigator's joint-coordinated rotation branch,
                # including its final 1.09 motor calibration and Webots'
                # per-wheel maxVelocity clamp.
                desired_angular = float(np.clip(
                    heading_error * self.config.heading_rotate_gain,
                    -self.config.max_angular_speed,
                    self.config.max_angular_speed)) * speed_scale
                _linear_speed, angular_speed = (
                    self._wheel_limited_body_velocity(
                        0.0, desired_angular))
                state["heading"] = (heading +
                    angular_speed * self._tick_seconds + math.pi) % (
                        2.0 * math.pi) - math.pi
                moved = 0.0
            else:
                # Joint-coordinated Webots navigation uses its dedicated
                # centreline follower, not the slower normal-navigation law.
                desired_linear = min(
                    self.config.linear_speed,
                    distance * self.config.distance_speed_gain) * speed_scale
                desired_angular = (
                    heading_error * self.config.heading_drive_gain *
                    speed_scale)
                linear_speed, angular_speed = (
                    self._wheel_limited_body_velocity(
                        desired_linear, desired_angular))
                next_heading = (heading +
                    angular_speed * self._tick_seconds + math.pi) % (
                        2.0 * math.pi) - math.pi
                moved = min(distance, linear_speed * self._tick_seconds)
                proposed = (
                    x + math.cos(next_heading) * moved,
                    y + math.sin(next_heading) * moved,
                )
                peer_distance = min((
                    math.hypot(proposed[0] - peer["position"][0],
                               proposed[1] - peer["position"][1])
                    for peer_id, peer in self.robots.items()
                    if peer_id != rid), default=float("inf"))
                if peer_distance < self.config.hard_minimum_distance:
                    # Analytic replacement for the controller's final
                    # peer/LiDAR safety envelope.  Never integrate a pose that
                    # crosses the 0.50 m physical collision threshold.
                    moved = 0.0
                    state["_headless_hold_until"] = max(
                        float(state.get("_headless_hold_until", 0.0)),
                        self.current_time + self.config.avoidance_scan_seconds)
                    self._next_joint_plan = min(
                        self._next_joint_plan, self.current_time)
                    self._joint_liveness_needed = True
                else:
                    state["position"] = proposed
                    state["heading"] = next_heading
        else:
            moved = 0.0
        state["total_distance"] = float(
            state.get("total_distance", 0.0)) + moved
        self.telemetry["distance_travelled"] += moved
        events = []
        task = state.get("current_task")
        if task is not None and moved > 0 and task.ideal_distance is not None:
            task.actual_distance += moved
            excess = max(0.0, task.actual_distance - task.ideal_distance)
            delta = excess - task.excess_distance_cursor
            if delta < -1e-9:
                raise ValueError("task excess distance cursor moved backwards")
            if delta > 0:
                task.excess_distance_cursor = excess
        if self._waypoint_index[rid] >= len(waypoints):
            state["_headless_joint_route_active"] = False
            self._active_joint_members.discard(rid)
            if state.get("_headless_recovery_active", False):
                state["_headless_recovery_active"] = False
                self._next_joint_plan = min(
                    self._next_joint_plan, self.current_time)
                events.append({"type": "joint_escape_completed",
                               "robot_id": rid})
            else:
                events.extend(self._handle_goal_reached(rid))
        return events

    def _wheel_limited_body_velocity(self, linear_speed: float,
                                     angular_speed: float
                                     ) -> Tuple[float, float]:
        """Apply Webots' differential-wheel velocity saturation exactly."""
        half_base = self.config.wheel_base / 2.0
        maximum_linear_wheel = (
            self.config.max_wheel_speed * self.config.wheel_radius)
        left = float(np.clip(
            linear_speed - angular_speed * half_base,
            -maximum_linear_wheel, maximum_linear_wheel))
        right = float(np.clip(
            linear_speed + angular_speed * half_base,
            -maximum_linear_wheel, maximum_linear_wheel))
        return (
            (left + right) / 2.0,
            (right - left) / self.config.wheel_base,
        )

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

    @staticmethod
    def _is_moving_state(state) -> bool:
        return state in {
            RobotState.EN_ROUTE_PICKUP, RobotState.CARRYING,
            RobotState.EN_ROUTE_DELIVERY, RobotState.RETURNING_HOME,
            RobotState.RETURNING_TO_CHARGE,
        }

    def _priority_keep_key(self, rid: int) -> tuple:
        state = self.robots[rid]
        clearance_origin = state.get("_headless_dock_clearance_origin")
        dock_clearance = bool(
            clearance_origin is not None and
            state.get("state") == RobotState.RETURNING_HOME and
            math.hypot(
                state["position"][0] - clearance_origin[0],
                state["position"][1] - clearance_origin[1]) <
                YIELD_RESUME_MIN_CLEARANCE)
        exempt = int(
            state.get("state") == RobotState.RETURNING_TO_CHARGE or
            dock_clearance or
            float(state.get("battery", BATTERY_CAPACITY)) <
                LOW_BATTERY_THRESHOLD)
        priority = float(getattr(
            state.get("current_task"), "priority", 0.0) or 0.0)
        return exempt, priority, float(rid)

    def _priority_pair(self, first: int, second: int) -> Optional[tuple]:
        """Return Webots' priority-safe (winner, yielder) pair."""
        if first not in self.robots or second not in self.robots:
            return None
        winner, yielder = (
            (first, second)
            if self._priority_keep_key(first) >= self._priority_keep_key(second)
            else (second, first))
        if not self._priority_keep_key(yielder)[0]:
            return winner, yielder
        winner, yielder = yielder, winner
        if self._priority_keep_key(yielder)[0]:
            return None
        return winner, yielder

    def _priority_ordered(self, winner: int,
                          yielder: int) -> Optional[tuple]:
        """Honour geometry unless its selected yielder is safety-exempt."""
        if winner not in self.robots or yielder not in self.robots:
            return None
        if not self._priority_keep_key(yielder)[0]:
            return winner, yielder
        if not self._priority_keep_key(winner)[0]:
            return yielder, winner
        return None

    def _priority_select(self, component, conflicts=None) -> Optional[tuple]:
        component = tuple(sorted(component))
        if len(component) < 2:
            return None
        pairs = set()
        if conflicts:
            pairs = {
                tuple(sorted((int(first), int(second))))
                for first, second, *_rest in conflicts
                if first in component and second in component}
        if not pairs:
            pairs = set(itertools.combinations(component, 2))
        ranked = []
        for first, second in pairs:
            selection = self._priority_pair(first, second)
            if selection is None:
                continue
            winner, yielder = selection
            distance = math.hypot(
                self.robots[first]["position"][0] -
                    self.robots[second]["position"][0],
                self.robots[first]["position"][1] -
                    self.robots[second]["position"][1])
            ranked.append((self._priority_keep_key(yielder), distance,
                           winner, yielder))
        if not ranked:
            return None
        ranked.sort(key=lambda row: (row[0], row[1]))
        return ranked[0][2], ranked[0][3]

    def _joint_staging_goal(self, goal_xy, robot_position, peer_positions,
                            reserved_positions=None):
        """Standalone copy of Webots' duplicate-goal staging rule."""
        grid = self.coordinator.grid
        reserved = list(reserved_positions or ())
        candidates = []
        for radius in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5):
            for step in range(32):
                angle = 2.0 * math.pi * step / 32.0
                target = (goal_xy[0] + radius * math.cos(angle),
                          goal_xy[1] + radius * math.sin(angle))
                col, row = grid.world_to_grid(*target)
                if not grid.in_bounds(col, row) or not grid.is_free(col, row):
                    continue
                nearby = [
                    math.hypot(target[0] - peer[0], target[1] - peer[1])
                    for peer in list(peer_positions) + reserved]
                clearance = min(nearby, default=float("inf"))
                if clearance < 0.80:
                    continue
                distance = math.hypot(
                    target[0] - robot_position[0],
                    target[1] - robot_position[1])
                candidates.append((
                    clearance + 0.4 * radius - 0.05 * distance, target))
        if not candidates:
            return goal_xy
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _joint_candidate_cache_key(self, planning_agents, priority_order,
                                   budget: float) -> tuple:
        """Return the exact discrete input consumed by JointGridPlanner."""
        grid = self.coordinator.grid
        endpoints = tuple(
            (int(rid), grid.world_to_grid(*start), grid.world_to_grid(*goal))
            for rid, (start, goal) in sorted(planning_agents.items()))
        return (endpoints, tuple(priority_order), round(float(budget), 6))

    def _cached_joint_candidate(self, planning_agents, priority_order,
                                budget: float):
        """Plan once per discrete state and reuse validated positive results."""
        key = self._joint_candidate_cache_key(
            planning_agents, priority_order, budget)
        if key in self._joint_candidate_cache:
            candidate = self._joint_candidate_cache.pop(key)
            self._joint_candidate_cache[key] = candidate
            self.telemetry["joint_plan_cache_hits"] += 1
            if candidate is None:
                self.telemetry["joint_plan_negative_cache_hits"] += 1
            return candidate
        self.telemetry["joint_plan_cache_misses"] += 1
        planner_by_tier = {
            "wide": self.coordinator.joint_grid_planner_wide,
            "primary": self.coordinator.joint_grid_planner,
            "short": self.coordinator.joint_grid_planner_short,
            "soft": self.coordinator.joint_grid_planner_soft,
        }
        attempts_before = {
            name: int(self.coordinator.joint_grid_tier_statistics[name][
                "attempted"])
            for name in planner_by_tier}
        started = time.perf_counter()
        candidate = self.coordinator.plan_joint_grid_candidate(
            planning_agents, max_seconds=budget,
            priority_order=priority_order)
        self._joint_planning_wall_seconds += (
            time.perf_counter() - started)
        attempted_reasons = [
            planner_by_tier[name].last_failure_reason
            for name in planner_by_tier
            if int(self.coordinator.joint_grid_tier_statistics[name][
                "attempted"]) > attempts_before[name]]
        deterministic_failure = bool(
            candidate is None and attempted_reasons and
            all(reason in {
                "no_solution", "expansion_budget", "expansion_limit"}
                for reason in attempted_reasons))
        if candidate is not None or deterministic_failure:
            self._joint_candidate_cache[key] = candidate
            if len(self._joint_candidate_cache) > self._joint_candidate_cache_limit:
                self._joint_candidate_cache.pop(
                    next(iter(self._joint_candidate_cache)))
        return candidate

    def _refresh_joint_grid_candidate(self) -> List[dict]:
        """Use the same rolling space-time planner as Webots, without radio."""
        if self._pending_joint_transaction is not None:
            # Webots permits only one prepare/arm/commit transaction at a
            # time.  The deterministic standalone barrier has the same
            # single-writer contract.
            return []
        unfinished_epoch = any(
            rid in self.robots and
            self.robots[rid].get("_headless_joint_route_active") and
            self._waypoint_index[rid] < len(self._waypoints[rid])
            for rid in self._active_joint_members)
        if (unfinished_epoch and not self._force_joint_replan and
                self.current_time <
                self._last_joint_activation_time + 2.0):
            # Exact counterpart of Webots' _joint_replan_min_interval.
            # Without this gate, a predictive conflict can prepare a new
            # epoch every 0.5 s and reset timed slots before motion occurs.
            return []
        self._force_joint_replan = False
        agents = {}
        for rid, state in self.robots.items():
            if not self._is_moving_state(state.get("state")):
                continue
            goal = self._goal_position(state.get("goal_location"))
            if goal is not None:
                agents[rid] = (state["position"], goal)
        if not agents:
            return []

        grouped_goals = {}
        for rid, (_start, goal) in agents.items():
            grouped_goals.setdefault(goal, []).append(rid)
        planning_agents = {}
        for goal, robot_ids in grouped_goals.items():
            primary = max(robot_ids, key=self._priority_keep_key)
            followers = sorted(
                (rid for rid in robot_ids if rid != primary),
                key=self._priority_keep_key, reverse=True)
            assigned_staging = []
            for rid in [primary] + followers:
                start = agents[rid][0]
                if rid == primary:
                    planning_agents[rid] = (start, goal)
                    continue
                peers = [self.robots[peer]["position"]
                         for peer in agents if peer != rid]
                staging = self._joint_staging_goal(
                    goal, start, peers, assigned_staging)
                if staging != goal:
                    assigned_staging.append(staging)
                planning_agents[rid] = (start, staging)

        failures = self._joint_candidate_failures
        budget = min(
            self.config.joint_candidate_max_budget_seconds,
            self.config.joint_candidate_budget_seconds + failures * 0.05)
        # FactorySupervisor gives the space-time planner active navigation
        # members only.  Adding idle/charging robots here as permanent wait
        # agents looked conservative, but changed both search complexity and
        # corridor feasibility in dense scenario C.  The controller-level
        # 0.55 m radial guard below remains the stationary-peer safety layer,
        # matching Webots' separation of planner and controller duties.
        priority_order = tuple(sorted(
            agents, key=self._priority_keep_key, reverse=True))
        self.telemetry["joint_plans_attempted"] += 1
        candidate = self._cached_joint_candidate(
            planning_agents, priority_order, budget)
        if candidate is None:
            self._joint_candidate_failures += 1
            self.telemetry["joint_plans_failed"] += 1
            events = [{"type": "joint_plan_failed",
                       "active_robots": sorted(agents)}]
            # Webots preserves the last committed joint prefix on a transient
            # search miss. Per-robot fallback is allowed only after its
            # collision/watchdog layers declare a liveness risk.
            if self._joint_liveness_needed:
                fallback_events = self._prepare_joint_fallback(
                    planning_agents)
                events.extend(fallback_events)
                if fallback_events:
                    self._joint_liveness_needed = False
            return events

        self._joint_candidate_failures = 0
        omitted = set(getattr(candidate, "omitted_robots", set()))
        for rid in omitted:
            if rid in self.robots:
                self.robots[rid]["_headless_hold_until"] = max(
                    float(self.robots[rid].get(
                        "_headless_hold_until", 0.0)),
                    self.current_time + 0.60)
        if omitted:
            self._next_joint_plan = min(
                self._next_joint_plan,
                self.current_time + 0.50)
        grid = self.coordinator.grid
        plans = {}
        offsets_by_robot = {}
        partial_by_robot = {}
        for rid, timed_path in candidate.paths.items():
            if rid not in agents:
                continue
            raw_points = [grid.grid_to_world(*step.cell)
                          for step in timed_path[1:]]
            raw_offsets = [
                max(0.0, (step.time_slot - 1) *
                    candidate.time_slot_seconds)
                for step in timed_path[1:]]
            points = []
            offsets = []
            if raw_points:
                first = raw_points[0]
                current = self.robots[rid]["position"]
                dx, dy = first[0] - current[0], first[1] - current[1]
                distance = math.hypot(dx, dy)
                if distance > 0.18:
                    ratio = min(0.10, distance * 0.45) / distance
                    points.append((current[0] + dx * ratio,
                                   current[1] + dy * ratio))
                    offsets.append(0.0)
            points.extend(raw_points)
            offsets.extend(raw_offsets)
            if not points:
                continue
            final = points[-1]
            current = self.robots[rid]["position"]
            if math.hypot(final[0] - current[0],
                          final[1] - current[1]) < 0.05:
                continue
            plans[rid] = points
            offsets_by_robot[rid] = offsets
            business_goal = agents[rid][1]
            partial_by_robot[rid] = bool(math.hypot(
                final[0] - business_goal[0],
                final[1] - business_goal[1]) > GOAL_TOLERANCE * 2.0)
        if not plans:
            events = [{"type": "joint_plan_empty",
                       "robot_ids": sorted(agents)}]
            if self._joint_liveness_needed:
                fallback_events = self._prepare_joint_fallback(agents)
                events.extend(fallback_events)
                if fallback_events:
                    self._joint_liveness_needed = False
            return events
        return self._prepare_joint_transaction(
            plans, offsets_by_robot,
            time_slot_seconds=candidate.time_slot_seconds,
            partial=partial_by_robot)

    def _trajectory_for_scale(self, rid: int, scale: float, *,
                              horizon: Optional[float] = None) -> List[tuple]:
        """Mirror FactorySupervisor._trajectory_for_scale sampling.

        The controller may approach a future timed waypoint before its slot;
        only a wait that is active *now* delays the Supervisor prediction.
        Applying every future waypoint deadline here used to manufacture a
        slow trajectory and hide conflicts from the standalone shield.
        """
        state = self.robots[rid]
        points = self._waypoints[rid]
        index = self._waypoint_index[rid]
        position = tuple(state["position"])
        now = self.current_time
        sample_dt = self.config.avoidance_sample_seconds
        prediction_horizon = (
            self.config.avoidance_horizon_seconds if horizon is None else
            float(horizon))
        start_at = max(
            self._dispatch_not_before[rid],
            float(state.get("_headless_hold_until", 0.0)))
        if index < len(points):
            current_target = points[index]
            current_deadline = (
                self._waypoint_not_before[rid][index]
                if index < len(self._waypoint_not_before[rid]) else now)
            at_current_target = math.hypot(
                position[0] - current_target[0],
                position[1] - current_target[1]) <= \
                float(state.get(
                    "_headless_waypoint_tolerance",
                    self.config.direct_waypoint_tolerance))
            if at_current_target and now < current_deadline:
                start_at = max(start_at, current_deadline)
            elif (at_current_target and
                  state.get("_headless_joint_plan_partial", False) and
                  index == len(points) - 1):
                # RobotNavigator reports a rolling 0.5 s endpoint wait in
                # each status packet; the next watchdog/epoch may replace it.
                start_at = max(start_at, now + 0.5)

        start_delay = max(
            0.0, min(start_at - now, prediction_horizon))
        speed = self.config.prediction_linear_speed * float(scale)
        if speed <= 0.0:
            samples = int(math.floor(
                prediction_horizon / sample_dt + 1e-9))
            return [position] * samples

        result = []
        cx, cy = position
        waypoint_index = index
        elapsed = start_delay
        next_sample = sample_dt
        while (next_sample <= start_delay + 1e-9 and
               next_sample <= prediction_horizon + 1e-9):
            result.append((cx, cy))
            next_sample += sample_dt

        while elapsed < prediction_horizon and waypoint_index < len(points):
            wx, wy = points[waypoint_index]
            dx, dy = wx - cx, wy - cy
            distance = math.hypot(dx, dy)
            if distance < 0.01:
                waypoint_index += 1
                continue
            travel_seconds = distance / speed
            while (next_sample <= elapsed + travel_seconds + 1e-9 and
                   next_sample <= prediction_horizon + 1e-9):
                fraction = (next_sample - elapsed) / travel_seconds
                result.append((cx + fraction * dx, cy + fraction * dy))
                next_sample += sample_dt
            elapsed += travel_seconds
            cx, cy = wx, wy
            waypoint_index += 1

        while next_sample <= prediction_horizon + 1e-9:
            result.append((cx, cy))
            next_sample += sample_dt
        return result

    def _motion_vector(self, rid: int) -> Tuple[float, float]:
        """Return Webots-equivalent intended motion direction."""
        state = self.robots[rid]
        points = self._waypoints[rid]
        for point in points[self._waypoint_index[rid]:]:
            dx = point[0] - state["position"][0]
            dy = point[1] - state["position"][1]
            distance = math.hypot(dx, dy)
            if distance > 0.05:
                return dx / distance, dy / distance
        goal = self._goal_position(state.get("goal_location"))
        if goal is not None:
            dx = goal[0] - state["position"][0]
            dy = goal[1] - state["position"][1]
            distance = math.hypot(dx, dy)
            if distance > 0.05:
                return dx / distance, dy / distance
        heading = float(state.get("heading", 0.0))
        return math.cos(heading), math.sin(heading)

    def _conflict_kind(self, component: tuple, conflicts=None) -> str:
        kinds = []
        pairs = ([(first, second) for first, second, *_rest in conflicts
                  if first in component and second in component]
                 if conflicts else
                 list(itertools.combinations(component, 2)))
        for first, second in pairs:
            a, b = self._motion_vector(first), self._motion_vector(second)
            dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
            kinds.append("same" if dot > 0.70 else
                         "head_on" if dot < -0.70 else "side")
        if "head_on" in kinds:
            return "head_on"
        if "side" in kinds:
            return "side"
        return "same"

    def _same_direction_pair(self, component: tuple) -> Optional[tuple]:
        """Return (follower, leader) using the production geometry rule."""
        for first, second in itertools.combinations(component, 2):
            a, b = self._motion_vector(first), self._motion_vector(second)
            if a[0] * b[0] + a[1] * b[1] <= 0.70:
                continue
            direction = (a[0] + b[0], a[1] + b[1])
            length = math.hypot(*direction)
            if length <= 1e-9:
                continue
            direction = (direction[0] / length, direction[1] / length)
            first_projection = sum(
                value * axis for value, axis in
                zip(self.robots[first]["position"], direction))
            second_projection = sum(
                value * axis for value, axis in
                zip(self.robots[second]["position"], direction))
            return ((first, second) if first_projection <= second_projection
                    else (second, first))
        selection = self._priority_select(component)
        if selection is None:
            return None
        winner, yielder = selection
        return yielder, winner

    def _side_crossing_pair(self, component: tuple) -> Optional[tuple]:
        """Return (winner, yielder) using Webots' crossing geometry rule."""
        component = tuple(sorted(component))
        best = None
        best_score = -2.0
        for first, second in itertools.combinations(component, 2):
            first_vector = self._motion_vector(first)
            second_vector = self._motion_vector(second)
            first_position = self.robots[first]["position"]
            second_position = self.robots[second]["position"]
            rel_x = second_position[0] - first_position[0]
            rel_y = second_position[1] - first_position[1]
            distance = math.hypot(rel_x, rel_y)
            if distance < 0.05:
                continue
            rel_x, rel_y = rel_x / distance, rel_y / distance
            ahead_first = first_vector[0] * rel_x + first_vector[1] * rel_y
            ahead_second = (
                second_vector[0] * -rel_x + second_vector[1] * -rel_y)
            score = max(ahead_first, ahead_second)
            if ahead_first >= ahead_second and ahead_first > 0.25:
                selection = self._priority_ordered(second, first)
            elif ahead_second > 0.25:
                selection = self._priority_ordered(first, second)
            else:
                continue
            if selection is None:
                continue
            winner, yielder = selection
            if score > best_score:
                best_score = score
                best = (winner, yielder)
        if best is not None:
            return best
        return self._priority_select(component)

    @staticmethod
    def _trajectory_pairs_conflicting(trajectories, minimum_distance,
                                      members=None):
        ids = sorted(trajectories)
        selected = set(members) if members is not None else None
        conflicts = []
        for offset, first in enumerate(ids):
            for second in ids[offset + 1:]:
                if selected is not None and not ({first, second} & selected):
                    continue
                length = min(len(trajectories[first]),
                             len(trajectories[second]))
                for index in range(length):
                    a, b = trajectories[first][index], trajectories[second][index]
                    distance = math.hypot(a[0] - b[0], a[1] - b[1])
                    if distance < minimum_distance:
                        conflicts.append((first, second, index, distance))
                        break
        return conflicts

    @staticmethod
    def _conflict_components(conflicts) -> List[tuple]:
        adjacency = {}
        for first, second, _index, _distance in conflicts:
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        components = []
        remaining = set(adjacency)
        while remaining:
            start = min(remaining)
            stack = [start]
            component = set()
            while stack:
                rid = stack.pop()
                if rid in component:
                    continue
                component.add(rid)
                stack.extend(adjacency.get(rid, ()))
            remaining.difference_update(component)
            components.append(tuple(sorted(component)))
        return components

    def _predictive_peer_avoidance(self) -> List[dict]:
        """Analytic counterpart of Webots' joint predictive speed shield."""
        active = [
            rid for rid, state in self.robots.items()
            if (self._is_moving_state(state.get("state")) and
                state.get("_headless_joint_route_active") and
                self._waypoint_index[rid] < len(self._waypoints[rid]))]
        for rid in active:
            state = self.robots[rid]
            if (state.get("speed_scale", 1.0) < 0.99 and
                    state.get("_headless_shield_until", 0.0) <=
                    self.current_time):
                state["speed_scale"] = 1.0
        if not active:
            return []
        active_set = set(active)
        trajectories = {
            rid: self._trajectory_for_scale(
                rid, float(self.robots[rid].get("speed_scale", 1.0)))
            for rid in active}
        actual_conflicts = []
        trajectory_ids = sorted(trajectories)
        for offset, first in enumerate(trajectory_ids):
            for second in trajectory_ids[offset + 1:]:
                if first not in active_set and second not in active_set:
                    continue
                distance = math.hypot(
                    self.robots[first]["position"][0] -
                        self.robots[second]["position"][0],
                    self.robots[first]["position"][1] -
                        self.robots[second]["position"][1])
                if distance < self.config.avoidance_minimum_distance:
                    actual_conflicts.append((first, second, 0, distance))
        if actual_conflicts:
            self.telemetry["predicted_peer_conflicts"] += len(
                actual_conflicts)
            events = []
            for component in self._conflict_components(actual_conflicts):
                moving = tuple(rid for rid in component if rid in active_set)
                if not moving:
                    continue
                kind = (self._conflict_kind(component)
                        if len(moving) == len(component) else "side")
                if kind == "same":
                    pair = self._same_direction_pair(component)
                    if pair is not None:
                        follower, leader = pair
                        self.robots[follower]["_headless_hold_until"] = max(
                            float(self.robots[follower].get(
                                "_headless_hold_until", 0.0)),
                            self.current_time + 0.45)
                        self.robots[follower]["speed_scale"] = 0.40
                        self.robots[leader]["speed_scale"] = 0.85
                        self.robots[follower]["_headless_shield_until"] = (
                            self.current_time + 1.5)
                        self.robots[leader]["_headless_shield_until"] = (
                            self.current_time + 1.0)
                        events.append({
                            "type": "same_direction_following",
                            "follower": follower, "leader": leader,
                        })
                        continue
                if kind == "side":
                    selection = self._side_crossing_pair(component)
                    if selection is not None:
                        winner, yielder = selection
                        deadline = self.current_time + 1.0
                        self.robots[yielder]["_headless_hold_until"] = max(
                            float(self.robots[yielder].get(
                                "_headless_hold_until", 0.0)), deadline)
                        self.robots[yielder]["speed_scale"] = 0.40
                        self.robots[yielder]["_headless_shield_until"] = deadline
                        self.robots[winner]["speed_scale"] = 1.0
                        self.robots[winner]["_headless_shield_until"] = deadline
                        for peer_id in moving:
                            if peer_id in (winner, yielder):
                                continue
                            self.robots[peer_id]["speed_scale"] = 0.60
                            self.robots[peer_id]["_headless_shield_until"] = deadline
                        events.append({
                            "type": "side_crossing_yield",
                            "robot_ids": list(component),
                            "winner": winner, "yielder": yielder,
                            "hold_seconds": 1.0,
                        })
                        continue
                closest = min(
                    distance for first, second, _index, distance
                    in actual_conflicts
                    if first in component and second in component)
                yielder = min(moving, key=self._priority_keep_key)
                selection = self._priority_select(
                    component, actual_conflicts)
                if selection is not None:
                    _winner, yielder = selection
                if closest < self.config.emergency_stop_distance:
                    if self._joint_try_escape_component(component):
                        events.append({
                            "type": "joint_escape_recovery",
                            "robot_ids": list(component),
                        })
                        continue
                    for rid in moving:
                        self.robots[rid]["_headless_hold_until"] = max(
                            float(self.robots[rid].get(
                                "_headless_hold_until", 0.0)),
                            self.current_time + 0.30)
                        self.robots[rid]["speed_scale"] = (
                            0.40 if rid == yielder else 0.65)
                        self.robots[rid]["_headless_shield_until"] = (
                            self.current_time + 2.0)
                    self.telemetry["emergency_peer_stops"] += 1
                    events.append({"type": "emergency_peer_stop",
                                   "robot_ids": list(component),
                                   "yielder": yielder})
                else:
                    profile = {
                        rid: (0.45 if rid == yielder else 0.75)
                        for rid in moving}
                    for rid, scale in profile.items():
                        self.robots[rid]["speed_scale"] = scale
                        self.robots[rid]["_headless_shield_until"] = (
                            self.current_time + 2.0)
                    events.append({"type": "actual_peer_speed_profile",
                                   "robot_ids": list(component),
                                   "profile": profile})
            self._next_joint_plan = min(
                self._next_joint_plan, self.current_time)
            if events:
                self._joint_liveness_needed = True
            return events
        conflicts = self._trajectory_pairs_conflicting(
            trajectories, self.config.avoidance_minimum_distance,
            active_set)
        if not conflicts:
            return []
        self.telemetry["predicted_peer_conflicts"] += len(conflicts)
        events = []
        for component in self._conflict_components(conflicts):
            moving = tuple(rid for rid in component if rid in active_set)
            if not moving:
                continue
            if (len(moving) == len(component) and
                    self._conflict_kind(component, conflicts) == "same"):
                pair = self._same_direction_pair(component)
                if pair is not None:
                    follower, leader = pair
                    self.robots[follower]["speed_scale"] = 0.55
                    self.robots[leader]["speed_scale"] = 1.0
                    self.robots[follower]["_headless_shield_until"] = (
                        self.current_time + 1.5)
                    self.robots[leader]["_headless_shield_until"] = (
                        self.current_time + 1.0)
                    events.append({
                        "type": "same_direction_following",
                        "follower": follower, "leader": leader,
                    })
                continue

            levels = (1.0, 0.85, 0.70, 0.55, 0.40)
            current_profile = {
                rid: float(self.robots[rid].get("speed_scale", 1.0))
                for rid in moving}
            # Match FactorySupervisor._find_joint_safety_profile's candidate
            # family. Webots stops this enumeration at an 80 ms wall deadline;
            # standalone evaluates the same bounded set deterministically so
            # host load cannot alter training trajectories.
            profiles = [current_profile]
            ordered_component = tuple(sorted(moving))
            if len(ordered_component) <= 4:
                profiles.extend(
                    dict(zip(ordered_component, scales))
                    for scales in itertools.product(
                        levels, repeat=len(ordered_component))
                    if dict(zip(ordered_component, scales)) != current_profile)
            else:
                profiles.extend({rid: level for rid in ordered_component}
                                for level in levels[1:])
                for offset, rid in enumerate(ordered_component):
                    profiles.append({
                        peer: (levels[(index + offset) % len(levels)]
                               if peer == rid else 1.0)
                        for index, peer in enumerate(ordered_component)
                    })

            def profile_rank(candidate):
                priority_score = sum(
                    scale * (
                        self._priority_keep_key(rid)[0] * 1e18 +
                        self._priority_keep_key(rid)[1] * 1e12 + rid)
                    for rid, scale in candidate.items())
                changes = sum(abs(
                    candidate[rid] - current_profile[rid]) for rid in moving)
                return (priority_score, sum(candidate.values()),
                        min(candidate.values()), -changes)

            profiles = profiles[:self.config.avoidance_profile_limit]
            safe = []
            trajectory_cache = {}
            for profile in profiles:
                candidate = dict(trajectories)
                for rid, scale in profile.items():
                    key = (rid, scale)
                    if key not in trajectory_cache:
                        trajectory_cache[key] = self._trajectory_for_scale(
                            rid, scale, horizon=4.0)
                    candidate[rid] = trajectory_cache[key]
                if self._trajectory_pairs_conflicting(
                        candidate, self.config.avoidance_minimum_distance,
                        component):
                    continue
                rank = profile_rank(profile)
                safe.append((*rank, profile))
            if safe:
                safe.sort(key=lambda row: row[:4], reverse=True)
                profile = safe[0][4]
            else:
                ordered = sorted(moving, key=self._priority_keep_key)
                profile = {rid: 1.0 for rid in moving}
                profile[ordered[0]] = 0.55
                if len(moving) > 1 and len(component) > 2:
                    profile[ordered[1]] = 0.70
            for rid, scale in profile.items():
                self.robots[rid]["speed_scale"] = scale
                self.robots[rid]["_headless_shield_until"] = (
                    self.current_time + 1.0)
            events.append({"type": "predictive_speed_profile",
                           "robot_ids": list(component),
                           "profile": profile})
        if events:
            self._next_joint_plan = min(
                self._next_joint_plan, self.current_time)
            self._joint_liveness_needed = True
        return events

    @staticmethod
    def _point_segment_distance(point, first, second) -> float:
        dx, dy = second[0] - first[0], second[1] - first[1]
        squared = dx * dx + dy * dy
        if squared <= 1e-12:
            return math.hypot(point[0] - first[0], point[1] - first[1])
        ratio = ((point[0] - first[0]) * dx +
                 (point[1] - first[1]) * dy) / squared
        ratio = max(0.0, min(1.0, ratio))
        projected = (first[0] + ratio * dx, first[1] + ratio * dy)
        return math.hypot(point[0] - projected[0],
                          point[1] - projected[1])

    @staticmethod
    def _point_along_polyline(points, target_distance):
        if not points:
            return None
        target_distance = max(0.0, float(target_distance))
        traversed = 0.0
        for first, second in zip(points, points[1:]):
            length = math.hypot(second[0] - first[0],
                                second[1] - first[1])
            if traversed + length >= target_distance:
                ratio = (0.0 if length <= 1e-12 else
                         (target_distance - traversed) / length)
                return (first[0] + ratio * (second[0] - first[0]),
                        first[1] + ratio * (second[1] - first[1]))
            traversed += length
        return tuple(points[-1])

    def _route_polyline(self, rid: int) -> List[tuple]:
        state = self.robots[rid]
        points = [tuple(state["position"])]
        for point in self._waypoints[rid][self._waypoint_index[rid]:]:
            point = tuple(point)
            if math.hypot(point[0] - points[-1][0],
                          point[1] - points[-1][1]) > 0.02:
                points.append(point)
        if len(points) < 2:
            goal = self._goal_position(state.get("goal_location"))
            if goal is not None:
                points.append(goal)
        return points

    def _distance_to_polyline(self, point, route) -> float:
        if not route:
            return math.inf
        if len(route) == 1:
            return math.hypot(point[0] - route[0][0],
                              point[1] - route[0][1])
        return min(self._point_segment_distance(point, first, second)
                   for first, second in zip(route, route[1:]))

    def _static_turning_path(self, start, goal):
        raw = self.coordinator.grid_planner.plan(start, goal, smooth=False)
        if not raw:
            return None
        cells = []
        for point in raw:
            cell = self.coordinator.grid.world_to_grid(*point)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        path = self.coordinator._cells_to_turning_waypoints(cells)
        if path and math.hypot(path[0][0] - start[0],
                               path[0][1] - start[1]) < 0.05:
            path = path[1:]
        return [tuple(point) for point in path] if path else None

    def _recovery_path_peer_clear(self, rid: int, path,
                                  clearance: float = 0.70) -> bool:
        if not path:
            return False
        start = tuple(self.robots[rid]["position"])
        peers = [tuple(peer["position"])
                 for peer_id, peer in self.robots.items() if peer_id != rid]
        departure = next((tuple(point) for point in path if math.hypot(
            point[0] - start[0], point[1] - start[1]) > 0.05), None)
        if departure is None:
            return False
        move_x, move_y = departure[0] - start[0], departure[1] - start[1]
        length = math.hypot(move_x, move_y)
        if length <= 1e-12:
            return False
        move_x, move_y = move_x / length, move_y / length
        for peer in peers:
            if math.hypot(peer[0] - start[0], peer[1] - start[1]) < 0.60:
                if (move_x * (peer[0] - start[0]) +
                        move_y * (peer[1] - start[1]) >= 0.0):
                    return False
            if any(math.hypot(point[0] - peer[0],
                              point[1] - peer[1]) < clearance
                   for point in path):
                return False
            # Match FactorySupervisor exactly: the current pose may already
            # be inside ``clearance`` during a physical escape.  Its first
            # segment is admitted only when the departure vector above points
            # away from a peer inside 0.60 m; applying the ordinary 0.70 m
            # segment check from the current pose makes every such escape
            # mathematically impossible.  Subsequent path segments still
            # require the full clearance.
            if any(self._point_segment_distance(peer, first, second) < clearance
                   for first, second in zip(path, path[1:])):
                return False
        return True

    def _install_recovery_path(self, rid: int, path, *,
                               escape_only: bool,
                               source: Optional[str] = None) -> bool:
        if not path:
            return False
        start = self.robots[rid]["position"]
        if math.hypot(path[-1][0] - start[0],
                      path[-1][1] - start[1]) < \
                self.config.minimum_navigation_displacement:
            return False
        self._abort_pending_joint_transaction(rid)
        state = self.robots[rid]
        state["_headless_hold_until"] = 0.0
        if not escape_only:
            state["speed_scale"] = 1.0
        self._install_path(
            rid, [tuple(point) for point in path], activate_joint=True,
            waypoint_offsets=[0.0] * len(path),
            activation_time=self.current_time, preserve_pending=True,
            partial=False, recovery=escape_only,
            source=(source or ("_joint_escape_robot" if escape_only else
                               "_joint_stall_recovery")))
        self._active_joint_members.add(rid)
        self._last_joint_activation_time = self.current_time
        state["_headless_stall_watch_pos"] = tuple(state["position"])
        state["_headless_stall_since"] = None
        state["_headless_joint_watch_pos"] = tuple(state["position"])
        state["_headless_joint_watch_since"] = self.current_time
        state["_headless_hard_stall_watch_pos"] = tuple(state["position"])
        state["_headless_hard_stall_since"] = None
        state["_headless_any_wait_since"] = None
        state["_headless_joint_wait_since"] = None
        state["_headless_route_less_since"] = None
        state["_headless_emergency_since"] = None
        return True

    def _joint_escape_robot(self, rid: int,
                            min_peer_clearance: float = 0.85) -> bool:
        state = self.robots[rid]
        origin = tuple(state["position"])
        goal = self._goal_position(state.get("goal_location"))
        peers = [tuple(peer["position"])
                 for peer_id, peer in self.robots.items() if peer_id != rid]
        candidates = []
        for distance in (0.45, 0.60, 0.80, 1.00, 1.25, 1.50, 1.75):
            for step in range(32):
                angle = 2.0 * math.pi * step / 32.0
                target = (origin[0] + distance * math.cos(angle),
                          origin[1] + distance * math.sin(angle))
                clearance = min((math.hypot(target[0] - peer[0],
                                             target[1] - peer[1])
                                 for peer in peers), default=math.inf)
                if clearance < min_peer_clearance:
                    continue
                cell = self.coordinator.grid.world_to_grid(*target)
                if (not self.coordinator.grid.in_bounds(*cell) or
                        not self.coordinator.grid.is_free(*cell) or
                        not self.coordinator._segment_clear(origin, target)):
                    continue
                progress = (0.0 if goal is None else
                    math.hypot(origin[0] - goal[0], origin[1] - goal[1]) -
                    math.hypot(target[0] - goal[0], target[1] - goal[1]))
                score = min(clearance, 1.6) + 0.55 * progress - 0.04 * distance
                candidates.append((score, target))
        for _score, target in sorted(candidates, reverse=True):
            path = self._static_turning_path(origin, target)
            if not path or not self._recovery_path_peer_clear(rid, path):
                # The production Supervisor retries through the reservation-
                # aware lifelong planner. Its answer can change as the
                # reservation clock advances, so a negative result must not
                # be cached solely by physical pose.
                path = self.coordinator.plan_grid_lifelong(
                    rid, origin, target)
            if not path or not self._recovery_path_peer_clear(rid, path):
                continue
            if self._install_recovery_path(rid, path, escape_only=True):
                self.telemetry["joint_escape_recoveries"] += 1
                return True
        return False

    def _detect_head_on_pair(self, component) -> Optional[tuple]:
        """Mirror the production corridor/head-on business-goal test."""
        stalled = tuple(sorted(component))
        best = None
        best_yielder_key = None
        best_distance = None
        for offset, first in enumerate(stalled):
            first_goal = self._goal_position(
                self.robots[first].get("goal_location"))
            if first_goal is None:
                continue
            first_position = self.robots[first]["position"]
            for second in stalled[offset + 1:]:
                second_goal = self._goal_position(
                    self.robots[second].get("goal_location"))
                if second_goal is None:
                    continue
                second_position = self.robots[second]["position"]
                dx = abs(first_position[0] - second_position[0])
                dy = abs(first_position[1] - second_position[1])
                head_on = bool(
                    (dx < 0.8 and
                     (first_goal[1] - first_position[1]) *
                     (second_goal[1] - second_position[1]) < 0.0) or
                    (dy < 0.8 and
                     (first_goal[0] - first_position[0]) *
                     (second_goal[0] - second_position[0]) < 0.0))
                if not head_on:
                    continue
                pair = self._priority_pair(first, second)
                if pair is None:
                    continue
                _winner, yielder = pair
                key = self._priority_keep_key(yielder)
                distance = math.hypot(
                    first_position[0] - second_position[0],
                    first_position[1] - second_position[1])
                if (best is None or key < best_yielder_key or
                        (key == best_yielder_key and
                         distance < best_distance)):
                    best = pair
                    best_yielder_key = key
                    best_distance = distance
        return best

    def _priority_direct_departure_clear(self, rid: int, path) -> bool:
        if not path:
            return False
        start = tuple(self.robots[rid]["position"])
        departure = next((tuple(point) for point in path if math.hypot(
            point[0] - start[0], point[1] - start[1]) > 0.05), None)
        if departure is None:
            return False
        move_x = departure[0] - start[0]
        move_y = departure[1] - start[1]
        length = math.hypot(move_x, move_y)
        if length <= 1e-12:
            return False
        move_x, move_y = move_x / length, move_y / length
        for peer_id, peer in self.robots.items():
            if peer_id == rid:
                continue
            relative_x = peer["position"][0] - start[0]
            relative_y = peer["position"][1] - start[1]
            if (math.hypot(relative_x, relative_y) < 0.65 and
                    move_x * relative_x + move_y * relative_y >= 0.0):
                return False
        return True

    def _joint_head_on_yield(self, component) -> bool:
        pair = self._detect_head_on_pair(component)
        if pair is None:
            return False
        winner, yielder = pair
        state = self.robots[yielder]
        if state.get("_headless_recovery_active", False):
            return False
        goal = state.get("goal_location")
        if self._goal_position(goal) is None:
            return False
        path = self.coordinator.plan_grid_lifelong(
            yielder, state["position"], goal,
            hard_peer_prefix=10.0,
            allow_unrestricted_fallback=False)
        if not path or not self._priority_direct_departure_clear(
                yielder, path):
            self.coordinator.rollback_robot_plan(yielder)
            return False
        state["_headless_hold_until"] = 0.0
        if not self._install_recovery_path(
                yielder, path, escape_only=False,
                source="_priority_yield_direct"):
            self.coordinator.rollback_robot_plan(yielder)
            return False
        state["speed_scale"] = 0.70
        self.robots[winner]["speed_scale"] = 1.0
        for peer_id in component:
            if peer_id not in (winner, yielder):
                self.robots[peer_id]["speed_scale"] = 0.60
        state["_headless_shield_until"] = self.current_time + 2.0
        self.telemetry["joint_direct_yield_replans"] += 1
        self._joint_liveness_needed = True
        self._next_joint_plan = min(
            self._next_joint_plan, self.current_time + 1.0)
        return True

    def _joint_try_escape_component(self, component) -> bool:
        if self._joint_head_on_yield(component):
            return True
        for rid in sorted(component, key=self._priority_keep_key):
            state = self.robots[rid]
            if self._priority_keep_key(rid)[0]:
                continue
            if state.get("_headless_escape_until", 0.0) > self.current_time:
                continue
            if state.get("_headless_recovery_active", False):
                continue
            if self._joint_escape_robot(rid, min_peer_clearance=0.90):
                state["_headless_escape_until"] = self.current_time + 3.0
                self._joint_liveness_needed = True
                self._next_joint_plan = min(
                    self._next_joint_plan, self.current_time + 2.0)
                for peer_id in component:
                    if peer_id != rid:
                        self.robots[peer_id]["speed_scale"] = 0.60
                return True
        return False

    def _joint_stall_recovery(self, rid: int) -> bool:
        state = self.robots[rid]
        now = self.current_time
        if state.get("_headless_stall_recovery_until", 0.0) > now:
            return False
        state["_headless_stall_recovery_until"] = (
            now + self.config.joint_stall_recovery_cooldown_seconds)
        goal = self._goal_position(state.get("goal_location"))
        route = self._route_polyline(rid)
        if goal is None or len(route) < 2:
            return False
        route_length = _polyline_length(route[0], route[1:])
        samples = [self._point_along_polyline(route, distance)
                   for distance in (0.4, 0.8, 1.2, 1.6, 2.0,
                                    2.5, 3.0, 3.5, 4.0)
                   if distance <= route_length]
        samples.extend(point for point in route[1:] if math.hypot(
            point[0] - route[0][0], point[1] - route[0][1]) <= 4.0)
        peers = [tuple(peer["position"])
                 for peer_id, peer in self.robots.items() if peer_id != rid]
        candidates = []
        seen = set()
        for sample in samples:
            cell = self.coordinator.grid.world_to_grid(*sample)
            sample_distance = math.hypot(
                sample[0] - route[0][0], sample[1] - route[0][1])
            tangent = ((1.0, 0.0) if sample_distance < 0.02 else (
                (sample[0] - route[0][0]) / max(1e-6, sample_distance),
                (sample[1] - route[0][1]) / max(1e-6, sample_distance)))
            normal = (-tangent[1], tangent[0])
            for lateral_cells in (0, 1, -1, 2, -2):
                candidate_cell = (
                    cell[0] + int(round(normal[0])) * lateral_cells,
                    cell[1] + int(round(normal[1])) * lateral_cells)
                if candidate_cell in seen:
                    continue
                seen.add(candidate_cell)
                if (not self.coordinator.grid.in_bounds(*candidate_cell) or
                        not self.coordinator.grid.is_free(*candidate_cell)):
                    continue
                target = self.coordinator.grid.grid_to_world(*candidate_cell)
                origin_distance = math.hypot(target[0] - route[0][0],
                                             target[1] - route[0][1])
                clearance = min((math.hypot(target[0] - peer[0],
                                             target[1] - peer[1])
                                 for peer in peers), default=math.inf)
                route_distance = self._distance_to_polyline(target, route)
                if (origin_distance < self.config.minimum_navigation_displacement or
                        clearance < self.config.relocation_peer_clearance or
                        route_distance > 1.0):
                    continue
                static_path = self.coordinator.grid_planner.plan(
                    target, goal, smooth=False)
                if not static_path:
                    continue
                static_length = sum(math.hypot(
                    second[0] - first[0], second[1] - first[1])
                    for first, second in zip(static_path, static_path[1:]))
                candidates.append((
                    (route_distance, -min(clearance, 2.0),
                     static_length, origin_distance), target))
        for _score, target in sorted(candidates, key=lambda row: row[0]):
            prefix = self._static_turning_path(route[0], target)
            if not prefix or not self._recovery_path_peer_clear(rid, prefix):
                prefix = self.coordinator.plan_grid_lifelong(
                    rid, route[0], target)
                if not prefix or not self._recovery_path_peer_clear(
                        rid, prefix):
                    continue
            suffix = []
            if math.hypot(target[0] - goal[0], target[1] - goal[1]) > \
                    GOAL_TOLERANCE * 1.5:
                suffix = self._static_turning_path(target, goal)
                if not suffix:
                    suffix = self.coordinator.plan_grid_lifelong(
                        rid, target, state.get("goal_location"))
                if not suffix:
                    continue
            if suffix and math.hypot(prefix[-1][0] - suffix[0][0],
                                     prefix[-1][1] - suffix[0][1]) < 0.05:
                suffix = suffix[1:]
            full_path = [*prefix, *suffix]
            if self._install_recovery_path(rid, full_path, escape_only=False):
                self.telemetry["joint_stall_recoveries"] += 1
                return True
        if self._joint_escape_robot(rid, min_peer_clearance=0.90):
            state["speed_scale"] = 1.0
            self.telemetry["joint_stall_recoveries"] += 1
            return True
        return False

    def _request_fresh_joint_plan(self) -> bool:
        """Mirror Webots' 1.5 s watchdog replan gate without stopping motion."""
        if self._joint_replan_cooldown_until > self.current_time:
            return False
        self._joint_replan_cooldown_until = self.current_time + 1.5
        self._abort_pending_joint_transaction()
        self._joint_candidate_failures = 0
        self._force_joint_replan = True
        self._next_joint_plan = min(self._next_joint_plan, self.current_time)
        return True

    def _escalate_stalled_robots(self, robot_ids) -> Optional[int]:
        """Run Webots' 5 s physical-progress backstop for one robot."""
        for rid in sorted(set(robot_ids), key=self._priority_keep_key):
            state = self.robots[rid]
            if state.get("_headless_escape_until", 0.0) > self.current_time:
                continue
            state["_headless_hold_until"] = 0.0
            self._dispatch_not_before[rid] = self.current_time
            if self._joint_escape_robot(rid):
                state["_headless_escape_until"] = self.current_time + 2.0
                self._joint_liveness_needed = True
                self._next_joint_plan = min(
                    self._next_joint_plan, self.current_time + 1.0)
                return rid
        return None

    def _joint_runtime_watchdog(self) -> List[dict]:
        """Use Webots' independent 3 s, 5 s and 8 s progress clocks."""
        events = []
        active = [rid for rid, state in self.robots.items()
                  if self._is_moving_state(state.get("state")) and
                  self._goal_position(state.get("goal_location")) is not None]
        if not active:
            return events

        recoveries = []
        hard_motion = []
        legal_waits = {}
        for rid in active:
            state = self.robots[rid]
            position = tuple(state["position"])
            threshold = self.config.stall_progress_distance * max(
                0.5, float(state.get("speed_scale", 1.0)))
            index = self._waypoint_index[rid]
            has_target = bool(
                self._waypoints[rid] and index < len(self._waypoints[rid]))
            at_target = bool(has_target and math.hypot(
                position[0] - self._waypoints[rid][index][0],
                position[1] - self._waypoints[rid][index][1]) <=
                float(state.get(
                    "_headless_waypoint_tolerance",
                    self.config.direct_waypoint_tolerance)))
            target_deadline = (
                self._waypoint_not_before[rid][index]
                if has_target and index < len(self._waypoint_not_before[rid])
                else self.current_time)
            timed_slot_wait = at_target and self.current_time < target_deadline
            endpoint_wait = bool(
                at_target and state.get("_headless_joint_plan_partial", False)
                and index == len(self._waypoints[rid]) - 1)
            pending_wait = bool(
                self._pending_joint_transaction is not None and
                rid in self._pending_joint_transaction["plans"])
            legal_wait = bool(
                self.current_time < self._dispatch_not_before[rid] or
                self.current_time < state.get("_headless_hold_until", 0.0) or
                pending_wait or timed_slot_wait or endpoint_wait)
            legal_waits[rid] = (legal_wait, endpoint_wait)

            # _update_hard_stall_clock: this clock always runs, but a legal
            # hold suppresses only its escalation, not the elapsed time.
            hard_watch = tuple(state.get(
                "_headless_hard_stall_watch_pos", position))
            hard_moved = math.hypot(
                position[0] - hard_watch[0], position[1] - hard_watch[1])
            if hard_moved >= threshold:
                state["_headless_hard_stall_watch_pos"] = position
                state["_headless_hard_stall_since"] = None
            elif state.get("_headless_hard_stall_since") is None:
                state["_headless_hard_stall_since"] = self.current_time
            hard_since = state.get("_headless_hard_stall_since")
            if (hard_since is not None and not legal_wait and
                    self.current_time - hard_since >=
                    self.config.joint_stall_relocation_seconds):
                hard_motion.append((rid, self.current_time - hard_since))

            # _update_joint_stall_clock is deliberately independent of legal
            # planned waits.  At 8 s Webots first tries a route-proximate
            # recovery; the shorter watchdog below handles ordinary replans.
            stall_watch = tuple(state.get(
                "_headless_stall_watch_pos", position))
            stall_moved = math.hypot(
                position[0] - stall_watch[0], position[1] - stall_watch[1])
            if stall_moved >= threshold:
                state["_headless_stall_watch_pos"] = position
                state["_headless_stall_since"] = None
            elif state.get("_headless_stall_since") is None:
                state["_headless_stall_since"] = self.current_time
            stall_since = state.get("_headless_stall_since")
            if (stall_since is not None and self.current_time - stall_since >=
                    self.config.joint_stall_relocation_seconds):
                recoveries.append((rid, self.current_time - stall_since))

        recoveries.sort(key=lambda item: (
            -item[1], self._priority_keep_key(item[0])))
        recovered_rid = None
        for rid, elapsed in recoveries:
            if (self.robots[rid].get(
                    "_headless_stall_recovery_until", 0.0) > self.current_time):
                continue
            if self._joint_stall_recovery(rid):
                recovered_rid = rid
                self._joint_liveness_needed = True
                events.append({"type": "joint_stall_recovery",
                               "robot_id": rid,
                               "stalled_seconds": elapsed})
                break

        stale_waits = []
        stalled = []
        emergency = []
        for rid in active:
            state = self.robots[rid]
            position = tuple(state["position"])
            threshold = self.config.stall_progress_distance * max(
                0.5, float(state.get("speed_scale", 1.0)))
            legal_wait, endpoint_wait = legal_waits[rid]
            route_less = (
                not state.get("_headless_joint_route_active", False) or
                not self._waypoints[rid] or
                self._waypoint_index[rid] >= len(self._waypoints[rid]))
            if route_less:
                if state.get("_headless_route_less_since") is None:
                    state["_headless_route_less_since"] = self.current_time
                elif (self.current_time - state["_headless_route_less_since"] >=
                      3.0):
                    stalled.append(rid)
            else:
                state["_headless_route_less_since"] = None

            if state.get("_headless_peer_stop_latched", False):
                emergency.append(rid)
                if state.get("_headless_emergency_since") is None:
                    state["_headless_emergency_since"] = self.current_time
                state["_headless_joint_watch_pos"] = position
                state["_headless_joint_watch_since"] = self.current_time
                continue
            state["_headless_emergency_since"] = None

            if legal_wait:
                if endpoint_wait:
                    wait_since = state.get("_headless_joint_wait_since")
                    if wait_since is None:
                        state["_headless_joint_wait_since"] = self.current_time
                    elif self.current_time - wait_since > 2.0:
                        stale_waits.append(rid)
                else:
                    state["_headless_joint_wait_since"] = None
                any_wait_since = state.get("_headless_any_wait_since")
                if any_wait_since is None:
                    state["_headless_any_wait_since"] = self.current_time
                elif (self.current_time - any_wait_since >
                      self.config.joint_stall_relocation_seconds):
                    stale_waits.append(rid)
                state["_headless_joint_watch_pos"] = position
                state["_headless_joint_watch_since"] = self.current_time
                continue

            state["_headless_joint_wait_since"] = None
            state["_headless_any_wait_since"] = None
            joint_watch = tuple(state.get(
                "_headless_joint_watch_pos", position))
            moved = math.hypot(
                position[0] - joint_watch[0], position[1] - joint_watch[1])
            if moved >= threshold:
                state["_headless_joint_watch_pos"] = position
                state["_headless_joint_watch_since"] = self.current_time
                continue
            if state.get("_headless_joint_watch_since") is None:
                state["_headless_joint_watch_since"] = self.current_time
                state["_headless_joint_watch_pos"] = position
                continue
            if self.current_time - state["_headless_joint_watch_since"] >= 3.0:
                stalled.append(rid)

        pending_hard_motion = [
            item for item in hard_motion if item[0] != recovered_rid]
        if emergency or stale_waits or stalled or pending_hard_motion:
            ids = set(emergency + stale_waits + stalled +
                      [rid for rid, _elapsed in pending_hard_motion])
            self._joint_liveness_needed = True
            self._request_fresh_joint_plan()
            hard_stalled = [
                rid for rid in stalled
                if self.robots[rid].get("_headless_route_less_since") is None
                and self.robots[rid].get("_headless_joint_watch_since") is not None
                and self.current_time - self.robots[rid][
                    "_headless_joint_watch_since"] >= 5.0]
            hard_stalled.extend(rid for rid, _elapsed in pending_hard_motion)
            hard_stalled.extend(
                rid for rid in stalled
                if self.robots[rid].get("_headless_route_less_since") is not None
                and self.current_time - self.robots[rid][
                    "_headless_route_less_since"] >= 3.0)
            hard_stalled.extend(
                rid for rid in emergency
                if self.robots[rid].get("_headless_emergency_since") is not None
                and self.current_time - self.robots[rid][
                    "_headless_emergency_since"] >= 3.0)
            for rid in stale_waits:
                wait_since = self.robots[rid].get("_headless_any_wait_since")
                if (wait_since is not None and
                        self.current_time - wait_since >=
                        self.config.joint_stall_relocation_seconds):
                    hard_stalled.append(rid)
            escaped = self._escalate_stalled_robots(hard_stalled)
            if escaped is not None:
                events.append({"type": "joint_stall_escape",
                               "robot_id": escaped})
            for rid in ids:
                state = self.robots[rid]
                state["_headless_joint_wait_since"] = None
                state["_headless_any_wait_since"] = None
                if rid in stalled:
                    continue
                state["_headless_joint_watch_pos"] = tuple(state["position"])
                state["_headless_joint_watch_since"] = self.current_time
        return events

    def _refresh_reservation_timing(self) -> None:
        for rid, state in self.robots.items():
            index = self._waypoint_index[rid]
            if not self._waypoints[rid] or index >= len(self._waypoints[rid]):
                continue
            not_before = max(
                self._dispatch_not_before[rid],
                float(state.get("_headless_hold_until", 0.0)))
            self.coordinator.refresh_active_plan_timing(
                rid, state["position"], self._waypoints[rid][index:],
                not_before=not_before)

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

    def _dock_clearance_required(self, rid: int, location: str) -> bool:
        if location not in ALL_LOCATIONS:
            return False
        for peer_id, peer in self.robots.items():
            task = peer.get("current_task")
            if peer_id == rid or task is None:
                continue
            if peer.get("state") == RobotState.EN_ROUTE_PICKUP:
                peer_goal = task.pickup_location
            elif peer.get("state") in {
                    RobotState.CARRYING, RobotState.EN_ROUTE_DELIVERY}:
                peer_goal = task.delivery_location
            else:
                continue
            if peer_goal == location:
                return True
        return False

    def _relocate_idle_robot(self, rid: int) -> List[dict]:
        """Mirror Webots' single-robot lazy relocation operation."""
        state = self.robots[rid]
        if (state.get("state") != RobotState.IDLE or
                state.get("current_task") is not None or
                self._waypoints[rid]):
            return []
        current_node = self.coordinator.graph.get_nearest_node(
            state["position"])
        if current_node in REST_NODES:
            return []
        result = self.coordinator.find_nearest_rest_node(
            state["position"], exclude_robot_id=rid)
        if result is None:
            return []
        target_name, target_xy = result
        if math.hypot(
                state["position"][0] - target_xy[0],
                state["position"][1] - target_xy[1]) < 0.4:
            return []
        self.coordinator.release_home(rid)
        node_path = self.coordinator.lifelong.plan(
            rid, current_node, target_name)
        if not node_path:
            self._reserve_idle_position(rid)
            return []
        path = [tuple(WAYPOINTS[node]) for node in node_path]
        if path and math.hypot(
                path[0][0] - state["position"][0],
                path[0][1] - state["position"][1]) < 0.05:
            path.pop(0)
        if not path:
            self._reserve_idle_position(rid)
            return []
        state["state"] = RobotState.RETURNING_HOME
        state["goal_location"] = tuple(target_xy)
        self._install_path(rid, path)
        return [{
            "type": "idle_relocation_started", "robot_id": rid,
            "target": target_name,
        }]

    def _relocate_idle_robots(self) -> List[dict]:
        """Mirror Webots lazy dock-to-rest relocation before scheduling."""
        events = []
        for rid in sorted(self.robots):
            events.extend(self._relocate_idle_robot(rid))
        return events

    def _check_joint_business_arrivals(self) -> List[dict]:
        """Mirror Webots' physical arrival promotion for partial prefixes."""
        if not self.config.joint_runtime:
            return []
        events = []
        for rid in sorted(self.robots):
            state = self.robots[rid]
            if (state.get("_headless_plan_source") !=
                    "joint_grid_transaction" or
                    state.get("current_task") is None or
                    state.get("state") not in {
                        RobotState.EN_ROUTE_PICKUP,
                        RobotState.EN_ROUTE_DELIVERY}):
                continue
            goal = self._goal_position(state.get("goal_location"))
            if goal is None:
                continue
            if math.hypot(
                    state["position"][0] - goal[0],
                    state["position"][1] - goal[1]) <= GOAL_TOLERANCE * 2.0:
                events.extend(self._handle_goal_reached(rid))
        return events

    def _observe_peer_clearance(self) -> None:
        """Record safety evidence without influencing motion decisions."""
        positions = {
            rid: tuple(state["position"]) for rid, state in self.robots.items()}
        if len(positions) < 2:
            return
        minimum = min(
            math.hypot(positions[first][0] - positions[second][0],
                       positions[first][1] - positions[second][1])
            for first, second in itertools.combinations(sorted(positions), 2))
        observed = self.telemetry["minimum_peer_distance"]
        self.telemetry["minimum_peer_distance"] = (
            minimum if observed is None else min(float(observed), minimum))
        if minimum + 1e-9 < self.config.hard_minimum_distance:
            self.telemetry["hard_distance_violations"] += 1

    def tick(self) -> List[dict]:
        """Advance by one Webots timestep, capped exactly at the horizon."""
        if self._horizon_finalized:
            return []
        remaining = self.config.episode_end_time - self.current_time
        if remaining <= 1e-9:
            return self._finalize_horizon()
        self._tick_seconds = min(self.config.timestep_seconds, remaining)
        self.current_time += self._tick_seconds
        self.telemetry["ticks"] += 1
        self.coordinator.set_sim_time(self.current_time)
        self._expire_failed_pairs()
        events = self._activate_pending_joint_transaction()

        while self.current_time + 1e-9 >= self._next_lifelong_tick:
            self.coordinator.lifelong_tick(1)
            self._next_lifelong_tick += self.config.lifelong_tick_seconds

        for rid in sorted(self.robots):
            events.extend(self._update_battery(rid))
            events.extend(self._move_robot(rid))

        self._observe_peer_clearance()
        events.extend(self._check_joint_business_arrivals())
        if self.config.joint_runtime:
            while self.current_time + 1e-9 >= self._next_avoidance_scan:
                events.extend(self._predictive_peer_avoidance())
                self._next_avoidance_scan += self.config.avoidance_scan_seconds
            while self.current_time + 1e-9 >= self._next_joint_watchdog:
                events.extend(self._joint_runtime_watchdog())
                self._next_joint_watchdog += self.config.joint_watchdog_seconds

        while self.current_time + 1e-9 >= self._next_reservation_refresh:
            self._refresh_reservation_timing()
            self._next_reservation_refresh += \
                self.config.reservation_refresh_seconds

        while self.current_time + 1e-9 >= self._next_relocation_scan:
            events.extend(self._relocate_idle_robots())
            self._next_relocation_scan += self.config.relocation_scan_seconds

        events.extend(self._retry_routes())
        if self.config.joint_runtime:
            if self.current_time + 1e-9 >= self._next_joint_plan:
                events.extend(self._refresh_joint_grid_candidate())
                self._next_joint_plan += self.config.joint_replan_seconds
        else:
            while self.current_time + 1e-9 >= self._next_deadlock_scan:
                events.extend(self._scan_deadlocks())
                self._next_deadlock_scan += self.config.deadlock_scan_seconds

        arrived = {
            task.task_id for task in self.tasks
            if task.status == TaskStatus.PENDING
            and float(task.arrival_time) <= self.current_time + 1e-9
            and float(task.arrival_time) < self.config.episode_end_time
        }
        for task_id in sorted(arrived - self._visible_task_ids):
            events.append({"type": "task_arrived", "task_id": task_id})
        self._visible_task_ids.update(arrived)
        for task in self.tasks:
            events.extend(task.deadline_events(self.current_time))
        if self.current_time + 1e-9 >= self.config.episode_end_time:
            events.extend(self._finalize_horizon())
        return events

    def _finalize_horizon(self) -> List[dict]:
        """Settle deadline state at H once, without draining the factory."""
        if self._horizon_finalized:
            return []
        self.current_time = self.config.episode_end_time
        events = []
        for task in self.tasks:
            if task.arrival_time < self.config.episode_end_time:
                events.extend(task.deadline_events(
                    self.current_time,
                    horizon_seconds=self.config.episode_end_time,
                    final=True))
        self._horizon_finalized = True
        events.append({
            "type": "episode_horizon",
            "current_time": float(self.current_time),
        })
        return events

    def advance_until_event(self, *, max_seconds: Optional[float] = None
                            ) -> List[dict]:
        """Advance until a policy-relevant event or a bounded stall."""
        if self.is_terminal():
            return []
        limit = (self.config.max_advance_seconds if max_seconds is None
                 else float(max_seconds))
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("max_seconds must be finite and positive")
        deadline = min(
            self.current_time + limit, self.config.episode_end_time)
        relevant = {
            "task_arrived", "task_completed", "task_failed_battery",
            "charge_swap_completed", "assignment_rejected",
            "deadlock_replan", "robot_idle", "task_deadline_base",
            "task_deadline_severity", "episode_horizon",
        }
        collected = []
        while self.current_time + 1e-9 < deadline:
            events = self.tick()
            collected.extend(events)
            if any(event.get("type") in relevant for event in events):
                break
            if self.is_terminal():
                break
            if (not self.config.fixed_horizon
                    and not self.has_active_execution()
                    and not self.has_future_arrivals()):
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
            "episode_end_time": self.config.episode_end_time,
            "fixed_horizon": self.config.fixed_horizon,
            "joint_runtime": self.config.joint_runtime,
            "route_planner": (
                "rolling_joint_grid" if self.config.joint_runtime else
                "per_robot_lifelong"),
            "peer_avoidance": (
                "analytic_joint_predictive_shield" if self.config.joint_runtime
                else "legacy_deadlock_monitor"),
            "joint_candidate_budget_seconds": (
                self.config.joint_candidate_budget_seconds),
            "joint_candidate_max_budget_seconds": (
                self.config.joint_candidate_max_budget_seconds),
            "pending_joint_epoch": (
                self._pending_joint_transaction["epoch"]
                if self._pending_joint_transaction is not None else None),
            "horizon_finalized": self._horizon_finalized,
            "termination_reason": (
                "episode_horizon" if self._horizon_finalized else
                "idle" if self.is_terminal() else None),
        })
        return result
