"""Versioned observation contract and headless Webots-logic trainer."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (BATTERY_CAPACITY, MAX_ROBOTS, RL_ENVIRONMENT_VERSION,
                    RobotState, TaskStatus)
from headless_training_runtime import (
    HEADLESS_DYNAMICS_VERSION, HeadlessFactoryRuntime, HeadlessRuntimeConfig,
)
from schedulers import (
    Assignment, SchedulingContext, build_cost_matrix, validate_assignment,
    validate_assignments,
)
from task_generator import TransportTask
from dual_objective import CountRewardProfile, UtilityV2RewardProfile


ENVIRONMENT_VERSION = RL_ENVIRONMENT_VERSION
@dataclass(frozen=True)
class RLEnvironmentConfig:
    max_robots: int = MAX_ROBOTS
    max_tasks: int = 20
    max_steps_per_episode: int = 64
    position_scale_x: float = 10.0
    position_scale_y: float = 8.0
    time_scale: float = 120.0
    distance_scale: float = 30.0


@dataclass(frozen=True)
class RewardConfig:
    task_completion: float = 10.0
    valid_assignment: float = 0.5
    priority: float = 0.5
    distance_weight: float = -0.08
    # A bounded age bonus prevents starvation, while the negative waiting
    # term still optimises mean delay.  Keeping these separate avoids the old
    # behaviour where unbounded waiting was positively rewarded.
    age_bonus: float = 0.5
    age_scale_seconds: float = 120.0
    max_age_bonus_units: float = 2.0
    waiting_weight: float = -0.03
    invalid_action: float = -5.0
    no_op: float = -2.0
    collision: float = -100.0
    # Waiting/deadlock recovery is reward-neutral unless a real collision is
    # reported separately.
    deadlock: float = 0.0


class SchedulingEnvironment:
    """Snapshot encoder plus a Webots-business-compatible headless runtime.

    Webots mode never commits domain state; scheduler adapters use observation
    and mask only. Headless mode operates on private deep copies for training.
    ``abstract`` remains a compatibility alias for ``headless`` and no longer
    uses the old distance/time event shortcut.
    """

    ROBOT_FEATURES = 8
    TASK_FEATURES = 9
    GLOBAL_FEATURES = 6

    def __init__(self, config: Optional[RLEnvironmentConfig] = None,
                 reward: Optional[RewardConfig] = None,
                 reward_profile=None,
                 simulation_mode: str = "headless",
                 runtime_config: Optional[HeadlessRuntimeConfig] = None):
        if simulation_mode not in {"abstract", "headless", "webots"}:
            raise ValueError(
                "simulation_mode must be abstract, headless or webots")
        if reward is not None and reward_profile is not None:
            raise ValueError("legacy reward and reward_profile are exclusive")
        if (reward_profile is not None and not isinstance(
                reward_profile, (CountRewardProfile,
                                 UtilityV2RewardProfile))):
            raise ValueError("unsupported reward profile")
        self.config = config or RLEnvironmentConfig()
        self.reward_config = reward or RewardConfig()
        self.reward_profile = reward_profile
        self.requested_simulation_mode = simulation_mode
        self.simulation_mode = (
            "headless" if simulation_mode == "abstract" else simulation_mode)
        self.runtime_config = runtime_config or HeadlessRuntimeConfig()
        if (self.reward_profile is not None and
                not self.runtime_config.fixed_horizon):
            raise ValueError("versioned reward profiles require fixed_horizon")
        if (self.reward_profile is not None and
                self.reward_profile.horizon_seconds
                != self.runtime_config.episode_end_time):
            raise ValueError("reward profile horizon differs from runtime")
        self.action_dim = self.config.max_robots * self.config.max_tasks + 1
        self.no_op_action = self.action_dim - 1
        self.observation_dim = (
            self.GLOBAL_FEATURES
            + self.config.max_robots * self.ROBOT_FEATURES
            + self.config.max_tasks * self.TASK_FEATURES
            + self.config.max_robots + self.config.max_tasks
        )
        self._robots: Dict[int, dict] = {}
        self._tasks: List[TransportTask] = []
        self._context = SchedulingContext()
        self._robot_slots: List[int] = []
        self._task_slots: List[TransportTask] = []
        self._step = 0
        self._completed_ids = set()
        self._cost_matrix = None
        self._runtime: Optional[HeadlessFactoryRuntime] = None
        self._reward_totals = {}
        self.rng = np.random.default_rng(0)

    def reset(self, robot_states: Optional[Dict[int, dict]] = None,
              tasks: Optional[List[TransportTask]] = None,
              context: Optional[SchedulingContext] = None,
              seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.simulation_mode == "headless":
            # Copy the snapshot as one object graph so an active robot's
            # current_task remains the canonical object in the copied queue.
            self._robots, self._tasks = deepcopy((
                robot_states or {}, tasks or []))
        else:
            self._robots = robot_states or {}
            self._tasks = tasks or []
        self._context = context or SchedulingContext()
        provider = self._context.path_cost_provider
        bind = getattr(provider, "bind_robot_states", None)
        if callable(bind):
            bind(self._robots)
        self._step = 0
        self._completed_ids.clear()
        self._reward_totals = {}
        self._cost_matrix = None
        if self.simulation_mode == "headless":
            self._runtime = HeadlessFactoryRuntime(
                self._robots, self._tasks, self._context,
                seed=0 if seed is None else int(seed),
                config=self.runtime_config)
            self._sync_runtime()
        else:
            self._runtime = None
        self._refresh_slots()
        return self.observe(), {
            "environment_version": ENVIRONMENT_VERSION,
            "simulation_mode": self.simulation_mode,
            "reward_profile_version": (
                self.reward_profile.version if self.reward_profile is not None
                else "legacy_reward"),
            "reward_profile_hash": (
                self.reward_profile.sha256
                if self.reward_profile is not None else None),
            "dynamics_version": (
                HEADLESS_DYNAMICS_VERSION
                if self.simulation_mode == "headless" else "webots-live"),
        }

    def set_snapshot(self, robot_states: Dict[int, dict],
                     tasks: List[TransportTask],
                     context: Optional[SchedulingContext] = None) -> np.ndarray:
        return self.reset(robot_states, tasks, context)[0]

    def _refresh_slots(self) -> None:
        self._robot_slots = sorted(self._robots)[:self.config.max_robots]
        now = float(self._context.current_time)
        self._task_slots = sorted(
            (task for task in self._tasks
             if task.status == TaskStatus.PENDING
             and float(task.arrival_time) <= now + 1e-9),
            key=lambda task: (task.task_id, task.arrival_time)
        )[:self.config.max_tasks]

    def encode_action(self, robot_slot: int, task_slot: int) -> int:
        if not (0 <= robot_slot < self.config.max_robots):
            raise ValueError("robot_slot out of range")
        if not (0 <= task_slot < self.config.max_tasks):
            raise ValueError("task_slot out of range")
        return robot_slot * self.config.max_tasks + task_slot

    def decode_action(self, action: int) -> Optional[Tuple[int, int]]:
        if action == self.no_op_action:
            return None
        if not (0 <= action < self.no_op_action):
            raise ValueError("action out of range")
        return divmod(action, self.config.max_tasks)

    def get_action_mask(self) -> np.ndarray:
        self._refresh_slots()
        self._cost_matrix = None
        mask = np.zeros(self.action_dim, dtype=bool)
        if self._robot_slots and self._task_slots:
            matrix = self._matrix()
            robot_index = {rid: index for index, rid in enumerate(matrix.robot_ids)}
            task_index = {
                task.task_id: index for index, task in enumerate(matrix.tasks)}
            for rslot, rid in enumerate(self._robot_slots):
                ri = robot_index.get(rid)
                if ri is None:
                    continue
                for tslot, task in enumerate(self._task_slots):
                    ti = task_index.get(task.task_id)
                    if ti is not None and matrix.feasible[ri, ti]:
                        mask[self.encode_action(rslot, tslot)] = True
        # NO_OP is a safety action, but is unavailable when useful work exists.
        mask[self.no_op_action] = not bool(mask[:-1].any())
        return mask

    def _matrix(self):
        if self._cost_matrix is None:
            self._cost_matrix = build_cost_matrix(
                self._task_slots, self._robots, self._context)
        return self._cost_matrix

    def assignment_for_action(self, action: int) -> Optional[Assignment]:
        decoded = self.decode_action(action)
        if decoded is None:
            return None
        rslot, tslot = decoded
        if rslot >= len(self._robot_slots) or tslot >= len(self._task_slots):
            return None
        rid = self._robot_slots[rslot]
        task = self._task_slots[tslot]
        matrix = self._matrix()
        try:
            ri = matrix.robot_ids.index(rid)
            ti = tuple(item.task_id for item in matrix.tasks).index(task.task_id)
        except ValueError:
            return None
        cost = float(matrix.values[ri, ti])
        if not matrix.feasible[ri, ti] or not np.isfinite(cost):
            return None
        return Assignment(rid, task, cost)

    def action_for_pair(self, robot_id: int, task_id: int) -> int:
        """Map canonical graph-policy IDs back to the fixed action space."""
        self._refresh_slots()
        try:
            robot_slot = self._robot_slots.index(int(robot_id))
            task_slot = next(
                index for index, task in enumerate(self._task_slots)
                if task.task_id == int(task_id))
        except (ValueError, StopIteration) as exc:
            raise ValueError("robot-task pair is not in the current slots") from exc
        action = self.encode_action(robot_slot, task_slot)
        if not self.get_action_mask()[action]:
            raise ValueError("robot-task pair is not a legal action")
        return action

    def policy_snapshot(self):
        """Return read-only-by-contract inputs for graph policy encoding.

        Callers must not mutate these objects.  Keeping the live references is
        necessary because a bound path-cost provider observes the same robot
        state objects as the abstract environment.  Only arrived task slots
        are exposed so graph edges have exactly the same semantics as the
        fixed action mask.
        """
        self._refresh_slots()
        return self._robots, tuple(self._task_slots), self._context

    def observe(self) -> np.ndarray:
        self._refresh_slots()
        self._cost_matrix = None
        cfg = self.config
        pending = len(self._task_slots)
        idle = sum(
            state.get("state") == RobotState.IDLE
            and state.get("current_task") is None
            for state in self._robots.values()
        )
        congestion = self._context.congestion_map
        mean_congestion = float(np.mean(congestion)) if congestion else 0.0
        mask = self.get_action_mask()
        global_features = [
            np.clip(self._context.current_time / cfg.time_scale, 0, 10),
            pending / max(1, cfg.max_tasks),
            idle / max(1, cfg.max_robots),
            np.clip(mean_congestion, 0, 10),
            float(mask[:-1].mean()) if self.no_op_action else 0.0,
            self._step / max(1, cfg.max_steps_per_episode),
        ]
        robot_values = []
        robot_mask = []
        for slot in range(cfg.max_robots):
            if slot < len(self._robot_slots):
                state = self._robots[self._robot_slots[slot]]
                x, y = state.get("position", (0.0, 0.0))
                robot_values.extend([
                    x / cfg.position_scale_x, y / cfg.position_scale_y,
                    float(state.get("state") == RobotState.IDLE),
                    np.clip(float(state.get("battery", 100.0)) / 100.0, 0, 1),
                    float(state.get("current_task") is not None),
                    float(state.get("faulted", False) or state.get("failed", False)),
                    min(float(state.get("tasks_completed", 0)) / 20.0, 10),
                    min(float(state.get("total_distance", 0.0)) / 200.0, 10),
                ])
                robot_mask.append(1.0)
            else:
                robot_values.extend([0.0] * self.ROBOT_FEATURES)
                robot_mask.append(0.0)
        task_values = []
        task_mask = []
        for slot in range(cfg.max_tasks):
            if slot < len(self._task_slots):
                task = self._task_slots[slot]
                px, py = task.pickup_position
                dx, dy = task.delivery_position
                wait = max(0.0, self._context.current_time - task.arrival_time)
                matrix = self._matrix()
                ti = next((index for index, item in enumerate(matrix.tasks)
                           if item.task_id == task.task_id), None)
                feasible = (
                    matrix.feasible[:, ti] if ti is not None
                    else np.zeros(0, dtype=bool))
                feasible_fraction = float(feasible.mean()) if feasible.size else 0.0
                costs = (
                    matrix.values[:, ti][feasible] if ti is not None
                    else np.zeros(0))
                expected = float(costs.min()) if costs.size else 0.0
                task_values.extend([
                    px / cfg.position_scale_x, py / cfg.position_scale_y,
                    dx / cfg.position_scale_x, dy / cfg.position_scale_y,
                    np.clip(float(task.priority) / 2.0, 0, 10),
                    min(wait / cfg.time_scale, 10),
                    float(task.status == TaskStatus.PENDING),
                    feasible_fraction,
                    min(expected / cfg.distance_scale, 10),
                ])
                task_mask.append(1.0)
            else:
                task_values.extend([0.0] * self.TASK_FEATURES)
                task_mask.append(0.0)
        result = np.asarray(
            global_features + robot_values + task_values
            + robot_mask + task_mask, dtype=np.float32)
        if result.shape != (self.observation_dim,) or not np.isfinite(result).all():
            raise ValueError("invalid RL observation")
        return result

    def _profile_reward(self, events, *, dispatch_task=None,
                        invalid_action=False, defer=False):
        if self.reward_profile is None:
            return None
        components = self.reward_profile.transition(
            self._tasks, events, dispatch_task=dispatch_task,
            invalid_action=invalid_action, defer=defer,
            horizon_seconds=self.runtime_config.episode_end_time,
            runtime_mode="headless_webots_logic")
        for name, value in components.items():
            if name.startswith("reward_") and name not in {
                    "reward_profile_version", "reward_profile_hash"}:
                if isinstance(value, (int, float)):
                    self._reward_totals[name] = (
                        self._reward_totals.get(name, 0.0) + float(value))
        return components

    def _truncation_info(self, truncated: bool) -> dict:
        invalid = bool(
            truncated and self._runtime is not None
            and self._runtime.current_time + 1e-9
            < self.runtime_config.episode_end_time)
        return {
            "invalid_truncation": invalid,
            "truncation_reason": (
                "max_steps_before_horizon" if invalid else None),
        }

    def step(self, action: int, *, candidate_attribution: bool = True):
        if self.simulation_mode != "headless" or self._runtime is None:
            raise RuntimeError("step is only available in headless mode")
        if not isinstance(candidate_attribution, bool):
            raise ValueError("candidate_attribution must be boolean")
        action_has_valid_type = (
            not isinstance(action, bool) and
            isinstance(action, (int, np.integer)))
        if action_has_valid_type:
            action = int(action)
        mask = self.get_action_mask()
        self._step += 1
        if (action_has_valid_type and self.reward_profile is not None
                and action == self.no_op_action
                and not mask[action]):
            components = self._profile_reward([], defer=True)
            truncated = self._step >= self.config.max_steps_per_episode
            return self.observe(), components["reward_total"], False, truncated, {
                "no_op": True,
                "deferred_feasible_assignment": True,
                "reward_components": components,
                **self._truncation_info(truncated),
                "dynamics_version": HEADLESS_DYNAMICS_VERSION,
            }
        if (not action_has_valid_type or not (0 <= action < self.action_dim)
                or not mask[action]):
            truncated = self._step >= self.config.max_steps_per_episode
            components = self._profile_reward([], invalid_action=True)
            reward = (self.reward_config.invalid_action if components is None
                      else components["reward_total"])
            return self.observe(), reward, False, truncated, {
                "invalid_action": True,
                "reward_components": components,
                **self._truncation_info(truncated),
                "dynamics_version": HEADLESS_DYNAMICS_VERSION}

        distance_before = float(
            self._runtime.telemetry["distance_travelled"])
        if action == self.no_op_action:
            had_active = self._has_active_executions()
            had_future = self._has_future_arrivals()
            events = self._runtime.advance_until_event()
            self._sync_runtime()
            completed = sum(
                event.get("type") == "task_completed" for event in events)
            distance = (float(self._runtime.telemetry["distance_travelled"])
                        - distance_before)
            components = self._profile_reward(events)
            reward = (self._event_reward(events, distance)
                      if components is None else components["reward_total"])
            if (components is None and not events and
                    not had_active and not had_future):
                reward += self.reward_config.no_op
            truncated = self._step >= self.config.max_steps_per_episode
            return self.observe(), float(reward), self._is_terminal_state(), truncated, {
                "no_op": True,
                "completed_this_step": completed,
                "completed_count": len(self._completed_ids),
                "runtime_events": events,
                "distance_travelled": distance,
                "reward_components": components,
                **self._truncation_info(truncated),
                "dynamics_version": HEADLESS_DYNAMICS_VERSION,
            }

        assignment = self.assignment_for_action(action)
        if assignment is None:
            components = self._profile_reward([], invalid_action=True)
            reward = (self.reward_config.invalid_action if components is None
                      else components["reward_total"])
            return self.observe(), reward, False, False, {
                "invalid_action": True,
                "reward_components": components,
                "dynamics_version": HEADLESS_DYNAMICS_VERSION}
        task = assignment.task
        wait = max(0.0, self._context.current_time - task.arrival_time)
        legacy_reward = (
            self.reward_config.valid_assignment
            + self.reward_config.priority * max(0.0, float(task.priority))
            + self.reward_config.age_bonus * min(
                wait / max(self.reward_config.age_scale_seconds, 1e-6),
                self.reward_config.max_age_bonus_units)
            + self.reward_config.waiting_weight * min(
                wait / max(self.reward_config.age_scale_seconds, 1e-6),
                self.reward_config.max_age_bonus_units)
        )
        dispatch_event = self._runtime.dispatch(assignment)
        events = [dispatch_event]
        self._sync_runtime()
        if dispatch_event.get("type") != "assignment_committed":
            components = self._profile_reward(
                events, invalid_action=True)
            reward = (self.reward_config.invalid_action if components is None
                      else components["reward_total"])
            truncated = self._step >= self.config.max_steps_per_episode
            return self.observe(), reward, (
                self._is_terminal_state()), (
                truncated), {
                    "assignment_rejected": True,
                    "reason": dispatch_event.get("reason", "unknown"),
                    "runtime_events": events,
                    "reward_components": components,
                    **self._truncation_info(truncated),
                    "completed_count": len(self._completed_ids),
                    "dynamics_version": HEADLESS_DYNAMICS_VERSION,
                }

        task.candidate_reward_eligible = candidate_attribution

        # Assign all currently possible work before advancing physical time.
        if not self.get_action_mask()[:-1].any():
            events.extend(self._runtime.advance_until_event())
            self._sync_runtime()
        completed = sum(
            event.get("type") == "task_completed" for event in events)
        distance = (float(self._runtime.telemetry["distance_travelled"])
                    - distance_before)
        components = self._profile_reward(
            events, dispatch_task=task,
            invalid_action=not candidate_attribution)
        reward = (legacy_reward + self._event_reward(events, distance)
                  if components is None else components["reward_total"])
        terminated = self._is_terminal_state()
        truncated = self._step >= self.config.max_steps_per_episode
        return self.observe(), float(reward), bool(terminated), bool(truncated), {
            "assignment": (assignment.robot_id, task.task_id),
            "completed_this_step": completed,
            "completed_count": len(self._completed_ids),
            "runtime_events": events,
            "distance_travelled": distance,
            "reward_components": components,
            **self._truncation_info(truncated),
            "dynamics_version": HEADLESS_DYNAMICS_VERSION,
        }

    def _sync_runtime(self) -> None:
        if self._runtime is None:
            return
        self._robots = self._runtime.robots
        self._tasks = self._runtime.tasks
        self._context = self._runtime.context
        self._completed_ids = set(self._runtime.completed_ids)
        self._cost_matrix = None

    def _event_reward(self, events, distance: float) -> float:
        reward = self.reward_config.distance_weight * max(0.0, float(distance))
        for event in events:
            kind = event.get("type")
            if kind == "task_completed":
                reward += self.reward_config.task_completion
            elif kind == "task_failed_battery":
                reward += self.reward_config.invalid_action
            elif kind == "deadlock_replan":
                reward += self.reward_config.deadlock
            elif kind == "collision":
                reward += self.reward_config.collision
        return float(reward)

    def _has_active_executions(self) -> bool:
        return bool(self._runtime and self._runtime.has_active_execution())

    def _has_future_arrivals(self) -> bool:
        return bool(self._runtime and self._runtime.has_future_arrivals())

    def _is_terminal_state(self) -> bool:
        return bool(self._runtime and self._runtime.is_terminal())

    def is_terminal(self) -> bool:
        return self._is_terminal_state()

    def runtime_telemetry(self) -> dict:
        """Return a detached, JSON-safe execution summary for reports."""
        if self._runtime is None:
            if self.simulation_mode == "headless":
                return {
                    "runtime_mode": "headless_webots_logic",
                    "dynamics_version": HEADLESS_DYNAMICS_VERSION,
                    "physics_fidelity": "business_logic_only",
                    "initialized": False,
                }
            return {
                "runtime_mode": "webots_snapshot",
                "dynamics_version": "webots-live",
            }
        result = self._runtime.telemetry_snapshot()
        result["runtime_mode"] = "headless_webots_logic"
        result["physics_fidelity"] = "business_logic_only"
        result["initialized"] = True
        if self.reward_profile is not None:
            result["reward_profile_version"] = self.reward_profile.version
            result["reward_profile_hash"] = self.reward_profile.sha256
            result["reward_totals"] = dict(self._reward_totals)
        else:
            result["reward_profile_version"] = "legacy_reward"
            result["reward_profile_hash"] = None
        return result

    def step_scheduler(self, scheduler):
        """Execute one decision from any project ``BaseScheduler``.

        Matching schedulers may propose multiple assignments, just as they do
        in Webots.  The Supervisor commits one result and schedules again from
        the updated snapshot, so this adapter validates the entire proposal
        then commits only its first assignment through ``step``.
        """
        if self.simulation_mode != "headless" or self._runtime is None:
            raise RuntimeError(
                "step_scheduler is only available in headless mode")
        mask = self.get_action_mask()
        if mask[self.no_op_action]:
            transition = self.step(self.no_op_action)
            info = dict(transition[4])
            info.update({
                "scheduler_name": getattr(scheduler, "name", type(scheduler).__name__),
                "scheduler_no_op": True,
            })
            return transition[:4] + (info,)

        pending = list(self._task_slots)
        try:
            decision = scheduler.assign(pending, self._robots, self._context)
            valid, reason = validate_assignments(
                decision.assignments, pending, self._robots, self._context)
            if not decision.is_feasible or not valid:
                selected = (decision.assignments[0]
                            if decision.assignments else None)
                scheduler.on_assignment_rejected(selected, reason)
                return self._scheduler_rejection(
                    scheduler, reason, getattr(decision, "diagnostics", {}))
            selected = decision.assignments[0]
            action = self.action_for_pair(
                selected.robot_id, selected.task.task_id)
        except Exception as exc:
            reason = f"scheduler_adapter_error:{type(exc).__name__}"
            try:
                scheduler.on_assignment_rejected(None, reason)
            except Exception:
                pass
            return self._scheduler_rejection(
                scheduler, reason, {})

        diagnostics = dict(getattr(decision, "diagnostics", {}) or {})
        transition = self.step(
            action,
            candidate_attribution=not bool(diagnostics.get("fallback", False)))
        info = dict(transition[4])
        if info.get("assignment_rejected") or info.get("invalid_action"):
            scheduler.on_assignment_rejected(
                selected, info.get("reason", "runtime_rejected"))
        else:
            scheduler.on_assignment_committed(selected)
        info.update({
            "scheduler_name": getattr(
                decision, "algorithm_name", None) or
                getattr(scheduler, "name", type(scheduler).__name__),
            "scheduler_diagnostics": diagnostics,
            "scheduler_computation_time": float(
                getattr(decision, "computation_time", 0.0)),
        })
        return transition[:4] + (info,)

    def _scheduler_rejection(self, scheduler, reason: str, diagnostics: dict):
        self._step += 1
        name = getattr(scheduler, "name", type(scheduler).__name__)
        components = self._profile_reward([], invalid_action=True)
        reward = (self.reward_config.invalid_action if components is None
                  else components["reward_total"])
        truncated = self._step >= self.config.max_steps_per_episode
        return self.observe(), float(reward), False, truncated, {
                "scheduler_output_rejected": True,
                "reason": str(reason),
                "scheduler_name": name,
                "scheduler_diagnostics": dict(diagnostics or {}),
                "reward_components": components,
                **self._truncation_info(truncated),
                "completed_count": len(self._completed_ids),
                "dynamics_version": HEADLESS_DYNAMICS_VERSION,
            }
