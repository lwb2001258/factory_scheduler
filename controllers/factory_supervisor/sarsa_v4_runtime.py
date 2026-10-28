"""Commit-aware runtime adapter for neural Expected SARSA."""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

import numpy as np

from schedulers import BaseScheduler, ModelValidationError, SchedulerResult
from sarsa_v3_runtime import EventRewardLedger
from sarsa_v4_agent import MODEL_VERSION, SarsaV4Agent
from sarsa_v4_features import WAIT_ACTION, build_snapshot, select_candidates


class SarsaV4Scheduler(BaseScheduler):
    uses_traffic = True
    sarsa_v4 = True
    pure_policy_required = True

    def __init__(self, model_path=None, *, agent=None, seed=42,
                 training=None, physical=True, device=None):
        super().__init__("SARSA")
        try:
            self.agent = agent or SarsaV4Agent.load(
                model_path, device=device)
        except Exception as exc:
            raise ModelValidationError(
                f"SARSA v4 checkpoint rejected: {exc}") from exc
        self.training = (
            os.environ.get("SMART_FACTORY_SARSA_V4_TRAINING", "0") == "1"
            if training is None else bool(training))
        self.physical = bool(physical)
        self.input_path = Path(model_path).resolve() if model_path else None
        self.initial_document = self.agent.document()
        self.initial_optimizer_updates = self.agent.optimizer_updates
        self.initial_transition_updates = self.agent.transition_updates
        if self.training:
            self.agent.start_episode(seed=seed)
        self.ledger = None
        self.sources = []
        self.now = 0.0
        self.collisions = 0
        self.previous = None
        self.previous_total = 0.0
        self.proposal = None
        self.initial_reward_total = 0.0
        self.wait_until = 0.0
        self.consecutive_waits = 0
        self.decisions = []
        self.td_errors = []
        self.latencies = []
        self.final_report = None

    def observe(self, tasks, now, collision_count=0, *, utility_profile=None):
        self.sources = list(tasks)
        self.now = float(now)
        self.collisions = int(collision_count)
        if self.training:
            if self.ledger is None:
                self.ledger = EventRewardLedger(
                    self.agent.objective, utility_profile)
            self.ledger.observe(self.sources, self.now, self.collisions)

    def _accept(self, decision, action, kind, robot_id=None, task_id=None):
        reward = None
        if self.training:
            if self.ledger is None:
                raise RuntimeError(
                    "observe tasks before neural SARSA training decisions")
            if self.previous is not None:
                reward = self.ledger.total - self.previous_total
                previous_decision, previous_action = self.previous
                self.td_errors.extend(self.agent.observe_transition(
                    previous_decision, previous_action, reward, decision,
                    physical=self.physical))
            else:
                self.initial_reward_total = self.ledger.total
            self.previous = (decision, int(action))
            self.previous_total = self.ledger.total
        self.decisions.append({
            "time": self.now, "action": int(action), "kind": kind,
            "robot_id": robot_id, "task_id": task_id,
            "reward_for_previous": reward})

    def assign(self, pending_tasks, robot_states, context=None):
        started = time.perf_counter()
        if context is None:
            raise ValueError("SARSA v4 requires runtime context")
        self.now = float(context.current_time)
        if self.now < self.wait_until:
            return SchedulerResult(
                is_feasible=True, algorithm_name=self.name,
                diagnostics={
                    "intentional_wait": True,
                    "reason": "sarsa_v4_wait_hold"})
        _candidates, pending = select_candidates(pending_tasks, self.now)
        horizon = float(context.configuration.get(
            "episode_end_time", 1800.0))
        oldest = max((self.now - task.arrival_time for task in pending),
                     default=0.0)
        busy = any(state.get("current_task") is not None
                   for state in robot_states.values())
        allow_wait = (
            busy and
            self.consecutive_waits < self.agent.config.max_consecutive_waits and
            oldest < 600.0 and
            horizon - self.now > self.agent.config.wait_seconds)
        snapshot = build_snapshot(
            robot_states, pending_tasks, context, allow_wait=allow_wait)
        if not snapshot.mask[:WAIT_ACTION].any():
            self.proposal = None
            return SchedulerResult(
                is_feasible=True, algorithm_name=self.name,
                computation_time=time.perf_counter() - started,
                diagnostics={
                    "intentional_wait": True,
                    "reason": "sarsa_v4_no_feasible_pair"})
        action, decision = self.agent.select_action(
            snapshot, training=self.training, physical=self.physical)
        assignment = snapshot.assignment(action)
        if action == WAIT_ACTION:
            self._accept(decision, action, "wait")
            if self.training:
                self.ledger.action_penalty("defer", self.now)
            self.wait_until = self.now + self.agent.config.wait_seconds
            self.consecutive_waits += 1
            self.proposal = None
        else:
            self.proposal = (
                assignment.robot_id, assignment.task.task_id,
                decision, action)
        elapsed = time.perf_counter() - started
        self.latencies.append(elapsed)
        return SchedulerResult(
            assignments=[assignment] if assignment else [],
            objective_value=float(self.agent.values(decision)[action]),
            computation_time=elapsed, is_feasible=True,
            algorithm_name=self.name,
            diagnostics={
                "model_version": MODEL_VERSION, "fallback": False,
                "action": action, "intentional_wait": action == WAIT_ACTION,
                "reason": "sarsa_v4_wait" if action == WAIT_ACTION else "valid",
                "inference_ms": elapsed * 1000.0,
                "candidate_count": len(snapshot.tasks),
                "pending_count": snapshot.pending_count,
                "device": self.agent.device.type})

    def _consume_proposal(self, assignment, kind):
        if (self.proposal is None or assignment is None or
                self.proposal[:2] != (
                    assignment.robot_id, assignment.task.task_id)):
            raise RuntimeError(
                "neural SARSA proposed action differs from runtime outcome")
        robot_id, task_id, decision, action = self.proposal
        self._accept(
            decision, action, kind, robot_id=robot_id, task_id=task_id)
        self.proposal = None

    def on_assignment_committed(self, assignment):
        self._consume_proposal(assignment, "commit")
        self.consecutive_waits = 0
        self.assignments_made += 1
        if self.training:
            self.ledger.note_commit(assignment, self.now)
            self.ledger.observe(self.sources, self.now, self.collisions)

    def on_assignment_rejected(self, assignment, reason):
        self._consume_proposal(assignment, "rejected")
        if self.training:
            self.ledger.action_penalty("invalid", self.now)

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        raise RuntimeError(
            "SARSA v4 requires the context-aware assign interface")

    def finalize(self, tasks, now, collision_count=0, *,
                 utility_profile=None, output_path=None):
        if self.final_report is not None:
            return self.final_report
        if not math.isclose(float(now), 1800.0, abs_tol=1e-8):
            raise RuntimeError(
                "neural SARSA episode must finish at the registered horizon")
        terminal_reward = None
        if self.training:
            if self.ledger is None:
                self.ledger = EventRewardLedger(
                    self.agent.objective, utility_profile)
            self.ledger.observe(
                tasks, float(now), int(collision_count), final=True)
            if self.previous is None:
                raise RuntimeError(
                    "neural SARSA episode contains no accepted decisions")
            terminal_reward = self.ledger.total - self.previous_total
            previous_decision, previous_action = self.previous
            self.td_errors.extend(self.agent.observe_transition(
                previous_decision, previous_action, terminal_reward,
                done=True, physical=self.physical))
            learned_return = sum(
                row["reward_for_previous"] or 0.0
                for row in self.decisions) + terminal_reward
            if not math.isclose(
                    learned_return + self.initial_reward_total,
                    self.ledger.total, abs_tol=1e-7):
                raise RuntimeError(
                    "neural SARSA lost or duplicated event rewards")
            self.agent.end_episode()
        elif self.agent.document() != self.initial_document:
            raise RuntimeError("frozen neural SARSA evaluation changed model state")
        if output_path is not None:
            if not self.training:
                raise RuntimeError("evaluation must not export updated weights")
            path = Path(output_path).resolve()
            run_root = Path(os.environ[
                "SMART_FACTORY_SARSA_V4_RUN_ROOT"]).resolve()
            if (path == self.input_path or not path.is_relative_to(run_root) or
                    path.exists()):
                raise RuntimeError(
                    "neural SARSA output would overwrite an existing model")
            self.agent.save(path)
        self.final_report = {
            "status": "complete", "model_version": MODEL_VERSION,
            "objective": self.agent.objective,
            "training": self.training,
            "optimizer_updates": (
                self.agent.optimizer_updates - self.initial_optimizer_updates),
            "transition_updates": (
                self.agent.transition_updates - self.initial_transition_updates),
            "mean_absolute_td": (
                float(np.mean(np.abs(self.td_errors)))
                if self.td_errors else 0.0),
            "max_absolute_td": (
                float(np.max(np.abs(self.td_errors)))
                if self.td_errors else 0.0),
            "last_loss": self.agent.last_loss,
            "decision_count": len(self.decisions),
            "decisions": self.decisions,
            "wait_count": sum(
                row["kind"] == "wait" for row in self.decisions),
            "rejected_count": sum(
                row["kind"] == "rejected" for row in self.decisions),
            "max_inference_ms": max(self.latencies, default=0.0) * 1000.0,
            "output_checkpoint": str(output_path) if output_path else None,
            "terminal_reward": terminal_reward,
            "initial_reward_unassigned": self.initial_reward_total,
            "reward_audit": self.ledger.report() if self.ledger else None,
            "device": self.agent.device.type,
        }
        return self.final_report


__all__ = ("EventRewardLedger", "SarsaV4Scheduler")
