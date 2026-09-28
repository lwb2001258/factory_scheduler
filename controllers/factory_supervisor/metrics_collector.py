"""
Metrics Collector for Smart Factory Simulation.

Collects, aggregates, and saves performance metrics for
experimental evaluation and comparison between scheduling methods.

Key Performance Indicators (KPIs):
1. System throughput (tasks completed per unit time)
2. Average task completion time
3. Robot idle time percentage
4. Number of coordination conflicts
5. Total distance travelled
"""

import json
import math
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from task_generator import TransportTask


@dataclass
class StepRecord:
    """Record of metrics at a single simulation timestep."""
    sim_time: float
    tasks_completed: int
    tasks_pending: int
    tasks_active: int
    robots_idle: int
    robots_active: int
    conflicts_total: int
    avg_battery: float
    min_pair_distance: Optional[float]
    robot_positions: Optional[dict] = None
    robot_task_states: Optional[dict] = None


@dataclass
class SafetyEvent:
    event_id: str
    sim_time: float
    event_type: str
    robot_id: int
    peer_id: Optional[int] = None
    path_version: int = 0
    decision: str = ""
    minimum_distance: Optional[float] = None
    ttc: Optional[float] = None


class MetricsCollector:
    """
    Collects and manages all performance metrics during simulation.
    """
    
    def __init__(self, scenario_name: str, scheduler_name: str, num_robots: int):
        self.scenario_name = scenario_name
        self.scheduler_name = scheduler_name
        self.num_robots = num_robots
        
        # Time-series data
        self.step_records: List[StepRecord] = []
        
        # Task-level data
        self.task_arrivals: List[dict] = []
        self.task_completions: List[dict] = []
        
        # Conflict/deadlock events
        self.conflict_events: List[dict] = []
        self.route_dispatch_events: List[dict] = []
        self.route_override_events: List[dict] = []
        self._active_conflicts: Dict[tuple, dict] = {}
        self._next_conflict_id = 1
        self._last_route_dispatch: Dict[int, dict] = {}
        self.conflict_episode_count = 0
        self.route_dispatch_count = 0
        self.route_override_count = 0
        self.unauthorized_route_write_count = 0
        self.unauthorized_route_write_events: List[dict] = []
        self.replan_events: List[dict] = []
        self.escape_events: List[dict] = []
        self.unplanned_stop_events: List[dict] = []
        self.replan_count = 0
        self.replan_request_count = 0
        self.replan_request_events: List[dict] = []
        self.escape_count = 0
        self.unplanned_stop_count = 0
        self._motion_watch: Dict[int, dict] = {}
        self.planned_wait_events: List[dict] = []
        self.planned_wait_count = 0
        self.planned_wait_total_seconds = 0.0
        self._active_planned_waits: Dict[int, dict] = {}
        self.yield_resume_events: List[dict] = []
        self.yield_start_count = 0
        self.yield_standoff_selected_count = 0
        self.yield_wait_start_count = 0
        self.yield_wait_cleared_count = 0
        self.yield_wait_timeout_count = 0
        self.yield_resume_proposed_count = 0
        self.yield_resume_rejected_count = 0
        self.yield_resume_committed_count = 0
        self.yield_resume_rollback_count = 0
        self.nonphysical_recoveries = 0
        self.nonphysical_recovery_events: List[dict] = []
        self.safety_event_count = 0
        self.minimum_pair_distance_seen = float('inf')
        self.pair_distance_violation_samples = 0
        self.deadlock_events: List[dict] = []
        self.safety_events: List[dict] = []
        self._last_safety_event: Dict[tuple, float] = {}
        self.scheduling_latencies_ms: List[float] = []
        self.invalid_scheduler_outputs = 0
        self.scheduler_fallbacks = 0
        self.native_scheduler_commits = 0
        self.fallback_scheduler_commits = 0
        self.commits_by_algorithm: Dict[str, int] = {}
        self.rl_inference_latencies_ms: List[float] = []
        self.rl_timeout_count = 0
        self.rl_policy_decisions = 0
        self.rl_fallback_decisions = 0
        
        # Output path
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "results"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(
            self.output_dir,
            f"experiment_{scenario_name}_{scheduler_name}_{timestamp}.json"
        )

    def record_scheduling_latency(self, seconds: float):
        if math.isfinite(seconds) and seconds >= 0:
            self.scheduling_latencies_ms.append(seconds * 1000.0)

    def record_scheduler_fallback(self, invalid_output: bool = True):
        self.scheduler_fallbacks += 1
        if invalid_output:
            self.invalid_scheduler_outputs += 1

    def record_rl_diagnostics(self, diagnostics: Optional[dict]):
        """Capture safety-wrapper diagnostics for auditable RL evaluation."""
        diagnostics = diagnostics or {}
        inference_ms = diagnostics.get("inference_ms")
        if isinstance(inference_ms, (int, float)) and math.isfinite(inference_ms):
            self.rl_inference_latencies_ms.append(float(inference_ms))
        self.rl_timeout_count = max(
            self.rl_timeout_count, int(diagnostics.get("timeout_count", 0)))
        self.rl_policy_decisions = max(
            self.rl_policy_decisions,
            int(diagnostics.get("policy_decisions", 0)))
        self.rl_fallback_decisions = max(
            self.rl_fallback_decisions,
            int(diagnostics.get("fallback_decisions", 0)))

    def record_scheduler_commit(self, algorithm_name: str,
                                native: bool = True):
        """Attribute committed work to the algorithm that actually chose it."""
        if native:
            self.native_scheduler_commits += 1
        else:
            self.fallback_scheduler_commits += 1
        name = str(algorithm_name or "unknown")
        self.commits_by_algorithm[name] = (
            self.commits_by_algorithm.get(name, 0) + 1)

    def record_unauthorized_route_write(self, sim_time: float, robot_id: int,
                                        source: str, path_version: int,
                                        plan_epoch=None):
        self.unauthorized_route_write_count += 1
        self.unauthorized_route_write_events.append({
            'sim_time': float(sim_time), 'robot_id': int(robot_id),
            'source': str(source), 'path_version': int(path_version),
            'plan_epoch': plan_epoch, 'decision': 'rejected',
        })
        self._cap_events(self.unauthorized_route_write_events)

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1,
                    max(0, math.ceil(percentile * len(ordered)) - 1))
        return ordered[index]
    
    def record_step(self, sim_time: float, robot_states: Dict[int, dict],
                    task_stats: dict, coord_stats: dict):
        """Record metrics at a simulation step."""
        from config import RobotState

        task_motion_states = {
            RobotState.EN_ROUTE_PICKUP, RobotState.CARRYING,
            RobotState.EN_ROUTE_DELIVERY,
        }
        for rid, state in robot_states.items():
            position = state.get('position')
            active = state.get('state') in task_motion_states
            planned_wait = state.get('planned_wait') is True
            wait_episode = self._active_planned_waits.get(rid)
            wait_ended = False
            if planned_wait:
                wait_identity = (
                    state.get('task_id'), state.get('plan_epoch'),
                    state.get('plan_source'),
                    int(state.get('path_version', 0)),
                    state.get('wait_reason'))
                if wait_episode is not None:
                    episode_identity = (
                        wait_episode.get('task_id'),
                        wait_episode.get('plan_epoch'),
                        wait_episode.get('plan_source'),
                        wait_episode.get('path_version'),
                        wait_episode.get('reason'))
                    if episode_identity != wait_identity:
                        wait_episode['ended_at'] = float(sim_time)
                        wait_episode['duration'] = max(
                            0.0, float(sim_time) -
                            wait_episode['started_at'])
                        wait_episode['outcome'] = 'superseded'
                        self.planned_wait_total_seconds += (
                            wait_episode['duration'])
                        self.planned_wait_events.append(wait_episode)
                        self._cap_events(self.planned_wait_events)
                        self._active_planned_waits.pop(rid, None)
                        wait_episode = None
                deadline = state.get('wait_deadline')
                if not isinstance(deadline, (int, float)) or not math.isfinite(
                        deadline):
                    deadline = None
                if wait_episode is None:
                    wait_episode = {
                        'robot_id': rid,
                        'started_at': float(sim_time),
                        'last_seen_at': float(sim_time),
                        'reason': state.get('wait_reason'),
                        'deadline': deadline,
                        'max_deadline': deadline,
                        'task_id': state.get('task_id'),
                        'plan_epoch': state.get('plan_epoch'),
                        'plan_source': state.get('plan_source'),
                        'path_version': int(state.get('path_version', 0)),
                        'goal_location': state.get('goal_location'),
                        'ended_at': None,
                        'duration': None,
                        'outcome': 'active',
                    }
                    self._active_planned_waits[rid] = wait_episode
                    self.planned_wait_count += 1
                else:
                    wait_episode['last_seen_at'] = float(sim_time)
                    if deadline is not None:
                        previous = wait_episode.get('max_deadline')
                        wait_episode['max_deadline'] = (
                            deadline if previous is None else
                            max(float(previous), deadline))
            elif wait_episode is not None:
                wait_episode = self._active_planned_waits.pop(rid)
                wait_episode['ended_at'] = float(sim_time)
                wait_episode['duration'] = max(
                    0.0, float(sim_time) - wait_episode['started_at'])
                max_deadline = wait_episode.get('max_deadline')
                wait_episode['outcome'] = (
                    'deadline_expired'
                    if max_deadline is not None and sim_time >= max_deadline
                    else 'resumed')
                self.planned_wait_total_seconds += wait_episode['duration']
                self.planned_wait_events.append(wait_episode)
                self._cap_events(self.planned_wait_events)
                wait_ended = True
            watch = self._motion_watch.get(rid)
            if not active or planned_wait or position is None:
                self._motion_watch[rid] = {
                    'position': position, 'progress_at': sim_time,
                    'reported': False}
                continue
            if wait_ended:
                self._motion_watch[rid] = {
                    'position': tuple(position), 'progress_at': sim_time,
                    'reported': False}
                continue
            if watch is None or watch.get('position') is None:
                self._motion_watch[rid] = {
                    'position': tuple(position), 'progress_at': sim_time,
                    'reported': False}
                continue
            distance = math.hypot(
                position[0] - watch['position'][0],
                position[1] - watch['position'][1])
            if distance >= 0.05:
                watch.update(position=tuple(position), progress_at=sim_time,
                             reported=False)
            elif (not watch['reported'] and
                  sim_time - watch['progress_at'] >= 10.0):
                event = {
                    'sim_time': sim_time, 'robot_id': rid,
                    'state': str(state.get('state')),
                    'stopped_seconds': sim_time - watch['progress_at'],
                    'position': list(position),
                    'path_version': int(state.get('path_version', 0)),
                    'plan_epoch': state.get('plan_epoch'),
                    'plan_source': state.get('plan_source'),
                    'task_id': state.get('task_id'),
                    'goal_location': state.get('goal_location'),
                    'planned_wait': False,
                }
                self.unplanned_stop_events.append(event)
                self.unplanned_stop_count += 1
                self._cap_events(self.unplanned_stop_events)
                watch['reported'] = True
        
        idle_count = sum(1 for rs in robot_states.values() 
                        if rs['state'] == RobotState.IDLE)
        active_count = len(robot_states) - idle_count
        avg_battery = sum(rs.get('battery', 100) for rs in robot_states.values()) / max(len(robot_states), 1)
        pair_distances = []
        robot_ids = list(robot_states)
        for index, rid_a in enumerate(robot_ids):
            pos_a = robot_states[rid_a].get('position')
            if pos_a is None:
                continue
            for rid_b in robot_ids[index + 1:]:
                pos_b = robot_states[rid_b].get('position')
                if pos_b is not None:
                    distance = ((pos_a[0] - pos_b[0]) ** 2 +
                                (pos_a[1] - pos_b[1]) ** 2) ** 0.5
                    pair_distances.append((distance, rid_a, rid_b))

        closest = min(pair_distances, default=(float('inf'), None, None))
        self.minimum_pair_distance_seen = min(
            self.minimum_pair_distance_seen, closest[0])
        if closest[0] < 0.50:
            self.pair_distance_violation_samples += 1
        for distance, rid_a, rid_b in pair_distances:
            if distance >= 0.50:
                continue
            self.record_safety_event(SafetyEvent(
                event_id=f"distance-{rid_a}-{rid_b}-{sim_time:.3f}",
                sim_time=sim_time,
                event_type="pair_distance_violation",
                robot_id=rid_a,
                peer_id=rid_b,
                decision="observed",
                minimum_distance=distance,
            ))
        
        record = StepRecord(
            sim_time=sim_time,
            tasks_completed=task_stats.get('completed', 0),
            tasks_pending=task_stats.get('pending', 0),
            tasks_active=task_stats.get('active', 0),
            robots_idle=idle_count,
            robots_active=active_count,
            conflicts_total=coord_stats.get('conflicts_resolved', 0),
            avg_battery=avg_battery,
            min_pair_distance=(closest[0]
                               if math.isfinite(closest[0]) else None),
            robot_positions={
                str(rid): [float(rs['position'][0]), float(rs['position'][1])]
                for rid, rs in robot_states.items()
                if rs.get('position') is not None
            },
            robot_task_states={
                str(rid): {
                    'state': str(rs.get('state')),
                    'task_id': rs.get('task_id'),
                    'goal_location': rs.get('goal_location'),
                    'plan_source': rs.get('plan_source'),
                    'plan_epoch': rs.get('plan_epoch'),
                    'path_version': int(rs.get('path_version', 0)),
                    'controller_waypoint_index': int(rs.get(
                        'controller_waypoint_index', 0)),
                    'planned_wait': bool(rs.get('planned_wait', False)),
                    'wait_reason': rs.get('wait_reason'),
                    'emergency_braking': bool(rs.get(
                        'emergency_braking', False)),
                    'speed_scale': float(rs.get('speed_scale', 1.0)),
                }
                for rid, rs in robot_states.items()
            },
        )
        self.step_records.append(record)

    @staticmethod
    def _cap_events(events: List[dict]):
        if len(events) > 10000:
            del events[:1000]

    def record_replan(self, sim_time: float, robot_id: int, source: str,
                      plan_epoch, result: str, dispatched: bool):
        if dispatched and result == 'succeeded':
            self.replan_count += 1
        self.replan_events.append({
            'sim_time': float(sim_time), 'robot_id': int(robot_id),
            'source': str(source), 'plan_epoch': plan_epoch,
            'result': str(result), 'dispatched': bool(dispatched),
        })
        self._cap_events(self.replan_events)

    def record_yield_event(self, event_type: str, sim_time: float,
                           robot_id: int, winner=None, target=None):
        """Record one deterministic yield/resume state-machine transition."""
        counter_attr = {
            'yield_start': 'yield_start_count',
            'yield_standoff_selected': 'yield_standoff_selected_count',
            'yield_wait_start': 'yield_wait_start_count',
            'yield_wait_cleared': 'yield_wait_cleared_count',
            'yield_wait_timeout': 'yield_wait_timeout_count',
            'yield_resume_proposed': 'yield_resume_proposed_count',
            'yield_resume_rejected': 'yield_resume_rejected_count',
            'yield_resume_committed': 'yield_resume_committed_count',
            'yield_resume_rollback': 'yield_resume_rollback_count',
        }.get(event_type)
        if counter_attr is not None:
            setattr(self, counter_attr, getattr(self, counter_attr, 0) + 1)
        self.yield_resume_events.append({
            'sim_time': float(sim_time), 'event_type': str(event_type),
            'robot_id': int(robot_id), 'winner': winner,
            'target': list(target) if target is not None else None,
        })
        self._cap_events(self.yield_resume_events)

    def record_nonphysical_recovery(self, sim_time: float, robot_id: int,
                                    target=None):
        """Audit teleport-style recoveries separately from physical escapes."""
        self.nonphysical_recoveries += 1
        self.nonphysical_recovery_events.append({
            'sim_time': float(sim_time), 'robot_id': int(robot_id),
            'target': list(target) if target is not None else None,
        })
        self._cap_events(self.nonphysical_recovery_events)

    def record_replan_request(self, sim_time: float, robot_id: int,
                              path_version: int, plan_epoch):
        self.replan_request_count += 1
        self.replan_request_events.append({
            'sim_time': float(sim_time), 'robot_id': int(robot_id),
            'path_version': int(path_version), 'plan_epoch': plan_epoch,
        })
        self._cap_events(self.replan_request_events)

    def record_escape(self, sim_time: float, robot_id: int,
                      source: str, target=None):
        self.escape_count += 1
        self.escape_events.append({
            'sim_time': float(sim_time), 'robot_id': int(robot_id),
            'source': str(source),
            'target': list(target) if target is not None else None,
        })
        self._cap_events(self.escape_events)
        if len(self.step_records) > 10000:
            del self.step_records[:1000]
    
    def record_task_arrival(self, task: TransportTask, sim_time: float):
        """Record a new task arrival."""
        self.task_arrivals.append({
            'task_id': task.task_id,
            'arrival_time': sim_time,
            'pickup': task.pickup_location,
            'delivery': task.delivery_location,
            'priority': task.priority,
        })
    
    def record_task_completion(self, task: TransportTask, robot_id: int, sim_time: float):
        """Record a task completion."""
        self.task_completions.append({
            'task_id': task.task_id,
            'robot_id': robot_id,
            'arrival_time': task.arrival_time,
            'assignment_time': task.assignment_time,
            'pickup_time': task.pickup_time,
            'completion_time': sim_time,
            'waiting_time': task.waiting_time,
            'completion_duration': task.completion_duration,
            'execution_time': task.execution_time,
            'pickup': task.pickup_location,
            'delivery': task.delivery_location,
        })
    
    def record_conflict_scan(self, conflicts, sim_time: float) -> None:
        """Track predicted conflicts as episodes from detection to resolve."""
        observed = {}
        for rid_a, rid_b, time_to_conflict, distance in conflicts:
            key = tuple(sorted((int(rid_a), int(rid_b))))
            observed[key] = (float(time_to_conflict), float(distance))
            episode = self._active_conflicts.get(key)
            if episode is None:
                episode = {
                    'conflict_id': self._next_conflict_id,
                    'type': 'predicted_trajectory',
                    'robots_involved': list(key),
                    'detected_at': float(sim_time),
                    'last_seen_at': float(sim_time),
                    'status': 'active',
                    'minimum_predicted_distance': float(distance),
                    'minimum_time_to_conflict': float(time_to_conflict),
                }
                self._next_conflict_id += 1
                self.conflict_episode_count += 1
                self._active_conflicts[key] = episode
                self.conflict_events.append(episode)
            else:
                episode['last_seen_at'] = float(sim_time)
                episode['minimum_predicted_distance'] = min(
                    episode['minimum_predicted_distance'], float(distance))
                episode['minimum_time_to_conflict'] = min(
                    episode['minimum_time_to_conflict'],
                    float(time_to_conflict))
        for key in set(self._active_conflicts) - set(observed):
            episode = self._active_conflicts.pop(key)
            episode['status'] = 'resolved'
            episode['resolved_at'] = float(sim_time)
        while len(self.conflict_events) > 10000:
            resolved_index = next(
                (i for i, item in enumerate(self.conflict_events)
                 if item.get('status') == 'resolved'), None)
            if resolved_index is None:
                break
            del self.conflict_events[resolved_index]

    def record_route_dispatch(self, robot_id: int, path_version: int,
                              plan_epoch: int, source: str, waypoint_count: int,
                              sim_time: float) -> bool:
        """Audit route ownership and rapid cross-source replacement."""
        event = {
            'sim_time': float(sim_time), 'robot_id': int(robot_id),
            'path_version': int(path_version), 'source': str(source),
            'plan_epoch': int(plan_epoch),
            'waypoint_count': int(waypoint_count),
        }
        previous = self._last_route_dispatch.get(int(robot_id))
        if (previous is not None and
                sim_time - previous['sim_time'] < 1.0 and
                previous['path_version'] != event['path_version']):
            self.route_override_events.append({
                'sim_time': float(sim_time), 'robot_id': int(robot_id),
                'previous_source': previous['source'],
                'new_source': event['source'],
                'previous_path_version': previous['path_version'],
                'new_path_version': int(path_version),
                'previous_plan_epoch': previous['plan_epoch'],
                'new_plan_epoch': int(plan_epoch),
            })
            self.route_override_count += 1
        self._last_route_dispatch[int(robot_id)] = event
        self.route_dispatch_events.append(event)
        self.route_dispatch_count += 1
        if len(self.route_dispatch_events) > 10000:
            del self.route_dispatch_events[:1000]
        if len(self.route_override_events) > 10000:
            del self.route_override_events[:1000]
        return True
    
    def record_deadlock(self, cycle: List[int], sim_time: float):
        """Record a deadlock detection event."""
        self.deadlock_events.append({
            'sim_time': sim_time,
            'robots_involved': cycle,
        })

    def record_safety_event(self, event: SafetyEvent,
                            throttle_seconds: float = 0.5) -> bool:
        """Record state changes while suppressing repeated per-frame noise."""
        key = (event.event_type, event.robot_id, event.peer_id, event.decision)
        previous = self._last_safety_event.get(key)
        if previous is not None and event.sim_time - previous < throttle_seconds:
            return False
        self._last_safety_event[key] = event.sim_time
        self.safety_events.append(asdict(event))
        self.safety_event_count += 1
        if len(self.safety_events) > 10000:
            del self.safety_events[:1000]
        return True
    
    def compute_final_metrics(self, robots: dict, task_stats: dict,
                              coord_stats: dict, total_time: float) -> dict:
        """
        Compute final summary metrics for the experiment.
        
        Returns comprehensive metrics dictionary.
        """
        # Throughput
        throughput = task_stats.get('completed', 0) / max(total_time / 60.0, 1)  # per minute
        
        # Average task completion time
        completion_times = [tc['completion_duration'] for tc in self.task_completions 
                          if tc['completion_duration'] is not None]
        avg_completion = sum(completion_times) / max(len(completion_times), 1)
        
        # Average waiting time
        waiting_times = [tc['waiting_time'] for tc in self.task_completions 
                        if tc['waiting_time'] is not None]
        avg_waiting = sum(waiting_times) / max(len(waiting_times), 1)
        
        # Robot idle time
        robot_idle_pcts = []
        total_distance = 0.0
        tasks_per_robot = []
        
        for rid, robot in robots.items():
            idle_pct = (robot.idle_time / max(total_time, 1)) * 100
            robot_idle_pcts.append(idle_pct)
            total_distance += robot.total_distance
            tasks_per_robot.append(robot.tasks_completed)
        
        avg_idle_pct = sum(robot_idle_pcts) / max(len(robot_idle_pcts), 1)
        
        # Workload balance (coefficient of variation)
        if tasks_per_robot and sum(tasks_per_robot) > 0:
            mean_tasks = sum(tasks_per_robot) / len(tasks_per_robot)
            variance = sum((t - mean_tasks)**2 for t in tasks_per_robot) / len(tasks_per_robot)
            workload_cv = (variance ** 0.5) / max(mean_tasks, 1)
        else:
            workload_cv = 0.0
        
        latency = self.scheduling_latencies_ms
        completion_marks = sorted(
            float(item['completion_time']) for item in self.task_completions
            if item.get('completion_time') is not None)
        progress_marks = [0.0] + completion_marks + [float(total_time)]
        completion_gaps = [later - earlier for earlier, later in zip(
            progress_marks, progress_marks[1:])]
        active_wait_seconds = sum(
            max(0.0, float(total_time) - episode['started_at'])
            for episode in self._active_planned_waits.values())
        return {
            # Primary KPIs
            "throughput_per_minute": throughput,
            "total_tasks_completed": task_stats.get('completed', 0),
            "total_tasks_generated": task_stats.get('total_generated', 0),
            "avg_task_completion_time": avg_completion,
            "avg_waiting_time": avg_waiting,
            "avg_robot_idle_pct": avg_idle_pct,
            "total_distance_all_robots": total_distance,
            "total_conflicts_resolved": coord_stats.get('conflicts_resolved', 0),
            "predicted_conflicts": self.conflict_episode_count,
            "route_dispatches": self.route_dispatch_count,
            "rapid_route_overrides": self.route_override_count,
            "unauthorized_route_writes": self.unauthorized_route_write_count,
            "audited_replans": self.replan_count,
            "replan_requests": self.replan_request_count,
            "physical_escapes": self.escape_count,
            "nonphysical_recoveries": self.nonphysical_recoveries,
            "yield_start": self.yield_start_count,
            "yield_standoff_selected": self.yield_standoff_selected_count,
            "yield_wait_start": self.yield_wait_start_count,
            "yield_wait_cleared": self.yield_wait_cleared_count,
            "yield_wait_timeout": self.yield_wait_timeout_count,
            "yield_resume_proposed": self.yield_resume_proposed_count,
            "yield_resume_rejected": self.yield_resume_rejected_count,
            "yield_resume_committed": self.yield_resume_committed_count,
            "yield_resume_rollback": self.yield_resume_rollback_count,
            "safety_event_count": self.safety_event_count,
            "unplanned_task_stops": self.unplanned_stop_count,
            "planned_wait_episodes": self.planned_wait_count,
            "planned_wait_total_seconds": (
                self.planned_wait_total_seconds + active_wait_seconds),
            "total_deadlocks": coord_stats.get('deadlocks_detected', 0),
            "min_pair_distance": (
                self.minimum_pair_distance_seen
                if math.isfinite(self.minimum_pair_distance_seen) else None),
            "pair_distance_violations": (
                self.pair_distance_violation_samples),
            
            # Secondary metrics
            "max_completion_time": max(completion_times) if completion_times else 0,
            "min_completion_time": min(completion_times) if completion_times else 0,
            "workload_balance_cv": workload_cv,
            "tasks_per_robot": tasks_per_robot,
            "robot_idle_percentages": robot_idle_pcts,
            "total_replans": coord_stats.get('total_replans', 0),
            "joint_plans_attempted": coord_stats.get(
                'joint_plans_attempted', 0),
            "joint_plans_accepted": coord_stats.get(
                'joint_plans_accepted', 0),
            "joint_plan_rollbacks": coord_stats.get(
                'joint_plan_rollbacks', 0),
            "longest_completion_plateau_seconds": max(
                completion_gaps, default=float(total_time)),
            "final_completion_plateau_seconds": (
                float(total_time) - completion_marks[-1]
                if completion_marks else float(total_time)),
            "robots_without_completed_tasks": sum(
                1 for count in tasks_per_robot if count == 0),
            "scheduling_latency_mean_ms": (
                statistics.fmean(latency) if latency else 0.0),
            "scheduling_latency_p50_ms": self._percentile(latency, 0.50),
            "scheduling_latency_p95_ms": self._percentile(latency, 0.95),
            "scheduling_latency_p99_ms": self._percentile(latency, 0.99),
            "scheduling_latency_max_ms": max(latency, default=0.0),
            "invalid_scheduler_outputs": self.invalid_scheduler_outputs,
            "scheduler_fallbacks": self.scheduler_fallbacks,
            "native_scheduler_commits": self.native_scheduler_commits,
            "fallback_scheduler_commits": self.fallback_scheduler_commits,
            "scheduler_commits_by_algorithm": dict(
                self.commits_by_algorithm),
            "rl_policy_decisions": self.rl_policy_decisions,
            "rl_fallback_decisions": self.rl_fallback_decisions,
            "rl_fallback_rate": (
                self.rl_fallback_decisions /
                max(1, self.rl_policy_decisions + self.rl_fallback_decisions)),
            "rl_timeout_count": self.rl_timeout_count,
            "rl_inference_mean_ms": (
                statistics.fmean(self.rl_inference_latencies_ms)
                if self.rl_inference_latencies_ms else 0.0),
            "rl_inference_p95_ms": self._percentile(
                self.rl_inference_latencies_ms, 0.95),
            "rl_inference_max_ms": max(
                self.rl_inference_latencies_ms, default=0.0),
        }
    
    def save_results(self, robots: dict, task_stats: dict,
                     coord_stats: dict, total_time: float):
        """Save all results to JSON file."""
        final_metrics = self.compute_final_metrics(
            robots, task_stats, coord_stats, total_time
        )
        
        active_wait_events = []
        for episode in self._active_planned_waits.values():
            snapshot = dict(episode)
            snapshot['duration'] = max(
                0.0, float(total_time) - snapshot['started_at'])
            active_wait_events.append(snapshot)

        results = {
            "experiment_info": {
                "scenario": self.scenario_name,
                "scheduler": self.scheduler_name,
                "num_robots": self.num_robots,
                "sim_duration": total_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary_metrics": final_metrics,
            "task_stats": task_stats,
            "coordination_stats": coord_stats,
            "task_completions": self.task_completions,
            "conflict_events": self.conflict_events,
            "route_dispatch_events": self.route_dispatch_events,
            "route_override_events": self.route_override_events,
            "unauthorized_route_write_events": (
                self.unauthorized_route_write_events),
            "replan_events": self.replan_events,
            "replan_request_events": self.replan_request_events,
            "escape_events": self.escape_events,
            "yield_resume_events": self.yield_resume_events,
            "nonphysical_recovery_events": self.nonphysical_recovery_events,
            "unplanned_stop_events": self.unplanned_stop_events,
            "planned_wait_events": (
                self.planned_wait_events + active_wait_events),
            "deadlock_events": self.deadlock_events,
            "safety_events": self.safety_events,
            "time_series": [asdict(sr) for sr in self.step_records[-100:]],  # last 100 steps
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str, allow_nan=False)
        
        return self.output_path
    
    def print_summary(self, metrics: dict):
        """Print a formatted summary of metrics."""
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        print(f"Scenario: {self.scenario_name} | Scheduler: {self.scheduler_name}")
        print(f"Robots: {self.num_robots}")
        print("-"*50)
        print(f"Throughput:           {metrics['throughput_per_minute']:.2f} tasks/min")
        print(f"Tasks completed:      {metrics['total_tasks_completed']}/{metrics['total_tasks_generated']}")
        print(f"Avg completion time:  {metrics['avg_task_completion_time']:.2f}s")
        print(f"Avg waiting time:     {metrics['avg_waiting_time']:.2f}s")
        print(f"Avg robot idle:       {metrics['avg_robot_idle_pct']:.1f}%")
        print(f"Total distance:       {metrics['total_distance_all_robots']:.1f}m")
        print(f"Conflicts resolved:   {metrics['total_conflicts_resolved']}")
        print(f"Deadlocks:           {metrics['total_deadlocks']}")
        print(f"Workload balance CV: {metrics['workload_balance_cv']:.3f}")
        print("="*50 + "\n")
