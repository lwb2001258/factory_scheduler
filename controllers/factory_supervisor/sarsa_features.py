"""SARSA v3 observations: all eight robots, twenty candidates, shared traffic.

Only information available at the decision time is used. Action indices retain
the 8*20 + WAIT layout; value parameters are shared across all pair actions.
Legacy RL observations and GraphImitation feature contracts are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from config import RobotState
from schedulers import Assignment, build_cost_matrix
from task_generator import TaskStatus
from traffic_features import TrafficSnapshot, TRAFFIC_FEATURE_NAMES, NOMINAL_SPEED

FEATURE_VERSION = "sarsa-pair-context-v3"
MAX_ROBOTS, MAX_TASKS, WAIT_ACTION = 8, 20, 160
ACTION_DIM = WAIT_ACTION + 1
ROBOT_FIELDS = ("present", "x", "y", "idle", "battery", "faulted", "busy",
                "completed", "distance", "route_seconds", "speed", "wait", "stops",
                "active_pickup_x", "active_pickup_y", "active_delivery_x", "active_delivery_y")
TASK_FIELDS = ("present", "pickup_x", "pickup_y", "delivery_x", "delivery_y",
               "priority", "waiting", "has_deadline", "deadline_remaining",
               "horizon_limited", "feasible_fraction", "minimum_route_distance")
GLOBAL_FIELDS = ("bias", "elapsed", "remaining", "robot_count", "idle_fraction",
                 "pending_count", "queue_overflow", "oldest_wait", "mean_battery")
PAIR_FIELDS = ("dispatch", "wait_action", "robot_x", "robot_y", "battery",
               "pickup_x", "pickup_y", "delivery_x", "delivery_y", "empty_distance",
               "loaded_distance", "route_distance", "estimated_seconds", "priority",
               "task_wait", "deadline_slack", "completion_margin", "misses_horizon",
               "cost_above_robot_best", "cost_above_task_best",
               "remaining_x_duration", "load_x_duration", "priority_x_duration",
               "waiting_x_duration", "traffic_x_duration", "battery_x_distance",
               "deadline_x_duration", "busy_nearest_pickup_distance",
               "wait_remaining", "wait_load", "wait_busy_fraction")
FEATURE_NAMES = (GLOBAL_FIELDS +
    tuple(f"robot_{i}_{f}" for i in range(MAX_ROBOTS) for f in ROBOT_FIELDS) +
    tuple(f"task_{i}_{f}" for i in range(MAX_TASKS) for f in TASK_FIELDS) +
    PAIR_FIELDS + TRAFFIC_FEATURE_NAMES +
    tuple(f"peer_{i}_pickup_distance" for i in range(MAX_ROBOTS)) +
    tuple(f"alternative_{i}_pickup_distance" for i in range(MAX_TASKS)))
FEATURE_DIM = len(FEATURE_NAMES)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
TRAFFIC_SCALE = np.asarray([1., 1., 8., 8., 1., 30., 3., 30., 10.])


def _finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite SARSA observation")
    return result


def _position(value):
    if value is None or len(value) != 2:
        raise ValueError("SARSA position must have two coordinates")
    return (_finite(value[0]), _finite(value[1]))


def select_candidates(tasks, now):
    """Bounded queue with aging; future arrivals cannot enter the policy."""
    pending = [t for t in tasks if t.status == TaskStatus.PENDING and
               _finite(t.arrival_time) <= now + 1e-9]
    if len({t.task_id for t in pending}) != len(pending):
        raise ValueError("duplicate SARSA task IDs")
    def key(task):
        age = max(0., now - task.arrival_time)
        deadline = getattr(task, "deadline_time", None)
        urgency = max(0., 1. - (deadline-now)/300.) if deadline is not None else 0.
        return (-(task.priority + age/300. + urgency), task.arrival_time, task.task_id)
    return tuple(sorted(pending, key=key)[:MAX_TASKS]), tuple(pending)


@dataclass(frozen=True)
class ActionSnapshot:
    time: float
    robot_ids: tuple
    tasks: tuple
    features: np.ndarray
    mask: np.ndarray
    route_costs: np.ndarray
    pending_count: int

    def assignment(self, action):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("invalid SARSA action type")
        if not 0 <= action < ACTION_DIM or not self.mask[action]:
            raise ValueError("illegal SARSA action")
        if action == WAIT_ACTION:
            return None
        row, column = divmod(int(action), MAX_TASKS)
        return Assignment(self.robot_ids[row], self.tasks[column],
                          float(self.route_costs[row, column]))


def build_snapshot(robots, tasks, context, *, allow_wait=False):
    now = _finite(context.current_time)
    horizon = _finite(context.configuration.get("episode_end_time", 1800.))
    if horizon <= 0 or not 0 <= now <= horizon + 1e-6:
        raise ValueError("invalid SARSA episode clock")
    robot_ids = tuple(sorted(robots))
    if len(robot_ids) > MAX_ROBOTS:
        raise ValueError("SARSA v3 supports at most eight robots")
    traffic = context.traffic_context
    if (not isinstance(traffic, TrafficSnapshot) or traffic.now != now or
            set(traffic.robots) != set(robot_ids)):
        raise ValueError("SARSA requires current traffic for every robot")
    candidates, pending = select_candidates(tasks, now)
    matrix = build_cost_matrix(list(candidates), robots, context)
    mr = {rid: i for i, rid in enumerate(matrix.robot_ids)}
    mt = {task.task_id: i for i, task in enumerate(matrix.tasks)}
    remaining = max(0., horizon-now)
    idle = sum(s.get("state") == RobotState.IDLE and s.get("current_task") is None
               for s in robots.values())
    positions = {rid: _position(robots[rid].get("position")) for rid in robot_ids}
    route_costs = np.full((MAX_ROBOTS, MAX_TASKS), np.inf)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for ri, rid in enumerate(robot_ids):
        for ti, task in enumerate(candidates):
            if rid in mr and matrix.feasible[mr[rid], mt[task.task_id]]:
                distance = (context.path_cost_provider(rid, task)
                            if context.path_cost_provider else
                            math.dist(positions[rid], task.pickup_position) +
                            math.dist(task.pickup_position, task.delivery_position))
                distance = _finite(distance)
                if distance < 0:
                    raise ValueError("negative SARSA route cost")
                route_costs[ri, ti] = distance
                mask[ri*MAX_TASKS+ti] = True
    mask[WAIT_ACTION] = bool(allow_wait or not mask[:WAIT_ACTION].any())
    context_values = [1., now/horizon, remaining/horizon, len(robot_ids)/8.,
        idle/max(1,len(robot_ids)), len(pending)/20., max(0,len(pending)-20)/20.,
        max((now-t.arrival_time for t in pending), default=0.)/horizon,
        sum(_finite(s.get("battery",100.)) for s in robots.values())/max(1,len(robots))/100.]
    for slot in range(MAX_ROBOTS):
        if slot >= len(robot_ids):
            context_values.extend([0.]*len(ROBOT_FIELDS)); continue
        rid = robot_ids[slot]; state = robots[rid]; motion = traffic.robots[rid]
        active = state.get("current_task")
        px, py = _position(getattr(active,"pickup_position",(0.,0.)))
        dx, dy = _position(getattr(active,"delivery_position",(0.,0.)))
        points = (motion.position,) + motion.route
        route_seconds = sum(math.dist(a,b) for a,b in zip(points,points[1:])) / max(.08,motion.speed)
        route_seconds += max(0.,motion.not_before-now)
        x,y = positions[rid]
        context_values.extend([1.,x/12.,y/8.,float(state.get("state")==RobotState.IDLE),
            _finite(state.get("battery",100.))/100.,
            float(bool(state.get("faulted",False) or state.get("failed",False))),
            float(active is not None),_finite(state.get("tasks_completed",0.))/100.,
            _finite(state.get("total_distance",0.))/1000.,route_seconds/horizon,
            motion.speed/.6,motion.history.wait_seconds/30.,motion.history.stop_events/10.,
            px/12.,py/8.,dx/12.,dy/8.])
    for slot in range(MAX_TASKS):
        if slot >= len(candidates):
            context_values.extend([0.]*len(TASK_FIELDS)); continue
        task = candidates[slot]; px,py = _position(task.pickup_position)
        dx,dy = _position(task.delivery_position); deadline = task.deadline_time
        finite = route_costs[:,slot][np.isfinite(route_costs[:,slot])]
        context_values.extend([1.,px/12.,py/8.,dx/12.,dy/8.,task.priority/3.,
            (now-task.arrival_time)/horizon,float(deadline is not None),
            (deadline-now)/horizon if deadline is not None else 0.,
            float(task.is_horizon_limited(horizon)),len(finite)/max(1,idle),
            float(np.min(finite))/60. if len(finite) else 0.])
    features = np.zeros((ACTION_DIM, FEATURE_DIM), dtype=np.float64)
    width = len(context_values)
    features[mask,:width] = np.asarray(context_values)
    busy_positions = [positions[rid] for rid in robot_ids if robots[rid].get("current_task") is not None]
    for action in np.flatnonzero(mask[:WAIT_ACTION]):
        ri,ti = divmod(int(action),MAX_TASKS);rid=robot_ids[ri];task=candidates[ti]
        position=positions[rid]; pickup=_position(task.pickup_position); delivery=_position(task.delivery_position)
        route=route_costs[ri,ti]; motion=traffic.robots[rid]
        traffic_values=np.asarray(traffic.features(rid,pickup,delivery),np.float64)
        duration=route/max(.08,motion.speed)+traffic_values[5]
        norm_duration=duration/horizon; load=len(pending)/20.
        deadline=task.deadline_time
        slack=(deadline-now-duration)/horizon if deadline is not None else 0.
        row=route_costs[ri]; col=route_costs[:,ti]
        pair=[1.,0.,position[0]/12.,position[1]/8.,robots[rid].get("battery",100.)/100.,
            pickup[0]/12.,pickup[1]/8.,delivery[0]/12.,delivery[1]/8.,
            math.dist(position,pickup)/30.,math.dist(pickup,delivery)/30.,route/60.,
            norm_duration,task.priority/3.,(now-task.arrival_time)/horizon,slack,
            (remaining-duration)/horizon,float(duration>remaining),
            (route-np.min(row[np.isfinite(row)]))/60.,(route-np.min(col[np.isfinite(col)]))/60.,
            remaining/horizon*norm_duration,load*norm_duration,task.priority/3.*norm_duration,
            (now-task.arrival_time)/horizon*norm_duration,traffic_values[2]/8.*norm_duration,
            robots[rid].get("battery",100.)/100.*route/60.,slack*norm_duration,
            min((math.dist(p,pickup) for p in busy_positions),default=0.)/30.,0.,0.,0.]
        peer=[math.dist(positions[r],pickup)/30. for r in robot_ids]+[0.]*(8-len(robot_ids))
        alternative=[math.dist(position,t.pickup_position)/30. for t in candidates]+[0.]*(20-len(candidates))
        features[action,width:]=np.concatenate((pair,traffic_values/TRAFFIC_SCALE,peer,alternative))
    if mask[WAIT_ACTION]:
        for name,value in {"wait_action":1.,"wait_remaining":remaining/horizon,
                           "wait_load":len(pending)/20.,"wait_busy_fraction":len(busy_positions)/max(1,len(robots))}.items():
            features[WAIT_ACTION,INDEX[name]]=value
    if not np.isfinite(features).all():
        raise ValueError("nonfinite SARSA feature tensor")
    # Bounded scaling makes linear trace updates stable without fitting a scaler
    # on validation/evaluation data. No discretization or early time saturation.
    features=np.clip(features,-5.,5.)
    for value in (features,mask,route_costs):
        value.setflags(write=False)
    return ActionSnapshot(now,robot_ids,candidates,features,mask,route_costs,len(pending))
