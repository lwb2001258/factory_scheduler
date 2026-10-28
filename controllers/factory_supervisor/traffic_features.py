"""Read-only, bounded traffic forecasts for task assignment.

Times are absolute simulation seconds. Forecasts are estimates, not motion
commands or collision guarantees. One snapshot is shared by every candidate
and its expert labels. Capacity is a coarse free-area proxy (robot slots),
not a measurement of corridor throughput.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from types import MappingProxyType


TRAFFIC_VERSION = "route-traffic-v1"
NOMINAL_SPEED = 0.22
HORIZON = 30.0
TIME_BIN = 1.0
CELL_SIZE = 1.0
HISTORY_SECONDS = 30.0
TRAFFIC_FEATURE_NAMES = (
    "traffic_route_available",
    "traffic_forecast_coverage",
    "traffic_peak_capacity_load",
    "traffic_mean_capacity_load",
    "traffic_peer_fleet_fraction",
    "traffic_expected_wait_seconds",
    "traffic_peer_speed_ratio",
    "traffic_peer_recent_wait_seconds",
    "traffic_peer_recent_stop_events",
)


def _xy(point):
    result = (float(point[0]), float(point[1]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("traffic position must be finite")
    return result


def cell_at(point):
    return (math.floor(point[0] / CELL_SIZE),
            math.floor(point[1] / CELL_SIZE))


def route_samples(points):
    """Sample every segment, returning cell-entry distance without duplicates.

    Conservative 0.1-cell sampling is independent of the supplied waypoint
    density. Re-entering a cell later in a route remains a distinct visit.
    """
    points = tuple(_xy(point) for point in points)
    if not points:
        return ()
    result = [(cell_at(points[0]), 0.0)]
    travelled = 0.0
    for first, second in zip(points, points[1:]):
        length = math.dist(first, second)
        count = max(1, math.ceil(length / (CELL_SIZE * 0.1)))
        for index in range(1, count + 1):
            ratio = index / count
            point = (first[0] + ratio * (second[0] - first[0]),
                     first[1] + ratio * (second[1] - first[1]))
            cell = cell_at(point)
            if cell != result[-1][0]:
                result.append((cell, travelled + ratio * length))
        travelled += length
    return tuple(result)


@dataclass(frozen=True)
class MotionHistory:
    speed: float = NOMINAL_SPEED
    wait_seconds: float = 0.0
    stop_events: int = 0


class TrafficMonitor:
    """Bounded observed history; identical sampling semantics in both runtimes."""

    def __init__(self):
        self._previous = {}
        self._samples = {}
        self._last_time = None

    def observe(self, now, states):
        now = float(now)
        if not math.isfinite(now) or now < 0:
            raise ValueError("traffic clock must be finite and nonnegative")
        if self._last_time is not None:
            if now < self._last_time:
                raise ValueError("traffic clock moved backwards")
            if now - self._last_time < 0.5:
                return
        self._last_time = now
        for rid in tuple(self._previous):
            if rid not in states:
                self._previous.pop(rid, None)
                self._samples.pop(rid, None)
        for rid, state in states.items():
            position = _xy(state["position"])
            heading = float(state.get("heading", 0.0))
            active = bool(state.get("traffic_moving", False))
            planned = bool(state.get("planned_wait", False))
            previous = self._previous.get(rid)
            samples = self._samples.setdefault(rid, deque())
            stopped = False
            if previous is not None:
                then, old_position, old_heading, was_active, was_stopped = previous
                elapsed = now - then
                speed = math.dist(position, old_position) / elapsed
                rotation = abs((heading - old_heading + math.pi) %
                               (2 * math.pi) - math.pi) / elapsed
                stopped = (active and was_active and not planned and
                           speed < 0.025 and rotation < 0.15)
                samples.append((now, elapsed, speed,
                                elapsed if stopped else 0.0,
                                int(stopped and not was_stopped),
                                bool(active and was_active and not planned)))
            while samples and samples[0][0] <= now - HISTORY_SECONDS:
                samples.popleft()
            self._previous[rid] = (now, position, heading, active, stopped)

    def snapshot(self, now):
        now = float(now)
        if not math.isfinite(now) or (self._last_time is not None and
                                     now < self._last_time):
            raise ValueError("cannot expose future observations to a snapshot")
        result = {}
        for rid, samples in self._samples.items():
            moving_time = distance = waiting = 0.0
            events = 0
            for end, duration, speed, wait, starts, active in samples:
                overlap = max(0.0, min(duration, end - (now - HISTORY_SECONDS)))
                if overlap <= 0:
                    continue
                if active and speed >= 0.025:
                    moving_time += overlap
                    distance += speed * overlap
                waiting += wait * overlap / duration
                events += starts if end - duration >= now - HISTORY_SECONDS else 0
            result[rid] = MotionHistory(
                min(0.6, distance / moving_time) if moving_time else NOMINAL_SPEED,
                waiting, events)
        return MappingProxyType(result)


@dataclass(frozen=True)
class TrafficRobot:
    robot_id: int
    position: tuple
    route: tuple
    not_before: float
    speed: float
    history: MotionHistory

    def __post_init__(self):
        object.__setattr__(self, "position", _xy(self.position))
        object.__setattr__(self, "route", tuple(_xy(p) for p in self.route))
        if not (math.isfinite(self.not_before) and
                math.isfinite(self.speed) and 0 <= self.speed <= 0.6):
            raise ValueError("invalid traffic motion estimate")


def forecast_positions(robot, now):
    """Near-term positions; active holds and measured speed affect arrival."""
    route = (robot.position,) + tuple(robot.route)
    lengths = [math.dist(a, b) for a, b in zip(route, route[1:])]
    prefix = [0.0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    result = []
    index = 0
    for slot in range(int(HORIZON / TIME_BIN) + 1):
        distance = max(0.0, now + slot * TIME_BIN -
                       max(now, robot.not_before)) * robot.speed
        while index < len(lengths) and distance >= prefix[index + 1]:
            index += 1
        if index >= len(lengths):
            result.append(route[-1])
        else:
            fraction = (distance - prefix[index]) / lengths[index]
            first, second = route[index:index + 2]
            result.append((first[0] + fraction * (second[0] - first[0]),
                           first[1] + fraction * (second[1] - first[1])))
    return tuple(result)


class TrafficSnapshot:
    """Frozen observations with private, per-snapshot memoized edge features."""

    version = TRAFFIC_VERSION

    def __init__(self, now, robots, route_provider, capacity=None):
        self.now = float(now)
        if not math.isfinite(self.now) or self.now < 0:
            raise ValueError("traffic clock must be finite and nonnegative")
        robots = tuple(robots)
        self.robots = MappingProxyType({robot.robot_id: robot for robot in robots})
        if len(self.robots) != len(robots):
            raise ValueError("duplicate traffic robot IDs")
        self.capacity = MappingProxyType(dict(capacity or {}))
        if any(not math.isfinite(value) or value < 1
               for value in self.capacity.values()):
            raise ValueError("traffic capacity must be finite and at least one")
        self._route_provider = route_provider
        self._features = {}
        cells = {}
        for robot in self.robots.values():
            positions = forecast_positions(robot, self.now)
            for slot, (start, end) in enumerate(zip(positions, positions[1:])):
                for cell, _distance in route_samples((start, end)):
                    cells.setdefault((slot, cell), set()).add(robot.robot_id)
        self.occupants = MappingProxyType({
            key: frozenset(owners) for key, owners in cells.items()})

    def _peers(self, slot, cell, robot_id):
        # One-second timing uncertainty avoids claiming exact physical ETAs.
        peers = set()
        for nearby in (slot - 1, slot, slot + 1):
            peers.update(self.occupants.get((nearby, cell), ()))
        peers.discard(robot_id)
        return peers

    def features(self, robot_id, pickup, delivery):
        pickup, delivery = _xy(pickup), _xy(delivery)
        key = (int(robot_id), pickup, delivery)
        if key in self._features:
            return self._features[key]
        robot = self.robots[robot_id]
        first = self._route_provider(robot.position, pickup)
        second = self._route_provider(pickup, delivery)
        if first is None or second is None:
            result = (0.0,) * len(TRAFFIC_FEATURE_NAMES)
            self._features[key] = result
            return result
        points = (robot.position,) + tuple(first) + (pickup,) + tuple(second) + (delivery,)
        samples = route_samples(points)
        speed = max(0.08, min(0.6, robot.speed))
        wait = 0.0
        loads = []
        encountered = set()
        covered = 0
        for cell, travelled in samples:
            arrival = max(0.0, robot.not_before - self.now) + travelled / speed + wait
            if arrival >= HORIZON:
                break
            covered += 1
            slot = int(arrival / TIME_BIN)
            peers = self._peers(slot, cell, robot_id)
            encountered.update(peers)
            capacity = max(1.0, float(self.capacity.get(cell, 1.0)))
            load = len(peers) / capacity
            loads.append(load)
            if load >= 1.0:
                original_slot = slot
                while slot < int(HORIZON / TIME_BIN) and len(
                        self._peers(slot, cell, robot_id)) >= capacity:
                    slot += 1
                wait += (slot - original_slot) * TIME_BIN
        histories = [self.robots[rid].history for rid in sorted(encountered)]
        count = max(1, len(histories))
        result = (
            1.0, covered / max(1, len(samples)),
            max(loads, default=0.0), sum(loads) / max(1, len(loads)),
            len(encountered) / max(1, len(self.robots) - 1),
            min(HORIZON, wait),
            sum(item.speed for item in histories) / count / NOMINAL_SPEED
                if histories else 1.0,
            sum(item.wait_seconds for item in histories) / count,
            sum(item.stop_events for item in histories) / count,
        )
        self._features[key] = result
        return result

    @staticmethod
    def expert_penalty(features):
        """Convert predicted delay/risk seconds to the baseline distance unit."""
        values = dict(zip(TRAFFIC_FEATURE_NAMES, features))
        exposure = values["traffic_peer_fleet_fraction"]
        historical_risk = exposure * (
            0.1 * values["traffic_peer_recent_wait_seconds"] +
            values["traffic_peer_recent_stop_events"])
        return NOMINAL_SPEED * (
            values["traffic_expected_wait_seconds"] + historical_risk)

    def endpoint_load(self, point):
        cell = cell_at(_xy(point))
        capacity = float(self.capacity.get(cell, 1.0))
        return sum(len(self.occupants.get((slot, cell), ())) / capacity
                   for slot in range(int(HORIZON / TIME_BIN))) / (
                       HORIZON / TIME_BIN)


class TrafficSnapshotBuilder:
    """Observe execution without changing routes, reservations or robot state."""

    def __init__(self):
        self.monitor = TrafficMonitor()
        self._capacity = None
        self._geometry = None
        self._last_key = None
        self._last_snapshot = None

    def observe(self, now, observations):
        self.monitor.observe(now, observations)

    def snapshot(self, now, observations, provider):
        self.observe(now, observations)
        route_provider = getattr(provider, "segment_route", None)
        geometry = getattr(getattr(provider, "coordinator", None), "grid", None)
        if not callable(route_provider) or geometry is None:
            raise ValueError("traffic features require a read-only geometry oracle")
        if geometry is not self._geometry:
            from config import ROBOT_RADIUS
            from grid_planner import GRID_RES
            area = {}
            for row in range(geometry.rows):
                for col in range(geometry.cols):
                    if geometry.is_free(col, row):
                        cell = cell_at(geometry.grid_to_world(col, row))
                        area[cell] = area.get(cell, 0.0) + GRID_RES ** 2
            # A coarse simultaneous-occupancy proxy, capped at four slots.
            # Keep absolute load and fleet fraction as separate features.
            slot_area = max(0.5, 2 * ROBOT_RADIUS + 0.25) ** 2
            self._capacity = {cell: max(1, min(4, math.floor(value / slot_area)))
                              for cell, value in area.items()}
            self._geometry = geometry
            self._last_key = None
        history = self.monitor.snapshot(now)
        robots = []
        for rid, state in sorted(observations.items()):
            past = history.get(rid, MotionHistory())
            scale = max(0.4, min(1.0, float(state.get("traffic_speed_scale", 1.0))))
            # Moving speed is measured; recent unplanned waits reduce its
            # effective value. The short horizon limits extrapolation error.
            speed = min(NOMINAL_SPEED * scale, past.speed) * max(
                0.2, 1.0 - past.wait_seconds / HISTORY_SECONDS)
            robots.append(TrafficRobot(
                rid, state["position"], tuple(state.get("traffic_route", ())),
                max(float(now), float(state.get("traffic_not_before", now))),
                speed, past))
        key = (float(now), tuple(robots))
        if key == self._last_key:
            return self._last_snapshot
        result = TrafficSnapshot(now, robots, route_provider, self._capacity)
        self._last_key, self._last_snapshot = key, result
        return result
