"""Pure, simulator-independent collision safety calculations."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple


class RiskLevel(str, Enum):
    CLEAR = "clear"
    CAUTION = "caution"
    BRAKE = "brake"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class SafetyConfig:
    collision_distance: float = 0.50
    static_clearance: float = 0.10
    prediction_horizon: float = 10.0
    reaction_latency: float = 0.20
    braking_deceleration: float = 0.60
    stale_caution_age: float = 0.10
    stale_stop_age: float = 0.30

    def __post_init__(self):
        positive = (
            self.collision_distance, self.static_clearance,
            self.prediction_horizon, self.braking_deceleration,
            self.stale_caution_age, self.stale_stop_age)
        if any(value <= 0 for value in positive):
            raise ValueError("safety distances and times must be positive")
        if self.stale_stop_age <= self.stale_caution_age:
            raise ValueError("stale_stop_age must exceed stale_caution_age")


@dataclass(frozen=True)
class MotionRisk:
    minimum_distance: float
    time_to_cpa: float
    time_to_collision: float
    stopping_distance: float
    level: RiskLevel


def braking_distance(speed: float, deceleration: float, latency: float = 0.0,
                     buffer: float = 0.0) -> float:
    """Distance needed to react and stop at constant deceleration."""
    if deceleration <= 0:
        raise ValueError("deceleration must be positive")
    v = max(0.0, float(speed))
    return v * max(0.0, latency) + v * v / (2.0 * deceleration) + max(0.0, buffer)


def cpa_ttc(position_a: Tuple[float, float], velocity_a: Tuple[float, float],
            position_b: Tuple[float, float], velocity_b: Tuple[float, float],
            collision_distance: float = 0.5,
            horizon: float = 10.0) -> Tuple[float, float, float]:
    """Return (minimum distance, time to CPA, first collision time)."""
    rx, ry = position_b[0] - position_a[0], position_b[1] - position_a[1]
    vx, vy = velocity_b[0] - velocity_a[0], velocity_b[1] - velocity_a[1]
    vv = vx * vx + vy * vy
    t_cpa = 0.0 if vv < 1e-12 else max(0.0, min(horizon, -(rx * vx + ry * vy) / vv))
    dx, dy = rx + vx * t_cpa, ry + vy * t_cpa
    minimum = math.hypot(dx, dy)

    # Solve |r + vt|² = collision_distance² for earliest future root.
    c = rx * rx + ry * ry - collision_distance * collision_distance
    if c <= 0:
        ttc = 0.0
    elif vv < 1e-12:
        ttc = math.inf
    else:
        b = 2.0 * (rx * vx + ry * vy)
        disc = b * b - 4.0 * vv * c
        if disc < 0:
            ttc = math.inf
        else:
            root = (-b - math.sqrt(disc)) / (2.0 * vv)
            ttc = root if 0.0 <= root <= horizon else math.inf
    return minimum, t_cpa, ttc


def assess_motion_risk(position_a, velocity_a, position_b, velocity_b,
                       speed: float, deceleration: float = 0.6,
                       latency: float = 0.2, collision_distance: float = 0.5,
                       horizon: float = 10.0) -> MotionRisk:
    minimum, t_cpa, ttc = cpa_ttc(
        position_a, velocity_a, position_b, velocity_b,
        collision_distance, horizon)
    d_stop = braking_distance(speed, deceleration, latency, buffer=0.1)
    if ttc <= 0.5:
        level = RiskLevel.EMERGENCY
    elif math.isfinite(ttc) and ttc <= 2.0:
        level = RiskLevel.BRAKE
    elif minimum < collision_distance + d_stop:
        level = RiskLevel.CAUTION
    else:
        level = RiskLevel.CLEAR
    return MotionRisk(minimum, t_cpa, ttc, d_stop, level)
