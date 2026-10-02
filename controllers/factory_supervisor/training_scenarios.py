"""Factory-grounded scenario generation shared by standalone trainers."""

import hashlib
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np

from config import (ALL_LOCATIONS, INITIAL_BATTERY_MAX, INITIAL_BATTERY_MIN,
                    PARKING_HEADINGS, PARKING_SPOTS, PLANNER_KEEP_OUT_BOXES,
                    REST_NODES, SCENARIOS, STORAGE_AREAS, WAYPOINTS,
                    WORKSTATIONS, RobotState)
from grid_planner import GRID_RES, OccupancyGrid
from motion_coordinator import MotionCoordinator
from schedulers import SchedulingContext
from task_generator import (TransportTask, canonical_task_manifest_document,
                            generate_task_manifest, tasks_from_manifest)


_STATIC_COORDINATORS = {}
_SEGMENT_CACHES = {}


def formal_scenario_config(scenario_id: str) -> dict:
    """Return the complete JSON-safe configuration covered by manifest hash."""
    scenario_id = str(scenario_id)
    if scenario_id not in SCENARIOS:
        raise ValueError("scenario_id must be A, B, or C")
    config = dict(SCENARIOS[scenario_id])
    count = int(config["num_robots"])
    config.update({
        "scenario_config_version": "factory-scenario-v1",
        "training_geometry": "factory-grid-astar-v3-online",
        "grid_resolution": float(GRID_RES),
        "locations": {
            name: [float(value) for value in ALL_LOCATIONS[name]]
            for name in sorted(ALL_LOCATIONS)
        },
        "planner_keep_out_boxes": [
            {key: box[key] for key in sorted(box)}
            for box in PLANNER_KEEP_OUT_BOXES
        ],
        "active_parking_states": {
            str(rid): {
                "position": [float(value) for value in PARKING_SPOTS[rid]],
                "heading": float(PARKING_HEADINGS[rid]),
            }
            for rid in range(1, count + 1)
        },
        "inactive_robot_semantics": "removed_from_webots_scene",
    })
    return config


def factory_free_positions() -> List[Tuple[float, float]]:
    """Return deduplicated parking/corridor positions that are grid-free.

    A graph/rest node can still lie inside an inflated grid keep-out. Check
    the same occupancy grid used by A* instead of trusting configuration
    comments when choosing physical robot centres for training.
    """
    values = list(PARKING_SPOTS.values())
    values.extend(WAYPOINTS[node] for node in REST_NODES if node in WAYPOINTS)
    candidates = list(dict.fromkeys((float(x), float(y)) for x, y in values))
    grid = OccupancyGrid(num_active_robots=8)
    positions = [point for point in candidates
                 if grid.is_free(*grid.world_to_grid(*point))]
    if len(positions) < 8:
        raise RuntimeError(
            "Factory layout has fewer than 8 verified free training starts")
    return positions


def _task_pair(rng):
    storage = list(STORAGE_AREAS)
    workstations = list(WORKSTATIONS)
    draw = float(rng.random())
    if draw < 0.40:
        return str(rng.choice(storage)), str(rng.choice(workstations))
    if draw < 0.75:
        return str(rng.choice(workstations)), str(rng.choice(storage))
    pickup = str(rng.choice(workstations))
    delivery = str(rng.choice([name for name in workstations
                               if name != pickup]))
    return pickup, delivery


def path_length(path, start):
    if path is None:
        return math.inf
    points = [tuple(start)] + [tuple(point) for point in path]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


class FactoryAStarCostOracle:
    """Cached static Grid-A* travel costs for scheduler observations/masks."""

    def __init__(self, robot_states: Dict[int, dict], num_robots: int):
        if (isinstance(num_robots, bool) or not isinstance(num_robots, int)
                or num_robots <= 0 or len(robot_states) != num_robots):
            raise ValueError("cost oracle robot count is inconsistent")
        self.robot_states = robot_states
        self.num_robots = num_robots
        if num_robots not in _STATIC_COORDINATORS:
            _STATIC_COORDINATORS[num_robots] = MotionCoordinator(
                num_active_robots=num_robots)
            _SEGMENT_CACHES[num_robots] = {}
        self.coordinator = _STATIC_COORDINATORS[num_robots]
        self.cache = _SEGMENT_CACHES[num_robots]

    def bind_robot_states(self, robot_states: Dict[int, dict]):
        """Rebind to the live/deep-copied simulator snapshot."""
        if len(robot_states) != self.num_robots:
            raise ValueError("bound robot count differs from cost oracle")
        self.robot_states = robot_states
        return self

    def segment(self, start, goal):
        start = (float(start[0]), float(start[1]))
        goal = (float(goal[0]), float(goal[1]))
        key = (start, goal)
        if key not in self.cache:
            path = self.coordinator.grid_planner.plan(start, goal, smooth=True)
            self.cache[key] = path_length(path, start)
        return self.cache[key]

    def __call__(self, robot_id: int, task: TransportTask) -> float:
        empty = self.segment(
            self.robot_states[robot_id]["position"], task.pickup_position)
        loaded = self.segment(task.pickup_position, task.delivery_position)
        total = empty + loaded
        return total if math.isfinite(total) else math.inf


def factory_scenario(seed: int, max_robots=8, max_tasks=20,
                     min_robots=2, max_generated_tasks=12, *,
                     scenario_id=None, duration_seconds=1800.0,
                     manifest=None):
    """Generate a legacy snapshot or an explicit canonical A/B/C run.

    Robot positions come only from verified parking/rest nodes. Task endpoints
    use the same WS/S docks as Webots. Unreachable pairs return infinity and
    are consequently removed by the scheduler action mask. Formal callers
    must pass ``scenario_id``; the no-ID branch is compatibility-only.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("scenario seed must be an integer")
    seed = int(seed)
    if scenario_id is not None:
        if (max_robots, max_tasks, min_robots, max_generated_tasks) != (
                8, 20, 2, 12):
            raise ValueError(
                "legacy random sizing options cannot be used formally")
        scenario_id = str(scenario_id)
        if scenario_id not in SCENARIOS:
            raise ValueError("scenario_id must be A, B, or C")
        scenario_config = formal_scenario_config(scenario_id)
        document = (generate_task_manifest(
            scenario_id, scenario_config, seed,
            duration_seconds=duration_seconds)
            if manifest is None else
            canonical_task_manifest_document(manifest))
        metadata = document["metadata"]
        if (metadata["scenario_id"] != scenario_id or
                metadata["seed"] != seed or
                metadata["scenario_config"] != scenario_config or
                not math.isclose(
                    metadata["duration_seconds"], float(duration_seconds),
                    rel_tol=0.0, abs_tol=1e-9)):
            raise ValueError("manifest does not match requested scenario run")
        robot_count = int(scenario_config["num_robots"])
        battery_rng = random.Random(seed)
        robots = {
            rid: {
                "position": tuple(PARKING_SPOTS[rid]),
                "heading": float(PARKING_HEADINGS[rid]),
                "state": RobotState.IDLE,
                "battery": battery_rng.uniform(
                    INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX),
                "current_task": None,
                "has_task": False,
                "goal_location": None,
                "tasks_completed": 0,
                "total_distance": 0.0,
            }
            for rid in range(1, robot_count + 1)
        }
        tasks = tasks_from_manifest(document)
        initial_state = [
            {
                "robot_id": rid,
                "position": list(robots[rid]["position"]),
                "heading": robots[rid]["heading"],
                "battery": robots[rid]["battery"],
            }
            for rid in sorted(robots)
        ]
        initial_state_payload = json.dumps(
            initial_state, sort_keys=True, separators=(",", ":"),
            allow_nan=False)
        initial_state_hash = hashlib.sha256(
            initial_state_payload.encode("utf-8")).hexdigest()
        oracle = FactoryAStarCostOracle(robots, robot_count)
        return robots, tasks, SchedulingContext(
            current_time=0.0, path_cost_provider=oracle,
            configuration={
                "training_geometry": "factory-grid-astar-v3-online",
                "scenario_mode": "canonical_manifest_full_horizon",
                "scenario_id": scenario_id,
                "num_robots": robot_count,
                "task_interval_seconds": scenario_config["task_interval"],
                "duration_seconds": metadata["duration_seconds"],
                "simulation_timestep_ms": metadata["timestep_ms"],
                "task_manifest_sha256": document["task_manifest_sha256"],
                "manifest_sha256": document["manifest_sha256"],
                "initial_state_version": (
                    "webots-parking-seeded-battery-v1"),
                "initial_robot_states": initial_state,
                "initial_robot_state_sha256": initial_state_hash,
                "parked_robot_obstacle_semantics": (
                    "static_oracle_excludes_robot_bodies_runtime_handles_peers"),
                "endpoint_semantics": (
                    "grid_astar_relaxes_inflated_start_and_dock_cells"),
            })

    # Compatibility-only random snapshot used by legacy trainer entry points.
    # Formal dual-objective runs must pass scenario_id and use the branch above.
    if manifest is not None:
        raise ValueError("manifest replay requires an explicit scenario_id")
    if not math.isclose(float(duration_seconds), 1800.0,
                        rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("duration_seconds requires an explicit scenario_id")
    rng = np.random.default_rng(seed)
    robot_count = int(rng.integers(min_robots, max_robots + 1))
    task_upper = min(max_tasks, max_generated_tasks)
    task_count = int(rng.integers(2, task_upper + 1))
    positions = factory_free_positions()
    chosen = rng.choice(len(positions), size=robot_count, replace=False)
    robots = {
        rid: {"position": positions[int(chosen[rid - 1])],
              "state": RobotState.IDLE,
              "battery": float(rng.uniform(30.0, 100.0)),
              "current_task": None, "tasks_completed": 0,
              "total_distance": 0.0}
        for rid in range(1, robot_count + 1)
    }
    tasks = []
    mean_interval = 30.0 if robot_count <= 3 else (15.0 if robot_count <= 5
                                                   else 8.0)
    arrival_time = 0.0
    for index in range(task_count):
        pickup, delivery = _task_pair(rng)
        locations = {**STORAGE_AREAS, **WORKSTATIONS}
        if index:
            arrival_time += float(rng.exponential(mean_interval))
        priority_roll = float(rng.random())
        tasks.append(TransportTask(
            index + 1, pickup, delivery, locations[pickup],
            locations[delivery], arrival_time,
            priority=(3 if priority_roll < 0.05 else
                      2 if priority_roll < 0.25 else 1)))
    oracle = FactoryAStarCostOracle(robots, robot_count)
    return robots, tasks, SchedulingContext(
        current_time=0.0, path_cost_provider=oracle,
        configuration={
            "training_geometry": "factory-grid-astar-v3-online",
            "scenario_mode": "legacy_random_snapshot",
        })
