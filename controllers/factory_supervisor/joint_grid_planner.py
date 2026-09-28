"""Rolling-window joint space-time planner on the factory occupancy grid."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from grid_planner import CELL_OBSTACLE, GRID_RES, OccupancyGrid
from config import line_intersects_obstacles


Cell = Tuple[int, int]


@dataclass(frozen=True)
class TimedCell:
    cell: Cell
    time_slot: int


@dataclass
class JointGridPlan:
    paths: Dict[int, List[TimedCell]]
    planning_seconds: float
    order: Tuple[int, ...]
    expanded_nodes: int
    time_slot_seconds: float


class JointGridPlanner:
    """Plan every supplied robot in one shared space-time reservation view.

    Multiple deterministic priority rotations are evaluated and the cheapest
    complete solution is selected. Each low-level search sees all paths
    already chosen in the same batch, including vertex footprints, reverse
    edges and goal residence through the rolling horizon.
    """

    # Cardinal moves only. A 3 s slot covers worst-case 180-degree heading
    # alignment plus a 0.25 m move at the controller's nominal 0.22 m/s.
    MOVES = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, grid: OccupancyGrid, *, horizon_slots: int = 4,
                 time_slot_seconds: float = 3.0,
                 separation_cells: int = 4,
                 minimum_distance_m: Optional[float] = None,
                 max_expansions_per_robot: int = 12000):
        self.grid = grid
        self.horizon_slots = int(horizon_slots)
        self.time_slot_seconds = float(time_slot_seconds)
        self.separation_cells = int(separation_cells)
        self.minimum_distance_m = (None if minimum_distance_m is None
                                   else float(minimum_distance_m))
        self.max_expansions_per_robot = int(max_expansions_per_robot)
        self.last_failure_reason = None
        self.last_expanded_nodes = 0
        self.last_orders_evaluated = 0
        self._static_heuristic_cache = {}
        self._vertex_conflict_offsets = self._build_vertex_conflict_offsets()

    def plan(self, agents: Dict[int, Tuple[Tuple[float, float],
                                           Tuple[float, float]]],
             *, max_seconds: float = 0.20,
             max_expansions: Optional[int] = None,
             priority_order: Optional[Tuple[int, ...]] = None
             ) -> Optional[JointGridPlan]:
        started = time.perf_counter()
        self.last_failure_reason = None
        self.last_expanded_nodes = 0
        self.last_orders_evaluated = 0
        if not agents:
            return JointGridPlan({}, 0.0, (), 0, self.time_slot_seconds)
        expansion_budget = (None if max_expansions is None else
                            max(1, int(max_expansions)))
        ids = tuple(sorted(agents))
        base_order = ids
        if priority_order is not None:
            preferred = tuple(priority_order)
            if set(preferred) == set(ids):
                base_order = preferred
        orders = [base_order]
        if len(ids) > 1:
            orders.append(tuple(reversed(base_order)))
            orders.extend(base_order[offset:] + base_order[:offset]
                          for offset in range(1, len(ids)))

        total_expanded = 0
        seen_orders = set()
        for order in orders:
            if order in seen_orders:
                continue
            seen_orders.add(order)
            if time.perf_counter() - started >= max_seconds:
                self.last_failure_reason = "timeout"
                break
            if (expansion_budget is not None and
                    total_expanded >= expansion_budget):
                self.last_failure_reason = "expansion_budget"
                break
            remaining_expansions = (None if expansion_budget is None else
                                    expansion_budget - total_expanded)
            self.last_orders_evaluated += 1
            candidate, expanded = self._plan_order(
                order, agents, started, max_seconds, remaining_expansions)
            total_expanded += expanded
            self.last_expanded_nodes = total_expanded
            if candidate is None:
                continue
            candidate_plan = JointGridPlan(
                candidate, 0.0, order, expanded, self.time_slot_seconds)
            if not self.validate(candidate_plan):
                continue
            elapsed = time.perf_counter() - started
            return JointGridPlan(candidate, elapsed, order, total_expanded,
                                 self.time_slot_seconds)

        if self.last_failure_reason is None:
            self.last_failure_reason = "no_solution"
        return None

    def _plan_order(self, order, agents, started, max_seconds,
                    max_expansions=None):
        vertex: Dict[Tuple[Cell, int], int] = {}
        edges: Dict[int, List[Tuple[Cell, Cell, int]]] = {}
        result: Dict[int, List[TimedCell]] = {}
        expanded_total = 0
        for rid in order:
            remaining_expansions = (
                None if max_expansions is None
                else max_expansions - expanded_total)
            if (remaining_expansions is not None and
                    remaining_expansions <= 0):
                self.last_failure_reason = "expansion_budget"
                return None, expanded_total
            start_xy, goal_xy = agents[rid]
            start = self.grid.world_to_grid(*start_xy)
            goal = self.grid.world_to_grid(*goal_xy)
            start_endpoint_cells = self._endpoint_cells(start)
            goal_endpoint_cells = self._endpoint_cells(goal)
            endpoint_cells = start_endpoint_cells | goal_endpoint_cells
            static_dist = self._static_heuristic_cache.get(goal)
            if static_dist is None:
                static_dist = self._build_static_heuristic(
                    goal, goal_endpoint_cells)
                self._static_heuristic_cache[goal] = static_dist
            path, expanded = self._space_time_astar(
                rid, start, goal, endpoint_cells, static_dist, vertex, edges,
                started, max_seconds, remaining_expansions)
            expanded_total += expanded
            if path is None:
                return None, expanded_total
            result[rid] = [TimedCell(cell, slot)
                           for slot, cell in enumerate(path)]
            self._reserve(rid, path, vertex, edges)
        return result, expanded_total

    def _space_time_astar(self, rid, start, goal, endpoint_cells, static_dist,
                          vertex, edges, started, max_seconds,
                          max_expansions=None):
        start_state = (start, 0)
        queue = [(self._heuristic_for(start, goal, static_dist),
                  0, 0.0, start_state)]
        came_from = {start_state: None}
        g_score = {start_state: 0.0}
        sequence = 0
        expanded = 0
        while queue:
            if ((max_expansions is not None and
                 expanded >= max_expansions) or
                    expanded >= self.max_expansions_per_robot or
                    time.perf_counter() - started >= max_seconds):
                if (max_expansions is not None and
                        expanded >= max_expansions):
                    self.last_failure_reason = "expansion_budget"
                elif expanded >= self.max_expansions_per_robot:
                    self.last_failure_reason = "expansion_limit"
                else:
                    self.last_failure_reason = "timeout"
                return None, expanded
            _, _, queued_g, (cell, slot) = heapq.heappop(queue)
            if queued_g != g_score.get((cell, slot)):
                continue
            expanded += 1
            if cell == goal and self._goal_residence_clear(
                    goal, slot, rid, vertex):
                return self._reconstruct(came_from, (cell, slot)), expanded
            if slot >= self.horizon_slots:
                # RHCR commits only a short prefix. Reaching the final task
                # goal inside one window is deliberately not required.
                return self._reconstruct(came_from, (cell, slot)), expanded
            moves = sorted(
                self.MOVES,
                key=lambda move: (
                    self._heuristic_for(
                        (cell[0] + move[0], cell[1] + move[1]), goal,
                        static_dist),
                    0 if move[0] < 0 else 1 if move[0] > 0 else 2,
                    0 if move[1] < 0 else 1 if move[1] > 0 else 2,
                ))
            for dc, dr in moves:
                nxt = (cell[0] + dc, cell[1] + dr)
                next_slot = slot + 1
                if not self._traversable(nxt, endpoint_cells):
                    continue
                if self._vertex_conflict(nxt, next_slot, rid, vertex):
                    continue
                if self._edge_conflict(cell, nxt, slot, rid, edges):
                    continue
                state = (nxt, next_slot)
                if dc == 0 and dr == 0:
                    step_cost = 1.5
                elif dc and dr:
                    step_cost = math.sqrt(2.0)
                else:
                    step_cost = 1.0
                tentative = g_score[(cell, slot)] + step_cost
                if tentative >= g_score.get(state, math.inf):
                    continue
                came_from[state] = (cell, slot)
                g_score[state] = tentative
                sequence += 1
                heapq.heappush(queue, (
                    tentative + self._heuristic_for(nxt, goal, static_dist),
                    sequence,
                    tentative, state))
        return None, expanded

    def _heuristic_for(self, cell, goal, static_dist):
        if static_dist is not None and cell in static_dist:
            return static_dist[cell]
        return self._heuristic(cell, goal)

    def _build_static_heuristic(self, goal, endpoint_cells):
        """Return obstacle-aware shortest-path distances to the goal."""
        distances = {}
        queue = []
        for cell in endpoint_cells:
            if not self.grid.in_bounds(*cell):
                continue
            if self.grid.is_free(*cell) or cell in endpoint_cells:
                distances[cell] = 0.0
                queue.append(cell)
        for cell in queue:
            current = distances[cell]
            for dc, dr in self.MOVES[1:]:
                nxt = (cell[0] + dc, cell[1] + dr)
                if nxt in distances or not self.grid.in_bounds(*nxt):
                    continue
                if not self.grid.is_free(*nxt) and nxt not in endpoint_cells:
                    continue
                distances[nxt] = current + 1.0
                queue.append(nxt)
        return distances

    def _goal_residence_clear(self, goal, arrival_slot, rid, vertex):
        """A reached goal must remain safe for the rest of this horizon."""
        return all(
            not self._vertex_conflict(goal, slot, rid, vertex)
            for slot in range(arrival_slot, self.horizon_slots + 1)
        )

    def _reserve(self, rid, path, vertex, edges):
        for slot, cell in enumerate(path):
            vertex[(cell, slot)] = rid
            if slot:
                edges.setdefault(slot - 1, []).append(
                    (path[slot - 1], cell, rid))
        goal = path[-1]
        for slot in range(len(path), self.horizon_slots + 1):
            vertex[(goal, slot)] = rid

    def _conflict_radius_cells(self):
        if self.minimum_distance_m is None:
            return self.separation_cells
        return int(math.ceil(self.minimum_distance_m / GRID_RES)) + 1

    def _cells_conflict(self, first: Cell, second: Cell) -> bool:
        if self.minimum_distance_m is None:
            return ((first[0] - second[0]) ** 2 +
                    (first[1] - second[1]) ** 2) <= self.separation_cells ** 2
        first_xy = self.grid.grid_to_world(*first)
        second_xy = self.grid.grid_to_world(*second)
        return (math.hypot(first_xy[0] - second_xy[0],
                           first_xy[1] - second_xy[1]) <=
                self.minimum_distance_m)

    def _build_vertex_conflict_offsets(self):
        """Precompute the translation-invariant safety footprint.

        The occupancy grid has a fixed linear world transform, so whether two
        cells violate the configured separation depends only on their offset.
        Building this mask once preserves ``_cells_conflict`` semantics while
        avoiding a full square scan and repeated world conversions at every
        A* successor.
        """
        radius = self._conflict_radius_cells()
        origin = (0, 0)
        return tuple(
            (dc, dr)
            for dc in range(-radius, radius + 1)
            for dr in range(-radius, radius + 1)
            if self._cells_conflict(origin, (dc, dr))
        )

    def _vertex_conflict(self, cell, slot, rid, vertex):
        for dc, dr in self._vertex_conflict_offsets:
            neighbour = (cell[0] + dc, cell[1] + dr)
            owner = vertex.get((neighbour, slot))
            if owner is not None and owner != rid:
                return True
        return False

    def _edge_conflict(self, first, second, slot, rid, edges):
        for other_first, other_second, owner in edges.get(slot, ()):
            if owner == rid:
                continue
            distance_cells = self._segment_distance(
                first, second, other_first, other_second)
            if self.minimum_distance_m is None:
                if distance_cells <= self.separation_cells:
                    return True
            elif distance_cells * GRID_RES <= self.minimum_distance_m:
                return True
        return False

    @staticmethod
    def _segment_distance(a, b, c, d):
        """Minimum distance between two 2-D movement segments."""
        def point_segment_distance(point, first, second):
            vx, vy = second[0] - first[0], second[1] - first[1]
            length_sq = vx * vx + vy * vy
            if length_sq == 0:
                return math.hypot(point[0] - first[0], point[1] - first[1])
            ratio = max(0.0, min(1.0, (
                (point[0] - first[0]) * vx +
                (point[1] - first[1]) * vy) / length_sq))
            projection = (first[0] + ratio * vx, first[1] + ratio * vy)
            return math.hypot(
                point[0] - projection[0], point[1] - projection[1])

        return min(
            point_segment_distance(a, c, d),
            point_segment_distance(b, c, d),
            point_segment_distance(c, a, b),
            point_segment_distance(d, a, b),
        )

    def _endpoint_cells(self, center: Cell) -> set:
        """Return one bounded cardinal connector from an endpoint to free space."""
        if not self.grid.in_bounds(*center):
            return set()
        if self.grid.is_free(*center):
            return {center}
        queue = [(center, (center,))]
        visited = {center}
        while queue:
            cell, path = queue.pop(0)
            if cell != center and self.grid.is_free(*cell):
                return set(path)
            if len(path) >= 4:
                continue
            for dc, dr in self.MOVES[1:]:
                nxt = (cell[0] + dc, cell[1] + dr)
                if nxt in visited or not self.grid.in_bounds(*nxt):
                    continue
                if self.grid.cells[nxt[1]][nxt[0]] == CELL_OBSTACLE:
                    continue
                if line_intersects_obstacles(
                        self.grid.grid_to_world(*cell),
                        self.grid.grid_to_world(*nxt)):
                    continue
                visited.add(nxt)
                queue.append((nxt, path + (nxt,)))
        return {center}

    def _traversable(self, cell, endpoint_cells):
        if not self.grid.in_bounds(*cell):
            return False
        if self.grid.cells[cell[1]][cell[0]] == CELL_OBSTACLE:
            return False
        return self.grid.is_free(*cell) or cell in endpoint_cells

    def _diagonal_clear(self, cell, dc, dr, endpoint_cells):
        return (self._traversable((cell[0] + dc, cell[1]), endpoint_cells)
                and self._traversable(
                    (cell[0], cell[1] + dr), endpoint_cells))

    @staticmethod
    def _heuristic(first: Cell, second: Cell) -> float:
        dx, dy = abs(first[0] - second[0]), abs(first[1] - second[1])
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    @staticmethod
    def _reconstruct(came_from, state):
        path = []
        while state is not None:
            path.append(state[0])
            state = came_from[state]
        path.reverse()
        return path

    def validate(self, plan: JointGridPlan,
                 separation_cells: Optional[int] = None) -> bool:
        """Independent timing, static-map and inter-robot validation."""
        old_minimum_distance_m = self.minimum_distance_m
        old_separation_cells = self.separation_cells
        if separation_cells is not None:
            self.minimum_distance_m = None
            self.separation_cells = int(separation_cells)
        try:
            return self._validate(plan)
        finally:
            self.minimum_distance_m = old_minimum_distance_m
            self.separation_cells = old_separation_cells

    def _validate(self, plan: JointGridPlan) -> bool:
        paths = plan.paths
        if not paths:
            return True
        for path in paths.values():
            if not path:
                return False
            relaxed = (self._endpoint_cells(path[0].cell) |
                       self._endpoint_cells(path[-1].cell))
            for index, timed in enumerate(path):
                if timed.time_slot != index:
                    return False
                if not self.grid.in_bounds(*timed.cell):
                    return False
                if (not self.grid.is_free(*timed.cell) and
                        timed.cell not in relaxed):
                    return False
                if index:
                    previous = path[index - 1].cell
                    if (abs(previous[0] - timed.cell[0]) +
                            abs(previous[1] - timed.cell[1]) > 1):
                        return False
        horizon = max(len(path) for path in paths.values())
        ids = sorted(paths)
        for slot in range(horizon):
            for index, rid_a in enumerate(ids):
                path_a = paths[rid_a]
                a = path_a[min(slot, len(path_a) - 1)].cell
                prev_a = path_a[min(max(0, slot - 1), len(path_a) - 1)].cell
                for rid_b in ids[index + 1:]:
                    path_b = paths[rid_b]
                    b = path_b[min(slot, len(path_b) - 1)].cell
                    prev_b = path_b[min(max(0, slot - 1), len(path_b) - 1)].cell
                    if self._cells_conflict(a, b):
                        return False
                    if slot and prev_a == b and prev_b == a:
                        return False
                    if slot:
                        distance_cells = JointGridPlanner._segment_distance(
                            prev_a, a, prev_b, b)
                        if self.minimum_distance_m is None:
                            if distance_cells <= self.separation_cells:
                                return False
                        elif distance_cells * GRID_RES <= self.minimum_distance_m:
                            return False
        return True
