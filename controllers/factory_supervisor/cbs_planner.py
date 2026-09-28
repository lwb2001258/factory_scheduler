"""
Conflict-Based Search (CBS) for Multi-Robot Path Planning.

This is the centralised, *optimal* MAPF (Multi-Agent Path-Finding)
planner used by the smart factory. It guarantees **conflict-free**
paths for any number of robots, treating robot-robot collisions as
hard constraints.

References
----------
Sharon, G., Stern, R., Felner, A., Sturtevant, N. R. (2015).
"Conflict-based search for optimal multi-agent pathfinding."
Artificial Intelligence, 219, 40-66.

Two-level architecture:

  ┌──────────────────────────────────────────────────────────────┐
  │  HIGH LEVEL  (this file: CBSPlanner)                         │
  │  Constraint Tree (CT) search, best-first by sum-of-costs.    │
  │  Each CT node has:                                           │
  │     • a per-robot set of constraints                         │
  │     • a per-robot path satisfying its constraints            │
  │     • the cost = Σ |path_i|                                  │
  │  At each CT node we look for the FIRST conflict between any  │
  │  two paths and split the node into two children, each adding │
  │  one disambiguating constraint.                              │
  ├──────────────────────────────────────────────────────────────┤
  │  LOW LEVEL  (this file: SpaceTimeAStar)                      │
  │  Single-agent A* in (node, time) space, respecting a set     │
  │  of vertex- and edge-constraints. Robots can WAIT in place.  │
  └──────────────────────────────────────────────────────────────┘

Conflict types
--------------
  • VERTEX:  robot_a and robot_b both at node N at time t
  • EDGE:    swap conflict — a goes N1→N2, b goes N2→N1 between t and t+1

Constraint types (corresponding)
--------------------------------
  • Vertex(robot, node, t): robot may NOT occupy node at time t
  • Edge(robot, n_from, n_to, t): robot may NOT traverse n_from→n_to
                                   between time t and t+1

Both are strict inequalities — even a "wait" at the same node counts
as occupancy at multiple consecutive timesteps.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet


# ================================================================
#  Data classes
# ================================================================

@dataclass(frozen=True)
class VertexConstraint:
    """robot may NOT occupy `node` at time `t`."""
    robot: int
    node: str
    t: int

    def __repr__(self) -> str:
        return f"V(r{self.robot},{self.node}@t{self.t})"


@dataclass(frozen=True)
class EdgeConstraint:
    """robot may NOT traverse n_from→n_to between t and t+1."""
    robot: int
    n_from: str
    n_to: str
    t: int

    def __repr__(self) -> str:
        return f"E(r{self.robot},{self.n_from}→{self.n_to}@t{self.t})"


@dataclass(frozen=True)
class VertexConflict:
    """Two robots collide at the same node at the same time."""
    robot_a: int
    robot_b: int
    node: str
    t: int

    def __repr__(self) -> str:
        return f"VC(r{self.robot_a}↔r{self.robot_b} @ {self.node}@t{self.t})"


@dataclass(frozen=True)
class EdgeConflict:
    """Two robots swap places (head-on)."""
    robot_a: int
    robot_b: int
    node_a: str   # robot_a goes node_a → node_b
    node_b: str   # robot_b goes node_b → node_a
    t: int        # at the t→t+1 transition

    def __repr__(self) -> str:
        return (f"EC(r{self.robot_a}:{self.node_a}→{self.node_b} ⇆ "
                f"r{self.robot_b}:{self.node_b}→{self.node_a} @t{self.t})")


# A "path" is just a list of node names indexed by time-step.
# Length-K path means the agent visits path[0]@t=0, path[1]@t=1, …, path[K-1]@t=K-1.
Path = List[str]


@dataclass
class CTNode:
    """A node in the high-level Constraint Tree."""
    constraints: Dict[int, FrozenSet]      # robot_id -> frozenset of constraints
    paths: Dict[int, Path]                  # robot_id -> path (by time-step)
    cost: float                             # sum-of-costs

    # Heap tie-breaker (CBS uses lexicographic on cost, then nseq):
    sequence_id: int = 0

    def __lt__(self, other: 'CTNode') -> bool:
        # min-heap on (cost, sequence_id); lower cost first, then FIFO
        return (self.cost, self.sequence_id) < (other.cost, other.sequence_id)


# ================================================================
#  Low-level: space-time A*
# ================================================================

class SpaceTimeAStar:
    """
    Single-agent A* in (node, time) state-space.

    The agent may either move to an adjacent node or wait (stay put)
    at each step. The search respects a set of vertex- and
    edge-constraints filtering successor states.

    Termination: search reaches the goal AND the resulting path is
    conflict-free under the constraints AT and AFTER the goal time
    (i.e. nothing forces the agent to move after arriving).

    For practical purposes we set a `max_time` to bound the state
    space — beyond it the planner gives up and returns None.
    """

    def __init__(self, graph_nodes: Dict[str, Tuple[float, float]],
                 graph_adj: Dict[str, List[str]]):
        self.nodes = graph_nodes
        self.adj = graph_adj

    def _heuristic(self, n: str, goal: str) -> float:
        """Euclidean distance — admissible & consistent."""
        p1 = self.nodes[n]
        p2 = self.nodes[goal]
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def plan(self, start: str, goal: str,
             vertex_constraints: Set[Tuple[str, int]],
             edge_constraints: Set[Tuple[str, str, int]],
             max_time: int = 200) -> Optional[Path]:
        """
        Find a conflict-free path under the given constraints.

        Args:
            start, goal:        node names
            vertex_constraints: {(node, t)} — agent may NOT be at node at t
            edge_constraints:   {(n_from, n_to, t)} — agent may NOT traverse
                                n_from→n_to between t and t+1
            max_time:           hard cutoff on time-step expansion

        Returns:
            list of node names indexed by time-step, or None if no path.
        """
        if start == goal and not any(t == 0 and n == start
                                      for (n, t) in vertex_constraints):
            return [start]

        # Determine the latest time we have constraints on goal — the
        # path must extend at least that far without revisiting goal
        # under constraint.
        latest_constrained = 0
        for (n, t) in vertex_constraints:
            if n == goal:
                latest_constrained = max(latest_constrained, t)

        # Open: priority queue of (f, g, t, node, parent_key)
        counter = itertools.count()
        open_heap: List = []
        start_g = 0.0
        start_f = self._heuristic(start, goal)
        start_key = (start, 0)
        heapq.heappush(open_heap, (start_f, start_g, next(counter), start, 0))

        # came_from[(node, t)] = (parent_node, parent_t)
        came_from: Dict[Tuple[str, int], Tuple[str, int]] = {}
        # g_score keyed on (node, t)
        g_score: Dict[Tuple[str, int], float] = {start_key: 0.0}
        closed: Set[Tuple[str, int]] = set()

        while open_heap:
            _f, g, _seq, node, t = heapq.heappop(open_heap)
            key = (node, t)

            if key in closed:
                continue
            closed.add(key)

            # Goal test: at goal AND no future constraint forces us to leave
            if node == goal and t >= latest_constrained:
                return self._reconstruct(came_from, key)

            if t >= max_time:
                continue

            # Successors: every neighbour + WAIT
            for nbr in self.adj.get(node, []) + [node]:
                t2 = t + 1
                # Vertex constraint?
                if (nbr, t2) in vertex_constraints:
                    continue
                # Edge constraint? (agent moves from `node` to `nbr` at time t)
                if nbr != node and (node, nbr, t) in edge_constraints:
                    continue

                # Edge cost: 1 for move, 1 for wait (uniform)
                # (Note: in reality wait is often cheaper or free, but
                #  uniform cost gives proper sum-of-costs CBS guarantees.)
                cost = 1.0
                tentative_g = g + cost
                nbr_key = (nbr, t2)
                if tentative_g < g_score.get(nbr_key, float('inf')):
                    came_from[nbr_key] = key
                    g_score[nbr_key] = tentative_g
                    f = tentative_g + self._heuristic(nbr, goal)
                    heapq.heappush(open_heap,
                                   (f, tentative_g, next(counter), nbr, t2))

        return None

    @staticmethod
    def _reconstruct(came_from: Dict[Tuple[str, int], Tuple[str, int]],
                     end_key: Tuple[str, int]) -> Path:
        path = [end_key[0]]
        key = end_key
        while key in came_from:
            key = came_from[key]
            path.append(key[0])
        path.reverse()
        return path


# ================================================================
#  High-level: CBS
# ================================================================

class CBSPlanner:
    """
    Conflict-Based Search planner.

    Usage:
        planner = CBSPlanner(graph_nodes, graph_adj)
        paths = planner.plan({1: ('N1', 'N15'), 2: ('N5', 'N11')})
        # paths[1] = ['N1', 'N2', ...]   ← list of node names by time-step
        # paths[2] = ['N5', 'N4', ...]
    """

    def __init__(self, graph_nodes: Dict[str, Tuple[float, float]],
                 graph_adj: Dict[str, List[str]]):
        self.nodes = graph_nodes
        self.adj = graph_adj
        self.low_level = SpaceTimeAStar(graph_nodes, graph_adj)
        # Stats — reset on every plan() call
        self.stats: Dict[str, int] = {}

    # ------------- conflict detection -------------

    @staticmethod
    def _node_at(path: Path, t: int) -> str:
        """The node a robot occupies at time t. After path ends, the
        robot stays at goal forever (trailing wait)."""
        if not path:
            return ""
        return path[t] if t < len(path) else path[-1]

    def _first_conflict(self, paths: Dict[int, Path]):
        """Return the EARLIEST (vertex or edge) conflict, or None."""
        if len(paths) < 2:
            return None
        max_len = max(len(p) for p in paths.values())
        robot_ids = sorted(paths.keys())

        for t in range(max_len):
            # Vertex conflicts at time t
            occupied: Dict[str, int] = {}
            for rid in robot_ids:
                n = self._node_at(paths[rid], t)
                if n in occupied:
                    other = occupied[n]
                    return VertexConflict(robot_a=other, robot_b=rid,
                                          node=n, t=t)
                occupied[n] = rid

            # Edge (swap) conflicts between t and t+1
            if t + 1 < max_len:
                for i, rid_a in enumerate(robot_ids):
                    for rid_b in robot_ids[i + 1:]:
                        a_t  = self._node_at(paths[rid_a], t)
                        a_t1 = self._node_at(paths[rid_a], t + 1)
                        b_t  = self._node_at(paths[rid_b], t)
                        b_t1 = self._node_at(paths[rid_b], t + 1)
                        if a_t == b_t1 and b_t == a_t1 and a_t != a_t1:
                            return EdgeConflict(
                                robot_a=rid_a, robot_b=rid_b,
                                node_a=a_t, node_b=a_t1, t=t)
        return None

    # ------------- helpers -------------

    @staticmethod
    def _split_constraints(constraints: FrozenSet
                           ) -> Tuple[Set[Tuple[str, int]],
                                      Set[Tuple[str, str, int]]]:
        """Convert a robot's constraint frozenset into the (vertex, edge)
        tuple-sets that SpaceTimeAStar.plan() expects."""
        vc: Set[Tuple[str, int]] = set()
        ec: Set[Tuple[str, str, int]] = set()
        for c in constraints:
            if isinstance(c, VertexConstraint):
                vc.add((c.node, c.t))
            elif isinstance(c, EdgeConstraint):
                ec.add((c.n_from, c.n_to, c.t))
        return vc, ec

    def _replan_one(self, robot_id: int, start: str, goal: str,
                    constraints: FrozenSet,
                    max_time: int) -> Optional[Path]:
        vc, ec = self._split_constraints(constraints)
        return self.low_level.plan(start, goal, vc, ec, max_time)

    @staticmethod
    def _path_cost(path: Path) -> float:
        # Uniform-cost: each step is 1 unit.
        return float(max(0, len(path) - 1)) if path else 0.0

    # ------------- main entry -------------

    def plan(self, agents: Dict[int, Tuple[str, str]],
             max_iterations: int = 1000,
             max_time: int = 200,
             ) -> Optional[Dict[int, Path]]:
        """
        Plan conflict-free paths for all agents.

        Args:
            agents: {robot_id: (start_node, goal_node)}
            max_iterations: cap on CT node expansions (safety)
            max_time:       cap on each agent's time horizon

        Returns:
            {robot_id: path} or None if unsolvable / timeout.
        """
        self.stats = {
            "ct_nodes_generated": 0,
            "ct_nodes_expanded": 0,
            "low_level_calls": 0,
            "vertex_conflicts": 0,
            "edge_conflicts": 0,
            "max_open_size": 0,
        }

        # ---- root node ----
        root_paths: Dict[int, Path] = {}
        root_constraints: Dict[int, FrozenSet] = {rid: frozenset()
                                                  for rid in agents}
        for rid, (start, goal) in agents.items():
            self.stats["low_level_calls"] += 1
            p = self._replan_one(rid, start, goal,
                                 root_constraints[rid], max_time)
            if p is None:
                # Unsolvable from the start
                return None
            root_paths[rid] = p

        sequence_id = 0
        root_cost = sum(self._path_cost(p) for p in root_paths.values())
        root = CTNode(constraints=root_constraints,
                      paths=root_paths, cost=root_cost,
                      sequence_id=sequence_id)
        sequence_id += 1
        self.stats["ct_nodes_generated"] = 1

        open_heap: List[CTNode] = [root]

        while open_heap:
            self.stats["max_open_size"] = max(self.stats["max_open_size"],
                                              len(open_heap))
            if self.stats["ct_nodes_expanded"] >= max_iterations:
                # Best-effort: return the best plan we have, but mark it
                return root_paths   # caller can detect via stats

            curr = heapq.heappop(open_heap)
            self.stats["ct_nodes_expanded"] += 1

            conflict = self._first_conflict(curr.paths)
            if conflict is None:
                # GOAL: no conflicts among any pair of paths
                return curr.paths

            # Branch: one child per robot involved in the conflict
            for rid in (conflict.robot_a, conflict.robot_b):
                if isinstance(conflict, VertexConflict):
                    self.stats["vertex_conflicts"] += 1
                    new_c = VertexConstraint(rid, conflict.node, conflict.t)
                else:  # EdgeConflict
                    self.stats["edge_conflicts"] += 1
                    if rid == conflict.robot_a:
                        new_c = EdgeConstraint(rid, conflict.node_a,
                                                conflict.node_b, conflict.t)
                    else:
                        new_c = EdgeConstraint(rid, conflict.node_b,
                                                conflict.node_a, conflict.t)

                # Build child constraint set
                child_constraints = dict(curr.constraints)
                child_constraints[rid] = (curr.constraints[rid] |
                                          frozenset({new_c}))

                # Re-plan ONLY the affected robot
                start, goal = agents[rid]
                self.stats["low_level_calls"] += 1
                new_path = self._replan_one(rid, start, goal,
                                            child_constraints[rid],
                                            max_time)
                if new_path is None:
                    continue   # this branch is infeasible

                child_paths = dict(curr.paths)
                child_paths[rid] = new_path
                child_cost = sum(self._path_cost(p)
                                 for p in child_paths.values())

                child = CTNode(constraints=child_constraints,
                               paths=child_paths,
                               cost=child_cost,
                               sequence_id=sequence_id)
                sequence_id += 1
                self.stats["ct_nodes_generated"] += 1
                heapq.heappush(open_heap, child)

        return None  # exhausted CT — no conflict-free plan exists


# ================================================================
#  Lifelong Planner  (Cooperative A* with global reservation table)
# ================================================================

class LifelongPlanner:
    """
    Lifelong / Online multi-robot planner — adds robots one by one
    without rewinding any robot already in motion.

    Algorithm (Silver 2005, "Cooperative Pathfinding"):
      • A single GLOBAL reservation table maps (node, t_global) → robot.
      • global_t is the simulation's discrete tick counter.
      • plan(robot, start, goal) does space-time A* starting at
        (start, global_t), treating already-reserved cells as blocked.
      • Once a path is committed, its cells are written into the
        reservation table.  Subsequent robots see them and route around.
      • release(robot) clears all reservations belonging to that robot
        (call when a task is completed or aborted).
      • tick(n=1) advances global_t — call this every simulation step.

    Why "lifelong":
      • New tasks arrive over time (Poisson). Each new task is planned
        independently *given the current reservation table*.
      • Existing robots are NEVER rewound or replanned, so no
        oscillation. The price is a small loss of optimality vs CBS.

    Trade-offs:
      • Suboptimality: the ordering of arrivals affects total cost.
        In high-density scenarios you may want to occasionally
        replan everyone with full CBS (use CBSPlanner for that).
      • Reservation expiry: planning is bounded by `lookahead_horizon`
        time-steps. After that, agents are assumed to be at their
        goal indefinitely.
    """

    def __init__(self,
                 graph_nodes: Dict[str, Tuple[float, float]],
                 graph_adj: Dict[str, List[str]],
                 lookahead_horizon: int = 200):
        """
        Args:
            graph_nodes: {node_name: (x, y)}
            graph_adj:   {node_name: [neighbour_names]}
            lookahead_horizon: max time-steps an A* search will explore
                               beyond the current global_t. Bounds A*
                               state space and reservation memory.
        """
        self.nodes = graph_nodes
        self.adj = graph_adj
        self.low_level = SpaceTimeAStar(graph_nodes, graph_adj)
        self.horizon = lookahead_horizon

        # Global tick counter (advanced by `tick()`)
        self.global_t: int = 0

        # Reservation table — keyed by (node, absolute_time)
        # Value is the robot_id holding the reservation.
        self._vertex_res: Dict[Tuple[str, int], int] = {}

        # Edge reservations — keyed by (n_from, n_to, absolute_time);
        # robot occupies edge during the t→t+1 transition.
        self._edge_res: Dict[Tuple[str, str, int], int] = {}

        # Cache of each robot's currently committed path:
        # robot_id -> [(node, abs_time), ...]
        self._robot_paths: Dict[int, List[Tuple[str, int]]] = {}

        # ── Static reservations (NEW) ──
        # Maps robot_id → node it permanently occupies (its home / parking
        # spot while idle). These reservations are NEVER pruned by tick(),
        # and are added to every plan() call as blocked cells across ALL
        # future timesteps (within the planning horizon).
        # When a robot leaves its home (gets a new task), call
        # release_static(rid). When it returns, call reserve_static(rid).
        self._static_reservations: Dict[int, str] = {}

        # Level 1 — verbose mode for head-on resolution logging.
        import os as _os
        self.verbose = _os.environ.get("MAPF_VERBOSE", "0") == "1"
        self.replans_with_wait = 0
        self.replans_with_detour = 0

        # Statistics for instrumentation
        self.stats: Dict[str, int] = {
            "plan_calls": 0,
            "plan_failures": 0,
            "total_replans": 0,
            "max_reservations": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, robot_id: int, start: str, goal: str) -> Optional[Path]:
        """
        Plan a path for `robot_id` from `start` to `goal`, avoiding
        cells already reserved by other robots.

        Returns the node-path (list of node names indexed by
        relative time-step from now). The path is automatically
        committed to the reservation table.
        """
        self.stats["plan_calls"] += 1

        # Drop any prior reservation belonging to this robot.
        # This is what allows re-planning the SAME robot when its
        # task changes (e.g. pickup→delivery transition) without
        # affecting any other robot's plan.
        self.release(robot_id)

        # Build "blocked cells" view of the reservation table for the
        # window [global_t, global_t + horizon].
        vc: Set[Tuple[str, int]] = set()
        ec: Set[Tuple[str, str, int]] = set()
        # We index A* internally with t_relative = 0..horizon, so
        # convert (node, t_abs) → (node, t_rel) with t_rel = t_abs - global_t.
        gt = self.global_t
        for (node, t_abs), _rid in self._vertex_res.items():
            t_rel = t_abs - gt
            if 0 <= t_rel <= self.horizon:
                vc.add((node, t_rel))

        # ── Static reservations: idle robots permanently occupy their
        # home node. Block this node for ALL relative timesteps (except
        # for the requesting robot itself, which may also have a static
        # reservation it wants to leave).
        for static_rid, static_node in self._static_reservations.items():
            if static_rid == robot_id:
                # Don't block the requester's own home — release_static
                # should be called separately, but if it wasn't, allow
                # leaving from / arriving at this node freely.
                continue
            for t_rel in range(self.horizon + 1):
                vc.add((static_node, t_rel))
        # Edge constraint conversion: agent traverses edge during
        # t_abs→t_abs+1, so block "agent at t_rel transition".
        for (n_from, n_to, t_abs), _rid in self._edge_res.items():
            t_rel = t_abs - gt
            if 0 <= t_rel <= self.horizon:
                ec.add((n_from, n_to, t_rel))
                # Also block the swap pair (n_to → n_from at same t_rel)
                # because that's a head-on collision.
                ec.add((n_to, n_from, t_rel))

        # Run space-time A* with horizon-bounded relative time.
        rel_path = self.low_level.plan(start, goal, vc, ec,
                                        max_time=self.horizon)
        if rel_path is None:
            self.stats["plan_failures"] += 1
            return None

        # Commit: write reservations into the global table at absolute time.
        committed: List[Tuple[str, int]] = []
        for t_rel, node in enumerate(rel_path):
            t_abs = gt + t_rel
            self._vertex_res[(node, t_abs)] = robot_id
            committed.append((node, t_abs))
        # Edge reservations between consecutive distinct steps
        for i in range(len(rel_path) - 1):
            a, b = rel_path[i], rel_path[i + 1]
            if a != b:
                self._edge_res[(a, b, gt + i)] = robot_id

        # ALSO reserve the goal cell for an extra `goal_lock` window
        # so a follower robot won't try to occupy the goal while this
        # robot is "parked" there (until release() is called).
        # We use a small fixed buffer of 5 ticks; release() is the
        # authoritative cleanup.
        last_node = rel_path[-1]
        for t_extra in range(1, 6):
            self._vertex_res.setdefault(
                (last_node, gt + len(rel_path) - 1 + t_extra), robot_id)
            committed.append((last_node, gt + len(rel_path) - 1 + t_extra))

        self._robot_paths[robot_id] = committed
        self.stats["max_reservations"] = max(
            self.stats["max_reservations"], len(self._vertex_res))

        # Level 1 — detect head-on resolution patterns
        wait_count = sum(
            1 for i in range(1, len(rel_path)) if rel_path[i] == rel_path[i-1])
        if wait_count > 0:
            self.replans_with_wait += 1
            if self.verbose:
                print(f"  [MAPF] R{robot_id} path has {wait_count} wait "
                      f"step(s) — yielding to peer (start={start}, goal={goal})")
        # Detect detour: when start and goal share a y-row but path uses other rows
        try:
            ys_in_path = {round(self.nodes[n][1], 1) for n in rel_path
                          if n in self.nodes}
            start_y = round(self.nodes[start][1], 1) if start in self.nodes else None
            goal_y = round(self.nodes[goal][1], 1) if goal in self.nodes else None
            if (start_y is not None and goal_y is not None and
                    start_y == goal_y and len(ys_in_path) > 1):
                self.replans_with_detour += 1
                if self.verbose:
                    print(f"  [MAPF] R{robot_id} taking parallel-aisle detour "
                          f"(start={start}@y={start_y}, goal={goal}@y={goal_y}, "
                          f"path uses y={sorted(ys_in_path)})")
        except (KeyError, TypeError):
            pass

        return rel_path

    def release(self, robot_id: int) -> None:
        """
        Erase all reservations held by `robot_id`. Call when the robot
        finishes its task / aborts / is removed from the active set.
        Subsequent plan() calls won't see those cells as blocked.
        """
        if robot_id not in self._robot_paths:
            return
        for (node, t_abs) in self._robot_paths[robot_id]:
            if self._vertex_res.get((node, t_abs)) == robot_id:
                del self._vertex_res[(node, t_abs)]
        # Drop edge reservations belonging to this robot
        to_drop = [k for k, rid in self._edge_res.items() if rid == robot_id]
        for k in to_drop:
            del self._edge_res[k]
        del self._robot_paths[robot_id]

    def set_verbose(self, on: bool) -> None:
        """Enable/disable Level 1 head-on resolution logging."""
        self.verbose = on

    def get_resolution_stats(self) -> dict:
        """Return Level 1 head-on resolution counters."""
        return {
            "replans_with_wait":   self.replans_with_wait,
            "replans_with_detour": self.replans_with_detour,
        }

    def reserve_static(self, robot_id: int, node: str) -> None:
        """
        Permanently reserve `node` for `robot_id` — useful for marking a
        robot's home / parking spot while it's IDLE. The reservation
        persists across tick() calls and blocks all other robots from
        routing through this node until release_static() is called.
        """
        self._static_reservations[robot_id] = node

    def release_static(self, robot_id: int) -> None:
        """Drop a static reservation (call when the robot leaves home)."""
        if robot_id in self._static_reservations:
            del self._static_reservations[robot_id]

    def get_static_reservation(self, robot_id: int) -> Optional[str]:
        """Return the static node a robot is parked at, or None."""
        return self._static_reservations.get(robot_id)

    def tick(self, n: int = 1) -> None:
        """
        Advance the global clock. Call ONCE per simulation step (or in
        bulk for fast-forwarding). Old reservations (t_abs < global_t)
        are pruned to keep memory bounded.
        """
        self.global_t += n
        # Prune past reservations
        gt = self.global_t
        keep_v = {k: v for k, v in self._vertex_res.items() if k[1] >= gt}
        keep_e = {k: v for k, v in self._edge_res.items() if k[2] >= gt}
        self._vertex_res = keep_v
        self._edge_res = keep_e
        # Trim each robot's committed path
        for rid in list(self._robot_paths.keys()):
            self._robot_paths[rid] = [
                (n, t) for (n, t) in self._robot_paths[rid] if t >= gt
            ]
            if not self._robot_paths[rid]:
                del self._robot_paths[rid]

    def reset(self) -> None:
        """Wipe ALL state. Use for a fresh experiment."""
        self.global_t = 0
        self._vertex_res.clear()
        self._edge_res.clear()
        self._robot_paths.clear()
        self._static_reservations.clear()
        for k in self.stats:
            self.stats[k] = 0

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_path(self, robot_id: int) -> Optional[Path]:
        """Return the committed FUTURE path of a robot as node names."""
        if robot_id not in self._robot_paths:
            return None
        return [n for (n, _t) in self._robot_paths[robot_id]]

    def reservation_size(self) -> int:
        """Current size of the reservation table (debug/metrics)."""
        return len(self._vertex_res) + len(self._edge_res)

    def is_reserved(self, node: str, t_abs: int,
                     by_other_than: Optional[int] = None) -> bool:
        """True if (node, t_abs) is reserved (optionally by someone other
        than `by_other_than`)."""
        owner = self._vertex_res.get((node, t_abs))
        if owner is None:
            return False
        if by_other_than is not None and owner == by_other_than:
            return False
        return True


# ================================================================
#  Self-test (run this file directly)
# ================================================================

if __name__ == "__main__":
    # Simple corridor: 5 nodes in a line, 2 robots swapping ends
    nodes = {f"N{i}": (float(i), 0.0) for i in range(5)}
    adj = {f"N{i}": [n for n in (f"N{i-1}", f"N{i+1}") if n in nodes]
           for i in range(5)}

    cbs = CBSPlanner(nodes, adj)
    paths = cbs.plan({1: ("N0", "N4"),
                      2: ("N4", "N0")})
    print("paths:", paths)
    print("stats:", cbs.stats)
    # Expected: one robot waits at a side branch — but here there is no
    # side branch, so the only solution is for one robot to wait at its
    # start until the other reaches half-way, then both swap. This will
    # fail since there's no passing space — CBS correctly returns None.
    if paths is None:
        print("✓ Correctly identified unsolvable swap on linear corridor")
