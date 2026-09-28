"""
Grid-based Path Planner for Factory Floor
==========================================

Replaces the old corridor graph (38 hand-placed nodes + 45 edges) with a
fine-grained occupancy grid that lets robots traverse the entire free
space — diagonals, around shelves, through gaps — without being limited
to north/south/middle corridors.

Architecture
------------
Factory: 18m × 11m (x ∈ [-9, 9], y ∈ [-5.5, 5.5])
Resolution: 0.25 m/cell (chosen so robot footprint = 1.4 cells diameter
            and shelves take ≥2 cells of obstacle padding)
Grid: 72 cols × 44 rows = 3168 cells

Each cell is labeled:
  FREE         — robot can occupy
  OBSTACLE     — physical object (shelf, table, factory boundary)
  INFLATED     — within ROBOT_RADIUS of an obstacle (forbidden)
  AVOID_ZONE   — inside a workstation/storage/charging zone (cost penalty
                 but not blocked — allows approach to docks)

The grid is built ONCE at startup from config.OBSTACLE_BOXES + dock zones,
then used by GridAStar for any-to-any path queries.

Key features
------------
* 8-connectivity (N, NE, E, SE, S, SW, W, NW) with √2 diagonal cost
* Octile heuristic (admissible & consistent for 8-conn grids)
* Theta*-style line-of-sight smoothing post-processing (reduces zigzag)
* Path waypoint subsampling (every 1.5m, for time-step granularity)
* Obstacle inflation by ROBOT_RADIUS + SAFETY_MARGIN
"""

import math
import heapq
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field

from config import (
    OBSTACLE_BOXES, SHELF_GAP_BOXES,
    line_intersects_obstacles,
    ROBOT_RADIUS, WORKSTATIONS, STORAGE_AREAS, CHARGING_STATIONS,
    ALL_LOCATIONS,
)


# Factory bounds (in metres)
FACTORY_X_MIN = -9.0
FACTORY_X_MAX =  9.0
FACTORY_Y_MIN = -5.5
FACTORY_Y_MAX =  5.5

# Grid resolution (smaller = more accurate, slower planning)
GRID_RES = 0.25
# Safety margin beyond ROBOT_RADIUS for inflation
SAFETY_MARGIN = 0.55   # increased from 0.32 — keeps path ≥1 grid cell  # Increased from 0.05 — robot edge stays ≥0.32m from raw obstacle

# Cell labels
CELL_FREE      = 0
CELL_OBSTACLE  = 1   # shelf / table / wall
CELL_INFLATED  = 2   # within ROBOT_RADIUS+safety of an obstacle
CELL_AVOID     = 3   # inside a "stay-out" zone (penalty, not blocked)
DYNAMIC_AVOID_COST = 8.0  # prefer a longer free corridor over a peer path


@dataclass(order=True)
class _PQItem:
    priority: float
    counter: int
    node: Tuple[int, int] = field(compare=False)


class OccupancyGrid:
    """
    2D grid representation of the factory floor.
    
    Conversion:
      world (x, y)  →  grid (col, row)
      col = round((x - FACTORY_X_MIN) / GRID_RES)
      row = round((y - FACTORY_Y_MIN) / GRID_RES)
    """
    
    def __init__(self, num_active_robots: int = 8):
        # Grid dimensions
        self.cols = int(round((FACTORY_X_MAX - FACTORY_X_MIN) / GRID_RES))
        self.rows = int(round((FACTORY_Y_MAX - FACTORY_Y_MIN) / GRID_RES))
        self.num_active_robots = num_active_robots
        # 2D array: cells[row][col] = CELL_*
        self.cells = [[CELL_FREE for _ in range(self.cols)]
                      for _ in range(self.rows)]
        self._build_obstacles()
        self._mark_inactive_robots()
        self._inflate_obstacles()
        self._carve_crossing_corridors()

    # ------------------------------------------------------------------
    # Crossing corridor carving
    # ------------------------------------------------------------------
    def _carve_crossing_corridors(self):
        """Disabled: shelf gaps (1.0m) too narrow for reliable DWA passage.
        Robots route around the shelf block via east/west corridors."""
        pass


    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(round((x - FACTORY_X_MIN) / GRID_RES))
        row = int(round((y - FACTORY_Y_MIN) / GRID_RES))
        return (col, row)
    
    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        x = FACTORY_X_MIN + col * GRID_RES
        y = FACTORY_Y_MIN + row * GRID_RES
        return (x, y)
    
    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows
    
    def is_free(self, col: int, row: int) -> bool:
        """Cell is traversable (FREE or AVOID, not OBSTACLE/INFLATED)."""
        if not self.in_bounds(col, row):
            return False
        return self.cells[row][col] not in (CELL_OBSTACLE, CELL_INFLATED)
    
    def cost_at(self, col: int, row: int) -> float:
        """Multiplier for traversing this cell (1.0=free, >1.0=avoid zone)."""
        if not self.in_bounds(col, row):
            return float('inf')
        c = self.cells[row][col]
        if c in (CELL_OBSTACLE, CELL_INFLATED):
            return float('inf')
        if c == CELL_AVOID:
            return DYNAMIC_AVOID_COST
        return 1.0
    
    # ------------------------------------------------------------------
    # Build obstacles from OBSTACLE_BOXES
    # ------------------------------------------------------------------
    def _build_obstacles(self):
        """
        Mark cells inside any OBSTACLE_BOX as CELL_OBSTACLE.
        Note: OBSTACLE_BOXES from config.py are ALREADY inflated by
        ROBOT_RADIUS for line-of-sight tests in motion_coordinator.
        For the grid we want the RAW obstacle, then re-inflate.
        Strip the inflation here.
        """
        # ── Physical obstacles (pre-inflated by ROBOT_RADIUS in config) ─
        for box in OBSTACLE_BOXES:
            cx, cy = box['cx'], box['cy']
            raw_hx = max(0.05, box['hx'] - ROBOT_RADIUS)
            raw_hy = max(0.05, box['hy'] - ROBOT_RADIUS)
            for col in range(self.cols):
                for row in range(self.rows):
                    cell_x, cell_y = self.grid_to_world(col, row)
                    if (abs(cell_x - cx) <= raw_hx and
                            abs(cell_y - cy) <= raw_hy):
                        self.cells[row][col] = CELL_OBSTACLE
        
        # ── Policy keep-outs (NOT pre-inflated — use raw hx/hy) ────────
        # SHELF_GAP_BOXES define the narrow aisles between shelves which
        # robots are NOT allowed to plan through. We mark them as
        # CELL_OBSTACLE so GridAStar treats them as solid; the standard
        # ROBOT_RADIUS+SAFETY inflation pass will then handle the buffer.
        for box in SHELF_GAP_BOXES:
            cx, cy = box['cx'], box['cy']
            raw_hx = box['hx']
            raw_hy = box['hy']
            for col in range(self.cols):
                for row in range(self.rows):
                    cell_x, cell_y = self.grid_to_world(col, row)
                    if (abs(cell_x - cx) <= raw_hx and
                            abs(cell_y - cy) <= raw_hy):
                        if self.cells[row][col] == CELL_FREE:
                            self.cells[row][col] = CELL_OBSTACLE
        # Also mark factory boundary ring (within ROBOT_RADIUS of edges)
        boundary_pad = 0.30   # robot can't stick into wall
        for col in range(self.cols):
            for row in range(self.rows):
                cx, cy = self.grid_to_world(col, row)
                if (cx - FACTORY_X_MIN < boundary_pad or
                        FACTORY_X_MAX - cx < boundary_pad or
                        cy - FACTORY_Y_MIN < boundary_pad or
                        FACTORY_Y_MAX - cy < boundary_pad):
                    if self.cells[row][col] == CELL_FREE:
                        self.cells[row][col] = CELL_OBSTACLE
    
    def _mark_inactive_robots(self):
        """Do not reserve parking spots for robots absent from the scene.

        FactorySupervisor removes surplus Webots Robot nodes before creating
        this grid. Standalone mode also creates only the selected fleet.
        Marking removed robots as obstacles made scenario A/B geometry differ
        from the corresponding Webots scene.
        """
        return

    def _inflate_obstacles(self):
        """
        Expand OBSTACLE cells by ROBOT_RADIUS+SAFETY into INFLATED cells.
        This is a true Minkowski sum: any cell within `inflate_cells` of
        an OBSTACLE cell becomes INFLATED (un-traversable).
        """
        inflate_dist = ROBOT_RADIUS + SAFETY_MARGIN
        # Ensure ≥1 cell of clearance even at low resolution
        inflate_cells = max(1, int(math.ceil(inflate_dist / GRID_RES)))
        # Find all OBSTACLE cells, then mark their neighbors
        obstacle_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c] == CELL_OBSTACLE:
                    obstacle_cells.append((c, r))
        for (oc, or_) in obstacle_cells:
            for dc in range(-inflate_cells, inflate_cells + 1):
                for dr in range(-inflate_cells, inflate_cells + 1):
                    nc, nr = oc + dc, or_ + dr
                    if not self.in_bounds(nc, nr):
                        continue
                    if self.cells[nr][nc] == CELL_FREE:
                        # Use Chebyshev distance (max(|dc|,|dr|)) — at
                        # low resolution this is more conservative than
                        # Euclidean and prevents the path from skimming
                        # obstacle corners.
                        if max(abs(dc), abs(dr)) <= inflate_cells:
                            self.cells[nr][nc] = CELL_INFLATED
    
    def stats(self) -> dict:
        """Return cell-count statistics."""
        counts = {CELL_FREE: 0, CELL_OBSTACLE: 0, CELL_INFLATED: 0,
                  CELL_AVOID: 0}
        for row in self.cells:
            for c in row:
                counts[c] = counts.get(c, 0) + 1
        return {
            "cols": self.cols, "rows": self.rows,
            "total": self.cols * self.rows,
            "free": counts.get(CELL_FREE, 0),
            "obstacle": counts.get(CELL_OBSTACLE, 0),
            "inflated": counts.get(CELL_INFLATED, 0),
            "avoid": counts.get(CELL_AVOID, 0),
            "free_pct": 100.0 * counts.get(CELL_FREE, 0) /
                        (self.cols * self.rows),
        }


class GridAStar:
    """
    8-connectivity A* on the OccupancyGrid with octile heuristic.
    Returns world-coordinate paths [(x,y), ...] from start to goal.
    """
    
    # 8-neighbour directions: (dc, dr, cost_multiplier)
    NEIGHBOURS = [
        (-1, -1, math.sqrt(2)),  # NW
        ( 0, -1, 1.0),           # N
        ( 1, -1, math.sqrt(2)),  # NE
        ( 1,  0, 1.0),           # E
        ( 1,  1, math.sqrt(2)),  # SE
        ( 0,  1, 1.0),           # S
        (-1,  1, math.sqrt(2)),  # SW
        (-1,  0, 1.0),           # W
    ]
    
    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
    
    def _octile(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (max(dx, dy) - min(dx, dy)) + math.sqrt(2) * min(dx, dy)
    
    def _snap_to_free(self,
                      col: int, row: int,
                      max_radius: int = 8) -> Optional[Tuple[int, int]]:
        """If (col, row) is blocked, find the nearest free cell within
        max_radius cells. Used for endpoints near obstacles (e.g. dock
        positions that may have been pushed into INFLATED zones)."""
        if self.grid.is_free(col, row):
            return (col, row)
        # Spiral search
        for r in range(1, max_radius + 1):
            for dc in range(-r, r + 1):
                for dr in range(-r, r + 1):
                    if max(abs(dc), abs(dr)) != r:
                        continue
                    nc, nr = col + dc, row + dr
                    if self.grid.is_free(nc, nr):
                        return (nc, nr)
        return None
    
    def plan(self,
             start_xy: Tuple[float, float],
             goal_xy: Tuple[float, float],
             smooth: bool = True,
             relax_endpoints: bool = True
             ) -> Optional[List[Tuple[float, float]]]:
        """
        Find a path from start_xy to goal_xy through the free grid.
        
        Args:
            start_xy:  world (x, y) start position
            goal_xy:   world (x, y) goal position
            smooth:    apply line-of-sight smoothing (default True)
        
        Returns:
            List of (x, y) world waypoints, or None if no path found.
        """
        start_cell_orig = self.grid.world_to_grid(*start_xy)
        goal_cell_orig = self.grid.world_to_grid(*goal_xy)
        
        # ── Endpoint relaxation ─────────────────────────────────
        # Inflated obstacles are conservative for paths in the middle
        # of free space, but they can also "swallow" the dock cells
        # (S/WS/CS positions that are near tables/shelves). To keep
        # docks reachable, we temporarily mark only the EXACT start
        # and goal cells (and a small ring around them) as FREE
        # during this plan() call, then restore afterwards.
        relaxed_cells = []
        if relax_endpoints:
            for (c0, r0) in [start_cell_orig, goal_cell_orig]:
                if c0 is None or r0 is None:
                    continue
                # Relax a 2-cell ring around endpoint (5×5 cells).
                # Charging stations near the boundary can be four cells from
                # permanent free space.  A 2-cell ring leaves them as isolated
                # islands and makes every return-to-charge replan fail.
                # Physical OBSTACLE cells below are still never relaxed.
                endpoint_xy = self.grid.grid_to_world(c0, r0)
                is_charging_endpoint = any(
                    math.hypot(endpoint_xy[0] - station[0],
                               endpoint_xy[1] - station[1]) <= GRID_RES
                    for station in CHARGING_STATIONS.values())
                relax_radius = 4 if is_charging_endpoint else 2
                for dc in range(-relax_radius, relax_radius + 1):
                    for dr in range(-relax_radius, relax_radius + 1):
                        nc, nr = c0 + dc, r0 + dr
                        if not self.grid.in_bounds(nc, nr):
                            continue
                        # Only relax INFLATED — never relax actual OBSTACLE
                        if self.grid.cells[nr][nc] == CELL_INFLATED:
                            relaxed_cells.append(
                                (nc, nr, self.grid.cells[nr][nc]))
                            self.grid.cells[nr][nc] = CELL_FREE
        
        # Now do the snap (after relaxation, dock cells should already be free)
        start = self._snap_to_free(*start_cell_orig)
        goal = self._snap_to_free(*goal_cell_orig)
        if start is None or goal is None:
            # Restore relaxed cells before returning
            for (c, r, original) in relaxed_cells:
                self.grid.cells[r][c] = original
            return None
        
        # Wrap the actual A* in a try/finally so relaxed cells are
        # always restored to their original INFLATED state.
        try:
            return self._plan_inner(start, goal, start_xy, goal_xy, smooth)
        finally:
            for (c, r, original) in relaxed_cells:
                self.grid.cells[r][c] = original
    
    def _plan_inner(self,
                     start: Tuple[int, int],
                     goal: Tuple[int, int],
                     start_xy: Tuple[float, float],
                     goal_xy: Tuple[float, float],
                     smooth: bool
                     ) -> Optional[List[Tuple[float, float]]]:
        """Internal A* search — called by plan() within a try/finally
        block that handles cell-relaxation cleanup."""
        if start == goal:
            # Same grid cell normally means the robot is already effectively
            # at its destination. Returning a [start, goal] pair can be
            # dispatched as a movement plan even when the two world points
            # coincide; controllers then report arrival without moving.
            if (math.hypot(start_xy[0] - goal_xy[0],
                           start_xy[1] - goal_xy[1]) < 0.05):
                return None
            return [start_xy, goal_xy]
        
        # A* with octile heuristic
        open_set: list = []
        counter = 0
        heapq.heappush(open_set, _PQItem(0.0, counter, start))
        came_from = {start: None}
        g_score = {start: 0.0}
        
        while open_set:
            item = heapq.heappop(open_set)
            current = item.node
            # Skip heap entries superseded by a lower-cost route.
            if (current != start and
                    item.priority > (g_score[current] +
                                     self._octile(current, goal)) + 1e-12):
                continue
            if current == goal:
                # Reconstruct
                cells: list = []
                while current is not None:
                    cells.append(current)
                    current = came_from[current]
                cells.reverse()
                # Convert to world coords
                path = [self.grid.grid_to_world(c, r) for (c, r) in cells]
                # Replace endpoints with exact start_xy / goal_xy
                if path:
                    path[0] = start_xy
                    path[-1] = goal_xy
                if smooth:
                    path = self._smooth(path)
                return path
            
            for dc, dr, base_cost in self.NEIGHBOURS:
                nc = current[0] + dc
                nr = current[1] + dr
                if not self.grid.is_free(nc, nr):
                    continue
                # Block diagonal cuts through corners
                if abs(dc) == 1 and abs(dr) == 1:
                    if (not self.grid.is_free(current[0] + dc, current[1]) or
                            not self.grid.is_free(current[0], current[1] + dr)):
                        continue
                
                cost = base_cost * self.grid.cost_at(nc, nr)
                tentative_g = g_score[current] + cost
                neighbour = (nc, nr)
                if tentative_g < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f = tentative_g + self._octile(neighbour, goal)
                    counter += 1
                    heapq.heappush(open_set, _PQItem(f, counter, neighbour))
        
        return None  # No path found
    
    def _smooth(self,
                path: List[Tuple[float, float]]
                ) -> List[Tuple[float, float]]:
        """
        Theta*-style line-of-sight smoothing: walk from start, keep
        extending the segment as long as the straight line is collision-
        free. Greatly reduces zigzag from grid quantization.
        """
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Find the farthest j such that straight line path[i] → path[j]
            # is collision-free (and stays within free cells)
            j = len(path) - 1
            while j > i + 1:
                if self._line_clear(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed
    
    def _line_clear(self,
                    p1: Tuple[float, float],
                    p2: Tuple[float, float]) -> bool:
        """Check whether the straight line p1→p2 stays in free grid cells.
        Walks along the line at GRID_RES/2 step size to detect obstacles
        regardless of orientation."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        n_steps = int(dist / (GRID_RES * 0.5)) + 1
        for k in range(n_steps + 1):
            t = k / n_steps
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            col, row = self.grid.world_to_grid(x, y)
            if not self.grid.is_free(col, row):
                return False
        # Also use config-level obstacle check as belt-and-suspenders
        return not line_intersects_obstacles(p1, p2)


# ----------------------------------------------------------------
# Convenience: subsample a long path into ~waypoints every 1.5m
# This is useful for the LifelongPlanner reservation table — too many
# waypoints would inflate the reservation set without benefit.
# ----------------------------------------------------------------
def subsample_path(path: List[Tuple[float, float]],
                   step_m: float = 1.5,
                   grid: Optional[OccupancyGrid] = None
                   ) -> List[Tuple[float, float]]:
    """
    Return a subset of `path` keeping waypoints far enough apart, but
    ALWAYS preserving any waypoint whose removal would create a non
    line-of-sight (i.e. collision-prone) segment.
    
    The naive distance-based subsample would drop corner-escape waypoints
    that route around shelves, causing the robot to cut diagonally
    through the obstacle. This LoS-aware version prevents that.
    """
    if len(path) <= 2:
        return list(path)
    if grid is None:
        grid = OccupancyGrid()

    def _los_clear(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        n_steps = int(dist / (GRID_RES * 0.5)) + 1
        for k in range(n_steps + 1):
            t = k / n_steps
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            col, row = grid.world_to_grid(x, y)
            if not grid.is_free(col, row):
                return False
        return not line_intersects_obstacles(p1, p2)

    result = [path[0]]
    last = path[0]
    for i, wp in enumerate(path[1:-1], start=1):
        far_enough = math.hypot(wp[0] - last[0], wp[1] - last[1]) >= step_m
        # Critical safety check: if dropping `wp` would create an unsafe
        # segment from `last` to the next path point, we MUST keep `wp`.
        next_wp = path[i + 1]
        must_keep = not _los_clear(last, next_wp)
        if far_enough or must_keep:
            result.append(wp)
            last = wp
    result.append(path[-1])

    # Final sanity pass: refuse to return an unsafe subsampling.
    for i in range(len(result) - 1):
        if not _los_clear(result[i], result[i + 1]):
            return list(path)
    return result
