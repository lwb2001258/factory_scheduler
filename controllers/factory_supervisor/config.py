"""
Configuration for Smart Factory Multi-Robot Coordination Simulation.
All factory layout, robot, scheduling, and experiment parameters.
"""

import math
import os

# ================================================================
# SIMULATION PARAMETERS
# ================================================================
TIMESTEP = 16          # Webots basic time step in ms
SIM_DURATION = float(os.environ.get("SMART_FACTORY_SIM_DURATION", "1800.0"))
# A duration is an automatic stop condition only for scripted experiments.
# Opening the world directly in the Webots GUI is interactive and must keep
# running until the user stops it.  Batch launchers set this flag explicitly.
AUTO_STOP_SIMULATION = os.environ.get(
    "SMART_FACTORY_AUTO_STOP", "0").strip().lower() in {"1", "true", "yes"}
# Runtime RHCR currently produces a read-only CBS candidate that is not
# transactionally dispatched to robots.  On an eight-robot scene the
# combinatorial CBS search can block Webots' synchronous controller loop for
# minutes, so keep it opt-in until candidate adoption is implemented.
ENABLE_RUNTIME_RHCR = os.environ.get(
    "SMART_FACTORY_ENABLE_RHCR", "0").strip().lower() in {"1", "true", "yes"}
# Production navigation is coordinated as one rolling space-time plan.  The
# legacy per-robot repair stack remains available only for A/B diagnostics.
ENABLE_JOINT_RUNTIME = os.environ.get(
    "SMART_FACTORY_ENABLE_JOINT_RUNTIME", "1"
).strip().lower() in {"1", "true", "yes"}
# The group speed optimizer remains available for controlled experiments, but
# repeated Webots validation showed throughput and replan regressions when it
# was enabled in the synchronous runtime loop. Keep the stable pairwise
# predictor as the production default until a kinematic/DWA-aware joint model
# is available.
ENABLE_PROACTIVE_JOINT_SPEED = os.environ.get(
    "SMART_FACTORY_ENABLE_JOINT_SPEED", "0"
).strip().lower() in {"1", "true", "yes"}
# The legacy 1.5-second interlock state machine overlaps the progress monitor
# and can misclassify robots waiting for a reached-goal packet as deadlocked.
# Keep the deterministic coordinator monitor plus 3s/10s recovery enabled;
# expose the legacy layer only for controlled comparison experiments.
ENABLE_LEGACY_INTERLOCK_RECOVERY = os.environ.get(
    "SMART_FACTORY_ENABLE_LEGACY_INTERLOCK", "0"
).strip().lower() in {"1", "true", "yes"}
# Moving Webots nodes hides coordination failures and is not physically
# deployable. Keep teleport-style recovery available only for diagnostics;
# formal experiments use planned retreat/replan recovery by default.
ENABLE_NONPHYSICAL_RECOVERY = os.environ.get(
    "SMART_FACTORY_ENABLE_NONPHYSICAL_RECOVERY", "0"
).strip().lower() in {"1", "true", "yes"}
# Deterministic priority-yield state machine used by joint runtime.  The
# legacy path is preserved when disabled so controlled A/B experiments can
# compare the new yield -> wait-for-clear -> resume loop against the existing
# head-on yield heuristic.
ENABLE_PRIORITY_YIELD_RESUME = os.environ.get(
    "SMART_FACTORY_ENABLE_PRIORITY_YIELD_RESUME", "0"
).strip().lower() in {"1", "true", "yes"}
YIELD_PREDICTION_HORIZON = 10.0
YIELD_PREDICTION_DT = 0.5
YIELD_RESUME_MIN_CLEARANCE = 1.00
YIELD_RESUME_HORIZON = 4.0
YIELD_RESUME_TIMEOUT = 10.0
YIELD_STANDOFF_SETTLE_SECONDS = 1.5
# Supervisor cadence controls. The logic is unchanged; these only change how
# often a few expensive bookkeeping scans run in wall-clock time.
JOINT_WATCHDOG_INTERVAL = float(os.environ.get(
    "SMART_FACTORY_JOINT_WATCHDOG_INTERVAL", "0.5"))
RESERVATION_REFRESH_INTERVAL = float(os.environ.get(
    "SMART_FACTORY_RESERVATION_REFRESH_INTERVAL", "1.0"))
NUM_REPEATS = 5        # Number of runs per experiment with different seeds

# ================================================================
# FACTORY LAYOUT
# ----------------------------------------------------------------
# Webots ENU coordinate system:
#   x = east  (left/right on floor plane)
#   y = north (up/down on floor plane, NOT vertical!)
#   z = up   (vertical axis)
#
# All location dictionaries below store (x, y) on the floor plane.
# Locations are *robot approach points* (where the robot stops to
# pickup/deliver), NOT the physical centres of workstation tables or
# shelves. Approach points are offset from the physical object so the
# robot's bounding box (radius 0.18 m) does not collide with it.
#
# Physical objects in the Webots world (.wbt) — for reference only:
#   Workstation tables: 2.0 x 1.5 x 0.8 m at y=±5  (front edge at y=±4.25)
#   Shelves:            1.0 x 2.0 x 1.0 m at y=0   (edges at y=±1)
# ================================================================

# Workstation approach points (robot stops here to pickup/deliver)
WORKSTATIONS = {
    "WS1": (-6.0, 3.5),   # Line A, Left   (table at y=+5)
    "WS2": (0.0, 3.5),    # Line A, Center
    "WS3": (6.0, 3.5),    # Line A, Right
    "WS4": (-6.0, -3.5),  # Line B, Left   (table at y=-5)
    "WS5": (0.0, -3.5),   # Line B, Center
    "WS6": (6.0, -3.5),   # Line B, Right
}

# Storage area approach points (robot stops here next to a shelf)
STORAGE_AREAS = {
    "S1": (-3.0, 1.5),    # North side of shelf 1
    "S2": (-1.0, 1.5),    # North side of shelf 2
    "S3": (1.0, 1.5),     # North side of shelf 3
    "S4": (3.0, 1.5),     # North side of shelf 4
    "S5": (-3.0, -1.5),   # South side of shelf 1
    "S6": (-1.0, -1.5),   # South side of shelf 2
    "S7": (1.0, -1.5),    # South side of shelf 3
    "S8": (3.0, -1.5),    # South side of shelf 4
}

# Charging station positions (match Webots .wbt exactly)
CHARGING_STATIONS = {
    "CS1": (-8.5, 0.0),   # Left side
    "CS2": (8.5, 0.0),    # Right side
}

# ================================================================
# ROBOT HOME / PARKING SPOTS
# ================================================================
# Each robot has a designated "home spot" — its initial position in
# the factory floor. After completing a delivery, robots return here
# instead of parking at the workstation, which prevents goal-blocking:
#
#   ❌ BEFORE: Robot 1 sits at WS2 → blocks Robot 2 trying to deliver
#              another task to WS2 → DEADLOCK on the workstation lane.
#   ✅ AFTER:  Robot 1 returns to (-7, 3) (its home spot) → WS2 is free
#              for any robot that needs it next.
#
# These coordinates EXACTLY mirror the ROBOT_N translation values in
# worlds/smart_factory.wbt. Each spot is in a quiet area of the
# corridor mesh (y=±3), well away from workstation/storage approach
# points (y=±3.5, ±1.5).
PARKING_SPOTS = {
    # Initial home spots — used at sim startup only. After completing
    # tasks, robots use task chaining + lazy relocation (see REST_NODES)
    # rather than always returning home.
    # Free graph nodes (8): N2, N4, N6, N10, N12, N14, N3, N13
    1: (-4.0,  3.0),  # ROBOT_1 → N2  (north corridor west)
    2: (-4.0, -3.0),  # ROBOT_2 → N12 (south corridor west)
    3: ( 4.0,  3.0),  # ROBOT_3 → N4  (north corridor east)
    4: ( 4.0, -3.0),  # ROBOT_4 → N14 (south corridor east)
    5: (-7.0,  0.0),  # ROBOT_5 → N6  (west vertical aisle)
    6: ( 7.0,  0.0),  # ROBOT_6 → N10 (east vertical aisle)
    7: (-8.5,  1.2),  # ROBOT_7 → near CS1, clear of north workstations
    8: ( 8.5, -1.2),  # ROBOT_8 → near CS2, clear of south workstations
}

# ----------------------------------------------------------------
# REST_NODES — corridor / aisle nodes where IDLE robots park
# while waiting for the next task.
# ----------------------------------------------------------------
# Used by lazy relocation: after finishing a delivery, a robot
# goes IDLE in place, then if no new task arrives within ~1s it
# relocates to the NEAREST rest node (not its original home).
# This dramatically reduces empty travel and keeps robots distributed
# across the factory floor — much more efficient than every robot
# returning to its assigned home spot.
#
# All rest nodes are NON-DOCK graph nodes that are NOT in
# LOCATION_TO_NODE.values() — meaning they're "free" corridor nodes
# that don't block any task pickup/delivery point.
REST_NODES = [
    "N2",      # (-4, 3)  north corridor mid-west
    "N3",      # ( 0, 3)  north corridor centre (between WS1/WS2)
    "N4",      # ( 4, 3)  north corridor mid-east
    "N6",      # (-7, 0)  west vertical aisle
    "N10",     # ( 7, 0)  east vertical aisle
    "N12",     # (-4, -3) south corridor mid-west
    "N13",     # ( 0, -3) south corridor centre (between WS4/WS5)
    "N14",     # ( 4, -3) south corridor mid-east
    "SA_N5",   # ( 4, 1.5) NE storage aisle end
    "SA_S5",   # ( 4, -1.5) SE storage aisle end
]

# All task-relevant locations (for pickup/delivery)
ALL_LOCATIONS = {}
ALL_LOCATIONS.update(WORKSTATIONS)
ALL_LOCATIONS.update(STORAGE_AREAS)

# ----------------------------------------------------------------
# Path planning graph (waypoint network).
#
# IMPORTANT: The original design had nodes N6..N10 along the y=0
# centre line, but that line is occupied by 4 shelves. Using those
# nodes makes the planner generate paths that physically pass through
# shelves. We replace them with two storage-aisle rows at y=±1.5
# (SA_N*/SA_S*) which sit *between* shelves and have collision-free
# straight-line connectivity.
#
# Layout:
#
#       N1───N2────N3────N4───N5             (top corridor   y=+3)
#       │    │     │     │    │
#       │  SA_N1─SA_N2─SA_N3─SA_N4─SA_N5    (north aisle    y=+1.5)
#       N6   │    │     │    │     N10      (charging-arm   y=  0)
#       │  SA_S1─SA_S2─SA_S3─SA_S4─SA_S5    (south aisle    y=-1.5)
#       │    │     │     │    │
#       N11──N12───N13───N14──N15            (bottom corridor y=-3)
#
# Vertical lanes connect through aisles to avoid the y=0 shelf row.
# ----------------------------------------------------------------
WAYPOINTS = {
    # Top horizontal corridor (free of obstacles)
    "N1":  (-7.0,  3.0),
    "N2":  (-4.0,  3.0),
    "N3":  ( 0.0,  3.0),
    "N4":  ( 4.0,  3.0),
    "N5":  ( 7.0,  3.0),
    # Outer middle waypoints (only at factory edges, where there are no shelves)
    "N6":  (-7.0,  0.0),
    "N10": ( 7.0,  0.0),
    # Bottom horizontal corridor (free of obstacles)
    "N11": (-7.0, -3.0),
    "N12": (-4.0, -3.0),
    "N13": ( 0.0, -3.0),
    "N14": ( 4.0, -3.0),
    "N15": ( 7.0, -3.0),
    # North storage aisle (y=+1.5) — sits in the gap between shelves and top corridor
    "SA_N1": (-4.0,  1.5),  # west of shelf_1
    "SA_N2": (-2.0,  1.5),  # between shelf_1 and shelf_2
    "SA_N3": ( 0.0,  1.5),  # between shelf_2 and shelf_3
    "SA_N4": ( 2.0,  1.5),  # between shelf_3 and shelf_4
    "SA_N5": ( 4.0,  1.5),  # east of shelf_4
    # South storage aisle (y=-1.5) — mirror of the above
    "SA_S1": (-4.0, -1.5),
    "SA_S2": (-2.0, -1.5),
    "SA_S3": ( 0.0, -1.5),
    "SA_S4": ( 2.0, -1.5),
    "SA_S5": ( 4.0, -1.5),
    # Charging station approach points
    "N_CS1": (-8.5,  0.0),
    "N_CS2": ( 8.5,  0.0),
    # ── Dock nodes (NEW): one per task LOCATION, sits exactly at
    # the robot stop point. Adding these gives A* a graph node to
    # terminate at, instead of appending an off-graph last hop.
    # WS docks are at y=±3.5 (between top corridor y=3 and table edge y=4.25).
    "DOCK_WS1": (-6.0,  3.5),
    "DOCK_WS2": ( 0.0,  3.5),
    "DOCK_WS3": ( 6.0,  3.5),
    "DOCK_WS4": (-6.0, -3.5),
    "DOCK_WS5": ( 0.0, -3.5),
    "DOCK_WS6": ( 6.0, -3.5),
    # S docks are at y=±1.5 next to the storage-aisle nodes; same y, ±1m east.
    "DOCK_S1": (-3.0,  1.5),
    "DOCK_S2": (-1.0,  1.5),
    "DOCK_S3": ( 1.0,  1.5),
    "DOCK_S4": ( 3.0,  1.5),
    "DOCK_S5": (-3.0, -1.5),
    "DOCK_S6": (-1.0, -1.5),
    "DOCK_S7": ( 1.0, -1.5),
    "DOCK_S8": ( 3.0, -1.5),
}

# Graph edges (bidirectional). All segments have been verified
# collision-free against physical shelves and workstation tables
# (see scripts/verify_layout.py).
GRAPH_EDGES = [
    # ---------- Top horizontal corridor ----------
    ("N1", "N2"), ("N2", "N3"), ("N3", "N4"), ("N4", "N5"),
    # ---------- Bottom horizontal corridor ----------
    ("N11", "N12"), ("N12", "N13"), ("N13", "N14"), ("N14", "N15"),
    # ---------- Far-left & far-right vertical corridors (outside the shelf zone) ----------
    ("N1", "N6"),   ("N6", "N11"),
    ("N5", "N10"),  ("N10", "N15"),
    # ---------- Vertical lanes through storage aisles (between shelves) ----------
    # x = -4 lane: top → SA_N1 → SA_S1 → bottom
    ("N2", "SA_N1"), ("SA_N1", "SA_S1"), ("SA_S1", "N12"),
    # x =  0 lane: top → SA_N3 → SA_S3 → bottom
    ("N3", "SA_N3"), ("SA_N3", "SA_S3"), ("SA_S3", "N13"),
    # x = +4 lane: top → SA_N5 → SA_S5 → bottom
    ("N4", "SA_N5"), ("SA_N5", "SA_S5"), ("SA_S5", "N14"),
    # ---------- Storage aisle horizontal connectivity ----------
    # North aisle (y=+1.5) — let robots traverse east-west between shelves
    ("SA_N1", "SA_N2"), ("SA_N2", "SA_N3"),
    ("SA_N3", "SA_N4"), ("SA_N4", "SA_N5"),
    # South aisle (y=-1.5)
    ("SA_S1", "SA_S2"), ("SA_S2", "SA_S3"),
    ("SA_S3", "SA_S4"), ("SA_S4", "SA_S5"),
    # ---------- Charging station connections ----------
    ("N_CS1", "N6"),
    ("N_CS2", "N10"),
    # ---------- Dock connections (NEW) ----------
    # Each WS dock connects to its nearest top/bottom corridor node.
    # Edge length is 0.5-1.12 m, all collision-free (verified by
    # verify_layout.py's Liang-Barsky test against tables).
    ("N1",  "DOCK_WS1"),  # (-7,3) → (-6,3.5)
    ("N3",  "DOCK_WS2"),  # ( 0,3) → ( 0,3.5)
    ("N5",  "DOCK_WS3"),  # ( 7,3) → ( 6,3.5)
    ("N11", "DOCK_WS4"),  # (-7,-3) → (-6,-3.5)
    ("N13", "DOCK_WS5"),  # ( 0,-3) → ( 0,-3.5)
    ("N15", "DOCK_WS6"),  # ( 7,-3) → ( 6,-3.5)
    # Each S dock connects to its nearest storage-aisle node (1m hop).
    # The hop is pure horizontal at y=±1.5 (same y as the dock and aisle),
    # so it cannot intersect any shelf (shelves are at y∈[-1, 1]).
    ("SA_N1", "DOCK_S1"),  # (-4,1.5) → (-3,1.5)
    ("SA_N2", "DOCK_S2"),  # (-2,1.5) → (-1,1.5)
    ("SA_N3", "DOCK_S3"),  # ( 0,1.5) → ( 1,1.5)
    ("SA_N4", "DOCK_S4"),  # ( 2,1.5) → ( 3,1.5)
    ("SA_S1", "DOCK_S5"),  # (-4,-1.5) → (-3,-1.5)
    ("SA_S2", "DOCK_S6"),  # (-2,-1.5) → (-1,-1.5)
    ("SA_S3", "DOCK_S7"),  # ( 0,-1.5) → ( 1,-1.5)
    ("SA_S4", "DOCK_S8"),  # ( 2,-1.5) → ( 3,-1.5)
]

# ----------------------------------------------------------------
# Physical obstacles for collision-aware path planning.
# Used by motion_coordinator.get_nearest_node() to skip nodes whose
# straight-line approach from the robot would clip a shelf/table.
# Each box is { center, half-extents }, ALREADY INFLATED by
# ROBOT_RADIUS. So a line through the inflated box = guaranteed collision.
# ----------------------------------------------------------------
# Forward-declare ROBOT_RADIUS here so OBSTACLE_BOXES can reference it.
# The authoritative definition is in the ROBOT PARAMETERS section below;
# this just makes Python happy with the module-level dict literal.
ROBOT_RADIUS = 0.18  # metres (bounding radius for collision)
OBSTACLE_BOXES = [
    # 4 shelves at y=0, x ∈ {-3, -1, +1, +3} — size 1.0 × 2.0 m
    {"name": "shelf_1", "cx": -3.0, "cy": 0.0,
     "hx": 0.5 + ROBOT_RADIUS, "hy": 1.0 + ROBOT_RADIUS},
    {"name": "shelf_2", "cx": -1.0, "cy": 0.0,
     "hx": 0.5 + ROBOT_RADIUS, "hy": 1.0 + ROBOT_RADIUS},
    {"name": "shelf_3", "cx":  1.0, "cy": 0.0,
     "hx": 0.5 + ROBOT_RADIUS, "hy": 1.0 + ROBOT_RADIUS},
    {"name": "shelf_4", "cx":  3.0, "cy": 0.0,
     "hx": 0.5 + ROBOT_RADIUS, "hy": 1.0 + ROBOT_RADIUS},
    # Workstation tables (Line A at y=+5, Line B at y=-5)
    # Each table: 2.0 × 1.5 m centered at (x, ±5) for x ∈ {-6, 0, +6}
    {"name": "WS1_table", "cx": -6.0, "cy":  5.0,
     "hx": 1.0 + ROBOT_RADIUS, "hy": 0.75 + ROBOT_RADIUS},
    {"name": "WS2_table", "cx":  0.0, "cy":  5.0,
     "hx": 1.0 + ROBOT_RADIUS, "hy": 0.75 + ROBOT_RADIUS},
    {"name": "WS3_table", "cx":  6.0, "cy":  5.0,
     "hx": 1.0 + ROBOT_RADIUS, "hy": 0.75 + ROBOT_RADIUS},
    {"name": "WS4_table", "cx": -6.0, "cy": -5.0,
     "hx": 1.0 + ROBOT_RADIUS, "hy": 0.75 + ROBOT_RADIUS},
    {"name": "WS5_table", "cx":  0.0, "cy": -5.0,
     "hx": 1.0 + ROBOT_RADIUS, "hy": 0.75 + ROBOT_RADIUS},
    {"name": "WS6_table", "cx":  6.0, "cy": -5.0,
     "hx": 1.0 + ROBOT_RADIUS, "hy": 0.75 + ROBOT_RADIUS},
]


# ----------------------------------------------------------------
# SHELF_GAP_BOXES — keep-out zones between adjacent shelves.
#
# Design rule: robots MAY NOT plan paths through the narrow aisles
# between shelves (which are storage-side spaces, not transit lanes).
# Robots should instead route around the shelves via the outer
# corridors (y > +1 or y < -1).
#
# Geometry:
#   • Shelves are at x∈{-3,-1,+1,+3}, each 1m wide, y∈[-1, +1]
#   • Gaps between adjacent shelves: x∈[-2.5,-1.5], [-0.5,+0.5], [+1.5,+2.5]
#   • We define each gap's y-extent SLIGHTLY SHORTER than the shelf
#     (y∈[-0.9, +0.9] instead of [-1,+1]) so that corner-escape waypoints
#     just outside the shelf (e.g. (-2.25, -1.75) or (-2, -1.5)) remain
#     in free space and the planner can still curve around shelf corners.
#   • No ROBOT_RADIUS inflation (unlike physical obstacles) because the
#     gap is a "policy" rule rather than a physical barrier; the
#     OccupancyGrid will still inflate it by ROBOT_RADIUS.
# ----------------------------------------------------------------
SHELF_GAP_BOXES = [
    # REMOVED: Shelf inflation (SAFETY_MARGIN=0.55) already closes all
    # inter-shelf gaps. Explicit gap boxes made cross-shelf routing
    # impossible (11m detours for 5m paths). Gap safety is now handled
    # by the OccupancyGrid's natural inflation from the 4 shelf boxes.
    # The 3 gap CROSSING WAYPOINTS (defined below) let robots transit
    # N↔S through the gap centers when needed.
]

# ----------------------------------------------------------------
# Combined list used by the planner. line_intersects_obstacles() and
# OccupancyGrid build their obstacle maps from this union, so paths
# avoid both physical objects AND policy keep-outs.
# ----------------------------------------------------------------
PLANNER_KEEP_OUT_BOXES = OBSTACLE_BOXES + SHELF_GAP_BOXES


def get_inactive_robot_obstacles(num_active_robots: int):
    """Generate obstacle boxes for INACTIVE robots (physically present in
    Webots but not participating in the current scenario).
    
    In Scenario A (3 robots), robots 4-8 sit at their parking spots as
    physical obstacles. The planner must route around them.
    
    Returns list of (cx, cy, half_w, half_h) boxes for inactive robots.
    """
    inactive_obstacles = []
    robot_obstacle_radius = 0.25  # slightly larger than physical robot (0.18m)
    for rid, pos in PARKING_SPOTS.items():
        if rid > num_active_robots:
            # This robot is inactive — add as obstacle
            inactive_obstacles.append(
                (pos[0], pos[1], robot_obstacle_radius, robot_obstacle_radius)
            )
    return inactive_obstacles


def get_full_keepout_boxes(num_active_robots: int):
    """Get complete list of keep-out boxes including inactive robots.
    Use this instead of PLANNER_KEEP_OUT_BOXES when scenario is known."""
    return PLANNER_KEEP_OUT_BOXES + get_inactive_robot_obstacles(num_active_robots)


def line_intersects_obstacles(p1, p2, boxes=None):
    """Liang-Barsky line-AABB intersection: True if segment p1→p2
    intersects ANY box in `boxes`. Defaults to PLANNER_KEEP_OUT_BOXES
    (i.e. physical obstacles + shelf-gap keep-outs) so callers get
    "is this segment OK to plan?" semantics.

    Pass `OBSTACLE_BOXES` explicitly if you only want physical-
    collision semantics (e.g. for verify_layout).
    """
    if boxes is None:
        boxes = PLANNER_KEEP_OUT_BOXES
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    for box in boxes:
        cx, cy = box['cx'], box['cy']
        hx, hy = box['hx'], box['hy']
        t_min = 0.0
        t_max = 1.0
        clipped_out = False
        for p, q in [(-dx, x1 - (cx - hx)),
                     ( dx, (cx + hx) - x1),
                     (-dy, y1 - (cy - hy)),
                     ( dy, (cy + hy) - y1)]:
            if abs(p) < 1e-9:
                if q < 0:
                    clipped_out = True
                    break
            else:
                t = q / p
                if p < 0:
                    if t > t_max:
                        clipped_out = True
                        break
                    if t > t_min:
                        t_min = t
                else:
                    if t < t_min:
                        clipped_out = True
                        break
                    if t < t_max:
                        t_max = t
        if not clipped_out:
            return True   # this box is intersected
    return False  # no box hit


# Map from task locations to nearest collision-free graph node.
# IMPORTANT: This must point to a node that has a *collision-free*
# straight line of sight to the actual location.
#
# Storage approach points S1-S8 map to dedicated storage-aisle nodes
# at the SAME y (y=±1.5) as the approach points themselves, so the
# final leg is a short pure-horizontal hop that cannot intersect a
# shelf. Workstations map to the nearest top/bottom corridor node.
LOCATION_TO_NODE = {
    # Each task LOCATION points at its DEDICATED dock node — these are
    # nodes positioned EXACTLY at the robot stop point, so when A*
    # plans a path, the final waypoint coincides with the dock pos
    # and there's no off-graph trailing leg.
    #
    # Each DOCK_* has a single edge connecting it to a corridor/aisle
    # node, so no other peer routes through it (it's a graph leaf).
    "WS1": "DOCK_WS1",  "WS2": "DOCK_WS2",  "WS3": "DOCK_WS3",
    "WS4": "DOCK_WS4",  "WS5": "DOCK_WS5",  "WS6": "DOCK_WS6",
    "S1": "DOCK_S1",    "S2": "DOCK_S2",    "S3": "DOCK_S3",
    "S4": "DOCK_S4",
    "S5": "DOCK_S5",    "S6": "DOCK_S6",    "S7": "DOCK_S7",
    "S8": "DOCK_S8",
    # Charging stations
    "CS1": "N_CS1",
    "CS2": "N_CS2",
}

# ================================================================
# ROBOT PARAMETERS
# ================================================================
MAX_ROBOTS = 8
# (ROBOT_RADIUS is defined above near OBSTACLE_BOXES; same value 0.18m)
WHEEL_RADIUS = 0.033         # metres
WHEEL_BASE = 0.287           # metres (distance between wheels)
MAX_LINEAR_SPEED = 0.26      # m/s (TurtleBot3 Waffle max)
MAX_ANGULAR_SPEED = 1.82     # rad/s
GOAL_TOLERANCE = 0.3         # metres - distance to consider goal reached
HEADING_TOLERANCE = 0.15     # radians - heading tolerance for rotation

# Battery parameters
BATTERY_CAPACITY = 100.0     # percentage
BATTERY_DEPLETION_TIME_SECONDS = 1800.0
BATTERY_DRAIN_RATE = BATTERY_CAPACITY / BATTERY_DEPLETION_TIME_SECONDS
BATTERY_CHARGE_RATE = BATTERY_DRAIN_RATE * 15.0
INITIAL_BATTERY_MIN = 25.0
INITIAL_BATTERY_MAX = BATTERY_CAPACITY
LOW_BATTERY_THRESHOLD = 25.0   # % - trigger RETURN to charging station
TASK_ABORT_BATTERY_THRESHOLD = 15.0  # % - abort an active task immediately
MIN_TASK_BATTERY     = LOW_BATTERY_THRESHOLD  # % - minimum charge for dispatch
                              # Must be lower than LOW_BATTERY_THRESHOLD so a
                              # robot returning to charge isn't immediately
                              # ineligible — it's allowed to finish its trip.
FULL_BATTERY_THRESHOLD = 95.0 # % - leave charging station once we hit this
ASSIGNMENT_FAILURE_TTL = 5.0  # seconds before retrying a failed robot/task pair

# Deadlock recovery.  An active robot that moves less than the progress
# distance for this long is considered physically stuck.  Nearby stuck
# robots are relocated as one group so their new positions are checked
# against each other before any Webots node is moved.
STALL_RELOCATION_TIMEOUT = 6.0  # reserve physical escape for sustained stalls
JOINT_STALL_RELOCATION_TIMEOUT = 8.0  # joint-mode route-proximate recovery deadline
JOINT_STALL_RECOVERY_COOLDOWN = 2.0   # retry interval when recovery routing fails
STALL_PROGRESS_DISTANCE = 0.10   # normal turning/slowdown still counts as progress
MIN_NAV_DISPLACEMENT = 0.15  # metres - reject routes that do not move the robot
STALL_GROUP_DISTANCE = 1.8       # metres joining stuck robots into one group
RELOCATION_PEER_CLEARANCE = 0.85  # minimum landing-point centre separation

# Planning runs inside Webots' synchronous controller step. These limits
# prevent a dense search from freezing simulation time.
SPACE_TIME_DETOUR_BUDGET_SECONDS = 0.05
SPACE_TIME_DETOUR_MAX_EXPANSIONS = 5000
PROACTIVE_SCAN_BUDGET_SECONDS = 0.10
PROACTIVE_SCAN_MAX_REPLANS = 1

# Robot states
class RobotState:
    """
    Robot lifecycle states.

    State transitions:
        IDLE → EN_ROUTE_PICKUP → CARRYING → EN_ROUTE_DELIVERY
              ↓                                      ↓
        RETURNING_TO_CHARGE                  RETURNING_HOME
              ↓                                      ↓
        CHARGING ────────────────────────────────→ IDLE

    All states are exposed as one-hot bits to the PPO scheduler
    (see RLScheduler.encode_state, 8-state encoding at indices 3..10).
    """
    IDLE = "idle"
    EN_ROUTE_PICKUP = "en_route_pickup"
    CARRYING = "carrying"
    EN_ROUTE_DELIVERY = "en_route_delivery"
    RETURNING_HOME = "returning_home"           # delivery done → home spot
    RETURNING_TO_CHARGE = "returning_to_charge" # low battery → CS, not charging yet
    CHARGING = "charging"                       # AT the CS, battery refilling

    # ──────────────────────────────────────────────────────────────
    # WAITING — reserved / placeholder state
    # ──────────────────────────────────────────────────────────────
    # Designed for "robot pauses while a path conflict is being
    # resolved by an external arbiter". NOT currently entered by any
    # active code path — Lifelong-CBS + DWA + static reservations
    # together prevent most conflicts at planning time, and
    # resolve_deadlock() simply clears the robot's path rather than
    # transitioning it to a distinct state.
    #
    # Kept here intentionally as a forward-compatible placeholder so
    # that the PPO state encoding (one-hot dim at index 10) and any
    # future explicit "yield-and-wait" coordinator logic can be
    # added without bumping the network's input dimension.
    #
    # Expected usage if/when implemented:
    #   robot.state = WAITING               # pause this robot
    #   robot.wait_until = sim_time + 5.0   # back-off timer
    #   ... 5 sec later ...
    #   robot.state = robot.previous_state  # resume
    WAITING = "waiting"

# ================================================================
# TASK PARAMETERS
# ================================================================
class TaskStatus:
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# ================================================================
# EXPERIMENTAL SCENARIOS
# ================================================================
SCENARIOS = {
    "A": {
        "name": "Small Fleet",
        "num_robots": 3,
        "task_interval": 30.0,  # seconds between task arrivals (Poisson mean)
        "initial_task_immediately": True,
        "description": "3 robots, low task arrival rate (1 task per 30 seconds)"
    },
    "B": {
        "name": "Medium Fleet",
        "num_robots": 5,
        "task_interval": 15.0,
        "initial_task_immediately": True,
        "description": "5 robots, moderate task arrival rate (1 task per 15 seconds)"
    },
    "C": {
        "name": "Large Fleet",
        "num_robots": 8,
        "task_interval": 8.0,
        "initial_task_immediately": True,
        "description": "8 robots, high task arrival rate (1 task per 8 seconds)"
    },
}

# Startup is event-driven: the timeout is only a failure guard, never a
# substitute for the per-robot READY handshake.
STARTUP_CONFIG = {
    "wait_for_all_robots": True,
    "robot_ready_timeout_seconds": 30.0,
    "minimum_stabilization_steps": 1,
    "dispatch_immediately_when_ready": True,
    "dispatch_when_no_tasks": False,
    "allow_partial_robot_startup": False,
}

# ================================================================
# RL TRAINING PARAMETERS (PPO)
# ================================================================
RL_ENVIRONMENT_VERSION = "rl-scheduling-v6-noop-bootstrap-ppo-validation"

RL_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,            # discount factor
    "gae_lambda": 0.95,       # GAE lambda
    "clip_epsilon": 0.2,      # PPO clipping
    "entropy_coeff": 0.01,    # entropy bonus
    "value_coeff": 0.5,       # value loss coefficient
    "max_grad_norm": 0.5,     # gradient clipping
    "num_epochs": 4,          # PPO epochs per update
    "batch_size": 64,
    "buffer_size": 2048,
    "hidden_size": 256,
    "num_layers": 2,
    "training_episodes": 5000,
    "eval_interval": 100,
    "save_interval": 500,
}

# Versioned DQN/SARSA scheduler defaults.  These models are opt-in and are
# never used without a compatible checkpoint.
RL_SCHEDULING_CONFIG = {
    "environment": {
        "version": RL_ENVIRONMENT_VERSION,
        "max_robots": MAX_ROBOTS,
        "max_tasks": 20,
        "max_steps_per_episode": 64,
        "no_op_enabled": True,
    },
    "reward": {
        "task_completion": 10.0,
        "valid_assignment": 0.5,
        "priority": 0.5,
        "invalid_action": -5.0,
        "no_op": -2.0,
        "collision": -100.0,
        # A deadlock is handled by runtime recovery and is not itself a
        # safety violation. Penalise only an actual collision.
        "deadlock": 0.0,
        "age_bonus": 0.5,
        "waiting_weight": -0.03,
        "distance_weight": -0.08,
    },
    "dqn": {
        "backend": "numpy-cpu",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_capacity": 100000,
        "warmup_steps": 5000,
        "target_update_interval": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.02,
        "epsilon_decay_steps": 50000,
        "double_dqn": True,
    },
    "sarsa": {
        "learning_rate": 0.05,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.02,
        "epsilon_decay": 0.999,
    },
}

# State space dimensions for RL
# Per robot: x, y, heading, state(one-hot 8), battery, has_task = 13
# + pending tasks: num_pending, avg_distance_to_tasks
# + congestion map: grid_cells
GRID_RESOLUTION = 1.0  # metres per grid cell for congestion map
GRID_WIDTH = 20        # cells
GRID_HEIGHT = 16       # cells

# Reward function weights
REWARD_TASK_COMPLETE = 10.0
REWARD_IDLE_PENALTY = -0.01   # per timestep per idle robot
REWARD_CONGESTION_PENALTY = -0.5
REWARD_DISTANCE_PENALTY = -0.1  # per metre of empty travel
REWARD_BALANCE_BONUS = 1.0    # for balanced workload distribution
REWARD_TASK_FAILURE = -10.0   # assigned task aborted after policy commitment

# ================================================================
# LOGGING & METRICS
# ================================================================
LOG_INTERVAL = 250  # timesteps between log outputs
METRICS_FILE_PREFIX = "experiment_results"


# ═══════════════════════════════════════════════════════════════════════
# GAP CROSSING WAYPOINTS — safe transit points between shelf rows
# ═══════════════════════════════════════════════════════════════════════
# These points are at the center of each inter-shelf gap (y=0).
# Robots traveling N↔S pass through these instead of detouring
# around the entire shelf block.
#
# Physical clearance at gap center:
#   Gap width (raw): 2m between shelf edges (e.g., shelf_2 edge at -0.3, shelf_3 at +0.3)
#   With robot radius 0.18m: edge clearance = 0.32m × 2 sides
#   Verdict: PASSABLE (same margin as dock approaches)
CROSSING_WAYPOINTS = {}  # Disabled: shelf gaps too narrow (0.32m clearance) for reliable DWA
