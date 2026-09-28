"""
Individual Robot Controller for Webots TurtleBot3 Waffle.

Each robot runs this controller independently. It handles:
1. Differential-drive motor control
2. LiDAR-based obstacle detection
3. GPS/Compass-based localisation
4. Navigation to waypoints received from the supervisor
5. Reactive collision avoidance (Dynamic Window Approach)
6. Communication with supervisor via Emitter/Receiver
"""

import sys
import json
import math
import random
import os
from typing import List, Tuple, Optional

try:
    from controller import Robot
except ImportError:
    print("[WARNING] Webots Robot module not found.")
    class Robot:
        def __init__(self):
            pass
        def getBasicTimeStep(self):
            return 16
        def step(self, ts):
            return -1


# ================================================================
# CONSTANTS
# ================================================================
MAX_SPEED = 7.27          # rad/s (max wheel angular velocity)
MAX_LINEAR_SPEED = 0.24   # m/s
MAX_ANGULAR_SPEED = 2.84  # rad/s
WHEEL_RADIUS = 0.033      # m
WHEEL_BASE = 0.287        # m (distance between wheels)
BATTERY_CAPACITY = 100.0
BATTERY_DEPLETION_TIME_SECONDS = 1800.0
BATTERY_DRAIN_RATE = BATTERY_CAPACITY / BATTERY_DEPLETION_TIME_SECONDS
INITIAL_BATTERY_MIN = 25.0

GOAL_THRESHOLD = 0.35     # m - distance to consider waypoint reached
HEADING_THRESHOLD = 0.15  # rad - heading alignment threshold

OBSTACLE_THRESHOLD = 0.8  # m - increased for earlier steering clearance
CRITICAL_DISTANCE = 0.40  # m - increased from 0.25: avoid bumping shelves

# DWA parameters
DWA_V_RESOLUTION = 0.01   # m/s
DWA_W_RESOLUTION = 0.05   # rad/s
DWA_PREDICTION_TIME = 1.5 # seconds
DWA_DT = 0.1              # seconds per prediction step

# Weights for DWA objective function
W_HEADING = 1.5    # Weight for heading toward goal
W_DISTANCE = 0.8   # Weight for distance to obstacles
W_VELOCITY = 0.5   # Weight for forward velocity
W_LATERAL = 2.0    # Weight for lateral bias when in sidestep mode

# Sidestep / lateral avoidance parameters
SIDESTEP_TRIGGER_DIST = 1.2   # m  �?head-on detected when LiDAR < this
SIDESTEP_RECOVER_DIST = 2.0   # m  �?exit sidestep once front clear beyond this
SIDESTEP_LATERAL_OFFSET = 0.85  # m �?wider lateral deviation for easier turns
SIDESTEP_TIMEOUT = 4.0        # seconds �?max time to spend in sidestep
SIDESTEP_FRONT_CONE_DEG = 30  # ±15° front cone for head-on detection
SIDESTEP_SIDE_CONE_DEG = 80   # ±40° each side for "where's the room?"

# ── Dock Zone (Shelf-Adjacent Band) ───────────────────────────────────
# Storage docks at y=±1.5 are only 0.32m from shelf faces (shelf top at y=±1.0).
# Anywhere in this band, LiDAR detects shelves at close range �?DWA panics.
# Solution: define "dock zones" as horizontal bands adjacent to shelves.
# ANY robot in a dock zone uses pure pursuit (no obstacle avoidance).
# Safe because: paths are pre-verified by the grid planner, and the first
# waypoint always escapes to y=±2.0 (open space above/below shelves).
#
# Shelf layout:  shelves at y∈[-1.0, 1.0], docks at y=±1.5
# Dock zone:     y∈[1.0, 2.0] (north) and y∈[-2.0, -1.0] (south)
#                AND x∈[-3.7, 3.7] (shelf x-extent)
DOCK_ZONE_Y_BANDS = [(1.0, 2.0), (-2.0, -1.0)]  # (y_min, y_max) bands
DOCK_ZONE_X_RANGE = (-3.7, 3.7)  # shelves span x∈[-3.5, 3.5] + margin

# ══════════════════════════════════════════════════════════════════════
# 物理参数 �?所有避障距离的计算基础
# ══════════════════════════════════════════════════════════════════════
ROBOT_RADIUS = 0.18             # m �?机器人碰撞半�?
ROBOT_DIAMETER = 2 * ROBOT_RADIUS  # 0.36m �?两机器人碰撞时的中心�?
MAX_LINEAR_SPEED = 0.22         # m/s �?最大线速度
BRAKING_DECEL = 0.5             # m/s² �?制动减速度(保守估计)
BRAKING_DISTANCE = MAX_LINEAR_SPEED**2 / (2 * BRAKING_DECEL)  # �?.048m
BRAKING_TIME = MAX_LINEAR_SPEED / BRAKING_DECEL                # �?.44s
SAFETY_MARGIN = 0.12            # m �?额外安全裕度

# ══════════════════════════════════════════════════════════════════════
# LAYER 0: 前瞻性路径冲突检�?+ 紧急制动（保险�?
# ══════════════════════════════════════════════════════════════════════
# 设计理念：提前预判路径冲�?�?请求重规划绕开 �?机器人始终在运动
# 紧急制动只作为最后保险（正常情况下不应触发）
#
# 前瞻检测距�?
PATH_CONFLICT_DIST = 1.4         # m �?前方路径上peer距离小于此值则"冲突"
PATH_CONFLICT_CHECK_INTERVAL = 5 # �?帧检测一�?160ms)
# 紧急制�?最后保�?:
EMERGENCY_STOP_DIST = 0.55       # m �?中心距小于此值才紧急制�?仅保�?

# ══════════════════════════════════════════════════════════════════════
# LAYER 1: 平滑减�?�?接近peer时减速（配合前瞻重规划）
# ══════════════════════════════════════════════════════════════════════
# 正常情况下，前瞻检测会提前重规划，机器人不会靠近peer�?
# 此层仅在重规划延迟期间提供平滑减速保护�?
PEER_SLOW_DIST = 1.2              # m �?开始减�?
PEER_STOP_DIST = 0.70             # m �?停止(保留0.50m硬冲突边界外的余�?
PEER_RADIAL_STOP_DIST = 0.80      # m �?独立欧氏距离保护，覆盖侧向接近和制动惯�?
# Legacy aliases for compatibility:
PEER_LOOKAHEAD_DIST = 2.0
PEER_REPLAN_DIST = 1.5
ROBOT_CONFLICT_DISTANCE = PEER_STOP_DIST
ROBOT_CONFLICT_SLOW_DIST = PEER_SLOW_DIST



class WaypointNavigator:
    """
    Waypoint-following navigation controller.
    Uses a combination of:
    - Proportional heading control for waypoint tracking
    - Dynamic Window Approach (DWA) for reactive collision avoidance
    """
    
    def __init__(self):
        self.quiet = os.environ.get('SMART_FACTORY_ROBOT_QUIET', '0') == '1'
        self.waypoints: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.navigation_active = False
        self.goal_reached = False
        
        # ── LAYER 0: 前瞻路径冲突检�?+ 紧急制�?保险) ──
        self._replan_requested = False
        self._conflict_check_counter = 0  # 帧计数器
        self._nav_step_count = 0          # 总帧�?用于冷却计时)
        self._replan_cooldown_until = 0.0 # 冷却结束时间
        self._emergency_stopped = False   # 紧急制动状�?
        self.robot_id = 0  # Set by RobotController
        
        # ── Sidestep state (lateral avoidance for head-on robots) ──
        # Used by Hybrid Topological + DWA navigation: when LiDAR
        # detects a robot directly ahead, we temporarily steer toward
        # the side that has more LiDAR clearance. Once the obstacle
        # has passed, we resume the planned path.
        self.sidestep_active = False
        self.sidestep_direction = 0    # +1 = left bias, -1 = right bias
        self.sidestep_start_t = 0.0    # seconds since simulation start
        self.sidestep_lateral_x = 0.0  # offset waypoint x
        self.sidestep_lateral_y = 0.0  # offset waypoint y
        
        # ── Peer robot positions (updated by supervisor broadcasts) ──
        self.peer_positions = {}  # {robot_id: (x, y)} �?all other robots
        self.peer_samples = {}
        self.paused_until = 0.0
        self.path_version = 0
        self.controller_time = 0.0
        self.speed_scale = 1.0
        self.waypoint_not_before: List[float] = []
        self.plan_is_partial = False
        self.joint_release_index = 10 ** 9
        self.waypoint_threshold = GOAL_THRESHOLD
        self.planned_wait_until = 0.0
        self.planned_wait_reason = None
        self.joint_epoch_wait_deadline = 0.0
        self.joint_coordinated = False
        self.direct_navigation = False
        # Preserve the legacy sample order so DWA tie-breaking is unchanged.
        self._dwa_v_samples = tuple(self._frange(
            0.0, MAX_LINEAR_SPEED, DWA_V_RESOLUTION * 5))
        self._dwa_w_samples = tuple(self._frange(
            -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED, DWA_W_RESOLUTION * 5))
        # Precompute one DWA integration delta per angular sample. The
        # controller uses the same algebra as the previous sin/cos calls,
        # but avoids two trig calls for every prediction substep.
        self._dwa_prediction_steps = int(DWA_PREDICTION_TIME / DWA_DT)
        self._dwa_w_deltas = tuple(
            (math.cos(w * DWA_DT), math.sin(w * DWA_DT))
            for w in self._dwa_w_samples)
    
    def set_waypoints(self, waypoints: List[Tuple[float, float]],
                      waypoint_not_before=None, partial=False):
        """Set a new list of waypoints to follow."""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.navigation_active = True
        self.goal_reached = False
        self.waypoint_not_before = list(waypoint_not_before or ())
        self.plan_is_partial = bool(partial)
        self.joint_coordinated = False
        self.direct_navigation = False
        self._reset_reactive_state()
        if not partial:
            self.joint_release_index = 10 ** 9
            self.waypoint_threshold = GOAL_THRESHOLD
            self.joint_epoch_wait_deadline = 0.0

    def _reset_reactive_state(self):
        """Clear all local obstacle-avoidance state after a route replacement."""
        self.sidestep_active = False
        self.sidestep_direction = 0
        self.sidestep_start_t = 0.0
        self.sidestep_lateral_x = 0.0
        self.sidestep_lateral_y = 0.0
        self._emergency_stopped = False
        self._replan_requested = False
    
    def get_current_target(self) -> Optional[Tuple[float, float]]:
        """Get the current target waypoint."""
        if not self.navigation_active or self.current_waypoint_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.current_waypoint_idx]
    
    def advance_waypoint(self) -> bool:
        """Move to next waypoint. Returns True if final goal reached."""
        self.current_waypoint_idx += 1
        if self.current_waypoint_idx >= len(self.waypoints):
            self.navigation_active = False
            self.goal_reached = True
            if not self.quiet:
                print(f"[Nav {self.robot_id}] \u2605 Final goal reached ({len(self.waypoints)} waypoints)")
            return True
        wp = self.waypoints[self.current_waypoint_idx]
        if not self.quiet:
            print(f"[Nav {self.robot_id}] \u2192 Waypoint {self.current_waypoint_idx}/{len(self.waypoints)-1} "
                  f"({wp[0]:.2f},{wp[1]:.2f})")
        return False
    
    def compute_control(self, robot_x: float, robot_y: float,
                        robot_heading: float,
                        lidar_ranges: Optional[List[float]] = None
                        ) -> Tuple[float, float]:
        """
        Compute motor velocities to navigate toward the current waypoint.

        Args:
            robot_x, robot_y: Current robot position in ENU ground plane.
            robot_heading: Current robot heading in radians (math convention:
                           0 = facing +x/east, +π/2 = facing +y/north).
            lidar_ranges:  Optional LiDAR scan for reactive obstacle avoidance.

        Returns:
            (left_speed, right_speed) motor velocities in rad/s.
        """
        self._nav_step_count += 1
        self.planned_wait_until = 0.0
        self.planned_wait_reason = None
        target = self.get_current_target()
        if target is None:
            return (0.0, 0.0)
        if self.current_waypoint_idx > self.joint_release_index:
            bounded_wait_until = min(
                self.controller_time + 0.5,
                self.joint_epoch_wait_deadline)
            if bounded_wait_until > self.controller_time:
                self.planned_wait_until = bounded_wait_until
                self.planned_wait_reason = 'joint_epoch_barrier'
            return (0.0, 0.0)

        tx, ty = target
        
        # ══════════════════════════════════════════════════════════════
        # LAYER 0+1: 前瞻冲突检�?+ 紧急制�?+ 平滑减�?
        # ══════════════════════════════════════════════════════════════
        conflict_result = self._check_path_conflict_and_emergency(
            robot_x, robot_y, robot_heading, lidar_ranges)
        
        peer_speed_factor = 1.0
        if conflict_result is not None:
            if len(conflict_result) == 2:
                # (0.0, 0.0) ? ????
                return conflict_result
            elif len(conflict_result) == 1:
                # (factor,) ? ????
                peer_speed_factor = conflict_result[0]
        if not getattr(self, 'joint_coordinated', False):
            peer_speed_factor = min(
                peer_speed_factor,
                self._peer_predictive_speed_factor(
                    robot_x, robot_y, robot_heading, self.controller_time))

        # Vector from robot to target
        dx = tx - robot_x
        dy = ty - robot_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if waypoint reached
        if distance < self.waypoint_threshold:
            deadline = (self.waypoint_not_before[self.current_waypoint_idx]
                        if self.current_waypoint_idx < len(
                            self.waypoint_not_before) else 0.0)
            if self.controller_time < deadline:
                self.planned_wait_until = deadline
                self.planned_wait_reason = 'joint_slot_deadline'
                return (0.0, 0.0)
            if (self.plan_is_partial and
                    self.current_waypoint_idx == len(self.waypoints) - 1):
                # A rolling-window endpoint is not a business-goal arrival.
                # Hold the reserved endpoint until the next joint epoch.
                bounded_wait_until = min(
                    self.controller_time + 0.5,
                    self.joint_epoch_wait_deadline)
                if bounded_wait_until > self.controller_time:
                    self.planned_wait_until = bounded_wait_until
                    self.planned_wait_reason = 'joint_window_endpoint'
                return (0.0, 0.0)
            final = self.advance_waypoint()
            if final:
                return (0.0, 0.0)
            if self.current_waypoint_idx > self.joint_release_index:
                bounded_wait_until = min(
                    self.controller_time + 0.5,
                    self.joint_epoch_wait_deadline)
                if bounded_wait_until > self.controller_time:
                    self.planned_wait_until = bounded_wait_until
                    self.planned_wait_reason = 'joint_epoch_barrier'
                return (0.0, 0.0)
            target = self.get_current_target()
            if target is None:
                return (0.0, 0.0)
            tx, ty = target
            dx = tx - robot_x
            dy = ty - robot_y
            distance = math.sqrt(dx*dx + dy*dy)

        # Desired heading in MATH convention: 0 = +x, π/2 = +y
        desired_heading = math.atan2(dy, dx)
        
        # Heading error (normalized to [-pi, pi])
        heading_error = desired_heading - robot_heading
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # ── Dock Zone Check ────────────────────────────────────────
        # When in the shelf-adjacent band (y∈[1.0,2.0] or y∈[-2.0,-1.0]
        # AND x∈[-3.7, 3.7]), bypass DWA entirely. The shelves are too
        # close for reactive avoidance. Use pure pursuit instead.
        if (getattr(self, 'joint_coordinated', False) or
                getattr(self, 'direct_navigation', False)):
            # A joint plan has already reserved conflict-free space-time
            # slots. Follow that centreline with pure pursuit instead of
            # running DWA/peer heuristics that deviate from the reservation
            # and create the exact dense clustering they are trying to fix.
            if abs(heading_error) > 0.30:
                angular_speed = max(
                    -MAX_ANGULAR_SPEED * 0.70,
                    min(MAX_ANGULAR_SPEED * 0.70, heading_error * 2.5))
                left_speed = -angular_speed * WHEEL_BASE / (2 * WHEEL_RADIUS)
                right_speed = angular_speed * WHEEL_BASE / (2 * WHEEL_RADIUS)
            else:
                forward_speed = min(
                    MAX_LINEAR_SPEED * 0.85, distance * 1.50)
                angular_speed = max(
                    -MAX_ANGULAR_SPEED * 0.60,
                    min(MAX_ANGULAR_SPEED * 0.60, heading_error * 2.0))
                left_speed = (
                    forward_speed - angular_speed * WHEEL_BASE / 2
                ) / WHEEL_RADIUS
                right_speed = (
                    forward_speed + angular_speed * WHEEL_BASE / 2
                ) / WHEEL_RADIUS
            # RobotController.run() applies speed_scale to every navigation
            # output. Keep this branch unscaled so coordinated motion is not
            # slowed by scale**2 while other navigation modes use scale once.
            return (left_speed, right_speed)

        near_dock = False
        if DOCK_ZONE_X_RANGE[0] <= robot_x <= DOCK_ZONE_X_RANGE[1]:
            for (y_min, y_max) in DOCK_ZONE_Y_BANDS:
                if y_min <= robot_y <= y_max:
                    near_dock = True
                    break
        
        if near_dock:
            # Dock zone: pure pursuit with MINIMAL safety checks.
            # DWA is bypassed (too sensitive for 0.32m shelf clearance),
            # but we still check LiDAR for IMMINENT collisions (< 0.25m)
            # which catches other robots or unexpected obstacles.
            
            # Safety check: is anything VERY close in front?
            dock_emergency = False
            if lidar_ranges:
                front_min_dock = self._get_front_min(lidar_ranges, robot_heading)
                if front_min_dock < 0.25:
                    dock_emergency = True
            
            if dock_emergency:
                # Something very close �?stop and rotate toward target
                heading_to_target = math.atan2(ty - robot_y, tx - robot_x)
                turn_err = heading_to_target - robot_heading
                while turn_err > math.pi: turn_err -= 2 * math.pi
                while turn_err < -math.pi: turn_err += 2 * math.pi
                rot_sign = 1.0 if turn_err > 0 else -1.0
                return (-MAX_SPEED * 0.2 * rot_sign, MAX_SPEED * 0.2 * rot_sign)
            
            # Pure pursuit (safe �?no imminent obstacle)
            if abs(heading_error) > 0.3:
                # Rotate toward waypoint
                angular_speed = max(min(heading_error * 2.5, MAX_ANGULAR_SPEED * 0.7),
                                    -MAX_ANGULAR_SPEED * 0.7)
                left_speed = -angular_speed * WHEEL_BASE / (2 * WHEEL_RADIUS)
                right_speed = angular_speed * WHEEL_BASE / (2 * WHEEL_RADIUS)
            else:
                # Drive forward at moderate speed
                forward_speed = min(MAX_LINEAR_SPEED * 0.5, distance * 0.8)
                angular_speed = heading_error * 2.0
                left_speed = (forward_speed - angular_speed * WHEEL_BASE / 2) / WHEEL_RADIUS
                right_speed = (forward_speed + angular_speed * WHEEL_BASE / 2) / WHEEL_RADIUS
            return (left_speed * peer_speed_factor, right_speed * peer_speed_factor)
        
        # ── Predictive Obstacle Avoidance (LiDAR-based) ─────────────
        # Strategy: detect obstacles EARLY via LiDAR and progressively
        # reduce speed BEFORE reaching the obstacle. The robot should
        # never actually hit anything �?it slows, steers, or replans.
        #
        # Priority layers:
        #   1. CRITICAL (< 0.25m): Emergency rotation (shouldn't happen)
        #   2. CLOSE (< 0.5m): Very slow + sidestep activation
        #   3. MODERATE (< 0.8m): DWA active, speed reduced
        #   4. EARLY (< 1.5m): Start reducing speed proportionally
        #   5. CLEAR (> 1.5m): Normal navigation
        obstacle_too_close = False
        min_front_dist = float('inf')
        # LiDAR-based speed reduction: gradual braking as obstacle gets closer
        lidar_speed_factor = 1.0

        if lidar_ranges:
            front_min, side_clear, front_in_narrow_cone = \
                self._scan_lidar_zones(lidar_ranges)
            min_front_dist = front_min
            
            # ── Predictive speed reduction based on LiDAR distance ──
            # Start slowing at 1.5m, proportional to distance
            LIDAR_LOOKAHEAD = 1.5
            LIDAR_DWA_DIST = 0.8  # DWA takes over below this
            if front_min < LIDAR_LOOKAHEAD and front_min >= LIDAR_DWA_DIST:
                # Linear: 100% at 1.5m �?60% at 0.8m
                t = (front_min - LIDAR_DWA_DIST) / (LIDAR_LOOKAHEAD - LIDAR_DWA_DIST)
                lidar_speed_factor = 0.6 + t * 0.4
            elif front_min < LIDAR_DWA_DIST and front_min >= CRITICAL_DISTANCE:
                # Further reduction: 60% at 0.8m �?30% at 0.25m
                t = (front_min - CRITICAL_DISTANCE) / (LIDAR_DWA_DIST - CRITICAL_DISTANCE)
                lidar_speed_factor = 0.3 + t * 0.3

            # ── 1. Emergency stop / rotate (LAST RESORT) ─────────
            if front_min < CRITICAL_DISTANCE:
                obstacle_too_close = True

            # ── 2. Head-on detection �?enter / continue sidestep ─
            elif (front_in_narrow_cone < SIDESTEP_TRIGGER_DIST
                  and not self.sidestep_active):
                self._replan_requested = True
                # Pick the side with more clearance
                left_clear, right_clear = side_clear
                if left_clear > right_clear and left_clear > 0.6:
                    self._enter_sidestep(direction=+1,
                                          robot_x=robot_x, robot_y=robot_y,
                                          robot_heading=robot_heading)
                elif right_clear > 0.6:
                    self._enter_sidestep(direction=-1,
                                          robot_x=robot_x, robot_y=robot_y,
                                          robot_heading=robot_heading)
                # If both sides too tight, fall through to DWA / emergency

            # ── 3. Already in sidestep �?check for exit ──────────
            if self.sidestep_active:
                # Auto-exit when front is clear OR timeout expires
                elapsed = self._sim_time() - self.sidestep_start_t
                if (front_in_narrow_cone > SIDESTEP_RECOVER_DIST
                        or elapsed > SIDESTEP_TIMEOUT):
                    self._exit_sidestep()
                    self._emergency_stopped = True
                    self._replan_requested = True
                    return (0.0, 0.0)
                else:
                    # Stay in sidestep �?let DWA with lateral bias steer
                    ss_l, ss_r = self._dwa_control(robot_x, robot_y, robot_heading,
                                                     tx, ty, lidar_ranges)
                    return (ss_l * peer_speed_factor, ss_r * peer_speed_factor)

            # ── 4. Standard DWA for moderate clearance ───────────
            if (not obstacle_too_close
                    and min_front_dist < OBSTACLE_THRESHOLD
                    and not self.sidestep_active):
                dwa_l, dwa_r = self._dwa_control(robot_x, robot_y, robot_heading,
                                                   tx, ty, lidar_ranges)
                return (dwa_l * peer_speed_factor * lidar_speed_factor, dwa_r * peer_speed_factor * lidar_speed_factor)
        
        if obstacle_too_close:
            # Emergency: rotate toward goal to clear the obstacle.
            heading_to_target = math.atan2(ty - robot_y, tx - robot_x)
            turn_err = heading_to_target - robot_heading
            while turn_err > math.pi: turn_err -= 2 * math.pi
            while turn_err < -math.pi: turn_err += 2 * math.pi
            rot_sign = 1.0 if turn_err > 0 else -1.0
            em_l = -MAX_SPEED * 0.3 * rot_sign
            em_r = MAX_SPEED * 0.3 * rot_sign
            return (em_l * peer_speed_factor, em_r * peer_speed_factor)
        
        # Normal navigation: proportional control
        if abs(heading_error) > HEADING_THRESHOLD:
            # Rotate toward target first
            angular_speed = max(min(heading_error * 3.0, MAX_ANGULAR_SPEED), 
                              -MAX_ANGULAR_SPEED)
            left_speed = -angular_speed * WHEEL_BASE / (2 * WHEEL_RADIUS)
            right_speed = angular_speed * WHEEL_BASE / (2 * WHEEL_RADIUS)
        else:
            # Drive forward with slight steering correction
            linear_speed = min(MAX_LINEAR_SPEED, distance * 0.5)
            angular_speed = heading_error * 2.0
            
            # Convert to differential drive
            left_speed = (linear_speed - angular_speed * WHEEL_BASE / 2) / WHEEL_RADIUS
            right_speed = (linear_speed + angular_speed * WHEEL_BASE / 2) / WHEEL_RADIUS
        
        # Clamp speeds
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
        
        # Apply peer conflict slowdown (works in all zones)
        # Apply combined speed reduction: peer proximity + LiDAR lookahead
        combined_factor = peer_speed_factor * lidar_speed_factor
        return (left_speed * combined_factor, right_speed * combined_factor)

    def _peer_predictive_speed_factor(self, robot_x, robot_y, robot_heading,
                                      now: float) -> float:
        """Apply stale-data fail-safe and continuous relative-motion TTC."""
        factor = 1.0
        estimated_speed = MAX_LINEAR_SPEED * self.speed_scale
        own_v = (estimated_speed * math.cos(robot_heading),
                 estimated_speed * math.sin(robot_heading))
        for peer_id, sample in self.peer_samples.items():
            position = sample.get('position')
            velocity = sample.get('velocity', (0.0, 0.0))
            if not position or len(position) < 2:
                continue
            distance = math.hypot(position[0] - robot_x, position[1] - robot_y)
            age = max(0.0, now - float(sample.get('sample_time', now)))
            if distance < PATH_CONFLICT_DIST:
                if age > 0.30:
                    # Stale peer data must reduce speed, not manufacture a
                    # stationary obstacle. The independent 0.65 m radial
                    # guard remains authoritative for a physical stop.
                    factor = min(factor, 0.2)
                if age > 0.10:
                    factor = min(factor, 0.5)
            rx, ry = position[0] - robot_x, position[1] - robot_y
            vx = float(velocity[0]) - own_v[0]
            vy = float(velocity[1]) - own_v[1]
            vv = vx * vx + vy * vy
            if vv < 1e-9:
                continue
            t_cpa = max(0.0, min(10.0, -(rx * vx + ry * vy) / vv))
            minimum = math.hypot(rx + vx * t_cpa, ry + vy * t_cpa)
            if minimum < 0.50:
                if t_cpa <= 0.5:
                    factor = min(factor, 0.2)
                if t_cpa <= 2.0:
                    factor = min(factor, 0.3)
                elif t_cpa <= 4.0:
                    factor = min(factor, 0.6)
        return factor
    
    # ══════════════════════════════════════════════════════════════
    #  LAYER 0: 前瞻路径冲突检�?+ 紧急制�?最后保�?
    # ══════════════════════════════════════════════════════════════
    
    def _check_path_conflict_and_emergency(self, robot_x, robot_y, 
                                             robot_heading, lidar_ranges):
        """
        基于"路径线段投影"的制动判�?+ 前瞻检测�?
        
        核心逻辑:
          仅当 peer �?我→当前waypoint"路径线段上才制动�?
          判定方法: peer到路径线段的垂直距离 < 机器人宽�?0.4m)
                   �?peer在线段正前方（投影距�?0�?
          侧面经过的peer �?完全忽略，不制动�?
        
        返回:
          None �?无危险，正常导航
          (0,0) �?紧急停�?
          (speed_factor,) �?减速因�?0~1)
        """
        # �? 不在这里重置 _replan_requested!
        # 该标志只在以下时机清�?
        #  - Supervisor读取并处理后(standalone模式)
        #  - 发送status消息�?Webots物理模式)
        #  - 收到新的navigate命令�?表示已经重规划完�?
        
        # ── 前瞻路径冲突检�? 已禁�?──
        # 原因: 该检测用peer当前位置判断未来路径段冲突，但不考虑peer也在移动�?
        # 导致: 远距离peer被误判为"在路径上" �?无效重规�?�?卡死循环�?
        # 替代: Supervisor层的轨迹预测(_proactive_path_conflict_scan)会预�?
        #       双方未来位置，正确判断是否有时空碰撞�?
        # 安全: 实时制动�?d_perp<0.30m)仍每帧运行，保障近距离安全�?
        # self._conflict_check_counter += 1
        # if self._conflict_check_counter >= PATH_CONFLICT_CHECK_INTERVAL:
        #     self._conflict_check_counter = 0
        #     self._check_path_for_conflicts(robot_x, robot_y)
        
        # ── LiDAR紧急制动（物理障碍物，非peer�?──
        if lidar_ranges:
            front_min = self._get_front_min(lidar_ranges, robot_heading)
            if front_min < 0.25:
                if not self._emergency_stopped:
                    if not self.quiet:
                        print(f"[Nav {self.robot_id}] ? LiDAR emergency braking! "
                              f"Obstacle ahead at {front_min:.2f} m")
                    self._replan_requested = True  # ?????????
                self._emergency_stopped = True
                return (0.0, 0.0)

        target = self.get_current_target()
        if target is None:
            if self._emergency_stopped:
                self._emergency_stopped = False
            return None

        # Hard radial safety is independent of path projection. The previous
        # projection-only rule ignored a peer once lateral offset exceeded
        # 0.40 m, even when centre distance was already below the 0.50 m hard
        # boundary. Stop early enough to cover controller/physics latency and
        # request a replacement route exactly once on entry.
        move_x, move_y = target[0] - robot_x, target[1] - robot_y
        coordinated = (getattr(self, 'joint_coordinated', False) or
                       getattr(self, 'direct_navigation', False))
        radial_stop = (EMERGENCY_STOP_DIST if coordinated else PEER_RADIAL_STOP_DIST)
        dangerous_radial = any(
            math.hypot(px - robot_x, py - robot_y) < radial_stop
            and move_x * (px - robot_x) + move_y * (py - robot_y) >= 0.0
            for px, py in self.peer_positions.values())
        if dangerous_radial:
            if not self._emergency_stopped:
                self._replan_requested = True
                self._emergency_stopped = True
            return (0.0, 0.0)

        if (getattr(self, 'joint_coordinated', False) or
                getattr(self, 'direct_navigation', False)):
            # A joint plan reserves conflict-free space-time slots. Keep the
            # hard radial peer stop above as a physical safety net, but do not
            # apply path-projection/predictive braking in coordinated mode.
            if self._emergency_stopped:
                self._emergency_stopped = False
            return None

        # ?? Peer????: ?"???????"?peer??? ??
        if not self.peer_positions:
            if self._emergency_stopped:
                self._emergency_stopped = False
            return None
        
        tx, ty = target
        # 路径向量: robot �?target
        path_dx = tx - robot_x
        path_dy = ty - robot_y
        path_len = math.sqrt(path_dx*path_dx + path_dy*path_dy)
        if path_len < 0.01:
            if self._emergency_stopped:
                self._emergency_stopped = False
            return None
        
        # 单位路径向量
        ux, uy = path_dx / path_len, path_dy / path_len
        
        min_along_dist = float('inf')
        blocking_peer = False
        
        for peer_id, (px, py) in self.peer_positions.items():
            # peer相对于robot的向�?
            rel_x, rel_y = px - robot_x, py - robot_y
            
            # 投影到路径方�?
            d_along = rel_x * ux + rel_y * uy   # 沿路径方向距�?
            d_perp = abs(rel_x * (-uy) + rel_y * ux)  # 垂直于路径距�?
            
            # 只关�? 在前�?d_along>0) �?在路径上(d_perp<0.4m=机器人宽�?
            if d_along <= 0:
                continue   # peer在后方，不管
            if d_perp > 0.40:
                continue   # peer在侧面，不管（不会碰撞）
            
            # peer在路径正前方�?
            blocking_peer = True
            min_along_dist = min(min_along_dist, d_along)
        
        if not blocking_peer:
            # 没有peer在路径上 �?解除紧急状�?
            if self._emergency_stopped:
                self._emergency_stopped = False
            return None
        
        # ── peer在路径正前方 �?根据距离决定制动力度 ──
        if min_along_dist < EMERGENCY_STOP_DIST:
            # 极近(<0.45m) �?紧急停�?
            if not self._emergency_stopped:
                if not self.quiet:
                    print(f"[Nav {self.robot_id}] �?Emergency braking! peer directly ahead "
                          f"d_along={min_along_dist:.2f} m")
                self._replan_requested = True  # �?只在首次进入时请求一�?
            self._emergency_stopped = True
            return (0.0, 0.0)
        elif min_along_dist < PEER_STOP_DIST:
            # <0.6m �?停止
            if not self._emergency_stopped:
                self._replan_requested = True  # �?只在首次进入时请求一�?
            self._emergency_stopped = True
            return (0.0, 0.0)
        elif min_along_dist < PEER_SLOW_DIST:
            # 0.6~1.2m �?线性减�?(不请求重规划,正常减速通过)
            factor = (min_along_dist - PEER_STOP_DIST) / (PEER_SLOW_DIST - PEER_STOP_DIST)
            factor = max(0.1, min(1.0, factor))
            return (factor,)
        
        # >1.2m但在路径�?�?不制动，supervisor轨迹预测会处�?
        return None
    
    def _check_path_for_conflicts(self, robot_x, robot_y):
        """
        前瞻路径冲突检测：检查前方N个waypoint线段上是否有peer�?
        
        只检�?未来�?(�?~4)，段0由实时制动层(_check_path_conflict_and_emergency)
        处理——避免两层重复触发导致无限重规划循环�?
        
        阈�? d_perp < 0.32m (两个robot半径0.18×2=0.36m�?.32m意味着必定碰撞)
        冷却: 请求重规划后5秒内不再触发
        """
        if not self.peer_positions or not self.waypoints:
            return
        
        # 冷却期检�? 避免反复重规划导致卡�?
        if not hasattr(self, '_replan_cooldown_until'):
            self._replan_cooldown_until = 0.0
        if hasattr(self, '_nav_step_count'):
            # 用帧数估算时�?(32ms/�?
            current_time_est = self._nav_step_count * 0.032
            if current_time_est < self._replan_cooldown_until:
                return
        
        # 收集前方路径�?
        idx = self.current_waypoint_idx
        upcoming = self.waypoints[idx:idx + 5]
        if not upcoming:
            return
        
        # 构建路径点序�? [当前位置, wp1, wp2, ...]
        points = [(robot_x, robot_y)] + [(w[0], w[1]) for w in upcoming]
        
        LOOKAHEAD_COLLISION_RADIUS = 0.32  # 只在必碰时触�?< 2×robot_radius)
        
        # 跳过�?! �?=当前位置→当前目标waypoint, 由实时制动层处理
        for i in range(1, len(points) - 1):
            ax, ay = points[i]
            bx, by = points[i + 1]
            seg_dx, seg_dy = bx - ax, by - ay
            seg_len = math.sqrt(seg_dx*seg_dx + seg_dy*seg_dy)
            if seg_len < 0.01:
                continue
            sux, suy = seg_dx / seg_len, seg_dy / seg_len
            
            for peer_id, (px, py) in self.peer_positions.items():
                rel_x, rel_y = px - ax, py - ay
                d_along = rel_x * sux + rel_y * suy
                d_perp = abs(rel_x * (-suy) + rel_y * sux)
                
                # peer在未来路径段�?投影在段�? 距离<碰撞半径)
                if 0 <= d_along <= seg_len and d_perp < LOOKAHEAD_COLLISION_RADIUS:
                    self._replan_requested = True
                    # 设置5秒冷�?�?56�?
                    if hasattr(self, '_nav_step_count'):
                        self._replan_cooldown_until = self._nav_step_count * 0.032 + 5.0
                    if not self.quiet:
                        print(f"[Nav {self.robot_id}] Look-ahead check: peer {peer_id} "
                              f"occupies path segment {i} (d_perp={d_perp:.2f} m < 0.32 m); requesting replan")
                    return
    
    def _get_front_min(self, lidar_ranges, robot_heading):
        """Get minimum LiDAR reading in the front ±30° cone."""
        n = len(lidar_ranges)
        if n == 0:
            return float('inf')
        min_r = float('inf')
        for i in range(n):
            # LiDAR angle relative to robot front
            ray_angle = (i / n) * 2 * math.pi
            if ray_angle > math.pi:
                ray_angle -= 2 * math.pi
            # Front cone: ±30° (±0.52 rad)
            if abs(ray_angle) < 0.52:
                min_r = min(min_r, lidar_ranges[i])
        return min_r

    def _scan_lidar_zones(self, lidar_ranges):
        """
        Partition the LiDAR scan into 3 zones used by the sidestep
        decision logic and return (front_min, (left_clear, right_clear),
        front_in_narrow_cone).
        
        Zones (assuming LiDAR is 360° starting at front, CCW):
          �?narrow front cone   ±SIDESTEP_FRONT_CONE_DEG/2 (e.g. ±15°)
                                �?used for head-on detection
          �?full front          ±45° �?used for general DWA trigger
          �?left side           +(15°…SIDESTEP_SIDE_CONE_DEG)
          �?right side          -(15°…SIDESTEP_SIDE_CONE_DEG)
        Returns the MIN distance in each zone (smaller = obstacle closer).
        """
        n = len(lidar_ranges)
        if n == 0:
            return (float('inf'), (float('inf'), float('inf')), float('inf'))
        
        # Helper: angle (deg) �?ray index (assuming ray 0 = front, CCW)
        def deg_to_idx(deg):
            # Webots LiDAR convention: ray 0 is front of sensor;
            # subsequent rays sweep CCW (left side first).
            d = deg % 360.0
            return int(round(d / 360.0 * n)) % n
        
        # Scan a sector and return min positive range
        def sector_min(start_deg, end_deg):
            i_start = deg_to_idx(start_deg)
            i_end = deg_to_idx(end_deg)
            best = float('inf')
            if i_start <= i_end:
                rng = range(i_start, i_end + 1)
            else:
                rng = list(range(i_start, n)) + list(range(0, i_end + 1))
            for i in rng:
                r = lidar_ranges[i]
                if r > 0 and r < best:
                    best = r
            return best
        
        # Narrow front cone: ±half the configured angle
        half_narrow = SIDESTEP_FRONT_CONE_DEG / 2
        narrow_front = min(
            sector_min(360 - half_narrow, 360),
            sector_min(0, half_narrow),
        )
        
        # Full front (±45°)
        front_full = min(
            sector_min(360 - 45, 360),
            sector_min(0, 45),
        )
        
        # Left side (CCW from narrow edge to side cone edge)
        left_clear = sector_min(half_narrow, SIDESTEP_SIDE_CONE_DEG / 2 + 45)
        # Right side (CW from narrow edge)
        right_clear = sector_min(360 - SIDESTEP_SIDE_CONE_DEG / 2 - 45,
                                  360 - half_narrow)
        
        return (front_full, (left_clear, right_clear), narrow_front)
    
    def _enter_sidestep(self, direction, robot_x, robot_y, robot_heading):
        """
        Begin a sidestep manoeuvre. `direction` is +1 (left of heading)
        or -1 (right of heading). We compute a virtual lateral target
        offset perpendicular to the robot's heading; DWA will then
        prefer trajectories toward this offset point while still
        making forward progress.
        """
        self.sidestep_active = True
        self.sidestep_direction = direction
        self.sidestep_start_t = self._sim_time()
        # Lateral offset is perpendicular to heading.
        # If heading=θ, then "left" unit vector = (-sin θ, cos θ)
        # and "right" = (sin θ, -cos θ).
        side_x = -math.sin(robot_heading) * direction
        side_y =  math.cos(robot_heading) * direction
        # Virtual sidestep waypoint: 1m forward + lateral offset
        forward_x = math.cos(robot_heading)
        forward_y = math.sin(robot_heading)
        self.sidestep_lateral_x = (robot_x + 1.0 * forward_x +
                                    SIDESTEP_LATERAL_OFFSET * side_x)
        self.sidestep_lateral_y = (robot_y + 1.0 * forward_y +
                                    SIDESTEP_LATERAL_OFFSET * side_y)
        side_word = "LEFT" if direction == +1 else "RIGHT"
        # Optional debug print (commented out to avoid log spam)
        # print(f"[Sidestep] entering {side_word}, "
        #       f"virtual target=({self.sidestep_lateral_x:.2f}, "
        #       f"{self.sidestep_lateral_y:.2f})")
    
    def _exit_sidestep(self):
        """End the current sidestep manoeuvre and resume normal navigation."""
        self.sidestep_active = False
        self.sidestep_direction = 0
    
    def _sim_time(self):
        """Approximate simulation time in seconds (for sidestep timeout).
        Uses Python's time module; acceptable since this is only used
        to bound how long we stay in sidestep, not for fine timing."""
        import time
        return time.monotonic()
    
    def _dwa_control(self, rx: float, ry: float, rh: float,
                     gx: float, gy: float,
                     lidar_ranges: List[float]) -> Tuple[float, float]:
        """
        Dynamic Window Approach for reactive collision avoidance.

        Evaluates candidate (v, w) pairs and selects the best based on:
          �?heading alignment with the goal
          �?distance to the nearest obstacle
          �?forward velocity preference

        All angles use math convention: 0 = facing +x, +π/2 = facing +y.
        Trajectory simulation uses (cos h, sin h) accordingly.
        """
        best_score = -float('inf')
        best_v = 0.0
        best_w = 0.0

        obstacle_points = self._lidar_obstacle_points(
            lidar_ranges, rx, ry, rh)
        start_obs_dist = self._distance_to_obstacle_points(
            rx, ry, obstacle_points)

        for v in self._dwa_v_samples:
            for w_index, w in enumerate(self._dwa_w_samples):
                # Simulate trajectory in math convention
                px, py, ph = rx, ry, rh
                min_dist_sq = float('inf')
                cos_ph = math.cos(rh)
                sin_ph = math.sin(rh)
                cos_delta, sin_delta = self._dwa_w_deltas[w_index]

                for _t_step in range(self._dwa_prediction_steps):
                    px += v * cos_ph * DWA_DT  # x = forward * cos
                    py += v * sin_ph * DWA_DT  # y = forward * sin
                    ph += w * DWA_DT

                    d_sq = self._distance_sq_to_obstacle_points(
                        px, py, obstacle_points)
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq

                    # Rotate the heading vector for the next substep using
                    # the recurrence cos(a+b), sin(a+b).
                    next_cos = cos_ph * cos_delta - sin_ph * sin_delta
                    next_sin = sin_ph * cos_delta + cos_ph * sin_delta
                    cos_ph, sin_ph = next_cos, next_sin

                min_dist = (math.sqrt(min_dist_sq)
                            if min_dist_sq < float('inf')
                            else float('inf'))

                if min_dist < CRITICAL_DISTANCE:
                    # Allow trajectory if it MOVES AWAY from the obstacle
                    # (end position further than start). This permits
                    # departure from docks where robot starts at 0.32m
                    # from shelf �?ANY trajectory heading away is valid.
                    if min_dist < start_obs_dist * 0.8:
                        # Trajectory gets CLOSER to obstacle �?reject
                        continue
                    # Otherwise: trajectory stays same or moves away �?allow

                # ── Heading score (toward goal) ─────────────────
                desired = math.atan2(gy - py, gx - px)
                heading_diff = abs(desired - ph)
                while heading_diff > math.pi:
                    heading_diff = abs(heading_diff - 2 * math.pi)
                heading_score = (math.pi - heading_diff) / math.pi
                
                # ── Distance score (avoid obstacles) ────────────
                dist_score = min(min_dist / OBSTACLE_THRESHOLD, 1.0)
                
                # ── Velocity score (prefer forward motion) ──────
                vel_score = v / MAX_LINEAR_SPEED
                
                # ── Lateral score (only in sidestep mode) ───────
                # Reward trajectories that move toward the virtual
                # lateral target. This biases the robot to drift
                # left/right past a head-on obstacle.
                lateral_score = 0.0
                if self.sidestep_active:
                    lat_dx = self.sidestep_lateral_x - px
                    lat_dy = self.sidestep_lateral_y - py
                    lat_dist = math.sqrt(lat_dx*lat_dx + lat_dy*lat_dy)
                    # Closer to lateral target = higher score (max 1.0)
                    lateral_score = max(0.0, 1.0 - lat_dist / 2.0)
                
                # ── Combined score ──────────────────────────────
                # When sidestepping, weigh heading-to-goal lower so
                # the robot is willing to deviate from the path; 
                # add lateral_score with high weight to pull it sideways.
                if self.sidestep_active:
                    score = (W_HEADING * 0.4 * heading_score +
                             W_DISTANCE * dist_score +
                             W_VELOCITY * vel_score +
                             W_LATERAL * lateral_score)
                else:
                    score = (W_HEADING * heading_score +
                             W_DISTANCE * dist_score +
                             W_VELOCITY * vel_score)
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
        
        # ── Recovery: if DWA found NO valid trajectory at all, the robot
        # would otherwise stay still forever. Rotate in place toward the
        # goal so LiDAR scans new directions on the next tick.
        if best_score == -float('inf'):
            desired = math.atan2(gy - ry, gx - rx)
            heading_err = desired - rh
            while heading_err > math.pi:
                heading_err -= 2 * math.pi
            while heading_err < -math.pi:
                heading_err += 2 * math.pi
            # Slow in-place rotation, sign of heading error
            best_v = 0.0
            best_w = 0.5 * (1.0 if heading_err > 0 else -1.0)

        # Convert to wheel speeds
        left_speed = (best_v - best_w * WHEEL_BASE / 2) / WHEEL_RADIUS
        right_speed = (best_v + best_w * WHEEL_BASE / 2) / WHEEL_RADIUS
        
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
        
        return (left_speed, right_speed)
    
    def _check_obstacle_distance(self, px, py, ph,
                                  lidar_ranges, rx, ry, rh):
        """Estimate distance to nearest obstacle from a predicted position."""
        points = self._lidar_obstacle_points(lidar_ranges, rx, ry, rh)
        return self._distance_to_obstacle_points(px, py, points)

    @staticmethod
    def _lidar_obstacle_points(lidar_ranges, rx, ry, rh):
        """Project the sampled scan once for all DWA candidates."""
        if not lidar_ranges:
            return ()
        
        # Simple approximation: use current LiDAR data offset by predicted motion
        points = []
        num_rays = len(lidar_ranges)
        
        for i in range(0, num_rays, max(1, num_rays // 36)):  # Check every 10 degrees
            r = lidar_ranges[i]
            if r <= 0 or r > 3.5:
                continue
            
            # Math convention: angle 0 = +x (forward when facing +x).
            # Webots LiDAR ray 0 points along the sensor's local +x; with
            # robot heading rh, world ray angle = rh + (2π·i/N).
            angle = (2 * math.pi * i / num_rays) + rh
            # Project ray endpoint to world coords using cos/sin (NOT sin/cos �?
            # this used to be a leftover from the old Webots NUE convention
            # where 'y' was treated as 'z'. Fixed here as part of task #38.)
            ox = rx + r * math.cos(angle)
            oy = ry + r * math.sin(angle)
            
            points.append((ox, oy))

        return tuple(points)

    @staticmethod
    def _distance_to_obstacle_points(px, py, obstacle_points):
        if not obstacle_points:
            return float('inf')
        min_dist = float('inf')
        for ox, oy in obstacle_points:
            d = math.sqrt((px - ox)**2 + (py - oy)**2)
            min_dist = min(min_dist, d)
        return min_dist

    @staticmethod
    def _distance_sq_to_obstacle_points(px, py, obstacle_points):
        """Squared-distance variant used by the DWA inner loop."""
        if not obstacle_points:
            return float('inf')
        min_dist_sq = float('inf')
        for ox, oy in obstacle_points:
            dx = px - ox
            dy = py - oy
            d_sq = dx * dx + dy * dy
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
        return min_dist_sq

    @staticmethod
    def _frange(start, stop, step):
        """Float range generator."""
        vals = []
        v = start
        while v <= stop:
            vals.append(v)
            v += step
        return vals


class RobotController:
    """
    Main robot controller class for Webots.
    
    Each instance controls one TurtleBot3 Waffle robot.
    """
    
    def __init__(self):
        # Initialize Webots Robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Get robot identity from customData
        self.robot_id = self._parse_robot_id()
        
        # Initialize devices
        self._motors_initialized = False
        self._sensors_initialized = False
        self._communication_initialized = False
        self._init_motors()
        self._init_sensors()
        self._init_communication()
        self._initialization_error = self._validate_initialization()
        self._ready_reported = False
        
        # Navigation
        self.quiet = os.environ.get('SMART_FACTORY_ROBOT_QUIET', '0') == '1'
        self.navigator = WaypointNavigator()
        self.navigator.robot_id = self.robot_id
        self.navigator.quiet = self.quiet
        self._prepared_joint_plans = {}
        self._active_plan_epoch = 0
        self._highest_prepared_epoch = 0
        self._scheduled_joint_plan = None
        self._armed_joint_epoch = None
        
        # State
        self.position = (0.0, 0.0)
        self.heading = 0.0
        self.battery = random.Random(self.robot_id).uniform(
            INITIAL_BATTERY_MIN, BATTERY_CAPACITY)
        
        print(f"[Robot {self.robot_id}] Initialized")

    def _validate_initialization(self):
        """Return an error string unless all dispatch-critical devices exist."""
        required = {
            "left_wheel_motor": getattr(self, "left_motor", None),
            "right_wheel_motor": getattr(self, "right_motor", None),
            "gps": getattr(self, "gps", None),
            "compass": getattr(self, "compass", None),
            "lidar": getattr(self, "lidar", None),
            "imu": getattr(self, "imu", None),
            "emitter": getattr(self, "emitter", None),
            "receiver": getattr(self, "receiver", None),
        }
        missing = [name for name, device in required.items() if device is None]
        if not self._motors_initialized:
            missing.append("motor_configuration")
        if not self._sensors_initialized:
            missing.append("sensor_configuration")
        if not self._communication_initialized:
            missing.append("communication_configuration")
        return "missing_devices:" + ",".join(missing) if missing else None

    def _send_ready(self):
        """Send exactly one READY/ERROR handshake after the first sensor read."""
        if self._ready_reported or not self.emitter:
            return
        self._ready_reported = True
        payload = {
            "type": "ROBOT_READY",
            "robot_id": self.robot_id,
            "robot_name": self.robot.getName(),
            "controller_name": "robot_controller",
            "initialization_timestamp": self.robot.getTime(),
            "position": list(self.position),
            "status": "READY" if self._initialization_error is None else "ERROR",
            "ready": self._initialization_error is None,
        }
        if self._initialization_error:
            payload["optional_error"] = self._initialization_error
        try:
            self.emitter.send(json.dumps(payload).encode("utf-8"))
            print(f"[Robot {self.robot_id}] READY handshake: "
                  f"{payload['status']}")
        except Exception as exc:
            self._ready_reported = False
            print(f"[Robot {self.robot_id}] READY handshake error: {exc}")
    
    def _parse_robot_id(self) -> int:
        """Parse robot ID from custom data."""
        try:
            custom_data = self.robot.getCustomData()
            if custom_data and "robot_id:" in custom_data:
                return int(custom_data.split("robot_id:")[1].strip())
        except Exception:
            pass
        
        # Fallback: try to extract from robot name
        try:
            name = self.robot.getName()
            if name and "_" in name:
                return int(name.split("_")[-1])
        except Exception:
            pass
        
        return 1  # Default
    
    def _init_motors(self):
        """Initialize wheel motors."""
        self._motors_initialized = False
        try:
            self.left_motor = self.robot.getDevice("left_wheel_motor")
            self.right_motor = self.robot.getDevice("right_wheel_motor")
            
            if self.left_motor:
                self.left_motor.setPosition(float('inf'))
                self.left_motor.setVelocity(0.0)
            if self.right_motor:
                self.right_motor.setPosition(float('inf'))
                self.right_motor.setVelocity(0.0)
            if self.left_motor is None or self.right_motor is None:
                raise RuntimeError("required wheel motor missing")
            self._motors_initialized = True
        except Exception as e:
            print(f"[Robot {self.robot_id}] Motor init error: {e}")
            self.left_motor = None
            self.right_motor = None
    
    def _init_sensors(self):
        """Initialize sensors (GPS, Compass, LiDAR, IMU)."""
        self._sensors_initialized = False
        try:
            # GPS
            self.gps = self.robot.getDevice("gps")
            if self.gps:
                self.gps.enable(self.timestep)
            
            # Compass
            self.compass = self.robot.getDevice("compass")
            if self.compass:
                self.compass.enable(self.timestep)
            
            # LiDAR
            self.lidar = self.robot.getDevice("lidar")
            if self.lidar:
                self.lidar.enable(self.timestep)
            
            # IMU
            self.imu = self.robot.getDevice("imu")
            if self.imu:
                self.imu.enable(self.timestep)
            
            # Position sensors (wheel encoders)
            self.left_encoder = self.robot.getDevice("left_wheel_sensor")
            if self.left_encoder:
                self.left_encoder.enable(self.timestep)
            
            self.right_encoder = self.robot.getDevice("right_wheel_sensor")
            if self.right_encoder:
                self.right_encoder.enable(self.timestep)
            required = (self.gps, self.compass, self.lidar, self.imu,
                        self.left_encoder, self.right_encoder)
            if any(device is None for device in required):
                raise RuntimeError("required sensor missing")
            self._sensors_initialized = True
        except Exception as e:
            print(f"[Robot {self.robot_id}] Sensor init error: {e}")
            self.gps = None
            self.compass = None
            self.lidar = None
            self.imu = None
    
    def _init_communication(self):
        """Initialize Emitter/Receiver for supervisor communication."""
        self._communication_initialized = False
        try:
            self.emitter = self.robot.getDevice("emitter")
            self.receiver = self.robot.getDevice("receiver")
            if self.emitter is None or self.receiver is None:
                raise RuntimeError("required communication device missing")
            self.receiver.enable(self.timestep)
            self._communication_initialized = True
        except Exception as e:
            print(f"[Robot {self.robot_id}] Communication init error: {e}")
            self.emitter = None
            self.receiver = None

    def _send_plan_ack(self, epoch, accepted, reason=""):
        if not self.emitter:
            return False
        payload = {
            "type": "PLAN_PREPARED",
            "robot_id": self.robot_id,
            "plan_epoch": int(epoch),
            "accepted": bool(accepted),
            "reason": reason,
        }
        try:
            self.emitter.send(json.dumps(payload).encode("utf-8"))
            return True
        except Exception:
            return False

    def _send_plan_armed(self, epoch):
        if not self.emitter:
            return False
        try:
            self.emitter.send(json.dumps({
                "type": "PLAN_ARMED", "robot_id": self.robot_id,
                "plan_epoch": int(epoch), "armed": True,
            }).encode("utf-8"))
            return True
        except Exception:
            return False

    def _send_plan_terminal(self, message_type, epoch):
        if not self.emitter:
            return False
        try:
            self.emitter.send(json.dumps({
                "type": message_type, "robot_id": self.robot_id,
                "plan_epoch": int(epoch),
                "active_plan_epoch": self._active_plan_epoch,
                "path_version": self.navigator.path_version,
            }).encode("utf-8"))
            return True
        except Exception:
            return False
    def _activate_scheduled_joint_plan(self):
        scheduled = self._scheduled_joint_plan
        if not scheduled or self.robot.getTime() < scheduled['activate_at']:
            return
        prepared = self._prepared_joint_plans.get(scheduled['epoch'])
        if (prepared is not None and prepared['path_version'] >=
                self.navigator.path_version):
            prepared['rollback_snapshot'] = {
                'waypoints': list(self.navigator.waypoints),
                'waypoint_index': self.navigator.current_waypoint_idx,
                'navigation_active': self.navigator.navigation_active,
                'paused_until': self.navigator.paused_until,
                'planned_wait_until': self.navigator.planned_wait_until,
                'planned_wait_reason': self.navigator.planned_wait_reason,
                'waypoint_not_before': list(
                    self.navigator.waypoint_not_before),
                'plan_is_partial': self.navigator.plan_is_partial,
                'joint_release_index': self.navigator.joint_release_index,
                'waypoint_threshold': self.navigator.waypoint_threshold,
                'joint_epoch_wait_deadline':
                    self.navigator.joint_epoch_wait_deadline,
                'replan_requested': self.navigator._replan_requested,
                'emergency_stopped': self.navigator._emergency_stopped,
                'active_epoch': self._active_plan_epoch,
            }
            self.navigator.path_version = prepared['path_version']
            self.navigator.paused_until = 0.0
            self.navigator.set_waypoints(prepared['waypoints'])
            self.navigator.waypoint_not_before = [
                scheduled['activate_at'] + float(offset)
                for offset in prepared.get('waypoint_offsets', ())]
            self.navigator.plan_is_partial = bool(
                prepared.get('partial_plan', False))
            # Execute the whole validated rolling prefix. Each waypoint still
            # has its scheduled not-before time, and a partial plan still
            # holds at its final cell until the next transaction is armed.
            self.navigator.joint_release_index = max(
                0, len(self.navigator.waypoints) - 1)
            self.navigator.waypoint_threshold = 0.22
            offsets = prepared.get('waypoint_offsets') or [0.0]
            last_offset = float(offsets[-1]) if offsets else 0.0
            self.navigator.joint_epoch_wait_deadline = (
                scheduled['activate_at'] + last_offset + 10.0)
            self.navigator.joint_coordinated = True
            self.navigator._replan_requested = False
            self.navigator._emergency_stopped = False
            self._active_plan_epoch = scheduled['epoch']
            self._send_plan_terminal("PLAN_ACTIVATED", scheduled['epoch'])
        self._scheduled_joint_plan = None
    
    def _read_sensors(self):
        """Read current sensor values."""
        # GPS position
        if self.gps:
            try:
                gps_values = self.gps.getValues()
                # ENU coordinate system: [x, y, z] where z is up
                # Ground-plane coordinates are (x, y)
                self.position = (gps_values[0], gps_values[1])
            except:
                pass
        
        # Compass heading
        if self.compass:
            try:
                compass_values = self.compass.getValues()
                # ENU coordinate system, robot rotates about +z axis by
                # angle θ (math convention: 0=facing east/+x, +π/2=north/+y).
                # Webots compass returns world-north vector in robot's local
                # frame. For a robot rotated by math angle θ around +z (CCW),
                # the world +y axis appears in robot frame as (sin θ, cos θ, 0).
                # Therefore the robot's math heading θ is recovered as:
                #     θ = atan2(compass[0], compass[1])
                # NOTE: previous code had `-compass[0]` which gave WRONG SIGN
                # for ±y headings, causing oscillation when targets required
                # north/south motion. Fixed via verify_dwa_lidar_coords audit.
                # All downstream code (compute_control, _dwa_control) uses
                # this same math-angle convention.
                self.heading = math.atan2(compass_values[0], compass_values[1])
            except:
                pass
    
    def _get_lidar_ranges(self) -> Optional[List[float]]:
        """Get current LiDAR range readings."""
        if self.lidar:
            try:
                ranges = self.lidar.getRangeImage()
                if ranges:
                    return list(ranges)
            except:
                pass
        return None
    
    def _set_motor_speeds(self, left: float, right: float):
        """Set wheel motor velocities."""
        if self.left_motor:
            self.left_motor.setVelocity(left)
        if self.right_motor:
            self.right_motor.setVelocity(right)
    
    def _receive_commands(self):
        """Process commands from the supervisor."""
        if not self.receiver:
            return
        
        while self.receiver.getQueueLength() > 0:
            try:
                data = self.receiver.getString()
                msg = json.loads(data)
                
                # Check if this message is for this robot
                target = msg.get('target_robot')
                if target is not None and target != self.robot_id:
                    self.receiver.nextPacket()
                    continue
                
                command = msg.get('command', {})
                cmd_type = command.get('type')

                if cmd_type == 'prepare_plan':
                    epoch = int(command.get('plan_epoch', 0))
                    version = int(command.get('path_version', 0))
                    waypoints = command.get('all_waypoints', [])
                    waypoint_offsets = command.get(
                        'waypoint_not_before_offsets', [])
                    accepted = bool(
                        epoch > self._active_plan_epoch and
                        epoch > self._highest_prepared_epoch and
                        version >= self.navigator.path_version and waypoints)
                    if accepted:
                        self._highest_prepared_epoch = epoch
                        self._prepared_joint_plans = {
                            epoch: {
                                'path_version': version,
                                'waypoints': [tuple(wp) for wp in waypoints],
                                'waypoint_offsets': [float(value)
                                                     for value in waypoint_offsets],
                                'partial_plan': bool(command.get(
                                    'partial_plan', False)),
                            }
                        }
                    self._send_plan_ack(
                        epoch, accepted,
                        "" if accepted else "stale_or_empty")

                elif cmd_type == 'arm_plan':
                    epoch = int(command.get('plan_epoch', 0))
                    prepared = self._prepared_joint_plans.get(epoch)
                    if (prepared and epoch > self._active_plan_epoch and
                            prepared['path_version'] >= self.navigator.path_version):
                        self._armed_joint_epoch = epoch
                        self._send_plan_armed(epoch)

                elif cmd_type == 'commit_plan':
                    epoch = int(command.get('plan_epoch', 0))
                    activate_at = float(command.get('activate_at', 0.0))
                    if (self._armed_joint_epoch == epoch and
                            activate_at > self.robot.getTime()):
                        self._scheduled_joint_plan = {
                            'epoch': epoch, 'activate_at': activate_at}
                        self._send_plan_terminal("PLAN_COMMITTED", epoch)

                elif cmd_type == 'abort_plan':
                    epoch = int(command.get('plan_epoch', 0))
                    prepared = self._prepared_joint_plans.pop(epoch, None)
                    if (prepared is not None and
                            self._active_plan_epoch == epoch):
                        snapshot = prepared.get('rollback_snapshot', {})
                        self.navigator.path_version = max(
                            self.navigator.path_version,
                            prepared['path_version']) + 1
                        self.navigator.waypoints = list(
                            snapshot.get('waypoints', ()))
                        self.navigator.current_waypoint_idx = min(
                            snapshot.get('waypoint_index', 0),
                            max(0, len(self.navigator.waypoints) - 1))
                        self.navigator.navigation_active = snapshot.get(
                            'navigation_active', False)
                        self.navigator.paused_until = snapshot.get(
                            'paused_until', 0.0)
                        self.navigator.waypoint_not_before = list(
                            snapshot.get('waypoint_not_before', ()))
                        self.navigator.plan_is_partial = bool(
                            snapshot.get('plan_is_partial', False))
                        self.navigator.joint_release_index = int(
                            snapshot.get('joint_release_index', 10 ** 9))
                        self.navigator.waypoint_threshold = float(
                            snapshot.get('waypoint_threshold', GOAL_THRESHOLD))
                        self.navigator.joint_epoch_wait_deadline = float(
                            snapshot.get('joint_epoch_wait_deadline', 0.0))
                        self.navigator._replan_requested = snapshot.get(
                            'replan_requested', False)
                        self.navigator._emergency_stopped = snapshot.get(
                            'emergency_stopped', False)
                        self._active_plan_epoch = snapshot.get(
                            'active_epoch', 0)
                    if (self._scheduled_joint_plan and
                            self._scheduled_joint_plan['epoch'] == epoch):
                        self._scheduled_joint_plan = None
                    if self._armed_joint_epoch == epoch:
                        self._armed_joint_epoch = None
                    self._send_plan_terminal("PLAN_ABORTED", epoch)
                
                elif cmd_type == 'navigate':
                    version = int(command.get('path_version', self.navigator.path_version + 1))
                    if version < self.navigator.path_version:
                        self.receiver.nextPacket()
                        continue
                    getattr(self, '_prepared_joint_plans', {}).clear()
                    self._scheduled_joint_plan = None
                    self.navigator.path_version = version
                    self.navigator.paused_until = 0.0
                    # Receive navigation waypoints
                    all_waypoints = command.get('all_waypoints', [])
                    direct = bool(command.get('direct_navigation', False))
                    if all_waypoints:
                        waypoints = [tuple(wp) for wp in all_waypoints]
                        self.navigator.set_waypoints(waypoints)
                        self.navigator.direct_navigation = direct
                        self.navigator._replan_requested = False  # 新路�?重规划完�?
                        self.navigator._emergency_stopped = False  # 解除停止
                        start = waypoints[0]
                        end = waypoints[-1]
                        print(f"[Robot {self.robot_id}] Received path: {len(waypoints)} steps, "
                              f"({start[0]:.2f},{start[1]:.2f})�?{end[0]:.2f},{end[1]:.2f})")
                    else:
                        target_pos = command.get('target')
                        if target_pos:
                            self.navigator.set_waypoints([tuple(target_pos)])
                            self.navigator.direct_navigation = direct

                elif cmd_type == 'release_joint_waypoint':
                    epoch = int(command.get('plan_epoch', 0))
                    if epoch == self._active_plan_epoch:
                        self.navigator.joint_release_index = max(
                            self.navigator.joint_release_index,
                            int(command.get('waypoint_index', 0)))
                
                elif cmd_type == 'stop':
                    self.navigator.navigation_active = False
                    self.navigator._emergency_stopped = False
                    self.navigator._replan_requested = False
                    self._set_motor_speeds(0, 0)

                elif cmd_type == 'hold':
                    version = int(command.get('path_version', self.navigator.path_version))
                    if version >= self.navigator.path_version:
                        self.navigator.path_version = version
                        self.navigator.paused_until = max(
                            self.navigator.paused_until,
                            float(command.get('until', 0.0)))
                        self._set_motor_speeds(0, 0)

                elif cmd_type == 'set_speed_scale':
                    self.navigator.speed_scale = max(
                        0.4, min(1.0, float(command.get('scale', 1.0))))
                
                elif cmd_type == 'charge':
                    # Navigate to charging station
                    target_pos = command.get('target')
                    if target_pos:
                        self.navigator.set_waypoints([tuple(target_pos)])

                elif cmd_type == 'battery_swap':
                    # Supervisor is authoritative for the Webots battery
                    # model; synchronize the controller telemetry after a
                    # completed station swap.
                    self.battery = max(
                        INITIAL_BATTERY_MIN,
                        min(BATTERY_CAPACITY, float(command['battery'])))
                    self.navigator.navigation_active = False
                    self._set_motor_speeds(0, 0)
                    print(f"[Robot {self.robot_id}] Battery swap synchronized: "
                          f"{self.battery:.1f}%")
                
                elif cmd_type == 'peer_positions':
                    # Update peer robot positions for conflict avoidance
                    # CRITICAL: Must set on NAVIGATOR (not just controller)
                    # because compute_control() runs on navigator's self
                    positions = command.get('positions', {})
                    peer_pos = {
                        int(rid): tuple(pos) for rid, pos in positions.items()
                    }
                    self.navigator.peer_positions = peer_pos
                    samples = command.get('samples', {})
                    for rid, sample in samples.items():
                        peer_id = int(rid)
                        seq = int(sample.get('seq', 0))
                        previous = self.navigator.peer_samples.get(peer_id)
                        if previous is None or seq >= int(previous.get('seq', -1)):
                            self.navigator.peer_samples[peer_id] = sample
                
                self.receiver.nextPacket()
                
            except Exception as e:
                print(f"[Robot {self.robot_id}] Command parse error: {e}")
                try:
                    self.receiver.nextPacket()
                except:
                    break
    
    def _send_status(self):
        """Send status report to supervisor."""
        if not self.emitter:
            return
        
        try:
            payload = {
                'robot_id': self.robot_id,
                'position': list(self.position),
                'heading': self.heading,
                'battery': self.battery,
                'navigating': self.navigator.navigation_active,
                'emergency_braking': self.navigator._emergency_stopped,
                'active_path_version': self.navigator.path_version,
                'active_waypoint_index': self.navigator.current_waypoint_idx,
                'active_plan_epoch': self._active_plan_epoch,
                'paused_until': self.navigator.paused_until,
                'planned_wait_until': self.navigator.planned_wait_until,
                'planned_wait_reason': self.navigator.planned_wait_reason,
                'speed_scale': self.navigator.speed_scale,
            }
            # Only INCLUDE reached_goal in the payload when actually True.
            # This way, supervisor's `msg.get('reached_goal') is True`
            # check is unambiguous AND the field's mere absence cannot
            # be misinterpreted as a positive flag in any future code.
            if self.navigator.goal_reached:
                payload['reached_goal'] = True
                # Reset BEFORE sending so we don't double-fire on the
                # next status tick if the message takes longer than
                # one Webots step to be picked up.
                self.navigator.goal_reached = False
            
            # 前瞻检测请求重规划 �?通知supervisor
            if self.navigator._replan_requested:
                payload['replan_requested'] = True
                self.navigator._replan_requested = False  # 发送后清除
            msg = json.dumps(payload)
            self.emitter.send(msg.encode('utf-8'))
                
        except Exception as e:
            print(f"[Robot {self.robot_id}] Status send error: {e}")
    
    def run(self):
        """Main control loop."""
        step_count = 0
        status_interval = 10  # Send status every N steps
        
        # ── One-shot sensor diagnostic (set WEBOTS_SENSOR_DIAG=1 to enable) ──
        import os
        if os.environ.get('WEBOTS_SENSOR_DIAG', '0') == '1':
            try:
                # Settle a few ticks for sensors to populate
                for _ in range(5):
                    self.robot.step(self.timestep)
                from sensor_diagnostics import run_sensor_diagnostics
                run_sensor_diagnostics(self.robot, self.robot_id,
                                        self.gps, self.compass,
                                        self.lidar, self.imu)
            except Exception as e:
                print(f"[Robot {self.robot_id}] Sensor diagnostic error: {e}")

        while self.robot.step(self.timestep) != -1:
            step_count += 1
            
            # 1. Read sensors
            self._read_sensors()
            # Devices and pose are now initialized; only this point may
            # advertise readiness to the Supervisor.
            self._send_ready()
            self.navigator.controller_time = (
                self.robot.getTime() if hasattr(self.robot, 'getTime')
                else step_count * self.timestep / 1000.0)
            
            # 2. Receive commands from supervisor
            self._receive_commands()
            self._activate_scheduled_joint_plan()
            
            # 3. Get LiDAR data
            lidar_ranges = self._get_lidar_ranges()
            
            # 4. Compute navigation control
            now = self.robot.getTime() if hasattr(self.robot, 'getTime') else 0.0
            if now < self.navigator.paused_until:
                self._set_motor_speeds(0.0, 0.0)
            elif self.navigator.navigation_active or self.navigator._emergency_stopped:
                left_speed, right_speed = self.navigator.compute_control(
                    self.position[0], self.position[1],
                    self.heading, lidar_ranges
                )
                # Use the validated 9% kinematic headroom while preserving
                # all planner speed-scale and emergency-stop decisions.
                left_speed *= self.navigator.speed_scale * 1.09
                right_speed *= self.navigator.speed_scale * 1.09
                self._set_motor_speeds(left_speed, right_speed)
                
                # Drain battery while moving
                self.battery = max(
                    0.0,
                    self.battery - BATTERY_DRAIN_RATE * self.timestep / 1000.0,
                )
            else:
                self._set_motor_speeds(0.0, 0.0)
            
            # 5. Send status to supervisor periodically
            if step_count % status_interval == 0:
                self._send_status()
            
            # 6. 周期性状态日�?�?00步≈10s)
            if step_count % 300 == 0 and self.navigator.navigation_active:
                if not self.quiet:
                    target = self.navigator.get_current_target()
                    wp_progress = f"{self.navigator.current_waypoint_idx}/{len(self.navigator.waypoints)-1}"
                    t_str = f"\u2192{target[0]:.1f},{target[1]:.1f})" if target else "no-target"
                    peers = len(self.navigator.peer_positions)
                    print(f"[Robot {self.robot_id}] pos=({self.position[0]:.2f},{self.position[1]:.2f}) "
                          f"wp={wp_progress} {t_str} peers={peers} bat={self.battery:.0f}%")


# ================================================================
# MAIN
# ================================================================

def main():
    controller = RobotController()
    controller.run()


if __name__ == "__main__":
    main()
