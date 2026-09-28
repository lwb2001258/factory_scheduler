"""
Sensor Test Controller
======================

Standalone Webots controller — drops a single robot in the factory,
runs the full sensor diagnostic, then continuously streams sensor data
so you can verify it changes correctly as you DRAG the robot around in
the Webots viewer (right-click the robot → Move Object).

Setup:
  1. In Webots, change one robot's controller to "sensor_test_controller"
  2. Hit play
  3. Watch the console
  4. Drag the robot around — verify GPS/compass values change
  5. Stop after a few seconds, change controller back

What it streams (every 0.5s):
  • GPS x, y, z + delta (how much you moved)
  • Compass vector + derived heading
  • IMU yaw + roll/pitch (should be ~0 on flat floor)
  • LiDAR: front/left/right/back min distances + total non-inf hits
  • Wheel encoder positions
"""

from controller import Robot
import math
import sys
import os

# Add path for sensor_diagnostics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                  'robot_controller'))
from sensor_diagnostics import run_sensor_diagnostics


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # ─── Acquire devices ────────────────────────────────────
    devs = {}
    for name, dev_type in [
        ('gps', 'GPS'),
        ('compass', 'Compass'),
        ('lidar', 'Lidar'),
        ('imu', 'InertialUnit'),
        ('left_wheel_sensor', 'PositionSensor'),
        ('right_wheel_sensor', 'PositionSensor'),
    ]:
        try:
            d = robot.getDevice(name)
            if d:
                d.enable(timestep)
                if dev_type == 'Lidar':
                    d.enablePointCloud()
                devs[name] = d
                print(f"  ✓ {name} ({dev_type}) acquired + enabled")
            else:
                print(f"  ❌ {name} ({dev_type}) — getDevice returned None")
        except Exception as e:
            print(f"  ❌ {name} — error: {e}")
            devs[name] = None
    
    # Settle 5 ticks
    print("\nSettling 5 ticks for sensors to populate…")
    for _ in range(5):
        robot.step(timestep)
    
    # ─── Run full diagnostic ────────────────────────────────
    run_sensor_diagnostics(robot, 0, devs.get('gps'), devs.get('compass'),
                            devs.get('lidar'), devs.get('imu'))
    
    # ─── Continuous streaming ───────────────────────────────
    print()
    print("═" * 68)
    print("  CONTINUOUS STREAM — drag the robot around to verify sensors")
    print("  (printing every 0.5s)")
    print("═" * 68)
    
    prev_pos = None
    step_count = 0
    print_interval = max(1, int(500 / timestep))  # ~500ms
    
    while robot.step(timestep) != -1:
        step_count += 1
        if step_count % print_interval != 0:
            continue
        
        # GPS
        g = devs.get('gps')
        if g:
            try:
                pos = g.getValues()
                delta = ""
                if prev_pos is not None:
                    dx = pos[0] - prev_pos[0]
                    dy = pos[1] - prev_pos[1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d > 0.001:
                        delta = f"  Δ={d:.3f}m"
                prev_pos = pos
                gps_str = f"GPS=({pos[0]:+6.2f}, {pos[1]:+6.2f}, {pos[2]:+5.3f}){delta}"
            except Exception as e:
                gps_str = f"GPS=ERROR({e})"
        else:
            gps_str = "GPS=N/A"
        
        # Compass + heading
        c = devs.get('compass')
        if c:
            try:
                cv = c.getValues()
                heading = math.degrees(math.atan2(cv[0], cv[1]))
                compass_str = (f"Compass=({cv[0]:+.2f},{cv[1]:+.2f})  "
                                f"heading={heading:+6.1f}°")
            except Exception as e:
                compass_str = f"Compass=ERROR"
        else:
            compass_str = "Compass=N/A"
        
        # IMU yaw
        i = devs.get('imu')
        if i:
            try:
                rpy = i.getRollPitchYaw()
                imu_str = (f"IMU(rpy)=({math.degrees(rpy[0]):+5.1f}°, "
                            f"{math.degrees(rpy[1]):+5.1f}°, "
                            f"{math.degrees(rpy[2]):+6.1f}°)")
            except Exception:
                imu_str = "IMU=ERROR"
        else:
            imu_str = "IMU=N/A"
        
        # LiDAR sectors
        l = devs.get('lidar')
        if l:
            try:
                ranges = l.getRangeImage()
                n = len(ranges)
                max_r = l.getMaxRange()
                # Sector min: front±15°, left 90°±15°, right 270°±15°, back 180°±15°
                def sec_min(start_deg, end_deg):
                    i_s = int(start_deg / 360 * n) % n
                    i_e = int(end_deg / 360 * n) % n
                    if i_s <= i_e:
                        rng = ranges[i_s:i_e+1]
                    else:
                        rng = list(ranges[i_s:]) + list(ranges[:i_e+1])
                    valid = [r for r in rng if 0 < r < max_r * 0.99]
                    return min(valid) if valid else float('inf')
                f = sec_min(345, 15)
                left = sec_min(75, 105)
                back = sec_min(165, 195)
                right = sec_min(255, 285)
                n_hits = sum(1 for r in ranges if 0 < r < max_r * 0.99)
                lidar_str = (f"LiDAR: F={f:.2f} L={left:.2f} B={back:.2f} "
                              f"R={right:.2f} ({n_hits}/{n} hits)")
            except Exception as e:
                lidar_str = f"LiDAR=ERROR({e})"
        else:
            lidar_str = "LiDAR=N/A"
        
        t = robot.getTime()
        print(f"t={t:6.1f}s | {gps_str}")
        print(f"           | {compass_str}")
        print(f"           | {imu_str}")
        print(f"           | {lidar_str}")
        print()


if __name__ == "__main__":
    main()
