"""
In-Robot Sensor Diagnostics for Webots
=======================================

Call `run_sensor_diagnostics(robot, robot_id)` ONCE at the top of your
robot controller's main loop. It will print a comprehensive sanity
report to the console.

Add to robot_controller.py:
    from sensor_diagnostics import run_sensor_diagnostics
    
    def run(self):
        # Settle 5 ticks for sensors to populate
        for _ in range(5):
            self.robot.step(self.timestep)
        
        run_sensor_diagnostics(self.robot, self.robot_id,
                                self.gps, self.compass, self.lidar,
                                self.imu)
        
        # ... rest of main loop ...

What it checks:
  • Each sensor exists and is enabled
  • Returns non-NaN, non-Inf, finite values
  • Values are within expected physical ranges
  • Compass + IMU cross-check (their heading estimates agree)
  • LiDAR returns expected number of rays + min/max range checks
  • Logs raw values for visual inspection
"""

import math


def _safe_get(getter, name):
    """Call a sensor getter, catching all errors."""
    try:
        return True, getter(), None
    except Exception as e:
        return False, None, str(e)


def check_finite(values, name, allow_zero=True):
    """Check every entry is finite (not NaN, not Inf)."""
    bad = []
    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            bad.append((i, f"non-numeric: {type(v).__name__}"))
            continue
        if math.isnan(v):
            bad.append((i, "NaN"))
        elif math.isinf(v):
            bad.append((i, "Inf"))
    if bad:
        print(f"    ❌ {name}: bad values at {bad[:3]}")
        return False
    return True


def run_sensor_diagnostics(robot, robot_id, gps, compass, lidar, imu):
    """
    Comprehensive sensor health check. Prints a multi-section report.
    Returns True if all sensors are healthy.
    """
    print()
    print("═" * 68)
    print(f"  SENSOR DIAGNOSTICS — Robot {robot_id}")
    print("═" * 68)
    
    all_ok = True
    
    # ───────────────────────────────────────────────────
    # 1. GPS
    # ───────────────────────────────────────────────────
    print()
    print("─── 1. GPS ────────────────────────────────────────────────────────")
    if gps is None:
        print("  ❌ GPS device handle is None")
        all_ok = False
    else:
        ok, vals, err = _safe_get(gps.getValues, "gps.getValues")
        if not ok:
            print(f"  ❌ GPS readout error: {err}")
            all_ok = False
        else:
            print(f"  Raw values: ({vals[0]:+.4f}, {vals[1]:+.4f}, {vals[2]:+.4f}) m")
            finite_ok = check_finite(vals, "GPS")
            all_ok &= finite_ok
            
            # Range check: factory is 18×11m centered around origin,
            # so |x| ≤ 9, |y| ≤ 5.5, z ≈ 0.03 (wheel radius)
            x, y, z = vals[0], vals[1], vals[2]
            in_range = (abs(x) <= 10 and abs(y) <= 6 and 0 <= z <= 0.5)
            if in_range:
                print(f"  ✓ GPS values within factory bounds")
            else:
                print(f"  ⚠️  GPS outside expected range: x∈[{x:.1f}], "
                       f"y∈[{y:.1f}], z∈[{z:.3f}]")
            
            # Z-axis sanity (wheel-mounted GPS ~3-5cm high)
            if not (0.0 <= z <= 0.15):
                print(f"  ⚠️  GPS z={z:.3f}m is unusual "
                       f"(expect 0.03–0.05m for ground robot)")
            
            # Sampling rate
            sampling = gps.getSamplingPeriod()
            print(f"  Sampling period: {sampling}ms "
                   f"({'enabled' if sampling > 0 else '❌ NOT enabled'})")
            if sampling <= 0:
                all_ok = False

    # ───────────────────────────────────────────────────
    # 2. Compass
    # ───────────────────────────────────────────────────
    print()
    print("─── 2. Compass ────────────────────────────────────────────────────")
    if compass is None:
        print("  ❌ Compass device handle is None")
        all_ok = False
    else:
        ok, vals, err = _safe_get(compass.getValues, "compass.getValues")
        if not ok:
            print(f"  ❌ Compass readout error: {err}")
            all_ok = False
        else:
            cx, cy, cz = vals[0], vals[1], vals[2]
            print(f"  Raw values: ({cx:+.4f}, {cy:+.4f}, {cz:+.4f})")
            
            finite_ok = check_finite(vals, "Compass")
            all_ok &= finite_ok
            
            # Compass should be unit vector
            magnitude = math.sqrt(cx*cx + cy*cy + cz*cz)
            print(f"  Magnitude: {magnitude:.4f} (expect ~1.0)")
            if not (0.95 <= magnitude <= 1.05):
                print(f"  ⚠️  Compass magnitude != 1; check northDirection")
            
            # Recover math heading: θ = atan2(compass[0], compass[1])
            heading = math.atan2(cx, cy)
            print(f"  Recovered heading: {math.degrees(heading):+6.1f}° "
                   f"(math: 0°=east, 90°=north)")
            
            # Sampling rate
            sampling = compass.getSamplingPeriod()
            print(f"  Sampling period: {sampling}ms "
                   f"({'enabled' if sampling > 0 else '❌ NOT enabled'})")
            if sampling <= 0:
                all_ok = False

    # ───────────────────────────────────────────────────
    # 3. IMU (cross-check vs compass)
    # ───────────────────────────────────────────────────
    print()
    print("─── 3. IMU (cross-check) ──────────────────────────────────────────")
    if imu is None:
        print("  ⚠️  IMU device handle is None (optional)")
    else:
        ok, vals, err = _safe_get(imu.getRollPitchYaw, "imu.getRollPitchYaw")
        if not ok:
            print(f"  ❌ IMU readout error: {err}")
            all_ok = False
        else:
            roll, pitch, yaw = vals
            print(f"  Roll/Pitch/Yaw: ({math.degrees(roll):+6.1f}°, "
                   f"{math.degrees(pitch):+6.1f}°, "
                   f"{math.degrees(yaw):+6.1f}°)")
            finite_ok = check_finite(vals, "IMU")
            all_ok &= finite_ok
            
            # Robot on flat ground: roll, pitch should be ~0
            if abs(roll) > 0.2 or abs(pitch) > 0.2:
                print(f"  ⚠️  Robot tilted: roll/pitch large "
                       f"(check it's on the floor)")
            
            # Compare IMU yaw vs compass heading
            if compass is not None:
                cvals = compass.getValues()
                compass_heading = math.atan2(cvals[0], cvals[1])
                # Webots yaw is also math angle from +x CCW
                diff = (yaw - compass_heading + math.pi) % (2*math.pi) - math.pi
                print(f"  IMU yaw - compass heading = "
                       f"{math.degrees(diff):+6.1f}° (expect ≈0)")
                if abs(diff) > math.radians(10):
                    print(f"  ⚠️  IMU and compass disagree by >10° — "
                           f"one of them is misconfigured")
                else:
                    print(f"  ✓ IMU and compass agree on heading")

    # ───────────────────────────────────────────────────
    # 4. LiDAR
    # ───────────────────────────────────────────────────
    print()
    print("─── 4. LiDAR ──────────────────────────────────────────────────────")
    if lidar is None:
        print("  ❌ LiDAR device handle is None")
        all_ok = False
    else:
        # Static properties
        h_res = lidar.getHorizontalResolution()
        n_layers = lidar.getNumberOfLayers()
        fov = lidar.getFov()
        max_r = lidar.getMaxRange()
        min_r = lidar.getMinRange()
        print(f"  Horizontal resolution: {h_res} rays")
        print(f"  Vertical layers: {n_layers}")
        print(f"  FoV: {math.degrees(fov):.1f}° "
               f"({'360°' if abs(fov - 2*math.pi) < 0.01 else 'limited'})")
        print(f"  Range: [{min_r}, {max_r}] m")
        
        sampling = lidar.getSamplingPeriod()
        print(f"  Sampling period: {sampling}ms "
               f"({'enabled' if sampling > 0 else '❌ NOT enabled'})")
        if sampling <= 0:
            all_ok = False
        
        # Read scan
        ok, ranges, err = _safe_get(lidar.getRangeImage, "lidar.getRangeImage")
        if not ok:
            print(f"  ❌ LiDAR getRangeImage error: {err}")
            all_ok = False
        else:
            if ranges is None:
                print(f"  ❌ getRangeImage returned None")
                all_ok = False
            else:
                n = len(ranges)
                if n != h_res * n_layers:
                    print(f"  ⚠️  Returned {n} rays, expected "
                           f"{h_res * n_layers}")
                
                finite_ok = check_finite(ranges, "LiDAR")
                all_ok &= finite_ok
                
                # Statistics
                valid = [r for r in ranges if 0 < r < max_r * 0.99]
                inf_count = sum(1 for r in ranges
                                if r >= max_r * 0.99 or r == float('inf'))
                neg_count = sum(1 for r in ranges if r <= 0)
                
                print(f"  Valid hits: {len(valid)} / {n}")
                print(f"  Inf/max-range hits (sky): {inf_count}")
                print(f"  Zero/negative: {neg_count}")
                
                if valid:
                    print(f"  Hit range stats: min={min(valid):.3f}m, "
                           f"max={max(valid):.3f}m, "
                           f"median={sorted(valid)[len(valid)//2]:.3f}m")
                
                # Check the 4 cardinal sectors
                def sector_mean(start_deg, end_deg):
                    i_s = int(start_deg / 360.0 * n) % n
                    i_e = int(end_deg / 360.0 * n) % n
                    if i_s <= i_e:
                        rng = ranges[i_s:i_e+1]
                    else:
                        rng = list(ranges[i_s:]) + list(ranges[:i_e+1])
                    valid = [r for r in rng if 0 < r < max_r * 0.99]
                    return sum(valid) / len(valid) if valid else float('inf')
                
                print(f"  Sector mean distances:")
                print(f"    Front (±15°):  {sector_mean(-15, 15):.3f}m")
                print(f"    Left  (75-105°): {sector_mean(75, 105):.3f}m")
                print(f"    Back  (165-195°): {sector_mean(165, 195):.3f}m")
                print(f"    Right (255-285°): {sector_mean(255, 285):.3f}m")

    # ───────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────
    print()
    print("═" * 68)
    if all_ok:
        print(f"  ✅ ALL SENSORS HEALTHY — Robot {robot_id} ready to navigate")
    else:
        print(f"  ❌ SENSOR ISSUES DETECTED — fix before running tasks")
    print("═" * 68)
    print()
    return all_ok
