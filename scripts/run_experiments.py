"""
Experiment Runner for Smart Factory Multi-Robot Coordination.

Runs systematic experiments comparing the RL-based scheduler against
baseline methods across all three scenarios.

Scenarios:
  A: 3 robots, 1 task per 30 seconds
  B: 5 robots, 1 task per 15 seconds
  C: 8 robots, 1 task per 8 seconds

Baselines: FCFS, NearestNeighbour, RoundRobin
Learned candidates include PPO/DQN variants, graph policies and LinUCB.

Each experiment is repeated 5 times with different random seeds.
"""

import os
import sys
import json
import subprocess
import time
import argparse
import statistics
import random
import math
import glob
import shutil
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 
                                'controllers', 'factory_supervisor'))

from config import SCENARIOS, NUM_REPEATS


# ================================================================
# EXPERIMENT CONFIGURATION
# ================================================================

SCHEDULER_TYPES = [
    "FCFS", "NearestNeighbour", "RoundRobin", "Greedy", "Random",
    "Hungarian", "Auction", "GA", "SA", "LearnedHungarian",
    "GraphImitation", "GraphPPO", "RainbowDQN", "QRDQN", "CQL",
    "LinUCB", "PPO_RL", "SARSA", "DQN"]
SCENARIO_KEYS = ["A", "B", "C"]
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

RESULTS_DIR = os.path.abspath(os.environ.get(
    "SMART_FACTORY_RESULTS_DIR",
    os.path.join(os.path.dirname(__file__), '..', 'results')))
WORLD_FILE = os.path.join(os.path.dirname(__file__), '..', 'worlds', 'smart_factory.wbt')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROBOT_CONTROLLER = os.path.join(
    PROJECT_ROOT, 'controllers', 'robot_controller', 'robot_controller.py')
SUPERVISOR_CONTROLLER = os.path.join(
    PROJECT_ROOT, 'controllers', 'factory_supervisor', 'factory_supervisor.py')

_WEBOTS_CHILD_PROCESS_FAILURES = (
    "QProcess: CreateFile failed",
    "was not found.\nWebots requires Python",
)
_EXTERNAL_WEBOTS_START_LOCK = threading.Lock()


def _tail(path: str, limit: int = 4000) -> str:
    """Read a bounded diagnostic tail without failing error handling."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            text = handle.read()
        return text[-limit:]
    except OSError:
        return ""


def _allocate_loopback_port() -> int:
    """Reserve an available TCP port for one Webots process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return int(probe.getsockname()[1])


def _find_webots_home(webots_path: str) -> str:
    """Find WEBOTS_HOME from either an executable path or PATH command."""
    executable = shutil.which(webots_path) or webots_path
    resolved = Path(executable).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / 'lib' / 'controller' / 'python').is_dir():
            return str(candidate)
    raise RuntimeError(
        f"Cannot locate Webots controller libraries from {webots_path!r}")


def _make_external_world(output_path: str) -> str:
    """Create a launch-only world using Webots external controllers.

    Only controller process ownership changes. Physics, sensors, nodes and all
    controller code remain identical to the production world.
    """
    with open(WORLD_FILE, 'r', encoding='utf-8') as handle:
        world = handle.read()
    robot_marker = 'controller "robot_controller"'
    supervisor_marker = 'controller "factory_supervisor"'
    expected_robots = max(int(cfg['num_robots']) for cfg in SCENARIOS.values())
    if (world.count(robot_marker) != expected_robots or world.count(
            supervisor_marker) != 1):
        raise RuntimeError(
            "Cannot build external-controller world: unexpected controller "
            "declarations in smart_factory.wbt")
    world = world.replace(robot_marker, 'controller "<extern>"')
    world = world.replace(supervisor_marker, 'controller "<extern>"')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(world)
    return output_path


def _stop_process(process: subprocess.Popen) -> None:
    """Stop one known child process and its Windows descendants."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(
            ['taskkill', '/PID', str(process.pid), '/T', '/F'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _wait_for_process_text(process: subprocess.Popen, log_path: str,
                           needle: str, timeout: float) -> bool:
    """Wait for an unbuffered controller readiness message."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if needle in _tail(log_path, limit=20000):
            return True
        if process.poll() is not None:
            return False
        time.sleep(0.1)
    return False


def _run_webots_with_external_controllers(
        webots_path: str, scenario: str, env: dict, show_webots: bool,
        wall_timeout: float) -> bool:
    """Run Webots physics with controller processes connected over TCP.

    Some managed Windows environments permit Webots itself but deny child
    process creation and local named pipes. Webots' supported external TCP
    mode preserves the exact controllers and world while avoiding both host
    restrictions.
    """
    runtime_dir = os.path.join(RESULTS_DIR, '.webots_runtime')
    os.makedirs(runtime_dir, exist_ok=True)
    run_token = f"{os.getpid()}_{time.time_ns()}"
    external_world = _make_external_world(os.path.join(
        runtime_dir, f'smart_factory_extern_{run_token}.wbt'))
    webots_log_path = os.path.join(runtime_dir, f'webots_{run_token}.log')
    webots_process = None
    controller_processes = []
    open_logs = []
    try:
        webots_log = open(
            webots_log_path, 'w', encoding='utf-8', errors='replace')
        open_logs.append(webots_log)
        # Serialize only the short bind/startup window.  Once the external
        # server owns its port, independent Webots simulations can run in
        # parallel without changing their controller or physics logic.
        with _EXTERNAL_WEBOTS_START_LOCK:
            port = _allocate_loopback_port()
            command = [
                webots_path, '--batch', '--mode=fast', '--minimize',
                '--extern-urls', f'--port={port}',
            ]
            if not show_webots:
                command.append('--no-rendering')
            command.append(external_world)
            webots_process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env,
                stdout=webots_log, stderr=subprocess.STDOUT, text=True,
            )
            # Webots buffers --extern-urls when stdout is a regular file, so
            # a log marker cannot be used as a readiness signal. Connecting
            # while the world is still loading can leave R2023b external
            # controllers pending indefinitely.
            startup_delay = float(env.get(
                'SMART_FACTORY_WEBOTS_STARTUP_DELAY', '3.0'))
            time.sleep(max(0.0, min(startup_delay, 30.0)))
        if webots_process.poll() is not None:
            print("Webots exited before its external-controller server "
                  f"started (code {webots_process.returncode}).")
            print(_tail(webots_log_path))
            return False

        webots_home = _find_webots_home(webots_path)
        controller_python = os.path.join(
            webots_home, 'lib', 'controller', 'python')
        base_controller_env = env.copy()
        existing_pythonpath = base_controller_env.get('PYTHONPATH', '')
        base_controller_env['WEBOTS_HOME'] = webots_home
        base_controller_env['PYTHONPATH'] = os.pathsep.join(
            value for value in (controller_python, existing_pythonpath)
            if value)
        base_controller_env['PYTHONIOENCODING'] = 'utf-8'
        base_controller_env['PYTHONUNBUFFERED'] = '1'

        def start_controller(name, controller_path):
            controller_env = base_controller_env.copy()
            controller_env['WEBOTS_CONTROLLER_URL'] = (
                f'tcp://127.0.0.1:{port}/{name}')
            controller_log_path = os.path.join(
                runtime_dir, f'{name}_{run_token}.log')
            controller_log = open(
                controller_log_path, 'w', encoding='utf-8', errors='replace')
            open_logs.append(controller_log)
            controller_processes.append((
                name,
                controller_log_path,
                subprocess.Popen(
                    [sys.executable, controller_path], cwd=PROJECT_ROOT,
                    env=controller_env, stdout=controller_log,
                    stderr=subprocess.STDOUT, text=True,
                ),
            ))
            return controller_processes[-1]

        # R2023b can deadlock world loading when a supervisor that removes
        # surplus robots races all robot connections. Establish one robot,
        # let the supervisor apply the A/B/C scene-tree count, then attach the
        # remaining active fleet. This is launch orchestration only; every
        # controller still executes the production source unchanged.
        name, log_path, process = start_controller(
            'robot_1', ROBOT_CONTROLLER)
        if not _wait_for_process_text(
                process, log_path, '[Robot 1] Initialized', timeout=45.0):
            print("robot_1 did not complete the Webots startup handshake:\n"
                  f"{_tail(log_path)}")
            return False

        name, log_path, process = start_controller(
            'factory_supervisor', SUPERVISOR_CONTROLLER)
        if not _wait_for_process_text(
                process, log_path, '[Supervisor] Initialized', timeout=45.0):
            print("factory_supervisor did not initialize the scenario:\n"
                  f"{_tail(log_path)}")
            return False

        for robot_id in range(
                2, int(SCENARIOS[scenario]['num_robots']) + 1):
            start_controller(f'robot_{robot_id}', ROBOT_CONTROLLER)

        deadline = time.monotonic() + wall_timeout
        return_code = None
        while time.monotonic() < deadline:
            return_code = webots_process.poll()
            if return_code is not None:
                break
            early_exit = next((
                (name, log_path, process.returncode)
                for name, log_path, process in controller_processes
                if process.poll() not in (None, 0)
            ), None)
            if early_exit is not None:
                name, log_path, controller_return_code = early_exit
                print(f"Controller {name} exited before Webots completed "
                      f"(code {controller_return_code}):\n{_tail(log_path)}")
                return False
            time.sleep(0.1)
        if return_code is None:
            print("WARNING: Webots simulation exceeded its dynamic wall-clock "
                  f"timeout ({wall_timeout:.0f}s) before completing!")
            return False

        failed_controllers = []
        for name, log_path, process in controller_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _stop_process(process)
            log_tail = _tail(log_path)
            if process.returncode not in (0, None) or 'Traceback (' in log_tail:
                failed_controllers.append((name, process.returncode, log_tail))
        if failed_controllers:
            for name, return_code_value, log_tail in failed_controllers:
                print(f"Controller {name} failed ({return_code_value}):\n"
                      f"{log_tail}")
            return False
        if return_code != 0:
            print(f"Webots exited with code {return_code}:\n"
                  f"{_tail(webots_log_path)}")
            return False
        return True
    finally:
        for _name, _log_path, process in controller_processes:
            _stop_process(process)
        if webots_process is not None:
            _stop_process(webots_process)
        for handle in open_logs:
            handle.close()
        try:
            os.remove(external_world)
        except OSError:
            pass


def webots_wall_timeout_seconds(sim_duration: float = None) -> float:
    """Return a conservative wall-clock timeout for a Webots batch run.

    Dense scenario-C physics can run close to real time even with --mode=fast.
    Allow two wall seconds per requested simulation second plus startup and
    shutdown overhead. Short smoke runs fail within two minutes rather than
    looking like a very slow simulation for ten minutes.
    """
    override = os.environ.get("SMART_FACTORY_WEBOTS_WALL_TIMEOUT_SECONDS")
    if override is not None:
        try:
            override_seconds = float(override)
        except (TypeError, ValueError):
            override_seconds = math.nan
        if math.isfinite(override_seconds) and override_seconds > 0:
            return max(120.0, override_seconds)
    if sim_duration is None:
        raw = os.environ.get("SMART_FACTORY_SIM_DURATION", "1800.0")
        try:
            sim_duration = float(raw)
        except (TypeError, ValueError):
            sim_duration = 1800.0
    if not math.isfinite(sim_duration) or sim_duration <= 0:
        sim_duration = 1800.0
    return max(120.0, sim_duration * 2.0 + 120.0)


def run_single_experiment(scenario: str, scheduler: str, seed: int,
                         webots_path: str = "webots",
                         model_path: str = None,
                         show_webots: bool = False,
                         controller_mode: str = "auto") -> dict:
    """
    Run a single experiment by launching Webots with the appropriate
    controller arguments.
    
    For standalone (non-Webots) testing, runs the simulation loop directly.
    """
    print(f"\n{'='*60}")
    print(f"Running: Scenario {scenario} | Scheduler: {scheduler} | Seed: {seed}")
    print(f"{'='*60}")
    
    # Try to run with Webots
    env = os.environ.copy()
    env["SCENARIO"] = scenario
    env["SCHEDULER"] = scheduler
    env["SEED"] = str(seed)
    # Distinguish automated experiments from opening the world interactively.
    # Only batch runs should stop and close Webots at SIM_DURATION.
    env["SMART_FACTORY_AUTO_STOP"] = "1"
    if model_path:
        env["MODEL_PATH"] = model_path
    else:
        env.pop("MODEL_PATH", None)
    
    run_started_at = time.time()
    preflight_output = ""
    try:
        # Check if webots is available
        result = subprocess.run(
            [webots_path, "--version"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10
        )
        webots_available = result.returncode == 0
        preflight_output = (result.stdout or "") + (result.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        webots_available = False

    if controller_mode not in {"auto", "internal", "external"}:
        raise ValueError(
            "controller_mode must be auto, internal, or external")
    child_process_blocked = any(
        marker in preflight_output
        for marker in _WEBOTS_CHILD_PROCESS_FAILURES)
    selected_controller_mode = controller_mode
    if selected_controller_mode == "auto":
        selected_controller_mode = (
            "external" if child_process_blocked else "internal")
    
    webots_run_ok = False
    if webots_available:
        # Launch Webots in batch mode (no GUI, faster)
        wall_timeout = webots_wall_timeout_seconds()
        print(f"Launching Webots simulation (wall timeout: "
              f"{wall_timeout:.0f}s)...")
        if selected_controller_mode == "external":
            reason = ("host blocks Webots child processes/local IPC"
                      if child_process_blocked else "explicit request")
            print("Controller mode: external TCP "
                  f"({reason}; simulation logic is unchanged)")
            try:
                webots_run_ok = _run_webots_with_external_controllers(
                    webots_path, scenario, env, show_webots, wall_timeout)
            except (OSError, RuntimeError) as exc:
                print(f"Webots external-controller launch failed: {exc}")
        else:
            print("Controller mode: internal Webots processes")
            try:
                webots_command = [
                    webots_path, "--batch", "--mode=fast", "--stdout",
                    "--stderr",
                ]
                if not show_webots:
                    webots_command.append("--no-rendering")
                webots_command.append(WORLD_FILE)
                result = subprocess.run(
                    webots_command,
                    env=env,
                    capture_output=True, encoding="utf-8", errors="replace",
                    text=True,
                    timeout=wall_timeout
                )
                print(result.stdout[-4000:])
                if result.returncode != 0:
                    print(f"Webots error: {result.stderr[-1000:]}")
                else:
                    webots_run_ok = True
            except subprocess.TimeoutExpired:
                print("WARNING: Webots simulation exceeded its dynamic "
                      f"wall-clock timeout ({wall_timeout:.0f}s) before "
                      "completing!")
        if not webots_run_ok:
            # Never load a stale result from a previous run after a failed
            # Webots launch; that would make validation appear successful.
            return None
    else:
        # Run standalone simulation (without Webots physics)
        print("Webots not found. Running standalone simulation...")
        run_standalone_simulation(
            scenario, scheduler, seed, model_path=model_path)
    
    # Find and load the most recent results file
    return load_latest_results(
        scenario, scheduler, seed=seed,
        runtime_mode="webots" if webots_available else "standalone",
        min_mtime=run_started_at - 1.0,
    )


def run_standalone_simulation(scenario: str, scheduler: str, seed: int,
                              model_path: str = None):
    """
    Run the simulation in standalone mode (without Webots).
    Uses simplified physics for testing algorithm logic.
    
    This implements the FULL state machine: IDLE → EN_ROUTE_PICKUP →
    EN_ROUTE_DELIVERY → IDLE, plus the charging cycle:
    IDLE (low battery) → RETURNING_TO_CHARGE → CHARGING → IDLE.
    
    Critical correctness fixes (vs the original buggy version):
      - Empty/None paths no longer leave the robot in a stuck state.
      - Battery thresholds are split (LOW = 25, MIN_TASK = 15, FULL = 95).
      - RETURNING_TO_CHARGE drains battery; only CHARGING refills it.
      - t=0 metrics are recorded before any drain happens.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 
                                    'controllers', 'factory_supervisor'))
    
    from config import (SCENARIOS, SIM_DURATION, RobotState, TIMESTEP,
                        TaskStatus, BATTERY_DRAIN_RATE, BATTERY_CHARGE_RATE,
                        INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX,
                        LOW_BATTERY_THRESHOLD, TASK_ABORT_BATTERY_THRESHOLD,
                        FULL_BATTERY_THRESHOLD,
                        CHARGING_STATIONS, PARKING_SPOTS, REST_NODES,
                        WAYPOINTS as _CFG_WAYPOINTS)
    from task_generator import TaskGenerator
    from motion_coordinator import MotionCoordinator
    from schedulers import (
        create_scheduler, GreedyScheduler, HungarianScheduler,
        NearestNeighbourScheduler, SchedulingContext)
    from learning_scheduler import attach_learning_trace
    from training_scenarios import FactoryAStarCostOracle, path_length
    from metrics_collector import MetricsCollector
    
    scenario_config = SCENARIOS[scenario]
    num_robots = scenario_config['num_robots']
    rng = random.Random(seed)
    
    # Initialise components
    task_gen = TaskGenerator(
        mean_interval=scenario_config['task_interval'],
        seed=seed,
        initial_task_immediately=scenario_config.get(
            'initial_task_immediately', False)
    )
    coordinator = MotionCoordinator(num_active_robots=num_robots)
    scheduler_obj = create_scheduler(
        scheduler, model_path=model_path, seed=seed)
    safe_schedulers = [
        HungarianScheduler(), GreedyScheduler(), NearestNeighbourScheduler()]
    failed_pairs = {}
    metrics = MetricsCollector(
        scenario, scheduler, num_robots, seed=seed,
        runtime_mode="standalone")
    
    # Robot starting positions (must match worlds/smart_factory.wbt)
    initial_positions = PARKING_SPOTS
    
    # Per-robot state structures
    robot_states = {}
    robot_tasks = {}
    robot_waypoints = {}
    robot_idle_time = {}
    robot_distance = {}
    robot_tasks_completed = {}
    
    for rid in range(1, num_robots + 1):
        pos = initial_positions[rid]
        robot_states[rid] = {
            'position': pos,
            'heading': 0.0,
            'state': RobotState.IDLE,
            'battery': 0.0,  # overwritten below with seeded initial charge
            'current_task': None,
            'has_task': False,
            'goal_location': None,
        }
        robot_states[rid]['battery'] = float(rng.uniform(
            INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX))
        robot_tasks[rid] = None
        robot_waypoints[rid] = []
        robot_idle_time[rid] = 0.0
        robot_distance[rid] = 0.0
        robot_tasks_completed[rid] = 0
    
    coordinator.set_priorities(list(range(1, num_robots + 1)))
    coordinator.lifelong_reset()  # fresh reservation table per experiment
    path_cost_oracle = FactoryAStarCostOracle(robot_states, num_robots)
    
    # Lifelong planner is discrete — its global_t advances 1 per
    # LIFELONG_TICK_PERIOD seconds of sim-time. Empirically 1.5 s
    # corresponds to one graph edge (~3 m at 0.22 m/s ≈ 13.6 s,
    # but our edges average ~2 m so 1.5 s is the right cadence).
    LIFELONG_TICK_PERIOD = 1.5  # seconds of sim-time per lifelong step
    next_lifelong_tick = LIFELONG_TICK_PERIOD
    
    # ----------------------------------------------------------------
    # Helper: send a robot to its nearest charging station
    # ----------------------------------------------------------------
    def send_to_charging(rid: int):
        """Route to the charging station with the shortest reachable A* path."""
        rs = robot_states[rid]
        coordinator.clear_robot_path(rid)
        coordinator.release_lifelong(rid); coordinator.release_robot_grid(rid)  # free old reservations first

        # Being physically at a station is the only case where an empty path
        # means arrival. A failed planner must never start charging remotely.
        arrived = [(name, xy) for name, xy in CHARGING_STATIONS.items()
                   if math.hypot(xy[0] - rs['position'][0],
                                 xy[1] - rs['position'][1]) <= 0.5]
        if arrived:
            nearest_cs = arrived[0][0]
            robot_waypoints[rid] = []
            rs['state'] = RobotState.CHARGING
            rs['goal_location'] = nearest_cs
            return

        candidates = []
        for station in CHARGING_STATIONS:
            candidate = coordinator.plan_grid_lifelong(
                rid, rs['position'], station)
            if candidate is None:
                candidate = coordinator.plan_path_for_robot(
                    rid, rs['position'], station)
            if candidate:
                candidates.append((path_length(candidate, rs['position']),
                                   station, candidate))
            # Planning another candidate must not leave reservations behind.
            coordinator.release_lifelong(rid)
            coordinator.release_robot_grid(rid)
        if not candidates:
            robot_waypoints[rid] = []
            rs['state'] = RobotState.WAITING
            rs['goal_location'] = None
            return

        _, nearest_cs, _ = min(candidates, key=lambda item: item[0])
        path = coordinator.plan_grid_lifelong(
            rid, rs['position'], nearest_cs)
        if path is None:
            path = coordinator.plan_path_for_robot(
                rid, rs['position'], nearest_cs)
        if not path:
            robot_waypoints[rid] = []
            rs['state'] = RobotState.WAITING
            rs['goal_location'] = None
            return
        robot_waypoints[rid] = list(path)
        rs['state'] = RobotState.RETURNING_TO_CHARGE
        rs['goal_location'] = nearest_cs
        print(f"  [T={sim_time:.1f}] Robot {rid} RETURNING TO CHARGE at "
              f"{nearest_cs} (battery={rs['battery']:.1f}%)")
    
    # ----------------------------------------------------------------
    # Helper: pop arrived waypoint, handle state transitions
    # ----------------------------------------------------------------
    def handle_arrival(rid: int):
        """Robot just popped its last waypoint. Decide what to do next."""
        rs = robot_states[rid]
        
        # Returning home / relocating to rest: pin static reservation,
        # go IDLE so the scheduler can immediately claim the robot for
        # the next task. This is "task chaining" — the robot can be
        # interrupted at any time during the relocation walk.
        if rs['state'] == RobotState.RETURNING_HOME:
            rs['state'] = RobotState.IDLE
            robot_waypoints[rid] = []
            coordinator.clear_robot_path(rid)
            # Reserve the actual current position's nearest node
            # (may be the rest node we just walked to, OR our origin
            # spot if we never moved). This prevents peers from
            # routing through our parked location.
            current_node = coordinator.graph.get_nearest_node(rs['position'])
            if current_node:
                coordinator.lifelong.reserve_static(rid, current_node)
            return
        
        task = robot_tasks[rid]
        
        if rs['state'] == RobotState.EN_ROUTE_PICKUP and task:
            # Reached pickup. Plan delivery path.
            task.pickup_time = sim_time
            rs['state'] = RobotState.EN_ROUTE_DELIVERY
            rs['goal_location'] = task.delivery_location
            
            path = coordinator.plan_grid_lifelong(
                rid, rs['position'], task.delivery_location
            )
            if path is None:
                path = coordinator.plan_path_for_robot(
                    rid, rs['position'], task.delivery_location
                )
            if path:
                robot_waypoints[rid] = list(path)
            else:
                # pickup == delivery — finish immediately
                robot_waypoints[rid] = []
                _complete_task(rid)
        
        elif rs['state'] == RobotState.EN_ROUTE_DELIVERY and task:
            _complete_task(rid)
        
        elif rs['state'] == RobotState.RETURNING_TO_CHARGE:
            # Reached the charging station — start charging
            rs['state'] = RobotState.CHARGING
            robot_waypoints[rid] = []
            coordinator.clear_robot_path(rid)
            coordinator.release_lifelong(rid); coordinator.release_robot_grid(rid)
            print(f"  [T={sim_time:.1f}] Robot {rid} ARRIVED at charging "
                  f"station, battery={rs['battery']:.1f}%")
    
    def _complete_task(rid: int):
        """Mark task complete, return robot to IDLE."""
        rs = robot_states[rid]
        task = robot_tasks[rid]
        task.status = TaskStatus.COMPLETED
        task.completion_time = sim_time
        robot_tasks_completed[rid] += 1
        metrics.record_task_completion(task, rid, sim_time)
        
        robot_tasks[rid] = None
        rs['state'] = RobotState.IDLE
        rs['current_task'] = None
        rs['has_task'] = False
        rs['goal_location'] = None
        coordinator.clear_robot_path(rid)
        # IMPORTANT: free the robot's reservations so other robots
        # may now route through what used to be its corridor.
        coordinator.release_lifelong(rid); coordinator.release_robot_grid(rid)
    
    # ----------------------------------------------------------------
    # Simulation loop
    # ----------------------------------------------------------------
    dt = TIMESTEP / 1000.0
    sim_time = 0.0
    step = 0
    
    print(f"  Starting standalone simulation ({SIM_DURATION}s)...")
    
    # Level 1/2/3 — coordination layer setup
    coordinator.init_deadlock_monitor()
    _next_rhcr_t = 5.0   # First RHCR replan after 5 sim-seconds
    
    # Bug 3 fix: record t=0 metrics BEFORE any drain happens
    metrics.record_step(
        sim_time, robot_states,
        task_gen.get_statistics(),
        coordinator.get_statistics(),
    )
    
    while sim_time < SIM_DURATION:
        sim_time += dt
        coordinator.set_sim_time(sim_time)
        step += 1
        
        # Advance lifelong planner clock once per LIFELONG_TICK_PERIOD
        if sim_time >= next_lifelong_tick:
            coordinator.lifelong_tick(1)
            next_lifelong_tick += LIFELONG_TICK_PERIOD
        
        # 1) Generate new tasks
        new_task = task_gen.update(sim_time)
        if new_task:
            metrics.record_task_arrival(new_task, sim_time)
        
        # 2) Update each robot
        for rid in range(1, num_robots + 1):
            rs = robot_states[rid]
            
            # ------- battery dynamics -------
            if rs['state'] in (RobotState.EN_ROUTE_PICKUP,
                               RobotState.EN_ROUTE_DELIVERY,
                               RobotState.RETURNING_HOME,
                               RobotState.RETURNING_TO_CHARGE):
                rs['battery'] = max(0.0, rs['battery'] - BATTERY_DRAIN_RATE * dt)
            elif rs['state'] == RobotState.CHARGING:
                rs['battery'] = min(100.0, rs['battery'] + BATTERY_CHARGE_RATE * dt)
                if rs['battery'] >= FULL_BATTERY_THRESHOLD:
                    rs['state'] = RobotState.IDLE
                    print(f"  [T={sim_time:.1f}] Robot {rid} CHARGED to "
                          f"{rs['battery']:.1f}%, returning to service")
            elif rs['state'] == RobotState.IDLE:
                robot_idle_time[rid] += dt

            # Idle low-battery robots charge before dispatch. Active robots
            # continue in the 15-25% band and abort only below 15%.
            if (rs['battery'] < LOW_BATTERY_THRESHOLD and
                    rs['state'] not in (RobotState.RETURNING_TO_CHARGE,
                                        RobotState.CHARGING)):
                active_task = robot_tasks[rid]
                if active_task is not None and rs['battery'] < TASK_ABORT_BATTERY_THRESHOLD:
                    active_task.status = (
                        TaskStatus.FAILED if rs['battery'] <
                        TASK_ABORT_BATTERY_THRESHOLD else TaskStatus.PENDING)
                    active_task.assigned_robot = None
                    active_task.assignment_time = None
                    robot_tasks[rid] = None
                    rs['current_task'] = None
                    rs['has_task'] = False
                    send_to_charging(rid)
                elif active_task is None:
                    send_to_charging(rid)
            
            # ------- movement -------
            if rs['state'] == RobotState.IDLE or rs['state'] == RobotState.CHARGING:
                continue
            
            if not robot_waypoints[rid]:
                # No waypoints but not idle — degenerate state.
                # Treat as "arrived" to recover gracefully (Bug 2 fix).
                handle_arrival(rid)
                continue
            
            target = robot_waypoints[rid][0]
            pos = rs['position']
            dx = target[0] - pos[0]
            dz = target[1] - pos[1]
            dist = (dx*dx + dz*dz) ** 0.5
            
            if dist < 0.35:
                # Reached this waypoint — pop and check next
                robot_waypoints[rid].pop(0)
                if not robot_waypoints[rid]:
                    handle_arrival(rid)
            else:
                # Step toward target
                speed = min(0.22, dist * 0.5) * dt
                rs['position'] = (
                    pos[0] + dx / dist * speed,
                    pos[1] + dz / dist * speed,
                )
                robot_distance[rid] += speed
        
        # 3) Assign pending tasks to idle robots
        pending = task_gen.get_pending_tasks()
        if pending:
            for _ in range(min(len(pending), num_robots)):
                pending = task_gen.get_pending_tasks()
                if not pending:
                    break
                
                states_for_scheduler = {
                    rid: {
                        **dict(robot_states[rid]),
                        "tasks_completed": robot_tasks_completed[rid],
                        "total_distance": robot_distance[rid],
                    }
                    for rid in range(1, num_robots + 1)
                }
                congestion = coordinator.get_congestion_map()
                failed_pairs = {
                    pair: expiry for pair, expiry in failed_pairs.items()
                    if expiry > sim_time
                }
                context = SchedulingContext(
                    current_time=sim_time,
                    congestion_map=congestion,
                    path_cost_provider=path_cost_oracle,
                    failed_pairs=frozenset(failed_pairs),
                    configuration={
                        "simulation_geometry": "factory-grid-astar-v1"},
                )
                decision = scheduler_obj.assign(
                    pending, states_for_scheduler, context
                )
                active_scheduler = scheduler_obj
                if not decision.is_feasible or not decision.assignments:
                    reason = decision.diagnostics.get("reason", "")
                    metrics.record_scheduler_fallback(
                        invalid_output=reason not in {
                            "no_candidates", "no_feasible_pair",
                            "empty_assignment", "empty_assignments"})
                    for fallback in safe_schedulers:
                        decision = fallback.assign(
                            pending, states_for_scheduler, context)
                        active_scheduler = fallback
                        if decision.is_feasible and decision.assignments:
                            break
                metrics.record_scheduling_latency(
                    decision.computation_time)
                if not decision.is_feasible or not decision.assignments:
                    break
                
                assignment = decision.assignments[0]
                rid, task = assignment.robot_id, assignment.task
                
                rs = robot_states[rid]
                
                # Robot is leaving its home → release static reservation
                
                # Use LIFELONG planner — guarantees no spatio-temporal
                # conflict with paths already committed by other robots.
                # Failure means the planner couldn't find a free corridor
                # within the lookahead horizon; fall back to legacy A*.
                path = coordinator.plan_grid_lifelong(
                    rid, rs['position'], task.pickup_location
                )
                if path is None:
                    path = coordinator.plan_path_for_robot(
                        rid, rs['position'], task.pickup_location
                    )
                if path:
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_robot = rid
                    task.assignment_time = sim_time
                    robot_tasks[rid] = task
                    rs['state'] = RobotState.EN_ROUTE_PICKUP
                    rs['current_task'] = task
                    rs['has_task'] = True
                    rs['goal_location'] = task.pickup_location
                    coordinator.release_home(rid)
                    robot_waypoints[rid] = list(path)
                    active_scheduler.on_assignment_committed(assignment)
                    attach_learning_trace(
                        task, assignment, states_for_scheduler, pending,
                        context, decision.algorithm_name or
                        active_scheduler.name)
                    metrics.record_scheduler_commit(
                        active_scheduler.name,
                        native=(active_scheduler is scheduler_obj))
                    failed_pairs.pop((rid, task.task_id), None)
                else:
                    active_scheduler.on_assignment_rejected(
                        assignment, "pickup_path_unreachable")
                    failed_pairs[(rid, task.task_id)] = sim_time + 5.0
        
        # ----------------------------------------------------------
        # Level 2 — Deadlock detection + priority-inheritance break.
        # Detects robots stuck >10 ticks; if 2+ are spatially close
        # the lower-priority one yields and re-plans.
        # ----------------------------------------------------------
        if step % 5 == 0:  # Run every ~50ms (less frequently than tick)
            stuck = coordinator.update_deadlock_monitor(robot_states)
            if stuck:
                broken = coordinator.break_deadlock(stuck, robot_states)
                # For each robot whose reservations were cleared, re-plan
                for rid in broken:
                    rs = robot_states[rid]
                    task = robot_tasks.get(rid)
                    if task and rs.get("goal_location"):
                        new_path = coordinator.plan_grid_lifelong(
                            rid, rs["position"], rs["goal_location"])
                        if new_path:
                            robot_waypoints[rid] = list(new_path)
        
        # ----------------------------------------------------------
        # Level 3 — Periodic RHCR (Rolling-Horizon Cooperative
        # Replanning). Every 5 sim-seconds, do a global CBS replan
        # over all active robots, accept only if total cost improves.
        # ----------------------------------------------------------
        if not hasattr(_replan_state := type('S', (), {})(), 'next_t'):
            pass   # noop sentinel for first run
        if step == 0:
            _next_rhcr_t = 5.0
        if sim_time >= _next_rhcr_t:
            active_goals = {}
            for rid in range(1, num_robots + 1):
                t = robot_tasks.get(rid)
                if t and t.status == TaskStatus.ASSIGNED:
                    rs = robot_states[rid]
                    if rs.get("state") == RobotState.EN_ROUTE_PICKUP:
                        active_goals[rid] = t.pickup_location
                    elif rs.get("state") == RobotState.EN_ROUTE_DELIVERY:
                        active_goals[rid] = t.delivery_location
            if len(active_goals) >= 2:
                new_paths = coordinator.rhcr_replan(robot_states, active_goals)
                # If plans changed, fetch updated paths from lifelong reservations
                # (rhcr_replan already wrote them; nothing else needed here —
                # the next plan_lifelong-equivalent path is already committed)
            _next_rhcr_t = sim_time + 5.0
        
        # ----------------------------------------------------------
        # Lazy relocation — IDLE robots that aren't currently at a
        # rest node walk to the nearest available rest node. This is
        # interruptible: the scheduler may grab the robot at any
        # moment if a new task arrives during this walk.
        # ----------------------------------------------------------
        for rid in range(1, num_robots + 1):
            rs = robot_states[rid]
            if rs['state'] != RobotState.IDLE:
                continue
            if robot_waypoints.get(rid):
                continue   # already moving (was relocated earlier)
            # Skip if already at (or very close to) a rest node
            current_node = coordinator.graph.get_nearest_node(rs['position'])
            if current_node in REST_NODES:
                # Make sure the static reservation is set
                if coordinator.get_home_node(rid) != current_node:
                    coordinator.lifelong.reserve_static(rid, current_node)
                continue
            # Find nearest collision-free, unreserved rest node
            result = coordinator.find_nearest_rest_node(
                rs['position'], exclude_robot_id=rid)
            if result is None:
                continue
            target_name, target_xy = result
            # Skip if we're already there
            d2 = ((rs['position'][0]-target_xy[0])**2 +
                   (rs['position'][1]-target_xy[1])**2) ** 0.5
            if d2 < 0.4:
                continue
            # Plan path; we use the underlying lifelong planner with
            # node-name target to guarantee it ends on the graph.
            coordinator.release_home(rid)
            start_node = current_node
            node_path = coordinator.lifelong.plan(rid, start_node, target_name)
            if node_path:
                wpts = [_CFG_WAYPOINTS[n] for n in node_path]
                # Drop redundant first waypoint if it equals current pos
                if wpts and (
                        abs(wpts[0][0] - rs['position'][0]) < 0.05 and
                        abs(wpts[0][1] - rs['position'][1]) < 0.05):
                    wpts.pop(0)
                if wpts:
                    robot_waypoints[rid] = wpts
                    rs['state'] = RobotState.RETURNING_HOME
        
        # ----------------------------------------------------------
        # CBS conflict check (lightweight).
        # We invoke CBS only when ≥2 robots are active AND their
        # current path-step destinations would cause a vertex/edge
        # conflict in the next few steps. Otherwise the robots'
        # individual A* paths are kept as-is.
        #
        # This avoids the "oscillation" pathology where CBS would
        # constantly re-plan and reset robots already in motion.
        # ----------------------------------------------------------
        # NOTE: deferred — currently disabled. The simple per-robot
        # A* (which is already collision-free against the static
        # graph) plus the LiDAR DWA reactive layer in robot_controller
        # are sufficient for scenarios A/B. CBS is available via
        # coordinator.replan_all_with_cbs() but invoked manually only.
        # Keeping this comment block as a hook for future activation.
        
        # 4) Periodic logging
        if step % 10000 == 0:
            stats = task_gen.get_statistics()
            avg_batt = sum(robot_states[r]['battery']
                           for r in range(1, num_robots + 1)) / num_robots
            print(f"  T={sim_time:.0f}s: completed={stats['completed']}, "
                  f"pending={stats['pending']}, generated={stats['total_generated']}, "
                  f"avg_battery={avg_batt:.1f}%")
        
        # 5) Periodic step metrics
        if step % 1000 == 0:
            metrics.record_step(
                sim_time, robot_states,
                task_gen.get_statistics(),
                coordinator.get_statistics(),
            )
    
    # ----------------------------------------------------------------
    # Build mock robot objects for metrics
    # ----------------------------------------------------------------
    class MockRobot:
        def __init__(self, rid):
            self.idle_time = robot_idle_time[rid]
            self.total_distance = robot_distance[rid]
            self.tasks_completed = robot_tasks_completed[rid]
            self.battery = robot_states[rid]['battery']
    
    mock_robots = {rid: MockRobot(rid) for rid in range(1, num_robots + 1)}
    
    task_stats = task_gen.get_statistics()
    coord_stats = coordinator.get_statistics()
    metrics.save_results(mock_robots, task_stats, coord_stats, sim_time)
    
    final_metrics = metrics.compute_final_metrics(
        mock_robots, task_stats, coord_stats, sim_time
    )
    metrics.print_summary(final_metrics)


def load_latest_results(scenario: str, scheduler: str, seed: int = None,
                        runtime_mode: str = None,
                        min_mtime: float = None) -> dict:
    """Load the newest result matching this exact experiment invocation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pattern = os.path.join(
        RESULTS_DIR, f"experiment_{scenario}_{scheduler}_*.json")
    candidates = sorted(
        glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for filepath in candidates:
        if min_mtime is not None and os.path.getmtime(filepath) < min_mtime:
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            continue
        info = document.get('experiment_info', {})
        if seed is not None and int(info.get('seed', -1)) != int(seed):
            continue
        if runtime_mode is not None and info.get(
                'runtime_mode') != runtime_mode:
            continue
        return document
    return {}


def run_all_experiments(scenarios: List[str] = None,
                       schedulers: List[str] = None,
                       seeds: List[int] = None,
                       webots_path: str = "webots",
                       model_paths: Dict[str, str] = None,
                       show_webots: bool = False,
                       controller_mode: str = "auto",
                       max_parallel: int = 1):
    """
    Run all experimental combinations.
    """
    if scenarios is None:
        scenarios = SCENARIO_KEYS
    if schedulers is None:
        schedulers = SCHEDULER_TYPES
    if seeds is None:
        seeds = RANDOM_SEEDS[:NUM_REPEATS]
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    model_paths = model_paths or {}
    
    total_experiments = len(scenarios) * len(schedulers) * len(seeds)
    all_results = {
        scenario: {
            scheduler: [None] * len(seeds)
            for scheduler in schedulers}
        for scenario in scenarios}
    
    print(f"\n{'#'*60}")
    print(f"SMART FACTORY MULTI-ROBOT EXPERIMENT SUITE")
    print(f"Scenarios: {scenarios}")
    print(f"Schedulers: {schedulers}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'#'*60}\n")
    
    jobs = []
    ordinal = 0
    for scenario in scenarios:
        for scheduler in schedulers:
            for seed_index, seed in enumerate(seeds):
                ordinal += 1
                jobs.append((ordinal, scenario, scheduler, seed_index, seed))

    def execute(job):
        job_number, scenario, scheduler, seed_index, seed = job
        print(f"\n[{job_number}/{total_experiments}] ", end="")
        result = run_single_experiment(
            scenario, scheduler, seed, webots_path,
            model_path=model_paths.get(scheduler),
            show_webots=show_webots,
            controller_mode=controller_mode,
        )
        return scenario, scheduler, seed_index, result

    if max_parallel == 1:
        completed_jobs = map(execute, jobs)
        for scenario, scheduler, seed_index, result in completed_jobs:
            all_results[scenario][scheduler][seed_index] = result
    else:
        workers = min(max_parallel, total_experiments)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(execute, job) for job in jobs]
            for future in as_completed(futures):
                scenario, scheduler, seed_index, result = future.result()
                all_results[scenario][scheduler][seed_index] = result
    
    # Generate comparison report
    generate_comparison_report(all_results)
    
    return all_results


def experiment_matrix_complete(all_results: Dict) -> bool:
    """Return whether every requested experiment produced usable metrics."""
    if not all_results:
        return False
    saw_result = False
    for scheduler_results in all_results.values():
        if not scheduler_results:
            return False
        for results in scheduler_results.values():
            if not results:
                return False
            for result in results:
                saw_result = True
                if not result or "summary_metrics" not in result:
                    return False
    return saw_result


def generate_comparison_report(all_results: Dict):
    """Generate a comparison table of all experiment results."""
    report_path = os.path.join(RESULTS_DIR, "comparison_report.txt")
    
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENTAL COMPARISON REPORT")
    lines.append("Learning-Based vs Rule-Based Scheduling for Multi-Robot Coordination")
    lines.append("=" * 80)
    
    for scenario in all_results:
        sc = SCENARIOS[scenario]
        lines.append(f"\n{'─'*80}")
        lines.append(f"SCENARIO {scenario}: {sc['description']}")
        lines.append(f"{'─'*80}")
        lines.append(f"{'Scheduler':<20} {'Throughput':>12} {'Avg Compl':>12} "
                     f"{'Avg Wait':>12} {'Idle %':>10} {'Conflicts':>10}")
        lines.append(f"{'':20} {'(tasks/min)':>12} {'(seconds)':>12} "
                     f"{'(seconds)':>12} {'':>10} {'':>10}")
        lines.append("-" * 80)
        
        for scheduler in all_results[scenario]:
            results = all_results[scenario][scheduler]
            
            if not results or not any(r for r in results if r):
                lines.append(f"{scheduler:<20} {'N/A':>12} {'N/A':>12} "
                            f"{'N/A':>12} {'N/A':>10} {'N/A':>10}")
                continue
            
            valid_results = [r for r in results if r and 'summary_metrics' in r]
            
            if not valid_results:
                lines.append(f"{scheduler:<20} {'No data':>12}")
                continue
            
            # Average metrics across seeds
            metrics_list = [r['summary_metrics'] for r in valid_results]
            
            avg_throughput = sum(m['throughput_per_minute'] for m in metrics_list) / len(metrics_list)
            avg_completion = sum(m['avg_task_completion_time'] for m in metrics_list) / len(metrics_list)
            avg_waiting = sum(m['avg_waiting_time'] for m in metrics_list) / len(metrics_list)
            avg_idle = sum(m['avg_robot_idle_pct'] for m in metrics_list) / len(metrics_list)
            avg_conflicts = sum(m['total_conflicts_resolved'] for m in metrics_list) / len(metrics_list)
            
            lines.append(f"{scheduler:<20} {avg_throughput:>12.2f} {avg_completion:>12.2f} "
                        f"{avg_waiting:>12.2f} {avg_idle:>10.1f} {avg_conflicts:>10.1f}")
            throughput_values = [
                m['throughput_per_minute'] for m in metrics_list]
            throughput_std = (
                statistics.stdev(throughput_values)
                if len(throughput_values) > 1 else 0.0)
            lines.append(
                f"  valid seeds={len(valid_results)}/{len(results)}; "
                f"throughput std={throughput_std:.3f}, "
                f"median={statistics.median(throughput_values):.3f}, "
                f"min={min(throughput_values):.3f}, "
                f"max={max(throughput_values):.3f}")
            latency_p95 = [
                m.get('scheduling_latency_p95_ms', 0.0)
                for m in metrics_list]
            latency_p99 = [
                m.get('scheduling_latency_p99_ms', 0.0)
                for m in metrics_list]
            lines.append(
                "  scheduling latency across runs: "
                f"P95 mean={statistics.fmean(latency_p95):.4f} ms, "
                f"P99 mean={statistics.fmean(latency_p99):.4f} ms")
    
    lines.append(f"\n{'='*80}")
    
    report = "\n".join(lines)
    print(report)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")


# ================================================================
# MAIN
# ================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Smart Factory Multi-Robot Coordination Experiments"
    )
    parser.add_argument("--scenario", "-s", nargs="+", default=None,
                       choices=["A", "B", "C"],
                       help="Scenario(s) to run (default: all)")
    parser.add_argument("--scheduler", "-r", nargs="+", default=None,
                       choices=SCHEDULER_TYPES,
                       help="Scheduler(s) to test (default: all)")
    parser.add_argument("--seeds", "-n", type=int, default=NUM_REPEATS,
                       help=f"Number of repetitions (default: {NUM_REPEATS})")
    parser.add_argument(
        "--seed-values", nargs="+", type=int, default=None,
        help=("Explicit evaluation seed values. When provided, this overrides "
              "--seeds and is useful for reproducible one-seed evaluations."))
    parser.add_argument("--webots", "-w", default="webots",
                       help="Path to Webots executable")
    parser.add_argument("--standalone", action="store_true",
                       help="Force standalone mode (no Webots)")
    parser.add_argument(
        "--show-webots", action="store_true",
        help="Show the Webots 3D window instead of using --no-rendering")
    parser.add_argument(
        "--controller-mode", choices=("auto", "internal", "external"),
        default="auto",
        help=("How Python controllers connect to Webots. 'auto' uses external "
              "TCP mode when the host blocks Webots child processes."))
    parser.add_argument(
        "--results-dir",
        help="Isolated directory for experiment JSON and comparison reports")
    parser.add_argument(
        "--max-parallel", type=int, default=1,
        help=("Maximum independent Webots experiments to run concurrently "
              "(default: 1)"))
    parser.add_argument("--sarsa-model")
    parser.add_argument("--dqn-model")
    parser.add_argument("--ppo-model")
    parser.add_argument("--learned-cost-model")
    parser.add_argument("--graph-imitation-model")
    parser.add_argument("--graph-ppo-model")
    parser.add_argument("--rainbow-dqn-model")
    parser.add_argument("--qrdqn-model")
    parser.add_argument("--cql-model")
    parser.add_argument("--linucb-model")
    
    args = parser.parse_args()

    if args.results_dir:
        resolved_results = os.path.abspath(args.results_dir)
        os.environ["SMART_FACTORY_RESULTS_DIR"] = resolved_results
        global RESULTS_DIR
        RESULTS_DIR = resolved_results
    
    seeds = (args.seed_values if args.seed_values is not None
             else RANDOM_SEEDS[:args.seeds])
    
    if args.standalone:
        webots_path = "nonexistent"  # Force standalone
    else:
        webots_path = args.webots
    
    all_results = run_all_experiments(
        scenarios=args.scenario,
        schedulers=args.scheduler,
        seeds=seeds,
        webots_path=webots_path,
        show_webots=args.show_webots,
        controller_mode=args.controller_mode,
        max_parallel=args.max_parallel,
        model_paths={
            name: path for name, path in (
                ("SARSA", args.sarsa_model),
                ("DQN", args.dqn_model),
                ("PPO_RL", args.ppo_model),
                ("LearnedHungarian", args.learned_cost_model),
                ("GraphImitation", args.graph_imitation_model),
                ("GraphPPO", args.graph_ppo_model),
                ("RainbowDQN", args.rainbow_dqn_model),
                ("QRDQN", args.qrdqn_model),
                ("CQL", args.cql_model),
                ("LinUCB", args.linucb_model),
            ) if path
        },
    )
    if not experiment_matrix_complete(all_results):
        print("ERROR: one or more experiments did not produce usable results.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
