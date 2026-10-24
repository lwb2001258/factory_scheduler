"""Monitor and resume one atomic standalone model lane without touching Webots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts",):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_strict_1800_workflow as workflow  # noqa: E402
from standalone_model_lane import _pid_alive, status_document  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=workflow.DEFAULT_RUN_DIR)
    parser.add_argument("--policy-id", required=True,
                        choices=workflow.PIPELINE_POLICY_IDS)
    parser.add_argument("--adopt-pid", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=3)
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.max_restarts < 0:
        raise ValueError("invalid monitor limits")
    return args


def _target_status(output_dir: Path, policy_id: str) -> dict:
    return next(
        row for row in status_document(output_dir)["models"]
        if row["policy_id"] == policy_id)


def _global_owns_policy(output_dir: Path, policy_id: str) -> bool:
    state = workflow.load_json(output_dir / "workflow_state.json")
    return bool(
        isinstance(state, dict) and
        policy_id in state.get("active_policy_ids", []))


def _spawn(output_dir: Path, policy_id: str, restart_index: int) -> int:
    lane_dir = workflow._headless_lane_state_path(
        output_dir, policy_id).parent
    lane_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stdout_path = lane_dir / (
        f"standalone.monitor_restart_{restart_index}.{stamp}.stdout.log")
    stderr_path = lane_dir / (
        f"standalone.monitor_restart_{restart_index}.{stamp}.stderr.log")
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NO_WINDOW |
            subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    stdout_stream = stdout_path.open("ab")
    stderr_stream = stderr_path.open("ab")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" /
             "standalone_model_lane.py"), "run", "--output-dir",
             str(output_dir), "--policy-id", policy_id],
            cwd=ROOT, env=environment,
            stdout=stdout_stream, stderr=stderr_stream,
            creationflags=creationflags)
    finally:
        stdout_stream.close()
        stderr_stream.close()
    print(
        f"[STANDALONE RESTART] policy={policy_id} pid={process.pid} "
        f"attempt={restart_index}", flush=True)
    return process.pid


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    monitor_path = workflow._headless_lane_state_path(
        output_dir, args.policy_id).parent / "monitor_state.json"
    tracked_pid = args.adopt_pid
    restarts = 0
    while True:
        target = _target_status(output_dir, args.policy_id)
        workflow.atomic_json(monitor_path, {
            "version": "standalone-lane-monitor-v1",
            "policy_id": args.policy_id,
            "status": target["status"],
            "tracked_pid": tracked_pid,
            "restart_count": restarts,
            "maximum_restarts": args.max_restarts,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        if target["status"] == "completed":
            print(f"[STANDALONE MONITOR COMPLETE] {args.policy_id}",
                  flush=True)
            return 0
        if target["status"] == "in_progress":
            tracked_pid = target["worker_pid"]
            time.sleep(args.poll_seconds)
            continue
        if tracked_pid and _pid_alive(tracked_pid):
            time.sleep(args.poll_seconds)
            continue
        if _global_owns_policy(output_dir, args.policy_id):
            tracked_pid = None
            time.sleep(args.poll_seconds)
            continue
        if restarts >= args.max_restarts:
            workflow.atomic_json(monitor_path, {
                "version": "standalone-lane-monitor-v1",
                "policy_id": args.policy_id,
                "status": "repair_required",
                "tracked_pid": tracked_pid,
                "restart_count": restarts,
                "maximum_restarts": args.max_restarts,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            raise RuntimeError(
                f"standalone restart budget exhausted: {args.policy_id}")
        restarts += 1
        tracked_pid = _spawn(output_dir, args.policy_id, restarts)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
