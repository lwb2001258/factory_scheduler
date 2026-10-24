"""Run or inspect one isolated standalone model-training lane.

This utility deliberately does not write ``workflow_state.json``.  It uses
the strict workflow's model-specific tuning, checkpoint, identity, and lane
state contracts, so a standalone pretraining process can run beside a Webots
evaluation without confusing their global stages.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_1800_workflow as workflow  # noqa: E402


def _pid_alive(pid) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _completion_valid(output_dir: Path, spec: dict, lane_state) -> bool:
    if isinstance(lane_state, dict) and lane_state.get("status") == "complete":
        try:
            workflow._validate_headless_lane_result(
                output_dir, spec, lane_state.get("result"))
            return True
        except (FileNotFoundError, KeyError, OSError, RuntimeError,
                TypeError, ValueError):
            return False
    report = workflow.load_json(
        output_dir / "headless" / spec["policy_id"] / "training_report.json")
    checkpoint = workflow.headless_final_path(output_dir, spec)
    if not (isinstance(report, dict) and report.get("status") == "complete" and
            report.get("policy_id") == spec["policy_id"] and
            report.get("scheduler") == spec["scheduler"] and
            report.get("objective") == spec["objective"] and
            checkpoint.is_file() and
            report.get("output_checkpoint_sha256") ==
                workflow.sha256_file(checkpoint)):
        return False
    try:
        workflow.validate_model_identity(spec, checkpoint)
    except (FileNotFoundError, KeyError, OSError, RuntimeError,
            TypeError, ValueError):
        return False
    return True


def status_document(output_dir: Path) -> dict:
    rows = []
    for spec in workflow.PIPELINE_AI_SPECS:
        spec = dict(spec)
        policy_id = spec["policy_id"]
        # Bind a completed tuning winner before validating its model identity.
        # This mutates only this process-local spec dictionary.
        try:
            workflow._apply_tuning_selection(output_dir, spec, required=False)
        except (FileNotFoundError, KeyError, OSError, RuntimeError,
                TypeError, ValueError):
            # A corrupt or partial selection is reported below as interrupted;
            # it must not make status collection for every other model fail.
            pass
        state_path = workflow._headless_lane_state_path(output_dir, policy_id)
        lane_state = workflow.load_json(state_path)
        completed = _completion_valid(output_dir, spec, lane_state)
        running = bool(
            not completed and isinstance(lane_state, dict) and
            lane_state.get("status") == "running" and
            _pid_alive(lane_state.get("worker_pid")))
        has_artifacts = bool(
            state_path.is_file() or
            (output_dir / "headless" / policy_id).exists())
        if completed:
            status = "completed"
        elif running:
            status = "in_progress"
        elif has_artifacts:
            status = "interrupted_or_failed"
        else:
            status = "not_started"
        rows.append({
            "pipeline_position": len(rows) + 1,
            "policy_id": policy_id,
            "scheduler": spec["scheduler"],
            "objective": spec["objective"],
            "status": status,
            "worker_pid": (
                lane_state.get("worker_pid")
                if isinstance(lane_state, dict) else None),
            "stage": (
                lane_state.get("stage")
                if isinstance(lane_state, dict) else None),
            "updated_at": (
                lane_state.get("updated_at")
                if isinstance(lane_state, dict) else None),
        })
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "completed", "in_progress", "not_started",
            "interrupted_or_failed")
    }
    return {
        "version": "standalone-model-status-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "counts": counts,
        "models": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("run", "status"),
        help="run exactly one model lane or inspect all model lanes")
    parser.add_argument("--output-dir", type=Path,
                        default=workflow.DEFAULT_RUN_DIR)
    parser.add_argument("--policy-id", choices=workflow.PIPELINE_POLICY_IDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.command == "status":
        print(json.dumps(
            status_document(output_dir), ensure_ascii=False, indent=2))
        return 0
    if not args.policy_id:
        raise ValueError("--policy-id is required for run")
    spec = next(
        dict(row) for row in workflow.PIPELINE_AI_SPECS
        if row["policy_id"] == args.policy_id)
    global_state = workflow.load_json(output_dir / "workflow_state.json")
    active_policy_ids = (
        global_state.get("active_policy_ids", [])
        if isinstance(global_state, dict) else [])
    if args.policy_id in active_policy_ids:
        raise RuntimeError(
            f"global workflow is already processing: {args.policy_id}")
    before = next(
        row for row in status_document(output_dir)["models"]
        if row["policy_id"] == args.policy_id)
    if before["status"] == "completed":
        print(f"[ALREADY COMPLETE] {args.policy_id}", flush=True)
        return 0
    if before["status"] == "in_progress":
        raise RuntimeError(
            f"standalone lane is already running: {args.policy_id} "
            f"pid={before['worker_pid']}")
    workflow._headless_model_lane_worker(output_dir, spec)
    after = next(
        row for row in status_document(output_dir)["models"]
        if row["policy_id"] == args.policy_id)
    if after["status"] != "completed":
        raise RuntimeError(
            f"standalone lane did not commit completion: {args.policy_id}")
    print(f"[STANDALONE COMPLETE] {args.policy_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
