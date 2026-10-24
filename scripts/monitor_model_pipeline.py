"""Continuously audit tuning groups and resume an interrupted workflow.

The monitor owns only the exact workflow PID that it starts (or an explicitly
adopted PID).  Completed hyperparameter groups are treated as immutable: every
episode result, final checkpoint, and the audit document itself must retain its
committed hash before the workflow is allowed to continue unattended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONTRACT = "hyperparameter-group-audit-v2-six-validation-episodes"
SCENARIOS = ("A", "B", "C")
ACTIVE_WINDOWS_EXIT_CODE = 259
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.025
ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.25
WORKFLOW_STARTUP_GRACE_SECONDS = 60.0
_RETRYABLE_WINDOWS_REPLACE_ERRORS = {5, 32, 33}


def canonical_sha256(document) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate short Windows reader/antivirus locks on state files."""
    delay = ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            retryable = (
                isinstance(error, PermissionError) or
                getattr(error, "winerror", None) in
                _RETRYABLE_WINDOWS_REPLACE_ERRORS)
            if not retryable or attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, ATOMIC_REPLACE_MAX_DELAY_SECONDS)


def atomic_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except (FileNotFoundError, OSError):
            pass


class SingleInstanceLock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)

    def acquire(self) -> bool:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(
                    self.stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def close(self) -> None:
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()


def process_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == ACTIVE_WINDOWS_EXIT_CODE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def should_terminate_stale_workflow(
        *, state_age_seconds: float, observed_process_seconds: float,
        stale_seconds: float) -> bool:
    """Ignore a predecessor's stale state while a new workflow starts.

    A resumed process needs a short window in which to replace the failed
    predecessor state.  Without this guard an adopted process can be killed
    before its Python imports finish merely because the prior state is old.
    """
    startup_grace = min(WORKFLOW_STARTUP_GRACE_SECONDS, stale_seconds)
    return (
        observed_process_seconds >= startup_grace and
        state_age_seconds > stale_seconds
    )


def terminate_process_tree(pid: int) -> None:
    if not process_running(pid):
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        os.kill(pid, 15)


def _contained_file(root: Path, relative_value, expected_sha256) -> bool:
    if not isinstance(relative_value, str) or not isinstance(
            expected_sha256, str):
        return False
    path = (root / relative_value).resolve()
    return bool(
        path.is_relative_to(root) and path.is_file() and
        sha256_file(path) == expected_sha256)


def validate_group_audit(output_dir: Path, path: Path) -> dict:
    output_root = output_dir.resolve()
    resolved_path = path.resolve()
    document = load_json(resolved_path)
    if not isinstance(document, dict):
        raise RuntimeError(f"unreadable group audit: {resolved_path}")
    try:
        relative = resolved_path.relative_to(output_root)
        policy_id = relative.parts[1]
        trial_id = relative.parts[3]
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"misplaced group audit: {resolved_path}") from exc
    unhashed = {
        key: value for key, value in document.items()
        if key != "report_sha256"
    }
    training = document.get("training_evidence")
    validation = document.get("validation_evidence")
    training_seeds = document.get("training_seed_contract")
    validation_seeds = document.get("validation_seed_contract")
    expected_training = [
        (scenario, seed, index)
        for index, (seed, scenario) in enumerate(
            ((seed, scenario) for seed in training_seeds
             for scenario in SCENARIOS), start=1)
    ] if isinstance(training_seeds, list) else []
    expected_validation = [
        (scenario, seed)
        for scenario in SCENARIOS for seed in validation_seeds
    ] if isinstance(validation_seeds, list) else []
    valid = bool(
        document.get("schema_version") == 1 and
        document.get("audit_contract") == AUDIT_CONTRACT and
        document.get("status") == "complete_and_verified" and
        document.get("policy_id") == policy_id and
        document.get("trial_id") == trial_id and
        document.get("duration_seconds_per_episode") == 1800.0 and
        isinstance(training_seeds, list) and len(training_seeds) == 5 and
        isinstance(validation_seeds, list) and len(validation_seeds) == 2 and
        all(isinstance(seed, int) for seed in training_seeds) and
        all(isinstance(seed, int) for seed in validation_seeds) and
        len(set(training_seeds)) == len(training_seeds) and
        len(set(validation_seeds)) == len(validation_seeds) and
        isinstance(training, list) and isinstance(validation, list) and
        document.get("training_episode_count") == len(expected_training) and
        document.get("validation_episode_count") == len(expected_validation) and
        len(training) == len(expected_training) and
        len(validation) == len(expected_validation) and
        document.get("training_evidence_sha256") ==
            canonical_sha256(training) and
        document.get("validation_evidence_sha256") ==
            canonical_sha256(validation) and
        document.get("report_sha256") == canonical_sha256(unhashed))
    if not valid:
        raise RuntimeError(f"group audit contract mismatch: {resolved_path}")
    actual_training = [
        (row.get("scenario"), row.get("seed"), row.get("episode_index"))
        for row in training if isinstance(row, dict)
    ]
    actual_validation = [
        (row.get("scenario"), row.get("seed"))
        for row in validation if isinstance(row, dict)
    ]
    if (actual_training != expected_training or
            actual_validation != expected_validation):
        raise RuntimeError(f"group audit unit order mismatch: {resolved_path}")
    trial_root = resolved_path.parent
    for evidence in (*training, *validation):
        evidence_path = (output_root / evidence["result_path"]).resolve()
        if (not evidence_path.is_relative_to(trial_root) or
                not _contained_file(
                    output_root, evidence["result_path"],
                    evidence["result_file_sha256"])):
            raise RuntimeError(
                f"group audit episode evidence changed: {evidence_path}")
    candidate = document.get("candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError(f"group candidate missing: {resolved_path}")
    checkpoint = Path(str(candidate.get("checkpoint", ""))).resolve()
    identity = checkpoint.with_name(checkpoint.name + ".identity.json")
    if (not checkpoint.is_relative_to(trial_root) or
            not checkpoint.is_file() or
            sha256_file(checkpoint) != candidate.get("checkpoint_sha256") or
            not identity.is_file() or
            sha256_file(identity) !=
                candidate.get("checkpoint_identity_sha256")):
        raise RuntimeError(f"group checkpoint evidence changed: {checkpoint}")
    return {
        "policy_id": policy_id,
        "trial_id": trial_id,
        "report_sha256": document["report_sha256"],
        "training_episode_count": len(training),
        "validation_episode_count": len(validation),
        "validation_metric": candidate.get("validation_metric"),
    }


def scan_group_audits(
        output_dir: Path, known: dict[str, dict] | None = None
        ) -> dict[str, dict]:
    audits = {}
    known = known or {}
    headless = output_dir / "headless"
    if not headless.is_dir():
        return audits
    for path in sorted(headless.glob("*/tuning/trial_*/group_audit.json")):
        relative = path.resolve().relative_to(output_dir.resolve())
        key = f"{relative.parts[1]}/{relative.parts[3]}"
        if key in audits:
            raise RuntimeError(f"duplicate group audit: {key}")
        if key in known:
            document = load_json(path)
            if not isinstance(document, dict):
                raise RuntimeError(f"unreadable group audit: {path}")
            unhashed = {
                name: value for name, value in document.items()
                if name != "report_sha256"
            }
            if (document.get("report_sha256") !=
                    known[key].get("report_sha256") or
                    document.get("report_sha256") !=
                    canonical_sha256(unhashed)):
                raise RuntimeError(f"previously verified group changed: {key}")
            audit = known[key]
        else:
            audit = validate_group_audit(output_dir, path)
        audits[key] = audit
    return audits


def stderr_diagnosis(output_dir: Path) -> str:
    candidates = sorted(
        (path for path in output_dir.glob("workflow*.stderr.log")
         if path.is_file() and path.stat().st_size),
        key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        return "process exited without a committed error"
    try:
        lines = candidates[0].read_text(
            encoding="utf-8", errors="replace").splitlines()[-20:]
    except OSError:
        return "process exited and stderr could not be read"
    return " | ".join(lines)[-4000:]


def failure_signature(workflow_state, diagnosis: str) -> str:
    error = (workflow_state or {}).get("error")
    basis = str(error or diagnosis or "unknown abnormal exit")
    return hashlib.sha256(basis.encode("utf-8", errors="replace")).hexdigest()


def append_history(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(
            document, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def start_workflow(args, restart_number: int) -> tuple[subprocess.Popen, dict]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"workflow.supervised.{restart_number:03d}.{stamp}"
    stdout_path = args.output_dir / f"{prefix}.stdout.log"
    stderr_path = args.output_dir / f"{prefix}.stderr.log"
    command = [
        str(args.python), "-u", str(args.workflow_script),
        "--output-dir", str(args.output_dir),
        "--report-path", str(args.report_path),
    ]
    stdout = stdout_path.open("a", encoding="utf-8", newline="\n")
    stderr = stderr_path.open("a", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stdout, stderr=stderr,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)
                           if os.name == "nt" else 0))
    finally:
        stdout.close()
        stderr.close()
    return process, {
        "command": command,
        "pid": process.pid,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "started_at": timestamp(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workflow-script", type=Path,
        default=ROOT / "scripts" / "run_strict_1800_workflow.py")
    parser.add_argument(
        "--report-path", type=Path,
        default=(ROOT / "docs" /
                 "all_scheduler_model_remediation_training_report_v6.md"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stale-minutes", type=float, default=180.0)
    parser.add_argument("--max-restarts-per-signature", type=int, default=3)
    parser.add_argument("--adopt-pid", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.workflow_script = args.workflow_script.resolve()
    args.report_path = args.report_path.resolve()
    args.python = args.python.resolve()
    if (args.poll_seconds < 1 or args.stale_minutes <= 0 or
            args.max_restarts_per_signature < 0):
        raise ValueError("monitor intervals and restart limit are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = SingleInstanceLock(args.output_dir / "monitor.lock")
    if not lock.acquire():
        print("[MONITOR] another monitor already owns this output directory",
              flush=True)
        return 0
    state_path = args.output_dir / "monitor_state.json"
    history_path = args.output_dir / "monitor_history.jsonl"
    prior = load_json(state_path)
    # A repair-required state records evidence from before the code fix. The
    # next monitor must fully revalidate current evidence instead of treating
    # the repaired audit envelope as an unauthorized live mutation.
    prior_audited = (
        {} if (prior or {}).get("monitor_status") == "repair_required"
        else dict((prior or {}).get("audited_groups", {})))
    # Revalidate all persisted evidence once whenever a new monitor process
    # takes ownership. Later polls only hash the compact audit envelope.
    audited = {}
    first_scan = True
    failure_counts = dict((prior or {}).get("failure_signature_counts", {}))
    restart_number = int((prior or {}).get("restart_count", 0))
    child = None
    pid = args.adopt_pid if process_running(args.adopt_pid) else None
    workflow_observed_at = time.monotonic()
    process_info = ({"pid": pid, "adopted_at": timestamp()}
                    if pid is not None else None)

    def record(action: str, **extra) -> dict:
        document = {
            "schema_version": 1,
            "monitor_status": "running",
            "action": action,
            "recorded_at": timestamp(),
            "workflow_pid": pid,
            "workflow_process": process_info,
            "restart_count": restart_number,
            "audited_group_count": len(audited),
            "audited_groups": audited,
            "failure_signature_counts": failure_counts,
            **extra,
        }
        atomic_json(state_path, document)
        if action != "healthy":
            append_history(history_path, {
                key: value for key, value in document.items()
                if key not in {"audited_groups", "failure_signature_counts"}
            })
        return document

    try:
        while True:
            try:
                current_audits = scan_group_audits(
                    args.output_dir, {} if first_scan else audited)
                previous_audits = prior_audited if first_scan else audited
                for key, previous in previous_audits.items():
                    if (key not in current_audits or
                            current_audits[key]["report_sha256"] !=
                            previous["report_sha256"]):
                        raise RuntimeError(
                            f"previously verified group changed: {key}")
                new_keys = sorted(set(current_audits) - set(previous_audits))
                audited = current_audits
                first_scan = False
                for key in new_keys:
                    record("hyperparameter_group_verified",
                           verified_group=current_audits[key])
                    print(
                        f"[MONITOR GROUP OK] {key} "
                        f"metric={current_audits[key]['validation_metric']}",
                        flush=True)
            except Exception as exc:
                if pid is not None:
                    terminate_process_tree(pid)
                record("repair_required", monitor_status="repair_required",
                       error=f"{type(exc).__name__}: {exc}")
                print(f"[MONITOR REPAIR REQUIRED] {exc}", flush=True)
                return 2

            workflow_state = load_json(args.output_dir / "workflow_state.json")
            if (isinstance(workflow_state, dict) and
                    workflow_state.get("status") == "complete" and
                    workflow_state.get("stage") == "complete"):
                record("workflow_complete", monitor_status="complete",
                       workflow_state=workflow_state)
                print("[MONITOR COMPLETE] workflow and group audits valid",
                      flush=True)
                return 0

            running = (child.poll() is None if child is not None and
                       child.pid == pid else process_running(pid))
            state_file = args.output_dir / "workflow_state.json"
            now = time.time()
            state_age_seconds = (
                now - state_file.stat().st_mtime
                if state_file.is_file() else 0.0)
            stale = bool(running and state_file.is_file() and
                         should_terminate_stale_workflow(
                             state_age_seconds=state_age_seconds,
                             observed_process_seconds=(
                                 time.monotonic() - workflow_observed_at),
                             stale_seconds=args.stale_minutes * 60.0))
            if stale:
                age_minutes = state_age_seconds / 60
                terminate_process_tree(pid)
                running = False
                record("stale_workflow_terminated",
                       state_age_minutes=round(age_minutes, 2))

            if running:
                record("healthy", workflow_state=workflow_state)
                time.sleep(args.poll_seconds)
                continue

            if (isinstance(workflow_state, dict) and
                    workflow_state.get("status") ==
                    "stopped_at_requested_run_limit"):
                record("requested_limit_reached", monitor_status="stopped",
                       workflow_state=workflow_state)
                return 0

            diagnosis = stderr_diagnosis(args.output_dir)
            signature = failure_signature(workflow_state, diagnosis)
            failure_counts[signature] = int(failure_counts.get(signature, 0)) + 1
            if failure_counts[signature] > args.max_restarts_per_signature:
                record(
                    "repair_required", monitor_status="repair_required",
                    failure_signature=signature, diagnosis=diagnosis,
                    workflow_state=workflow_state)
                print(
                    "[MONITOR REPAIR REQUIRED] repeated abnormal exit: "
                    f"{diagnosis}", flush=True)
                return 3

            restart_number += 1
            child, process_info = start_workflow(args, restart_number)
            pid = child.pid
            workflow_observed_at = time.monotonic()
            record(
                "resumed_from_checkpoint", failure_signature=signature,
                diagnosis=diagnosis, workflow_state=workflow_state)
            print(
                f"[MONITOR RESUME] pid={pid} restart={restart_number} "
                f"audited_groups={len(audited)}", flush=True)
            time.sleep(min(args.poll_seconds, 5.0))
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
