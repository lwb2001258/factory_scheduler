"""Run deterministic Count/Utility GraphPPO auto-optimization."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "controllers" / "factory_supervisor"
if str(SUPERVISOR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR))

from dual_auto_optimizer import (  # noqa: E402
    AutoTuningContract, GraphPPOSearchSpace, OptimizationCalibration,
    SeedPartitions, SuccessiveHalvingPlan,
    run_dual_auto_optimization_workflow,
)


def build_contract(mode: str, duration: float = None,
                   max_trials: int = None) -> AutoTuningContract:
    if mode == "smoke":
        duration = 1.0 if duration is None else duration
        max_trials = 2 if max_trials is None else max_trials
        training = tuple(range(21000, 21005))
        validation = (31000,)
        halving = SuccessiveHalvingPlan(
            training_episode_budgets=(3, 15),
            validation_seed_budgets=(1, 1),
            reduction_factor=2)
    elif mode == "formal":
        if duration not in (None, 1800, 1800.0):
            raise ValueError("formal mode requires exactly 1800 seconds")
        duration = 1800.0
        max_trials = 12 if max_trials is None else max_trials
        training = tuple(range(21000, 21010))
        validation = tuple(range(31000, 31005))
        halving = SuccessiveHalvingPlan(
            training_episode_budgets=(3, 12, 30),
            validation_seed_budgets=(1, 3, 5),
            reduction_factor=3)
    else:
        raise ValueError("mode must be smoke or formal")
    return AutoTuningContract(
        mode=mode, duration_seconds=duration, max_trials=max_trials,
        sampler_seed=20261002,
        partitions=SeedPartitions(
            training=training, validation=validation,
            webots_finetune=training[:5]),
        search_space=GraphPPOSearchSpace(),
        halving_plan=halving)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--max-trials", type=int)
    args = parser.parse_args()
    contract = build_contract(args.mode, args.duration, args.max_trials)
    calibration = OptimizationCalibration.recommended(
        contract.duration_seconds)
    source_paths = [Path(__file__).resolve(), *SUPERVISOR.glob("*.py")]
    report = run_dual_auto_optimization_workflow(
        contract, args.output_dir, calibration=calibration,
        source_paths=source_paths)
    print(json.dumps({
        "status": report["status"],
        "mode": report["mode"],
        "headless_optimization_complete":
            report["headless_optimization_complete"],
        "count_winner": report["objectives"]["count"][
            "winner_trial_id"],
        "utility_winner": report["objectives"]["utility_v2"][
            "winner_trial_id"],
        "webots_finetune_status": report["webots_finetune"]["status"],
        "report": str(
            (args.output_dir / "workflow_report.json").resolve()),
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
