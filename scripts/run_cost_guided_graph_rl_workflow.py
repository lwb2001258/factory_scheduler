"""Resume-safe CostGraphPPO/GraphRainbow training and Webots workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "controllers/factory_supervisor"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CONTROLLER) not in sys.path:
    sys.path.insert(0, str(CONTROLLER))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cost_graph_ppo_model import (
    CostGraphPPOConfig, CostGraphPPOModel, CostGraphPPOProvenance,
)
from cost_graph_ppo_training import train_cost_graph_ppo
from cost_guided_graph import CostPriorIdentity
from dual_objective import CountRewardProfile, UtilityV2RewardProfile
from graph_rainbow_model import (
    GraphRainbowConfig, GraphRainbowModel, GraphRainbowProvenance,
)
from graph_rainbow_training import (
    GraphRainbowReplayBuffer, graph_rainbow_replay_identity,
    train_graph_rainbow,
)
from headless_training_runtime import HeadlessRuntimeConfig
from rl_environment import RLEnvironmentConfig
import cost_guided_graph_rl_workflow_contract as contract


CALIBRATION = ROOT / "config/formal_1800/calibration.json"
CALIBRATION_SHA256 = (
    "1d7c3a1b418b52af4aeeb1f131c263e661c7d5129e4ce5d8022eb3c548577055")
HORIZON_SECONDS = 1800.0
STANDALONE_PLAN_VERSION = "cost-guided-standalone-plan-v1"
EXECUTION_REVIEW_VERSION = "cost-guided-execution-review-v1"
POLICY_MODEL_SEEDS = {
    "CostGraphPPO_count": 81001,
    "CostGraphPPO_utility_v2": 81002,
    "GraphRainbow_count": 82001,
    "GraphRainbow_utility_v2": 82002,
}


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def utility_profile(path=CALIBRATION) -> UtilityV2RewardProfile:
    document = load_json(path)
    if (not isinstance(document, dict) or
            contract.sha256(path) != CALIBRATION_SHA256):
        raise ValueError("formal calibration identity differs")
    raw = document.get("utility_reward_profile")
    if not isinstance(raw, dict):
        raise ValueError("utility reward profile is missing")
    allowed = {item.name for item in fields(UtilityV2RewardProfile)}
    profile = UtilityV2RewardProfile(**{
        key: value for key, value in raw.items() if key in allowed})
    if (profile.sha256 != document.get(
            "utility_reward_profile_sha256") or
            profile.horizon_seconds != HORIZON_SECONDS):
        raise ValueError("utility reward profile hash differs")
    return profile


def reward_profile(objective: str):
    if objective == "count":
        return CountRewardProfile(horizon_seconds=HORIZON_SECONDS)
    if objective == "utility_v2":
        return utility_profile()
    raise ValueError("unknown standalone objective")


def model_config(policy_id: str):
    if policy_id.startswith("CostGraphPPO_"):
        return CostGraphPPOConfig(
            hidden_size=64, update_epochs=2,
            learning_rate=0.0003, prior_logit_scale=1.0)
    if policy_id == "GraphRainbow_count":
        return GraphRainbowConfig(
            hidden_size=64, atom_count=51, v_min=0.0, v_max=100.0,
            learning_rate=0.0003, n_step=3,
            replay_capacity=5000, batch_size=8,
            target_update_interval=100, prior_q_scale=1.0)
    if policy_id == "GraphRainbow_utility_v2":
        return GraphRainbowConfig(
            hidden_size=64, atom_count=51, v_min=-500.0, v_max=500.0,
            learning_rate=0.0003, n_step=3,
            replay_capacity=5000, batch_size=8,
            target_update_interval=100, prior_q_scale=1.0)
    raise ValueError(f"unknown cost-guided policy: {policy_id}")


def standalone_plan(workflow: dict) -> dict:
    policies = {}
    for policy_id in contract.POLICIES:
        mapping = contract.TEACHER_MAPPING[policy_id]
        profile = reward_profile(mapping["objective"])
        config = model_config(policy_id)
        policies[policy_id] = {
            **mapping,
            "model_seed": POLICY_MODEL_SEEDS[policy_id],
            "model_config": asdict(config),
            "reward_profile": profile.canonical(),
            "reward_profile_version": profile.version,
            "reward_profile_sha256": profile.sha256,
            "prior_bundle_sha256":
                workflow["teacher_bundles"][
                    mapping["primary"]]["bundle_sha256"],
            "secondary_bundle_sha256":
                workflow["teacher_bundles"][
                    mapping["secondary"]]["bundle_sha256"],
        }
    return {
        "version": STANDALONE_PLAN_VERSION,
        "workflow_version": workflow["version"],
        "workflow_source_selection_sha256":
            workflow["source_selection_sha256"],
        "calibration_path": str(CALIBRATION),
        "calibration_sha256": CALIBRATION_SHA256,
        "horizon_seconds": HORIZON_SECONDS,
        "episode_order": "policy_then_seed_then_scenario",
        "standalone_seeds": list(contract.STANDALONE_SEEDS),
        "scenarios": list(contract.SCENARIOS),
        "policies": policies,
    }


def _copy_atomic(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".",
            suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _policy_directory(root, policy_id):
    return Path(root) / "standalone" / policy_id


def _state_path(root, policy_id):
    return _policy_directory(root, policy_id) / "state.json"


def _validate_state(root, policy_id, state, plan):
    expected = plan["policies"][policy_id]
    expected_keys = {
        _episode_key(seed, scenario)
        for seed in contract.STANDALONE_SEEDS
        for scenario in contract.SCENARIOS}
    if (not isinstance(state, dict) or
            state.get("version") != STANDALONE_PLAN_VERSION or
            state.get("policy_id") != policy_id or
            state.get("objective") != expected["objective"] or
            state.get("reward_profile_sha256") !=
            expected["reward_profile_sha256"] or
            state.get("completed") != sorted(
                state.get("completed", [])) or
            len(state.get("completed", [])) !=
            len(set(state.get("completed", []))) or
            not set(state.get("completed", [])).issubset(expected_keys) or
            len(state.get("episode_reports", [])) !=
            len(state.get("completed", [])) or
            state.get("generation") != len(state.get("completed", [])) or
            state.get("status") not in {"pending", "running", "complete"}):
        raise ValueError(f"standalone state identity differs: {policy_id}")
    checkpoint = Path(state["checkpoint"])
    if (not checkpoint.is_file() or
            contract.sha256(checkpoint) != state["checkpoint_sha256"]):
        raise ValueError(f"standalone checkpoint differs: {policy_id}")
    if policy_id.startswith("GraphRainbow_"):
        replay = Path(state["replay"])
        if (not replay.is_file() or
                contract.sha256(replay) != state["replay_sha256"]):
            raise ValueError(f"standalone replay differs: {policy_id}")
    if state.get("status") == "complete":
        final = Path(state.get("final_checkpoint", ""))
        if (set(state["completed"]) != expected_keys or
                not final.is_file() or
                contract.sha256(final) !=
                state.get("final_checkpoint_sha256")):
            raise ValueError(
                f"standalone final checkpoint differs: {policy_id}")
        if policy_id.startswith("GraphRainbow_"):
            final_replay = Path(state.get("final_replay", ""))
            if (not final_replay.is_file() or
                    contract.sha256(final_replay) !=
                    state.get("final_replay_sha256")):
                raise ValueError(
                    f"standalone final replay differs: {policy_id}")


def _new_policy_state(root, policy_id, workflow, plan):
    directory = _policy_directory(root, policy_id)
    directory.mkdir(parents=True, exist_ok=True)
    row = plan["policies"][policy_id]
    profile = reward_profile(row["objective"])
    identity = CostPriorIdentity.from_selected_bundles(
        policy_id, workflow["teacher_bundles"])
    checkpoint = directory / "generations/0000_model.npz"
    if policy_id.startswith("CostGraphPPO_"):
        provenance = CostGraphPPOProvenance(
            row["objective"], profile.version, profile.sha256,
            "standalone", tuple(contract.STANDALONE_SEEDS))
        model = CostGraphPPOModel(
            identity, provenance, model_config(policy_id),
            POLICY_MODEL_SEEDS[policy_id])
        model.save(checkpoint)
        replay_path = replay_hash = None
    else:
        provenance = GraphRainbowProvenance(
            row["objective"], profile.version, profile.sha256,
            "standalone", tuple(contract.STANDALONE_SEEDS))
        model = GraphRainbowModel(
            identity, provenance, model_config(policy_id),
            POLICY_MODEL_SEEDS[policy_id])
        model.save(checkpoint)
        replay = GraphRainbowReplayBuffer(
            model.config.replay_capacity,
            alpha=model.config.per_alpha,
            priority_epsilon=model.config.priority_epsilon,
            prior_identity_sha256=identity.sha256,
            training_identity_sha256=
                graph_rainbow_replay_identity(model),
            seed=POLICY_MODEL_SEEDS[policy_id])
        replay_path = directory / "generations/0000_replay.npz"
        replay.save(replay_path)
        replay_hash = contract.sha256(replay_path)
    state = {
        "version": STANDALONE_PLAN_VERSION,
        "policy_id": policy_id,
        "objective": row["objective"],
        "status": "pending",
        "generation": 0,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": contract.sha256(checkpoint),
        "replay": (
            str(replay_path.resolve()) if replay_path else None),
        "replay_sha256": replay_hash,
        "reward_profile_sha256": profile.sha256,
        "completed": [],
        "episode_reports": [],
    }
    contract.write(root, _state_path(root, policy_id), state)
    return state


def prepare_standalone(root=contract.DEFAULT_RUN) -> dict:
    root, workflow = contract.validate_root(root)
    contract.require_reviews(root, 8)
    plan = standalone_plan(workflow)
    plan_path = root / "standalone/plan.json"
    existing = load_json(plan_path)
    if existing is None:
        frozen_calibration = root / "frozen/calibration.json"
        _copy_atomic(CALIBRATION, frozen_calibration)
        if contract.sha256(frozen_calibration) != CALIBRATION_SHA256:
            raise RuntimeError("frozen calibration differs")
        contract.write(root, plan_path, plan)
    elif existing != plan:
        raise ValueError("standalone plan identity differs")
    frozen_calibration = root / "frozen/calibration.json"
    if (not frozen_calibration.is_file() or
            contract.sha256(frozen_calibration) != CALIBRATION_SHA256):
        raise ValueError("frozen standalone calibration identity differs")
    states = {}
    for policy_id in contract.POLICIES:
        state = load_json(_state_path(root, policy_id))
        if state is None:
            state = _new_policy_state(
                root, policy_id, workflow, plan)
        _validate_state(root, policy_id, state, plan)
        states[policy_id] = state
    return {"plan": plan, "states": states}


def _episode_key(seed, scenario):
    return f"{int(seed)}:{scenario}"


def _load_policy(root, policy_id, state, plan):
    row = plan["policies"][policy_id]
    checkpoint = Path(state["checkpoint"])
    if policy_id.startswith("CostGraphPPO_"):
        model = CostGraphPPOModel.load(
            checkpoint, expected_policy_id=policy_id,
            expected_reward_profile_sha256=
                row["reward_profile_sha256"])
        replay = None
    else:
        model = GraphRainbowModel.load(
            checkpoint, expected_policy_id=policy_id,
            expected_reward_profile_sha256=
                row["reward_profile_sha256"])
        replay = GraphRainbowReplayBuffer.load(
            state["replay"],
            expected_prior_identity_sha256=model.prior_identity.sha256,
            expected_training_identity_sha256=
                graph_rainbow_replay_identity(model))
    if (asdict(model.config) != row["model_config"] or
            model.seed != row["model_seed"] or
            model.provenance.training_phase != "standalone" or
            model.provenance.standalone_seeds !=
            tuple(contract.STANDALONE_SEEDS) or
            model.provenance.webots_fine_tune_seeds):
        raise ValueError(
            f"standalone checkpoint plan differs: {policy_id}")
    return model, replay


def _save_generation(root, policy_id, state, model, replay, report):
    generation = int(state["generation"]) + 1
    directory = _policy_directory(root, policy_id) / "generations"
    checkpoint = directory / f"{generation:04d}_model.npz"
    model.save(checkpoint)
    if replay is not None:
        replay_path = directory / f"{generation:04d}_replay.npz"
        replay.save(replay_path)
        replay_hash = contract.sha256(replay_path)
    else:
        replay_path = None
        replay_hash = None
    key = _episode_key(
        report["episode_seeds"][0], report["scenarios"][0])
    state = dict(state)
    state.update({
        "generation": generation,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": contract.sha256(checkpoint),
        "replay": (
            str(replay_path.resolve()) if replay_path else None),
        "replay_sha256": replay_hash,
        "completed": sorted(state["completed"] + [key]),
        "episode_reports": state["episode_reports"] + [
            report["episodes"][0]],
        "status": "running",
        "training_step": model.training_step,
        "training_episodes": model.training_episodes,
    })
    contract.write(root, _state_path(root, policy_id), state)
    return state


def _finalize_policy(root, policy_id, state, model, replay):
    final_dir = Path(root) / "standalone/final"
    final_model = final_dir / f"{policy_id}.npz"
    model.save(final_model)
    final_replay = None
    if replay is not None:
        final_replay = final_dir / f"{policy_id}.replay.npz"
        replay.save(final_replay)
    state = dict(state)
    state.update({
        "status": "complete",
        "final_checkpoint": str(final_model.resolve()),
        "final_checkpoint_sha256": contract.sha256(final_model),
        "final_replay": (
            str(final_replay.resolve()) if final_replay else None),
        "final_replay_sha256": (
            contract.sha256(final_replay) if final_replay else None),
    })
    contract.write(root, _state_path(root, policy_id), state)
    return state


def train_standalone(root=contract.DEFAULT_RUN) -> dict:
    root, _workflow = contract.validate_root(root)
    contract.require_reviews(root, 9)
    prepared = prepare_standalone(root)
    plan = prepared["plan"]
    states = prepared["states"]
    expected_keys = {
        _episode_key(seed, scenario)
        for seed in contract.STANDALONE_SEEDS
        for scenario in contract.SCENARIOS}
    for policy_id in contract.POLICIES:
        state = states[policy_id]
        if state.get("status") == "complete":
            if set(state["completed"]) != expected_keys:
                raise ValueError(
                    f"completed standalone state is incomplete: {policy_id}")
            continue
        model, replay = _load_policy(
            root, policy_id, state, plan)
        profile = reward_profile(
            plan["policies"][policy_id]["objective"])
        for seed in contract.STANDALONE_SEEDS:
            for scenario in contract.SCENARIOS:
                key = _episode_key(seed, scenario)
                if key in state["completed"]:
                    continue
                print(
                    f"[STANDALONE] {policy_id} {key} "
                    f"generation={state['generation'] + 1}",
                    flush=True)
                if policy_id.startswith("CostGraphPPO_"):
                    report = train_cost_graph_ppo(
                        model, (seed,), reward_profile=profile,
                        scenario_ids=(scenario,),
                        env_config=RLEnvironmentConfig(
                            max_robots=8, max_tasks=20,
                            max_steps_per_episode=10000),
                        runtime_config=HeadlessRuntimeConfig(
                            episode_end_time=HORIZON_SECONDS,
                            max_advance_seconds=HORIZON_SECONDS,
                            fixed_horizon=True))
                else:
                    report = train_graph_rainbow(
                        model, replay, (seed,), reward_profile=profile,
                        scenario_ids=(scenario,),
                        env_config=RLEnvironmentConfig(
                            max_robots=8, max_tasks=20,
                            max_steps_per_episode=10000),
                        runtime_config=HeadlessRuntimeConfig(
                            episode_end_time=HORIZON_SECONDS,
                            max_advance_seconds=HORIZON_SECONDS,
                            fixed_horizon=True))
                state = _save_generation(
                    root, policy_id, state, model, replay, report)
        if set(state["completed"]) != expected_keys:
            raise RuntimeError(
                f"standalone training incomplete: {policy_id}")
        state = _finalize_policy(
            root, policy_id, state, model, replay)
        states[policy_id] = state
    report = {
        "version": STANDALONE_PLAN_VERSION,
        "status": "complete",
        "episode_count": sum(
            len(state["completed"]) for state in states.values()),
        "expected_episode_count": (
            len(contract.POLICIES) * len(contract.STANDALONE_SEEDS)
            * len(contract.SCENARIOS)),
        "policies": {
            policy_id: {
                "objective": state["objective"],
                "episode_count": len(state["completed"]),
                "training_step": state.get("training_step", 0),
                "training_episodes": state.get("training_episodes", 0),
                "final_checkpoint": state["final_checkpoint"],
                "final_checkpoint_sha256":
                    state["final_checkpoint_sha256"],
                "final_replay": state.get("final_replay"),
                "final_replay_sha256":
                    state.get("final_replay_sha256"),
                "max_reward_conservation_error": max(
                    row["reward_conservation_error"]
                    for row in state["episode_reports"]),
                "mean_return": sum(
                    row["return"] for row in state["episode_reports"]
                ) / len(state["episode_reports"]),
            }
            for policy_id, state in states.items()
        },
    }
    if report["episode_count"] != report["expected_episode_count"]:
        raise RuntimeError("standalone workflow episode count differs")
    contract.write(root, root / "standalone/report.json", report)
    return report


def execution_review_path(root, phase):
    return Path(root) / phase / "execution_reviews.json"


def record_execution_review(root, phase, *, perspective, files,
                            findings, checks):
    root, _ = contract.validate_root(root)
    if perspective not in contract.REVIEW_PERSPECTIVES:
        raise ValueError("execution review perspective is invalid")
    paths = [Path(path).resolve() for path in files]
    if (not paths or any(
            not path.is_file() or not path.is_relative_to(ROOT)
            for path in paths)):
        raise ValueError("execution review files are invalid")
    hashes = {
        path.relative_to(ROOT).as_posix(): contract.sha256(path)
        for path in sorted(paths)}
    path = execution_review_path(root, phase)
    receipt = load_json(path) or {
        "version": EXECUTION_REVIEW_VERSION,
        "phase": phase, "files": hashes, "reviews": []}
    if receipt.get("files") != hashes:
        receipt = {
            "version": EXECUTION_REVIEW_VERSION,
            "phase": phase, "files": hashes, "reviews": []}
    if any(row["perspective"] == perspective
           for row in receipt["reviews"]):
        raise ValueError("execution review perspective already exists")
    receipt["reviews"].append({
        "round": len(receipt["reviews"]) + 1,
        "perspective": perspective,
        "status": "pass",
        "reviewed_at": contract.now(),
        "findings": list(findings),
        "checks": list(checks),
        "files": hashes,
    })
    contract.write(root, path, receipt)
    return receipt


def require_execution_reviews(root, phase):
    root, _ = contract.validate_root(root)
    receipt = load_json(execution_review_path(root, phase))
    if (not isinstance(receipt, dict) or
            receipt.get("version") != EXECUTION_REVIEW_VERSION or
            len(receipt.get("reviews", [])) !=
            len(contract.REVIEW_PERSPECTIVES) or
            {row.get("perspective") for row in receipt["reviews"]} !=
            set(contract.REVIEW_PERSPECTIVES) or
            any(row.get("status") != "pass"
                for row in receipt["reviews"])):
        raise RuntimeError(f"{phase} requires three execution reviews")
    for relative, expected in receipt["files"].items():
        path = (ROOT / relative).resolve()
        if (not path.is_file() or not path.is_relative_to(ROOT) or
                contract.sha256(path) != expected):
            raise RuntimeError(f"reviewed execution file changed: {relative}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare-standalone", "train-standalone",
                            "status"))
    parser.add_argument("--root", type=Path, default=contract.DEFAULT_RUN)
    args = parser.parse_args(argv)
    if args.command == "prepare-standalone":
        value = prepare_standalone(args.root)
        print(json.dumps({
            "status": "prepared",
            "policies": list(value["states"])}, indent=2))
        return 0
    if args.command == "train-standalone":
        print(json.dumps(
            train_standalone(args.root), indent=2, sort_keys=True))
        return 0
    root, workflow = contract.validate_root(args.root)
    print(json.dumps({
        "workflow": workflow,
        "standalone_plan": load_json(root / "standalone/plan.json"),
        "standalone_report": load_json(root / "standalone/report.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
