"""Webots fine-tuning and same-batch evaluation for cost-guided graph RL.

The runner is deliberately separate from the standalone runner: physical
episodes, manifests, phase-labelled checkpoints, and final evaluation have
their own resumable state and review gates.  Evaluation uses the supervisor's
physical action/reward attribution in deterministic collection mode; no
checkpoint or optimiser state is written during evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "controllers/factory_supervisor"
for value in (ROOT, CONTROLLER, ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cost_graph_ppo_model import (
    CostGraphPPOModel, CostGraphPPOProvenance,
)
from cost_graph_ppo_training import (
    CostGraphPPORolloutStep, update_cost_graph_ppo,
)
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_FEATURE_DIM, COST_GUIDED_GRAPH_SCHEMA_SHA256,
    CostGuidedGraphObservation,
)
from graph_rainbow_model import GraphRainbowModel, GraphRainbowProvenance
from graph_rainbow_training import (
    GraphRainbowNStepAccumulator, GraphRainbowReplayBuffer,
    MAX_REPLAY_ROBOTS, MAX_REPLAY_TASKS,
    graph_rainbow_replay_identity, update_graph_rainbow,
)
from schedulers import CostMatrix
from task_generator import generate_task_manifest, load_task_manifest
from training_scenarios import formal_scenario_config

import cost_guided_graph_rl_workflow_contract as contract
import run_cost_guided_graph_rl_workflow as standalone
import run_webots_1800_workflow as webots_workflow


WEBOTS_WORKFLOW_VERSION = "cost-guided-graph-rl-webots-v1"
FINE_TUNE_PLAN_VERSION = "cost-guided-graph-rl-webots-fine-tune-v1"
EVALUATION_PLAN_VERSION = "cost-guided-graph-rl-webots-evaluation-v1"
LEARNING_RATE_MULTIPLIER = 0.1
DEFAULT_WEBOTS = Path(
    r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe")
SCHEDULER_BY_POLICY = {
    policy_id: (
        "CostGraphPPO" if policy_id.startswith("CostGraphPPO_")
        else "GraphRainbow")
    for policy_id in contract.POLICIES
}
ROLLOUT_KIND_BY_SCHEDULER = {
    "CostGraphPPO": "cost_graph_ppo",
    "GraphRainbow": "graph_rainbow",
}


def _copy_atomic(source: Path, destination: Path) -> None:
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _environment(values: dict[str, str]):
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in values.items()):
        raise ValueError("Webots environment must contain string pairs")
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def physical_reward_profile(objective: str):
    profile = standalone.reward_profile(objective)
    if objective == "utility_v2":
        profile = replace(profile, safety_observation="webots_physical")
        if profile.sha256 != webots_workflow._physical_utility_reward_hash():
            raise RuntimeError("physical Utility reward identity differs")
    return profile


def manifest_path(root, partition: str, scenario: str, seed: int) -> Path:
    return (Path(root) / "webots/manifests" / partition / scenario /
            f"{int(seed)}.json")


def prepare_manifests(root, partition: str, seeds) -> list[Path]:
    root, _workflow = contract.validate_root(root)
    expected = {
        "fine_tune": tuple(contract.WEBOTS_FINE_TUNE_SEEDS),
        "evaluation": tuple(contract.EVALUATION_SEEDS),
    }
    seeds = tuple(int(value) for value in seeds)
    if partition not in expected or seeds != expected[partition]:
        raise ValueError("manifest seed partition differs")
    paths = []
    for seed in seeds:
        for scenario in contract.SCENARIOS:
            document = generate_task_manifest(
                scenario, formal_scenario_config(scenario), seed,
                duration_seconds=standalone.HORIZON_SECONDS)
            path = manifest_path(root, partition, scenario, seed)
            existing = standalone.load_json(path)
            if existing is None:
                contract.write(root, path, document)
            elif existing != document:
                raise ValueError(
                    f"canonical manifest identity differs: {partition}/"
                    f"{scenario}/{seed}")
            if load_task_manifest(path) != document:
                raise RuntimeError("written manifest failed strict reload")
            paths.append(path)
    return paths


def observation_from_payload(payload: dict, *,
                             expected_prior_sha256: str
                             ) -> CostGuidedGraphObservation:
    expected_fields = {
        "feature_schema_sha256", "prior_identity_sha256", "current_time",
        "matrix_shape", "coordinates", "action_features", "action_mask",
        "no_op_index",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("physical observation payload fields differ")
    if (payload["feature_schema_sha256"] !=
            COST_GUIDED_GRAPH_SCHEMA_SHA256 or
            payload["prior_identity_sha256"] != expected_prior_sha256):
        raise ValueError("physical observation identity differs")
    shape = payload["matrix_shape"]
    if (not isinstance(shape, list) or len(shape) != 2 or
            any(isinstance(value, bool) or not isinstance(value, int)
                for value in shape) or
            not 0 < shape[0] <= MAX_REPLAY_ROBOTS or
            not 0 < shape[1] <= MAX_REPLAY_TASKS):
        raise ValueError("physical observation matrix shape is invalid")
    coordinates = np.asarray(payload["coordinates"], dtype=np.int64)
    features = np.asarray(payload["action_features"], dtype=np.float32)
    mask = np.asarray(payload["action_mask"], dtype=bool)
    if (coordinates.ndim != 2 or coordinates.shape[1:] != (2,) or
            features.shape != (
                len(coordinates) + 1, COST_GUIDED_GRAPH_FEATURE_DIM) or
            mask.shape != (len(coordinates) + 1,) or
            payload["no_op_index"] != len(coordinates) or
            not np.all(np.isfinite(features))):
        raise ValueError("physical observation arrays are invalid")
    feasible = np.zeros(tuple(shape), dtype=bool)
    if len(coordinates):
        if (np.any(coordinates < 0) or
                np.any(coordinates[:, 0] >= shape[0]) or
                np.any(coordinates[:, 1] >= shape[1])):
            raise ValueError("physical observation coordinate is out of range")
        feasible[coordinates[:, 0], coordinates[:, 1]] = True
    matrix = CostMatrix(
        tuple(range(shape[0])), tuple(range(shape[1])),
        np.zeros(tuple(shape), dtype=np.float64), feasible,
        np.zeros(tuple(shape), dtype=np.float64))
    return CostGuidedGraphObservation(
        matrix, coordinates, features, mask, int(payload["no_op_index"]),
        expected_prior_sha256, COST_GUIDED_GRAPH_SCHEMA_SHA256,
        float(payload["current_time"]))


def _physical(envelope: dict) -> dict:
    try:
        physical = envelope["result"]["formal_evaluation"][
            "physical_fine_tune"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Webots physical reward payload is missing") from exc
    if not webots_workflow.valid_action_reward_rollout(
            physical, require_shared_supervision=True):
        raise ValueError("Webots physical action/reward payload is invalid")
    return physical


def _validated_rollout_rows(envelope, model, scheduler: str) -> list[dict]:
    physical = _physical(envelope)
    rows = physical["rollout"]
    expected_kind = ROLLOUT_KIND_BY_SCHEDULER[scheduler]
    for index, row in enumerate(rows):
        if (row.get("kind") != expected_kind or
                row.get("done") is not (index == len(rows) - 1)):
            raise ValueError("Webots rollout algorithm/boundary differs")
        observation = observation_from_payload(
            row.get("observation"),
            expected_prior_sha256=model.prior_identity.sha256)
        action = row.get("action_index")
        if (isinstance(action, bool) or not isinstance(action, int) or
                not 0 <= action < observation.action_count or
                not observation.action_mask[action] or
                action == observation.no_op_index):
            raise ValueError("Webots rollout action is invalid")
    expected = float(physical["reward_audit"]["reward_total"])
    actual = math.fsum(float(row["reward"]) for row in rows)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("Webots rollout reward is not conserved")
    return rows


def update_ppo_from_envelope(model: CostGraphPPOModel,
                             envelope: dict) -> dict:
    rows = _validated_rollout_rows(envelope, model, "CostGraphPPO")
    rollout = []
    for index, row in enumerate(rows):
        observation = observation_from_payload(
            row["observation"],
            expected_prior_sha256=model.prior_identity.sha256)
        rollout.append(CostGraphPPORolloutStep(
            observation, int(row["action_index"]),
            float(row["old_log_probability"]), float(row["value"]),
            float(row["reward"]), index == len(rows) - 1, False))
    metrics = update_cost_graph_ppo(
        model, rollout, bootstrap_value=0.0)
    metrics.update({
        "rollout_steps": len(rows),
        "episode_return": math.fsum(float(row["reward"]) for row in rows),
        "reward_conservation_error": abs(
            math.fsum(float(row["reward"]) for row in rows) -
            float(_physical(envelope)["reward_audit"]["reward_total"])),
    })
    return metrics


def update_rainbow_from_envelope(
        model: GraphRainbowModel, replay: GraphRainbowReplayBuffer,
        envelope: dict) -> dict:
    rows = _validated_rollout_rows(envelope, model, "GraphRainbow")
    observations = [observation_from_payload(
        row["observation"],
        expected_prior_sha256=model.prior_identity.sha256)
        for row in rows]
    accumulator = GraphRainbowNStepAccumulator(
        model.config.n_step, model.config.gamma)
    before = model.training_step
    losses = []
    for index, row in enumerate(rows):
        terminal = index == len(rows) - 1
        next_observation = (
            observations[index + 1] if not terminal else observations[index])
        transitions = accumulator.append(
            observations[index], int(row["action_index"]),
            float(row["reward"]), next_observation,
            terminated=terminal, episode_boundary=terminal)
        for transition in transitions:
            replay.add(transition)
            metrics = update_graph_rainbow(model, replay)
            if metrics["mean_loss"] is not None:
                losses.append(float(metrics["mean_loss"]))
    model.training_episodes += 1
    actual = math.fsum(float(row["reward"]) for row in rows)
    expected = float(_physical(envelope)["reward_audit"]["reward_total"])
    return {
        "rollout_steps": len(rows),
        "episode_return": actual,
        "reward_conservation_error": abs(actual - expected),
        "updates": model.training_step - before,
        "mean_loss": statistics.fmean(losses) if losses else None,
        "replay_size": len(replay),
    }


def _load_model(path: Path, policy_id: str, expected_reward_hash: str):
    model_class = (
        CostGraphPPOModel if policy_id.startswith("CostGraphPPO_")
        else GraphRainbowModel)
    return model_class.load(
        path, expected_policy_id=policy_id,
        expected_reward_profile_sha256=expected_reward_hash)


def _transition_to_physical(model, objective: str) -> None:
    physical = physical_reward_profile(objective)
    if model.provenance.training_phase == "standalone":
        if (model.provenance.standalone_seeds !=
                tuple(contract.STANDALONE_SEEDS) or
                model.provenance.webots_fine_tune_seeds):
            raise ValueError("standalone checkpoint provenance differs")
        provenance_class = (
            CostGraphPPOProvenance
            if isinstance(model, CostGraphPPOModel)
            else GraphRainbowProvenance)
        model.provenance = provenance_class(
            objective, physical.version, physical.sha256,
            "webots_fine_tuned", tuple(contract.STANDALONE_SEEDS),
            tuple(contract.WEBOTS_FINE_TUNE_SEEDS))
        tuned = replace(
            model.config,
            learning_rate=(
                model.config.learning_rate * LEARNING_RATE_MULTIPLIER))
        model.config = tuned
        model.optimizer.learning_rate = tuned.learning_rate
        if isinstance(model, CostGraphPPOModel):
            model.network.config = tuned
        else:
            model.online.config = tuned
            model.target.config = tuned
    elif (model.provenance.training_phase != "webots_fine_tuned" or
          model.provenance.reward_profile_version != physical.version or
          model.provenance.reward_profile_sha256 != physical.sha256 or
          model.provenance.webots_fine_tune_seeds !=
          tuple(contract.WEBOTS_FINE_TUNE_SEEDS)):
        raise ValueError("fine-tuned checkpoint provenance differs")


def _run_episode(*, root, canonical_path, webots, scenario, seed,
                 manifest, policy_id, checkpoint, objective,
                 deterministic: bool):
    del root
    scheduler = SCHEDULER_BY_POLICY[policy_id]
    model_class = (
        CostGraphPPOModel if scheduler == "CostGraphPPO"
        else GraphRainbowModel)
    current = model_class.load(checkpoint, expected_policy_id=policy_id)
    environment = {
        "SMART_FACTORY_COST_GUIDED_POLICY_ID": policy_id,
        "SMART_FACTORY_EXPECTED_REWARD_PROFILE_SHA256":
            current.provenance.reward_profile_sha256,
        "SMART_FACTORY_PURE_POLICY_REQUIRED": "1",
        "SMART_FACTORY_COST_GRAPH_RL_DETERMINISTIC": (
            "1" if deterministic else "0"),
    }
    with _environment(environment):
        envelope, cached = webots_workflow.run_or_resume_episode(
            canonical_path=canonical_path, webots=Path(webots),
            scenario=scenario, seed=int(seed), manifest=manifest,
            scheduler=scheduler, model_path=checkpoint,
            fine_tune_objective=objective,
            fine_tune_algorithm=scheduler,
            expected_result_scheduler=policy_id)
    if not webots_workflow.valid_episode(
            envelope, scenario=scenario, seed=int(seed), manifest=manifest,
            input_model_sha256=contract.sha256(checkpoint),
            fine_tune_objective=objective,
            fine_tune_algorithm=scheduler,
            expected_scheduler=policy_id):
        raise RuntimeError("Webots episode failed strict validation")
    _validated_rollout_rows(envelope, current, scheduler)
    return envelope, cached


def _fine_tune_state_path(root, policy_id: str) -> Path:
    return Path(root) / "webots_fine_tune" / policy_id / "state.json"


def _fine_tune_plan(root, standalone_report: dict) -> dict:
    policies = {}
    for policy_id in contract.POLICIES:
        objective = contract.TEACHER_MAPPING[policy_id]["objective"]
        source = Path(
            standalone_report["policies"][policy_id]["final_checkpoint"])
        physical = physical_reward_profile(objective)
        policies[policy_id] = {
            "algorithm": SCHEDULER_BY_POLICY[policy_id],
            "objective": objective,
            "input_checkpoint": str(source.resolve()),
            "input_checkpoint_sha256": contract.sha256(source),
            "standalone_reward_profile_sha256":
                standalone.reward_profile(objective).sha256,
            "physical_reward_profile_version": physical.version,
            "physical_reward_profile_sha256": physical.sha256,
            "learning_rate_multiplier": LEARNING_RATE_MULTIPLIER,
        }
    return {
        "version": FINE_TUNE_PLAN_VERSION,
        "workflow_version": WEBOTS_WORKFLOW_VERSION,
        "episode_order": "policy_then_seed_round_then_A_B_C",
        "balanced_round_update": True,
        "seeds": list(contract.WEBOTS_FINE_TUNE_SEEDS),
        "scenarios": list(contract.SCENARIOS),
        "policies": policies,
    }


def _new_fine_tune_state(root, policy_id: str, row: dict) -> dict:
    state = {
        "version": FINE_TUNE_PLAN_VERSION,
        "policy_id": policy_id,
        "algorithm": row["algorithm"],
        "objective": row["objective"],
        "status": "pending",
        "completed_seeds": [],
        "round_reports": [],
        "current_checkpoint": row["input_checkpoint"],
        "current_checkpoint_sha256": row["input_checkpoint_sha256"],
        "current_replay": None,
        "current_replay_sha256": None,
        "input_checkpoint_sha256": row["input_checkpoint_sha256"],
        "physical_reward_profile_sha256":
            row["physical_reward_profile_sha256"],
    }
    contract.write(root, _fine_tune_state_path(root, policy_id), state)
    return state


def _validate_fine_tune_state(state: dict, policy_id: str, row: dict,
                              root: Path) -> None:
    seeds = tuple(state.get("completed_seeds", ())) if isinstance(
        state, dict) else ()
    expected_prefix = tuple(contract.WEBOTS_FINE_TUNE_SEEDS[:len(seeds)])
    if (not isinstance(state, dict) or
            state.get("version") != FINE_TUNE_PLAN_VERSION or
            state.get("policy_id") != policy_id or
            state.get("algorithm") != row["algorithm"] or
            state.get("objective") != row["objective"] or
            state.get("input_checkpoint_sha256") !=
            row["input_checkpoint_sha256"] or
            state.get("physical_reward_profile_sha256") !=
            row["physical_reward_profile_sha256"] or
            seeds != expected_prefix or
            len(state.get("round_reports", ())) != len(seeds) or
            state.get("status") not in {"pending", "running", "complete"}):
        raise ValueError(f"Webots fine-tune state differs: {policy_id}")
    checkpoint = Path(state.get("current_checkpoint", ""))
    if (not checkpoint.is_file() or
            contract.sha256(checkpoint) !=
            state.get("current_checkpoint_sha256")):
        raise ValueError(f"fine-tune checkpoint pointer differs: {policy_id}")
    replay_value = state.get("current_replay")
    if policy_id.startswith("GraphRainbow_") and seeds:
        replay = Path(replay_value or "")
        if (not replay.is_file() or contract.sha256(replay) !=
                state.get("current_replay_sha256")):
            raise ValueError(f"fine-tune replay pointer differs: {policy_id}")
    elif replay_value is not None or state.get("current_replay_sha256") is not None:
        raise ValueError(f"unexpected fine-tune replay: {policy_id}")
    for round_index, (seed, report) in enumerate(
            zip(seeds, state["round_reports"]), start=1):
        episode_paths = [
            Path(root) / "webots_fine_tune" / policy_id /
            "episodes" / f"seed_{seed}" / scenario / "result.json"
            for scenario in contract.SCENARIOS]
        output = Path(report.get("output_checkpoint", ""))
        if (report.get("version") != FINE_TUNE_PLAN_VERSION or
                report.get("policy_id") != policy_id or
                report.get("round") != round_index or
                report.get("seed") != seed or
                report.get("scenarios") != list(contract.SCENARIOS) or
                not output.is_file() or contract.sha256(output) !=
                report.get("output_checkpoint_sha256") or
                report.get("episode_result_sha256") != [
                    contract.sha256(path) if path.is_file() else None
                    for path in episode_paths] or
                float(report.get(
                    "max_reward_conservation_error", math.inf)) > 1e-9):
            raise ValueError(f"fine-tune round differs: {policy_id}/{seed}")
        if policy_id.startswith("GraphRainbow_"):
            output_replay = Path(report.get("output_replay", ""))
            if (not output_replay.is_file() or
                    contract.sha256(output_replay) !=
                    report.get("output_replay_sha256")):
                raise ValueError(
                    f"fine-tune round replay differs: {policy_id}/{seed}")
    if state.get("status") == "complete":
        final = Path(state.get("final_checkpoint", ""))
        if (seeds != tuple(contract.WEBOTS_FINE_TUNE_SEEDS) or
                not final.is_file() or contract.sha256(final) !=
                state.get("final_checkpoint_sha256")):
            raise ValueError(f"final fine-tune checkpoint differs: {policy_id}")
        if policy_id.startswith("GraphRainbow_"):
            final_replay = Path(state.get("final_replay", ""))
            if (not final_replay.is_file() or
                    contract.sha256(final_replay) !=
                    state.get("final_replay_sha256")):
                raise ValueError(
                    f"final fine-tune replay differs: {policy_id}")
        elif (state.get("final_replay") is not None or
              state.get("final_replay_sha256") is not None):
            raise ValueError(f"unexpected final replay: {policy_id}")


def prepare_fine_tune(root=contract.DEFAULT_RUN) -> dict:
    root, _workflow = contract.validate_root(root)
    contract.require_reviews(root, 9)
    standalone.require_execution_reviews(root, "standalone")
    standalone_report = standalone.load_json(root / "standalone/report.json")
    if (not isinstance(standalone_report, dict) or
            standalone_report.get("status") != "complete" or
            standalone_report.get("episode_count") != 72):
        raise ValueError("standalone training report is incomplete")
    manifests = prepare_manifests(
        root, "fine_tune", contract.WEBOTS_FINE_TUNE_SEEDS)
    plan = _fine_tune_plan(root, standalone_report)
    plan_path = root / "webots_fine_tune/plan.json"
    existing = standalone.load_json(plan_path)
    if existing is None:
        contract.write(root, plan_path, plan)
    elif existing != plan:
        raise ValueError("Webots fine-tune plan identity differs")
    states = {}
    for policy_id, row in plan["policies"].items():
        state = standalone.load_json(_fine_tune_state_path(root, policy_id))
        if state is None:
            state = _new_fine_tune_state(root, policy_id, row)
        _validate_fine_tune_state(state, policy_id, row, root)
        states[policy_id] = state
    return {"plan": plan, "states": states, "manifests": manifests}


def _load_fine_tune_model_and_replay(policy_id, state, row):
    checkpoint = Path(state["current_checkpoint"])
    expected_hash = (
        row["standalone_reward_profile_sha256"]
        if not state["completed_seeds"] else
        row["physical_reward_profile_sha256"])
    model = _load_model(checkpoint, policy_id, expected_hash)
    if not state["completed_seeds"]:
        if model.provenance.training_phase != "standalone":
            raise ValueError("first Webots round requires standalone model")
        replay = None
    else:
        if model.provenance.training_phase != "webots_fine_tuned":
            raise ValueError("resumed Webots round requires fine-tuned model")
        expected_learning_rate = (
            standalone.model_config(policy_id).learning_rate *
            LEARNING_RATE_MULTIPLIER)
        if not math.isclose(
                model.config.learning_rate, expected_learning_rate,
                rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("fine-tuned learning rate differs")
        if policy_id.startswith("GraphRainbow_"):
            replay = GraphRainbowReplayBuffer.load(
                state["current_replay"],
                expected_prior_identity_sha256=model.prior_identity.sha256,
                expected_training_identity_sha256=
                    graph_rainbow_replay_identity(model))
        else:
            replay = None
    return model, replay


def _new_rainbow_replay(model: GraphRainbowModel) -> GraphRainbowReplayBuffer:
    return GraphRainbowReplayBuffer(
        model.config.replay_capacity, alpha=model.config.per_alpha,
        priority_epsilon=model.config.priority_epsilon,
        prior_identity_sha256=model.prior_identity.sha256,
        training_identity_sha256=graph_rainbow_replay_identity(model),
        seed=model.seed + 73000)


def _save_fine_tune_round(root, policy_id, state, model, replay,
                          seed, input_checkpoint, envelopes, updates):
    round_index = len(state["completed_seeds"]) + 1
    directory = Path(root) / "webots_fine_tune" / policy_id / "generations"
    checkpoint = directory / f"round_{round_index:02d}_model.npz"
    model.save(checkpoint)
    replay_path = None
    if replay is not None:
        replay_path = directory / f"round_{round_index:02d}_replay.npz"
        replay.save(replay_path)
    episode_paths = [
        Path(root) / "webots_fine_tune" / policy_id / "episodes" /
        f"seed_{seed}" / scenario / "result.json"
        for scenario in contract.SCENARIOS]
    record = {
        "version": FINE_TUNE_PLAN_VERSION,
        "policy_id": policy_id,
        "round": round_index,
        "seed": int(seed),
        "scenarios": list(contract.SCENARIOS),
        "input_checkpoint_sha256": contract.sha256(input_checkpoint),
        "output_checkpoint": str(checkpoint.resolve()),
        "output_checkpoint_sha256": contract.sha256(checkpoint),
        "output_replay": str(replay_path.resolve()) if replay_path else None,
        "output_replay_sha256": (
            contract.sha256(replay_path) if replay_path else None),
        "episode_result_sha256": [
            contract.sha256(path) for path in episode_paths],
        "updates": updates,
        "max_reward_conservation_error": max(
            float(row["reward_conservation_error"]) for row in updates),
        "training_step": model.training_step,
        "training_episodes": model.training_episodes,
    }
    round_path = directory / f"round_{round_index:02d}.json"
    contract.write(root, round_path, record)
    state = dict(state)
    state.update({
        "status": "running",
        "completed_seeds": state["completed_seeds"] + [int(seed)],
        "round_reports": state["round_reports"] + [record],
        "current_checkpoint": str(checkpoint.resolve()),
        "current_checkpoint_sha256": contract.sha256(checkpoint),
        "current_replay": (
            str(replay_path.resolve()) if replay_path else None),
        "current_replay_sha256": (
            contract.sha256(replay_path) if replay_path else None),
        "training_step": model.training_step,
        "training_episodes": model.training_episodes,
    })
    contract.write(root, _fine_tune_state_path(root, policy_id), state)
    return state


def _finalize_fine_tune_policy(root, policy_id, state, model, replay):
    final_dir = Path(root) / "webots_fine_tune/final"
    checkpoint = final_dir / f"{policy_id}.npz"
    _copy_atomic(Path(state["current_checkpoint"]), checkpoint)
    final_replay = None
    if replay is not None:
        final_replay = final_dir / f"{policy_id}.replay.npz"
        _copy_atomic(Path(state["current_replay"]), final_replay)
    state = dict(state)
    state.update({
        "status": "complete",
        "final_checkpoint": str(checkpoint.resolve()),
        "final_checkpoint_sha256": contract.sha256(checkpoint),
        "final_replay": (
            str(final_replay.resolve()) if final_replay else None),
        "final_replay_sha256": (
            contract.sha256(final_replay) if final_replay else None),
    })
    contract.write(root, _fine_tune_state_path(root, policy_id), state)
    return state


def fine_tune_webots(root=contract.DEFAULT_RUN, *,
                     webots=DEFAULT_WEBOTS) -> dict:
    root, _workflow = contract.validate_root(root)
    contract.require_reviews(root, 10)
    prepared = prepare_fine_tune(root)
    plan, states = prepared["plan"], prepared["states"]
    webots_workflow.validate_webots(Path(webots))
    for policy_id in contract.POLICIES:
        row, state = plan["policies"][policy_id], states[policy_id]
        if state["status"] == "complete":
            continue
        model, replay = _load_fine_tune_model_and_replay(
            policy_id, state, row)
        for seed in contract.WEBOTS_FINE_TUNE_SEEDS:
            if seed in state["completed_seeds"]:
                continue
            input_checkpoint = Path(state["current_checkpoint"])
            envelopes = []
            for scenario in contract.SCENARIOS:
                canonical = (
                    root / "webots_fine_tune" / policy_id / "episodes" /
                    f"seed_{seed}" / scenario / "result.json")
                envelope, _cached = _run_episode(
                    root=root, canonical_path=canonical, webots=webots,
                    scenario=scenario, seed=seed,
                    manifest=manifest_path(
                        root, "fine_tune", scenario, seed),
                    policy_id=policy_id, checkpoint=input_checkpoint,
                    objective=row["objective"], deterministic=False)
                envelopes.append(envelope)
            _transition_to_physical(model, row["objective"])
            if (policy_id.startswith("GraphRainbow_") and replay is None):
                replay = _new_rainbow_replay(model)
            updates = []
            for envelope in envelopes:
                if isinstance(model, CostGraphPPOModel):
                    updates.append(update_ppo_from_envelope(model, envelope))
                else:
                    updates.append(update_rainbow_from_envelope(
                        model, replay, envelope))
            if max(row["reward_conservation_error"] for row in updates) > 1e-9:
                raise RuntimeError("Webots fine-tune reward conservation failed")
            state = _save_fine_tune_round(
                root, policy_id, state, model, replay, seed,
                input_checkpoint, envelopes, updates)
        state = _finalize_fine_tune_policy(
            root, policy_id, state, model, replay)
        states[policy_id] = state
    report = {
        "version": FINE_TUNE_PLAN_VERSION,
        "status": "complete",
        "episode_count": sum(
            len(state["completed_seeds"]) * len(contract.SCENARIOS)
            for state in states.values()),
        "expected_episode_count": (
            len(contract.POLICIES) *
            len(contract.WEBOTS_FINE_TUNE_SEEDS) *
            len(contract.SCENARIOS)),
        "policies": {
            policy_id: {
                "algorithm": state["algorithm"],
                "objective": state["objective"],
                "completed_seeds": state["completed_seeds"],
                "physical_episode_count": (
                    len(state["completed_seeds"]) * len(contract.SCENARIOS)),
                "training_step": state["training_step"],
                "training_episodes": state["training_episodes"],
                "final_checkpoint": state["final_checkpoint"],
                "final_checkpoint_sha256":
                    state["final_checkpoint_sha256"],
                "final_replay": state.get("final_replay"),
                "final_replay_sha256": state.get("final_replay_sha256"),
                "max_reward_conservation_error": max(
                    round_row["max_reward_conservation_error"]
                    for round_row in state["round_reports"]),
            }
            for policy_id, state in states.items()
        },
    }
    if report["episode_count"] != report["expected_episode_count"]:
        raise RuntimeError("Webots fine-tune episode count differs")
    contract.write(root, root / "webots_fine_tune/report.json", report)
    return report


def _evaluation_plan(root, fine_report: dict) -> dict:
    policies = {}
    for policy_id in contract.POLICIES:
        row = fine_report["policies"][policy_id]
        checkpoint = Path(row["final_checkpoint"])
        objective = contract.TEACHER_MAPPING[policy_id]["objective"]
        profile = physical_reward_profile(objective)
        model = _load_model(checkpoint, policy_id, profile.sha256)
        if (model.provenance.training_phase != "webots_fine_tuned" or
                model.provenance.webots_fine_tune_seeds !=
                tuple(contract.WEBOTS_FINE_TUNE_SEEDS)):
            raise ValueError("evaluation checkpoint provenance differs")
        policies[policy_id] = {
            "algorithm": SCHEDULER_BY_POLICY[policy_id],
            "objective": objective,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": contract.sha256(checkpoint),
            "prior_identity_sha256": model.prior_identity.sha256,
            "reward_profile_version": profile.version,
            "reward_profile_sha256": profile.sha256,
            "training_step": model.training_step,
            "training_episodes": model.training_episodes,
        }
    return {
        "version": EVALUATION_PLAN_VERSION,
        "workflow_version": WEBOTS_WORKFLOW_VERSION,
        "episode_order": "policy_then_seed_then_scenario",
        "deterministic_policy": True,
        "physical_reward_audit": True,
        "gradient_updates": False,
        "common_seed_batch": list(contract.EVALUATION_SEEDS),
        "scenarios": list(contract.SCENARIOS),
        "expected_episode_count": (
            len(contract.POLICIES) * len(contract.EVALUATION_SEEDS) *
            len(contract.SCENARIOS)),
        "policies": policies,
    }


def prepare_evaluation(root=contract.DEFAULT_RUN) -> dict:
    root, _workflow = contract.validate_root(root)
    contract.require_reviews(root, 10)
    standalone.require_execution_reviews(root, "webots_fine_tune")
    fine_report = standalone.load_json(root / "webots_fine_tune/report.json")
    if (not isinstance(fine_report, dict) or
            fine_report.get("status") != "complete" or
            fine_report.get("episode_count") !=
            fine_report.get("expected_episode_count")):
        raise ValueError("Webots fine-tune report is incomplete")
    manifests = prepare_manifests(
        root, "evaluation", contract.EVALUATION_SEEDS)
    plan = _evaluation_plan(root, fine_report)
    path = root / "evaluation/plan.json"
    existing = standalone.load_json(path)
    if existing is None:
        contract.write(root, path, plan)
    elif existing != plan:
        raise ValueError("Webots evaluation plan identity differs")
    state_path = root / "evaluation/state.json"
    state = standalone.load_json(state_path)
    if state is None:
        state = {
            "version": EVALUATION_PLAN_VERSION,
            "status": "pending",
            "completed": [],
            "episodes": [],
        }
        contract.write(root, state_path, state)
    expected = {
        f"{policy_id}:{seed}:{scenario}"
        for policy_id in contract.POLICIES
        for seed in contract.EVALUATION_SEEDS
        for scenario in contract.SCENARIOS}
    if (not isinstance(state, dict) or
            state.get("version") != EVALUATION_PLAN_VERSION or
            state.get("status") not in {"pending", "running", "complete"} or
            state.get("completed") != sorted(state.get("completed", ())) or
            len(state.get("completed", ())) !=
            len(set(state.get("completed", ()))) or
            not set(state.get("completed", ())).issubset(expected) or
            len(state.get("episodes", ())) != len(state.get("completed", ()))):
        raise ValueError("Webots evaluation state differs")
    rows_by_key = {}
    for row in state["episodes"]:
        if not isinstance(row, dict):
            raise ValueError("Webots evaluation row is invalid")
        key = f"{row.get('policy_id')}:{row.get('seed')}:{row.get('scenario')}"
        if key in rows_by_key or key not in expected:
            raise ValueError("Webots evaluation row key differs")
        result_path = Path(row.get("result_path", ""))
        if (not result_path.is_file() or
                contract.sha256(result_path) != row.get("result_sha256")):
            raise ValueError("Webots evaluation result pointer differs")
        marker = standalone.load_json(result_path)
        marker = (marker or {}).get("cost_guided_evaluation", {})
        plan_row = plan["policies"][row["policy_id"]]
        if (marker.get("version") != EVALUATION_PLAN_VERSION or
                marker.get("deterministic_policy") is not True or
                marker.get("gradient_updates") is not False or
                marker.get("checkpoint_sha256_before") !=
                plan_row["checkpoint_sha256"] or
                marker.get("checkpoint_sha256_after") !=
                plan_row["checkpoint_sha256"]):
            raise ValueError("Webots evaluation result marker differs")
        rows_by_key[key] = row
    if set(rows_by_key) != set(state["completed"]):
        raise ValueError("Webots evaluation completed rows differ")
    return {"plan": plan, "state": state, "manifests": manifests}


def _episode_metrics(envelope, policy_id, plan_row, seed, scenario) -> dict:
    result = envelope["result"]
    summary = result["summary_metrics"]
    formal = result["formal_evaluation"]
    physical = _physical(envelope)
    rewards = [float(row["reward"]) for row in physical["rollout"]]
    reward_total = float(physical["reward_audit"]["reward_total"])
    objective = plan_row["objective"]
    if objective == "count":
        score = float(formal["count_evaluation"]["total_tasks_completed"])
    else:
        score = float(formal["utility_v2_evaluation"][
            "utility_score_deadline_v2"])
    values = {
        "collision_count": summary.get("physical_collision_count"),
        "fallback_count": summary.get("fallback_scheduler_commits"),
        "invalid_count": summary.get("invalid_scheduler_outputs"),
        "timeout_count": summary.get("scheduler_timeout_count"),
        "latency_p99_ms": summary.get("scheduling_latency_p99_ms"),
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("evaluation safety/latency metric is invalid")
    reward_error = abs(math.fsum(rewards) - reward_total)
    gates = {
        "reward_conservation": reward_error <= 1e-9,
        "zero_collision": values["collision_count"] == 0,
        "zero_fallback": values["fallback_count"] == 0,
        "zero_invalid": values["invalid_count"] == 0,
        "zero_timeout": values["timeout_count"] == 0,
        "latency_p99_below_40ms": values["latency_p99_ms"] < 40.0,
        "checkpoint_unchanged": (
            envelope["input_model_sha256"] ==
            plan_row["checkpoint_sha256"]),
    }
    return {
        "policy_id": policy_id,
        "algorithm": plan_row["algorithm"],
        "objective": objective,
        "seed": int(seed),
        "scenario": scenario,
        "score": score,
        "checkpoint_sha256": plan_row["checkpoint_sha256"],
        "manifest_sha256": envelope["manifest_sha256"],
        "reward_profile_sha256": plan_row["reward_profile_sha256"],
        "reward_total": reward_total,
        "reward_conservation_error": reward_error,
        "rollout_step_count": len(rewards),
        **values,
        "configured_scheduler_timeout_seconds":
            webots_workflow.PHYSICAL_FINE_TUNE_TIMEOUT_SECONDS,
        "timeout_margin_ms": (
            webots_workflow.PHYSICAL_FINE_TUNE_TIMEOUT_SECONDS * 1000.0 -
            float(values["latency_p99_ms"])),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _paired_comparisons(rows: list[dict]) -> dict:
    result = {}
    for objective in contract.OBJECTIVES:
        policies = [
            policy_id for policy_id in contract.POLICIES
            if contract.TEACHER_MAPPING[policy_id]["objective"] == objective]
        if len(policies) != 2:
            raise RuntimeError("paired comparison requires two policies")
        by_policy = {
            policy_id: {
                (row["seed"], row["scenario"]): row["score"]
                for row in rows if row["policy_id"] == policy_id}
            for policy_id in policies}
        keys = sorted(set(by_policy[policies[0]]) & set(by_policy[policies[1]]))
        expected = len(contract.EVALUATION_SEEDS) * len(contract.SCENARIOS)
        if len(keys) != expected:
            raise RuntimeError("paired evaluation key set differs")
        differences = [
            by_policy[policies[0]][key] - by_policy[policies[1]][key]
            for key in keys]
        mean = statistics.fmean(differences)
        standard_error = (
            statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1 else 0.0)
        # df=17 because the frozen protocol always has 6 seeds x 3 scenarios.
        half_width = 2.109816 * standard_error
        result[objective] = {
            "left_policy": policies[0],
            "right_policy": policies[1],
            "paired_count": len(differences),
            "difference": "left_minus_right",
            "mean_difference": mean,
            "confidence_interval_95": [
                mean - half_width, mean + half_width],
            "method": "paired_t_interval_df17",
        }
    return result


def evaluate_webots(root=contract.DEFAULT_RUN, *,
                    webots=DEFAULT_WEBOTS) -> dict:
    root, _workflow = contract.validate_root(root)
    contract.require_reviews(root, 11)
    prepared = prepare_evaluation(root)
    plan, state = prepared["plan"], prepared["state"]
    webots_workflow.validate_webots(Path(webots))
    for policy_id in contract.POLICIES:
        row = plan["policies"][policy_id]
        checkpoint = Path(row["checkpoint"])
        for seed in contract.EVALUATION_SEEDS:
            for scenario in contract.SCENARIOS:
                key = f"{policy_id}:{seed}:{scenario}"
                if key in state["completed"]:
                    continue
                before_hash = contract.sha256(checkpoint)
                canonical = (
                    root / "evaluation" / policy_id / "episodes" /
                    f"seed_{seed}" / scenario / "result.json")
                envelope, _cached = _run_episode(
                    root=root, canonical_path=canonical, webots=webots,
                    scenario=scenario, seed=seed,
                    manifest=manifest_path(
                        root, "evaluation", scenario, seed),
                    policy_id=policy_id, checkpoint=checkpoint,
                    objective=row["objective"], deterministic=True)
                after_hash = contract.sha256(checkpoint)
                if before_hash != after_hash or after_hash != row[
                        "checkpoint_sha256"]:
                    raise RuntimeError("evaluation mutated frozen checkpoint")
                marker = {
                    "version": EVALUATION_PLAN_VERSION,
                    "deterministic_policy": True,
                    "physical_reward_audit": True,
                    "gradient_updates": False,
                    "policy_id": policy_id,
                    "prior_identity_sha256": row[
                        "prior_identity_sha256"],
                    "reward_profile_sha256": row[
                        "reward_profile_sha256"],
                    "checkpoint_sha256_before": before_hash,
                    "checkpoint_sha256_after": after_hash,
                }
                if (envelope.get("cost_guided_evaluation") not in
                        (None, marker)):
                    raise ValueError("evaluation marker identity differs")
                envelope["cost_guided_evaluation"] = marker
                contract.write(root, canonical, envelope)
                metrics = _episode_metrics(
                    envelope, policy_id, row, seed, scenario)
                metrics.update({
                    "result_path": str(canonical.resolve()),
                    "result_sha256": contract.sha256(canonical),
                })
                state = dict(state)
                state.update({
                    "status": "running",
                    "completed": sorted(state["completed"] + [key]),
                    "episodes": state["episodes"] + [metrics],
                })
                contract.write(root, root / "evaluation/state.json", state)
    expected = plan["expected_episode_count"]
    if len(state["completed"]) != expected:
        raise RuntimeError("same-batch evaluation is incomplete")
    state = dict(state)
    state["status"] = "complete"
    contract.write(root, root / "evaluation/state.json", state)
    rows = state["episodes"]
    report = {
        "version": EVALUATION_PLAN_VERSION,
        "status": "complete",
        "episode_count": len(rows),
        "expected_episode_count": expected,
        "common_seed_batch": list(contract.EVALUATION_SEEDS),
        "scenarios": list(contract.SCENARIOS),
        "all_formal_gates_passed": all(row["passed"] for row in rows),
        "failed_episode_keys": [
            f"{row['policy_id']}:{row['seed']}:{row['scenario']}"
            for row in rows if not row["passed"]],
        "policies": {
            policy_id: {
                "objective": plan["policies"][policy_id]["objective"],
                "checkpoint_sha256":
                    plan["policies"][policy_id]["checkpoint_sha256"],
                "episode_count": sum(
                    item["policy_id"] == policy_id for item in rows),
                "mean_score": statistics.fmean(
                    item["score"] for item in rows
                    if item["policy_id"] == policy_id),
                "max_reward_conservation_error": max(
                    item["reward_conservation_error"] for item in rows
                    if item["policy_id"] == policy_id),
                "max_latency_p99_ms": max(
                    item["latency_p99_ms"] for item in rows
                    if item["policy_id"] == policy_id),
                "collision_count": sum(
                    item["collision_count"] for item in rows
                    if item["policy_id"] == policy_id),
                "passed": all(
                    item["passed"] for item in rows
                    if item["policy_id"] == policy_id),
            }
            for policy_id in contract.POLICIES
        },
        "paired_comparisons": _paired_comparisons(rows),
        "episodes": rows,
    }
    contract.write(root, root / "evaluation/report.json", report)
    return report


def formal_gate_report(root=contract.DEFAULT_RUN) -> dict:
    root, workflow = contract.validate_root(root)
    contract.require_reviews(root, 12)
    standalone_report = standalone.load_json(root / "standalone/report.json")
    fine_report = standalone.load_json(root / "webots_fine_tune/report.json")
    evaluation_report = standalone.load_json(root / "evaluation/report.json")
    for name, report in (
            ("standalone", standalone_report),
            ("webots_fine_tune", fine_report),
            ("evaluation", evaluation_report)):
        if not isinstance(report, dict) or report.get("status") != "complete":
            raise ValueError(f"{name} report is incomplete")
    report = {
        "version": WEBOTS_WORKFLOW_VERSION,
        "status": (
            "pass" if evaluation_report["all_formal_gates_passed"]
            else "fail"),
        "workflow_sha256": contract.sha256(root / "workflow.json"),
        "frozen_selection_sha256": workflow[
            "frozen_selection_sha256"],
        "calibration_sha256": standalone.CALIBRATION_SHA256,
        "standalone_report_sha256": contract.sha256(
            root / "standalone/report.json"),
        "webots_fine_tune_report_sha256": contract.sha256(
            root / "webots_fine_tune/report.json"),
        "evaluation_report_sha256": contract.sha256(
            root / "evaluation/report.json"),
        "seed_partitions": workflow["seed_partitions"],
        "same_batch_six_seed_evaluation": (
            evaluation_report["common_seed_batch"] ==
            list(contract.EVALUATION_SEEDS)),
        "formal_reward_modified": False,
        "all_formal_gates_passed": evaluation_report[
            "all_formal_gates_passed"],
        "failed_episode_keys": evaluation_report[
            "failed_episode_keys"],
        "policies": evaluation_report["policies"],
        "paired_comparisons": evaluation_report["paired_comparisons"],
    }
    contract.write(root, root / "formal_gate_report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "prepare-fine-tune", "fine-tune", "prepare-evaluation",
        "evaluate", "formal-report", "status"))
    parser.add_argument("--root", type=Path, default=contract.DEFAULT_RUN)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    args = parser.parse_args(argv)
    if args.command == "prepare-fine-tune":
        prepared = prepare_fine_tune(args.root)
        print({"status": "prepared", "policies": list(prepared["states"])})
    elif args.command == "fine-tune":
        print(fine_tune_webots(args.root, webots=args.webots))
    elif args.command == "prepare-evaluation":
        prepared = prepare_evaluation(args.root)
        print({
            "status": "prepared",
            "completed": len(prepared["state"]["completed"])})
    elif args.command == "evaluate":
        print(evaluate_webots(args.root, webots=args.webots))
    elif args.command == "formal-report":
        print(formal_gate_report(args.root))
    else:
        root, workflow = contract.validate_root(args.root)
        print({
            "workflow": workflow,
            "fine_tune": standalone.load_json(
                root / "webots_fine_tune/report.json"),
            "evaluation": standalone.load_json(
                root / "evaluation/report.json"),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
