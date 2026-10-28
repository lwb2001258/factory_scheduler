"""Evaluate the current GraphImitation expert without training or fine tuning.

The prepared experiment supplies preserved baseline results and exact manifests.
Steps are gated by three documented code reviews; incomplete attempts are kept.
"""
from contextlib import contextmanager
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'controllers/factory_supervisor'))
import numpy as np
import run_experiments
import run_webots_1800_workflow as physical
from graph_expert_scheduler import expert_descriptor

VERSION = 'graph-expert-evaluation-v1'
OBJECTIVES = ('count', 'utility_v2')
SCENARIOS = ('A', 'B', 'C')
SEEDS = tuple(range(61000, 61006))
DEFAULT_ROOT = ROOT / 'results/graph_imitation_expert_20261027'
DEFAULT_WEBOTS = Path('C:/Program Files/Webots/msys64/mingw64/bin/webots.exe')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def confined(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('output escapes isolated experiment')
    return path


def write(root, path, value):
    physical.atomic_json(confined(root, path), value)


def validate_root(root):
    root = Path(root).resolve()
    if root.parent != (ROOT / 'results').resolve():
        raise ValueError('experiment must be a direct child of results')
    protocol = read(root / 'protocol.json')
    expected = {'version': VERSION, 'training': False, 'fine_tuning': False,
                'regression_validation': False, 'scheduler': 'GraphExpert',
                'objectives': list(OBJECTIVES), 'scenarios': list(SCENARIOS),
                'evaluation_seeds': list(SEEDS), 'duration_seconds': 1800,
                'episodes': 36, 'preserve_existing_artifacts': True,
                'sarsa_remains_frozen': True, 'minimum_review_rounds_per_step': 3}
    if not isinstance(protocol, dict) or any(protocol.get(k) != v for k, v in expected.items()):
        raise RuntimeError('missing or incompatible evaluation-only protocol')
    if root == Path(protocol['baseline_run']).resolve():
        raise RuntimeError('baseline is read-only')
    for name in ('protected_hashes.json', 'baseline_pairs.json', 'source_before_hashes.json'):
        if not read(root / 'audit' / name):
            raise RuntimeError('preservation inventory is missing')
    return root


def require_reviews(root, through):
    sources, artifacts = {}, {}
    for step in range(1, through + 1):
        receipt = read(Path(root) / 'audit' / f'step_{step:02d}_reviews.json') or {}
        rounds = receipt.get('reviews', [])
        if (len(rounds) < 3 or any(row.get('status') != 'pass' for row in rounds[-3:])
                or len({row.get('round') for row in rounds[-3:]}) != 3):
            raise RuntimeError(f'step {step} needs three passing review rounds')
        sources.update(receipt.get('source_sha256', {}))
        artifacts.update(receipt.get('artifact_sha256', {}))
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f'reviewed source changed: {rel}')
    for rel, expected in artifacts.items():
        if sha(confined(root, Path(root) / rel)) != expected:
            raise RuntimeError(f'reviewed artifact changed: {rel}')


@contextmanager
def run_lock(root):
    root = validate_root(root)
    with (root / 'workflow.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield root
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def source_hashes():
    files = {p for folder in ('controllers', 'scripts', 'tests')
             for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts}
    files.update((ROOT / 'worlds').glob('*.wbt'))
    files.update((ROOT / 'protos').rglob('*.proto'))
    files.add(ROOT / 'config/formal_1800/calibration.json')
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def verify_preservation(root):
    protected = read(Path(root) / 'audit/protected_hashes.json')
    for rel, expected in protected.items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f'protected artifact changed: {rel}')
    return len(protected)


def baseline_pairs(root):
    pairs = read(Path(root) / 'audit/baseline_pairs.json')
    expected = {(o, s, n) for o in OBJECTIVES for s in SCENARIOS for n in SEEDS}
    if len(pairs) != 36 or {(r['objective'], r['scenario'], r['seed']) for r in pairs} != expected:
        raise RuntimeError('evaluation matrix must contain exactly 36 distinct pairs')
    return pairs


def prepare(root):
    root = validate_root(root)
    require_reviews(root, 2)
    verify_preservation(root)
    descriptors = {}
    for objective in OBJECTIVES:
        path = confined(root, root / 'models' / f'{objective}.json')
        value = expert_descriptor(objective)
        if path.exists() and read(path) != value:
            raise RuntimeError('existing descriptor differs; refusing overwrite')
        if not path.exists():
            write(root, path, value)
        descriptors[objective] = {'path': path.relative_to(root).as_posix(), 'sha256': sha(path)}
    pairs = baseline_pairs(root)
    for pair in pairs:
        validate_baseline(root, pair)
    contract = {
        'version': VERSION, 'source_sha256': source_hashes(), 'models': descriptors,
        'input_sha256': {rel: sha(root / rel) for rel in (
            'protocol.json', 'audit/baseline_pairs.json', 'audit/protected_hashes.json')},
        'manifest_sha256': {r['manifest']: r['manifest_sha256'] for r in pairs},
        'evaluation_seeds': list(SEEDS), 'episodes': 36, 'duration_seconds': 1800,
        'training': False, 'fine_tuning': False,
    }
    existing = read(root / 'run_contract.json')
    if existing is not None and existing != contract:
        raise RuntimeError('existing run contract changed; refusing stale resume')
    if existing is None:
        for rel, expected in contract['source_sha256'].items():
            dest = confined(root, root / 'source_frozen' / rel)
            if dest.exists() and sha(dest) != expected:
                raise RuntimeError('frozen source already differs')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(ROOT / rel, dest)
        write(root, root / 'run_contract.json', contract)
    return contract


def verify_contract(root):
    contract = read(Path(root) / 'run_contract.json')
    if not contract or contract['source_sha256'] != source_hashes():
        raise RuntimeError('frozen source changed or contract missing')
    for rel, expected in {**contract['input_sha256'], **contract['manifest_sha256']}.items():
        if sha(confined(root, Path(root) / rel)) != expected:
            raise RuntimeError(f'frozen input changed: {rel}')
    for model in contract['models'].values():
        if sha(confined(root, Path(root) / model['path'])) != model['sha256']:
            raise RuntimeError('expert descriptor changed')
    return contract


def validate_baseline(root, pair):
    path = Path(pair['baseline_result'])
    baseline_root = Path(read(Path(root) / 'protocol.json')['baseline_run']).resolve()
    if not path.resolve().is_relative_to(baseline_root) or sha(path) != pair['baseline_result_sha256']:
        raise RuntimeError('baseline identity changed')
    envelope = read(path)
    manifest = confined(root, Path(root) / pair['manifest'])
    if sha(manifest) != pair['manifest_sha256']:
        raise RuntimeError('paired manifest changed')
    models = read(baseline_root / 'workflow_complete.json')['models']
    model = models[pair['objective']]
    if sha(model['checkpoint']) != model['sha256'] or not physical.valid_episode(
            envelope, scenario=pair['scenario'], seed=pair['seed'], manifest=manifest,
            input_model_sha256=model['sha256'], fine_tune_objective=None,
            expected_scheduler='GraphImitation') or not physical.valid_pure_policy_result(envelope):
        raise RuntimeError('baseline fails physical/identity/pure-policy checks')
    return envelope


def valid_expert(envelope, *, pair, manifest, descriptor_hash, contract_hash):
    if not physical.valid_episode(
            envelope, scenario=pair['scenario'], seed=pair['seed'], manifest=manifest,
            input_model_sha256=descriptor_hash, fine_tune_objective=None,
            expected_scheduler='GraphExpert') or not physical.valid_pure_policy_result(envelope):
        return False
    formal = envelope['result']['formal_evaluation']
    info = envelope['result']['experiment_info']
    summary = envelope['result']['summary_metrics']
    expert = formal.get('graph_expert', {})
    return (
        envelope.get('contract_sha256') == contract_hash
        and envelope.get('baseline_result_sha256') == pair['baseline_result_sha256']
        and envelope.get('manifest_sha256') == sha(manifest)
        and info.get('num_robots') == pair['num_robots']
        and formal.get('graph_expert') == {
            **expert_descriptor(pair['objective']), 'descriptor_sha256': descriptor_hash,
            'parameter_updates': 0}
        and expert.get('training_enabled') is False
        and type(expert.get('parameter_updates')) is int
        and 'physical_fine_tune' not in formal
        and summary.get('native_scheduler_commits', 0) > 0
        and set(summary.get('scheduler_commits_by_algorithm', {})) == {'GraphExpert'}
    )


def evaluate_episode(root, pair, webots):
    root = validate_root(root)
    contract = verify_contract(root)
    validate_baseline(root, pair)
    objective, scenario, seed = (pair[k] for k in ('objective', 'scenario', 'seed'))
    model = contract['models'][objective]
    descriptor = confined(root, root / model['path'])
    manifest = confined(root, root / pair['manifest'])
    path = confined(root, root / 'evaluation' / objective / scenario / str(seed) / 'result.json')
    checks = dict(pair=pair, manifest=manifest, descriptor_hash=model['sha256'],
                  contract_hash=sha(root / 'run_contract.json'))
    if path.exists():
        envelope = read(path)
        if not valid_expert(envelope, **checks):
            raise RuntimeError('invalid existing result; refusing overwrite')
        return envelope
    attempt = confined(root, path.parent / 'attempts' / uuid.uuid4().hex)
    environment = {**physical.common_environment(manifest),
        'PYTHONDONTWRITEBYTECODE': '1', 'SMART_FACTORY_PHYSICAL_FINE_TUNE': '0',
        'SMART_FACTORY_GRAPH_PPO_FINE_TUNE': '0', 'SMART_FACTORY_SARSA_V3': '0',
        'SMART_FACTORY_SARSA_V3_TRAINING': '0', 'SMART_FACTORY_SARSA_V3_OUTPUT_MODEL': '',
        'SMART_FACTORY_FINE_TUNE_ALGORITHM': '', 'SMART_FACTORY_FINE_TUNE_OBJECTIVE': '',
        'RL_SCHEDULER_TIMEOUT_SECONDS': '0.05'}
    write(root, attempt / 'request.json', {'pair': pair, 'environment': environment,
          'contract_sha256': checks['contract_hash'], 'descriptor_sha256': model['sha256']})
    write(root, root / 'progress.json', {'status': 'evaluating', 'objective': objective,
          'scenario': scenario, 'seed': seed, 'attempt': str(attempt), 'pid': os.getpid()})
    print(f'[RUN] GraphExpert/{objective}/{scenario}/{seed}', flush=True)
    # Explicit availability check prevents an intentional headless fallback.
    physical.validate_webots(webots)
    result = run_experiments.run_single_experiment(
        scenario, 'GraphExpert', seed, webots_path=str(webots), model_path=str(descriptor),
        controller_mode='external', extra_env=environment, results_dir=str(attempt / 'raw'))
    envelope = {'schema_version': 1, 'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'contract_sha256': checks['contract_hash'], 'baseline_result_sha256': pair['baseline_result_sha256'],
        'input_model_path': str(descriptor), 'input_model_sha256': model['sha256'],
        'manifest_path': str(manifest), 'manifest_sha256': sha(manifest), 'result': result}
    write(root, attempt / 'envelope.json', envelope)
    verify_contract(root)
    if not valid_expert(envelope, **checks):
        raise RuntimeError(f'expert episode failed formal checks: {objective}/{scenario}/{seed}')
    write(root, path, envelope)
    print(f'[DONE] {objective}/{scenario}/{seed} Count=' + str(
        result['formal_evaluation']['count_evaluation']['total_tasks_completed']), flush=True)
    return envelope


def evaluate(root, webots):
    root = validate_root(root)
    require_reviews(root, 3)
    verify_contract(root)
    verify_preservation(root)
    physical.validate_webots(webots)
    for index, pair in enumerate(baseline_pairs(root), 1):
        evaluate_episode(root, pair, webots)
        print(f'[PROGRESS] {index}/36 complete', flush=True)
    verified = verify_preservation(root)
    write(root, root / 'evaluation_complete.json', {
        'status': 'awaiting_step_04_reviews', 'episode_count': 36,
        'contract_sha256': sha(root / 'run_contract.json'), 'protected_files_verified': verified})
    write(root, root / 'progress.json', {'status': 'awaiting_step_04_reviews', 'completed': 36})


METRICS = ('count', 'utility', 'collisions', 'latency_mean_ms', 'latency_p95_ms',
           'planned_wait_seconds', 'predicted_conflicts', 'replans')


def metrics(envelope):
    result = envelope['result']
    formal, s = result['formal_evaluation'], result['summary_metrics']
    values = dict(count=formal['count_evaluation']['total_tasks_completed'],
        utility=formal['utility_v2_evaluation']['utility_score_deadline_v2'],
        collisions=s['physical_collision_count'], latency_mean_ms=s['scheduling_latency_mean_ms'],
        latency_p95_ms=s['scheduling_latency_p95_ms'], planned_wait_seconds=s['planned_wait_total_seconds'],
        predicted_conflicts=s['predicted_conflicts'], replans=s['total_replans'])
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values.values()):
        raise RuntimeError('comparison metric missing or non-finite')
    return values


def summarize(rows):
    """Pair before averaging; resample six seeds as clusters across scenarios."""
    summary = {}
    for objective in OBJECTIVES:
        for scenario in (*SCENARIOS, 'all'):
            group = [r for r in rows if r['objective'] == objective
                     and (scenario == 'all' or r['scenario'] == scenario)]
            if len(group) != (18 if scenario == 'all' else 6):
                raise RuntimeError('comparison group incomplete')
            values = {}
            for metric in METRICS:
                old = statistics.fmean(r['graph_' + metric] for r in group)
                new = statistics.fmean(r['expert_' + metric] for r in group)
                deltas = [statistics.fmean(r['delta_' + metric] for r in group if r['seed'] == seed)
                          for seed in SEEDS]
                rng = np.random.default_rng(271027)
                boot = np.mean(np.asarray(deltas)[rng.integers(0, 6, size=(10000, 6))], axis=1)
                values[metric] = {'graph_mean': old, 'expert_mean': new,
                    'paired_mean_difference': statistics.fmean(deltas),
                    'relative_difference_percent': 100 * (new - old) / old if old != 0 else None,
                    'seed_cluster_bootstrap_95_percent_interval': np.quantile(boot, [.025, .975]).tolist(),
                    'seed_mean_positive_zero_negative': [sum(d > 0 for d in deltas),
                        sum(d == 0 for d in deltas), sum(d < 0 for d in deltas)]}
            summary[f'{objective}/{scenario}'] = values
    return summary


def report(root):
    root = validate_root(root)
    require_reviews(root, 4)
    contract = verify_contract(root)
    rows = []
    result_hashes = {}
    for pair in baseline_pairs(root):
        o, s, n = (pair[k] for k in ('objective', 'scenario', 'seed'))
        path = root / f'evaluation/{o}/{s}/{n}/result.json'
        envelope = read(path)
        if not valid_expert(envelope, pair=pair, manifest=root / pair['manifest'],
                descriptor_hash=contract['models'][o]['sha256'], contract_hash=sha(root / 'run_contract.json')):
            raise RuntimeError('cannot compare an incomplete or mismatched evaluation')
        old, new = metrics(validate_baseline(root, pair)), metrics(envelope)
        row = {'objective': o, 'scenario': s, 'seed': n}
        for metric in METRICS:
            row.update({f'graph_{metric}': old[metric], f'expert_{metric}': new[metric],
                        f'delta_{metric}': new[metric] - old[metric]})
        rows.append(row)
        result_hashes[path.relative_to(root).as_posix()] = sha(path)
    preserved = verify_preservation(root)
    document = {'status': 'awaiting_step_05_reviews', 'episode_pairs': 36,
        'baseline': read(root / 'protocol.json')['baseline_run'],
        'comparison_kind': 'paired historical Webots results, baseline not rerun',
        'training': False, 'fine_tuning': False, 'parameter_updates': 0,
        'seeds': list(SEEDS), 'duration_seconds': 1800,
        'contract_sha256': sha(root / 'run_contract.json'), 'expert_result_sha256': result_hashes,
        'protected_files_verified': preserved, 'summary': summarize(rows), 'pairs': rows,
        'limitations': ['Six seed clusters; bootstrap intervals are descriptive and based on a small sample.',
                       'Historical baseline latency can differ with machine load.',
                       'The expert is a fixed one-step cost heuristic, not a globally optimal scheduling oracle.']}
    write(root, root / 'comparison.json', document)
    path = confined(root, root / 'paired_results.csv')
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'evaluate', 'report'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--webots', type=Path, default=DEFAULT_WEBOTS)
    args = parser.parse_args()
    with run_lock(args.root) as root:
        if args.phase == 'prepare':
            prepare(root)
        elif args.phase == 'evaluate':
            evaluate(root, args.webots)
        else:
            report(root)


if __name__ == '__main__':
    main()
