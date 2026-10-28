"""Isolated SARSA v3 training -> physical fine tuning -> six-seed evaluation.

No intermediate nonregression/rollback phase is executed. Each phase stops at
its three-review gate. Resume is episode-granular and identity checked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
import uuid

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT/'scripts',ROOT/'controllers/factory_supervisor'):
    sys.path.insert(0,str(directory))

import numpy as np
import sarsa_workflow_contract as audit
import run_strict_1800_workflow as strict
import run_webots_1800_workflow as physical
import run_experiments
from config import RobotState
from dual_objective import count_evaluation,utility_v2_evaluation
from headless_training_runtime import HeadlessFactoryRuntime,HeadlessRuntimeConfig
from sarsa_v3_agent import SarsaV3Agent,SarsaV3Config,MODEL_VERSION
from sarsa_v3_runtime import SarsaV3Scheduler
from sarsa_features import FEATURE_DIM,FEATURE_VERSION
from training_scenarios import factory_scenario

SCENARIOS=('A','B','C')
OBJECTIVES=('count','utility_v2')
DEFAULT_WEBOTS=Path('C:/Program Files/Webots/msys64/mingw64/bin/webots.exe')


def digest(document):
    return hashlib.sha256(json.dumps(document,sort_keys=True,allow_nan=False,
                                    separators=(',',':')).encode()).hexdigest()


@contextmanager
def lock(root):
    root=audit.validate_root(root)
    path=root/'workflow.lock'
    with path.open('a+b') as stream:
        if path.stat().st_size==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def contract(root, *, amend_execution=False):
    root=audit.validate_root(root)
    files=sorted({p for directory in (ROOT/'controllers',ROOT/'scripts')
                  for p in directory.rglob('*.py') if '__pycache__' not in p.parts})
    document={'version':'sarsa-v3-run-v1','model_version':MODEL_VERSION,
        'feature_version':FEATURE_VERSION,'feature_dim':FEATURE_DIM,
        'sources':{p.relative_to(ROOT).as_posix():audit.sha256(p) for p in files},
        'calibration_sha256':audit.sha256(strict.CALIBRATION),
        'objectives':list(OBJECTIVES),'scenarios':list(SCENARIOS),'horizon':1800,
        'training_min_rounds':20,'training_max_rounds':60,
        'fine_tune_min_rounds':6,'fine_tune_max_rounds':10,
        'training_seeds':list(range(21000,21060)),
        'reserved_validation_seeds':list(range(31000,31006)),
        'fine_tune_seeds':list(range(41000,41010)),
        'evaluation_seeds':list(range(61000,61006)),
        'nonregression_validation':False,'validation_model_selection':False,
        'selection':'latest cumulative checkpoint; no holdout/evaluation-based selection',
        'baseline_final_evaluation':'rerun preserved old selected model on identical manifests',
        'config':SarsaV3Config().__dict__,'physical_learning_rate_multiplier':.1,
        'convergence':{'patience':3,'prediction_relative_change':.02,
                       'normalized_absolute_td':.1,'metric_relative_range':.1}}
    amendment=audit.read(root/'audit/execution_amendment.json')
    if amendment is not None:
        document['execution_amendment_sha256']=digest(amendment)
    path=root/'run_contract.json';existing=audit.read(path)
    if existing is not None and existing!=document:
        if not amend_execution:
            raise RuntimeError('source/configuration changed after run freeze; refusing stale resume')
        validate_execution_amendment(root,existing,document,amendment)
        audit.write(root,path,document)
    if existing is None:audit.write(root,path,document)
    return document


def validate_execution_amendment(root,previous,current,amendment):
    """Only an explicitly reviewed execution repair can reuse pinned records."""
    allowed={'scripts/sarsa_workflow_contract.py','scripts/run_sarsa_v3_workflow.py'}
    if (not isinstance(amendment,dict) or
            amendment.get('version')!='sarsa-execution-amendment-v1' or
            amendment.get('parent_contract_sha256')!=digest(previous) or
            current.get('execution_amendment_sha256')!=digest(amendment) or
            not amendment.get('reason') or
            set(amendment.get('approved_source_changes',{}))!=allowed):
        raise RuntimeError('invalid execution amendment provenance')
    if 'execution_amendment_sha256' in previous:
        raise RuntimeError('execution amendments cannot be chained implicitly')
    old_settings={k:v for k,v in previous.items() if k!='sources'}
    new_settings={k:v for k,v in current.items()
                  if k not in {'sources','execution_amendment_sha256'}}
    if old_settings!=new_settings:
        raise RuntimeError('execution amendment cannot change training settings')
    changed={name for name in set(previous['sources'])|set(current['sources'])
             if previous['sources'].get(name)!=current['sources'].get(name)}
    if changed!=allowed:
        raise RuntimeError('execution amendment cannot change algorithm sources')
    for name in allowed:
        if amendment['approved_source_changes'][name]!={
                'before':previous['sources'][name],'after':current['sources'][name]}:
            raise RuntimeError('execution repair source hashes differ from review')
    archive=Path(root)/'audit/contracts'/f'{digest(previous)}.json'
    if audit.read(archive)!=previous:
        raise RuntimeError('original contract was not preserved')
    records=amendment.get('resumable_records',{})
    actual={p.relative_to(root).as_posix():audit.sha256(p)
            for p in (Path(root)/'standalone').glob('*/episodes/*/*/result.json')}
    if not records or actual!=records:
        raise RuntimeError('completed episode inventory changed before amendment')
    for name in records:
        row=audit.read(Path(root)/name)
        if (row.get('contract_sha256')!=digest(previous) or
                row.get('status')!='complete' or
                audit.sha256(row['checkpoint'])!=row.get('checkpoint_sha256')):
            raise RuntimeError('amendment references an invalid old episode')


def reusable_predecessor(path,existing,requested_hash):
    """Recognize only unchanged, individually pinned pre-repair episodes."""
    path=Path(path).resolve()
    for root in path.parents:
        if not (root/'run_contract.json').is_file():continue
        current=audit.read(root/'run_contract.json')
        amendment=audit.read(root/'audit/execution_amendment.json')
        return bool(isinstance(amendment,dict) and requested_hash==digest(current) and
            current.get('execution_amendment_sha256')==digest(amendment) and
            existing.get('contract_sha256')==amendment.get('parent_contract_sha256') and
            amendment.get('resumable_records',{}).get(path.relative_to(root).as_posix())==audit.sha256(path))
    return False


def progress(root,phase,objective,**values):
    document={'at':audit.now(),'phase':phase,'objective':objective,**values}
    audit.write(root,Path(root)/'progress'/f'{phase}_{objective}.json',document)
    print(json.dumps(document,ensure_ascii=False),flush=True)


def manifest(root,partition,scenario,seed):
    allowed={'standalone_training':range(21000,21060),'webots_finetune':range(41000,41010),
             'final_evaluation':range(61000,61006)}
    if scenario not in SCENARIOS or seed not in allowed.get(partition,()):
        raise ValueError('unregistered SARSA seed partition')
    source=strict.manifest_path(partition,scenario,seed)
    strict._load_manifest(source)
    destination=audit.confined(root,Path(root)/'manifests'/partition/scenario/f'{seed}.json')
    if destination.exists():
        if audit.sha256(destination)!=audit.sha256(source):raise RuntimeError('manifest changed')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    return destination


def checked_record(path,identity):
    existing=audit.read(path)
    if existing is None:return None
    different={k for k,v in identity.items() if existing.get(k)!=v}
    if different and not (different=={'contract_sha256'} and
            reusable_predecessor(path,existing,identity['contract_sha256'])):
        raise RuntimeError(f'episode/phase resume identity mismatch: {path}')
    checkpoint=existing.get('checkpoint')
    if checkpoint and (not Path(checkpoint).is_file() or
                        audit.sha256(checkpoint)!=existing.get('checkpoint_sha256')):
        raise RuntimeError(f'resume checkpoint missing or changed: {path}')
    return existing


def summarize_round(rows,before,after,round_index):
    probes=[];rewards=[]
    for row in rows:
        report=row['learning']
        decisions=report['decisions']
        stride=max(1,len(decisions)//80)
        probes.extend(item['features'] for item in decisions[::stride])
        rewards.extend(abs(item['reward_for_previous']) for item in decisions
                       if item['reward_for_previous'] is not None)
        if report['terminal_reward'] is not None:rewards.append(abs(report['terminal_reward']))
    x=np.asarray(probes,np.float64)/math.sqrt(FEATURE_DIM)
    old=x@before.weights;new=x@after.weights
    change=float(np.mean(np.abs(new-old)))/max(1.,float(np.mean(np.abs(new))))
    absolute_td=statistics.fmean(row['learning']['mean_absolute_td'] for row in rows)
    mean_reward=statistics.fmean(rewards) if rewards else 0.
    return {'round':round_index,'mean_metric':statistics.fmean(row['objective_metric'] for row in rows),
        'mean_absolute_td':absolute_td,'normalized_absolute_td':absolute_td/max(1.,mean_reward),
        'visited_prediction_relative_change':change,'probe_count':len(probes),
        'decisions':sum(row['learning']['decision_count'] for row in rows),
        'updates':sum(row['learning']['updates'] for row in rows)}


def converged(rounds,minimum,settings):
    recent=rounds[-settings['patience']:]
    if len(rounds)<minimum or len(recent)<settings['patience']:return False
    values=[row['mean_metric'] for row in recent]
    stable=(max(values)-min(values))/max(1.,abs(statistics.fmean(values)))<=settings['metric_relative_range']
    return bool(stable and all(row['decisions']>=30 and row['updates']>=30 and
        row['visited_prediction_relative_change']<=settings['prediction_relative_change'] and
        row['normalized_absolute_td']<=settings['normalized_absolute_td'] for row in recent))


def headless_episode(root,objective,input_checkpoint,scenario,seed,contract_hash):
    root=Path(root);task_manifest=manifest(root,'standalone_training',scenario,seed)
    directory=root/'standalone'/objective/'episodes'/scenario/str(seed)
    result_path=directory/'result.json'
    identity={'contract_sha256':contract_hash,'input_checkpoint_sha256':audit.sha256(input_checkpoint),
              'manifest_sha256':audit.sha256(task_manifest),'scenario':scenario,'seed':seed,'objective':objective}
    previous=checked_record(result_path,identity)
    if previous:return previous
    # Interrupted attempts are preserved. Only a fully audited episode receives
    # a canonical result pointing to its immutable output checkpoint.
    attempt=directory/'attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    agent=SarsaV3Agent.load(input_checkpoint,objective=objective)
    robots,tasks,context=factory_scenario(seed,scenario_id=scenario,duration_seconds=1800.,
                                       manifest=strict._load_manifest(task_manifest))
    context=replace(context,configuration={**context.configuration,'graph_traffic_enabled':True})
    runtime=HeadlessFactoryRuntime(robots,tasks,context,seed=seed,
        config=HeadlessRuntimeConfig(episode_end_time=1800.,fixed_horizon=True))
    scheduler=SarsaV3Scheduler(agent=agent,seed=seed,training=True,physical=False)
    profile=strict.utility_profile() if objective=='utility_v2' else None
    start=time.perf_counter();heartbeat=0.;iterations=0
    progress(root,'standalone',objective,scenario=scenario,seed=seed,sim_time=0.,status='running')
    while not runtime.is_terminal():
        iterations+=1
        if iterations>50000:raise RuntimeError('headless SARSA made no bounded progress')
        scheduler.observe(runtime.tasks,runtime.current_time,utility_profile=profile)
        pending=runtime.arrived_pending_tasks()
        idle=any(s.get('state')==RobotState.IDLE and s.get('current_task') is None for s in runtime.robots.values())
        if idle and pending:
            decision=scheduler.assign(pending,runtime.robots,runtime.context)
            if decision.assignments:
                selected=decision.assignments[0];event=runtime.dispatch(selected)
                if event.get('type')=='assignment_committed':scheduler.on_assignment_committed(selected)
                else:scheduler.on_assignment_rejected(selected,event.get('reason','runtime_rejected'))
            else:
                delay=max(.032,scheduler.wait_until-runtime.current_time) if scheduler.wait_until>runtime.current_time else 2.
                runtime.advance_until_event(max_seconds=delay)
        else:runtime.advance_until_event(max_seconds=30.)
        if runtime.current_time>=heartbeat+300.:
            heartbeat=runtime.current_time
            progress(root,'standalone',objective,scenario=scenario,seed=seed,
                     sim_time=round(runtime.current_time,3),updates=agent.updates-scheduler.initial_updates,status='running')
    learning=scheduler.finalize(runtime.tasks,runtime.current_time,utility_profile=profile)
    count=count_evaluation(runtime.tasks,horizon_seconds=1800.)
    utility=utility_v2_evaluation(runtime.tasks,strict.utility_score_config(),route_distances={
        t.task_id:{'ideal_distance':t.ideal_distance,'actual_distance':t.actual_distance}
        for t in runtime.tasks if t.assignment_time is not None})
    checkpoint=attempt/'model.json';agent.save(checkpoint)
    record={**identity,'status':'complete','checkpoint':str(checkpoint),
        'checkpoint_sha256':audit.sha256(checkpoint),'learning':learning,
        'completed_tasks':count['total_tasks_completed'],'utility':utility['utility_score_deadline_v2'],
        'objective_metric':count['total_tasks_completed'] if objective=='count' else utility['utility_score_deadline_v2'],
        'wall_seconds':time.perf_counter()-start,'physics_fidelity':'business_logic_only',
        'runtime':runtime.telemetry_snapshot()}
    audit.write(root,result_path,record)
    progress(root,'standalone',objective,scenario=scenario,seed=seed,status='episode_complete',
             metric=record['objective_metric'],checkpoint=str(checkpoint))
    return record


def standalone_worker(root,objective,document):
    try:
        return _standalone_worker(root,objective,document)
    except Exception as exc:
        # Report before the other objective finishes, including if disk writes
        # remain unavailable after the bounded retry budget.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        try:
            progress(root,'standalone',objective,status='failed',
                     error=f'{type(exc).__name__}: {exc}')
        except Exception:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        raise


def _standalone_worker(root,objective,document):
    root=Path(root);contract_hash=digest(document)
    report_path=root/'standalone'/objective/'training_report.json'
    identity={'contract_sha256':contract_hash,'objective':objective}
    previous=checked_record(report_path,identity)
    if previous:return previous
    initial=root/'standalone'/objective/'initial.json'
    expected_initial=SarsaV3Agent(objective,config=SarsaV3Config(**document['config']))
    if not initial.exists():expected_initial.save(initial)
    elif SarsaV3Agent.load(initial,objective=objective).document()!=expected_initial.document():
        raise RuntimeError('standalone initialization changed after registration')
    current=initial;rounds=[]
    for index,seed in enumerate(document['training_seeds'],1):
        before=SarsaV3Agent.load(current,objective=objective);rows=[]
        for scenario in SCENARIOS:
            row=headless_episode(root,objective,current,scenario,seed,contract_hash)
            rows.append(row);current=Path(row['checkpoint'])
        summary=summarize_round(rows,before,SarsaV3Agent.load(current),index)
        summary.update({'seed':seed,'checkpoint':str(current),'checkpoint_sha256':audit.sha256(current)})
        rounds.append(summary)
        audit.write(root,root/'standalone'/objective/'rounds'/f'{index:03d}.json',summary)
        progress(root,'standalone',objective,status='round_complete',**summary)
        if converged(rounds,document['training_min_rounds'],document['convergence']):break
    did_converge=converged(rounds,document['training_min_rounds'],document['convergence'])
    report={**identity,'status':'complete','checkpoint':str(current),'checkpoint_sha256':audit.sha256(current),
        'balanced_rounds':len(rounds),'episodes':len(rounds)*3,'rounds':rounds,
        'converged':did_converge,'completion_reason':'joint_learning_criteria' if did_converge else 'maximum_rounds_reached',
        'intermediate_nonregression':'skipped_by_user_instruction','selection':document['selection']}
    audit.write(root,report_path,report)
    return report


def physical_episode(root,objective,checkpoint,scenario,seed,phase,document,webots,*,legacy=False):
    root=Path(root);fine=phase=='fine_tune'
    if fine and legacy:raise ValueError('legacy baseline is evaluation-only')
    partition='webots_finetune' if fine else 'final_evaluation'
    task_manifest=manifest(root,partition,scenario,seed)
    variant='baseline' if legacy else 'v3'
    directory=root/phase/objective/variant/scenario/str(seed)
    result_path=directory/'result.json'
    identity={'contract_sha256':digest(document),'objective':objective,
        'input_checkpoint_sha256':audit.sha256(checkpoint),'manifest_sha256':audit.sha256(task_manifest),
        'scenario':scenario,'seed':seed,'variant':variant,'phase':phase}
    previous=checked_record(result_path,identity)
    if previous:return previous
    attempt=directory/'attempts'/uuid.uuid4().hex
    output=attempt/'updated_model.json'
    environment=physical.common_environment(task_manifest)
    environment.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8',
        'SMART_FACTORY_PHYSICAL_FINE_TUNE':'0','SMART_FACTORY_GRAPH_PPO_FINE_TUNE':'0',
        'SMART_FACTORY_SARSA_V3':'0' if legacy else '1',
        'SMART_FACTORY_SARSA_V3_TRAINING':'1' if fine else '0',
        'SMART_FACTORY_SARSA_V3_RUN_ROOT':str(root),
        'SMART_FACTORY_SARSA_V3_OUTPUT_MODEL':str(output) if fine else ''})
    progress(root,phase,objective,scenario=scenario,seed=seed,variant=variant,status='running')
    result=run_experiments.run_single_experiment(scenario,'SARSA',seed,webots_path=str(webots),
        model_path=str(checkpoint),controller_mode='external',extra_env=environment,results_dir=str(attempt/'raw'))
    envelope={'schema_version':1,'completed_at':audit.now(),'input_model_path':str(Path(checkpoint).resolve()),
        'input_model_sha256':identity['input_checkpoint_sha256'],'manifest_path':str(task_manifest),
        'manifest_sha256':identity['manifest_sha256'],'result':result}
    if not physical.valid_episode(envelope,scenario=scenario,seed=seed,manifest=task_manifest,
            input_model_sha256=identity['input_checkpoint_sha256'],fine_tune_objective=None,
            expected_scheduler='SARSA'):
        raise RuntimeError('physical SARSA result failed horizon/manifest/objective integrity checks')
    if audit.sha256(checkpoint)!=identity['input_checkpoint_sha256']:
        raise RuntimeError('physical run changed its input checkpoint')
    formal=result['formal_evaluation'];metrics=result['summary_metrics']
    if not legacy:
        if not physical.valid_pure_policy_result(envelope):raise RuntimeError('SARSA v3 evaluation contains substituted/invalid decisions')
        learning=formal.get('sarsa_v3',{})
        if learning.get('training') is not fine or learning.get('model_version')!=MODEL_VERSION:
            raise RuntimeError('SARSA physical learning mode/model mismatch')
        if fine:
            if learning.get('updates',0)<=0 or not output.is_file():raise RuntimeError('physical SARSA did not update/export model')
            SarsaV3Agent.load(output,objective=objective)
        elif learning.get('updates')!=0:raise RuntimeError('evaluation performed learning')
    else:learning={}
    final_checkpoint=output if fine else Path(checkpoint)
    count=formal['count_evaluation']['total_tasks_completed']
    utility=formal['utility_v2_evaluation']['utility_score_deadline_v2']
    row={**identity,'status':'complete','checkpoint':str(final_checkpoint),
        'checkpoint_sha256':audit.sha256(final_checkpoint),'learning':learning,
        'completed_tasks':count,'utility':utility,'objective_metric':count if objective=='count' else utility,
        'pure_policy':physical.valid_pure_policy_result(envelope),
        'fallback_commits':metrics.get('fallback_scheduler_commits'),
        'physical_collisions':metrics.get('physical_collision_count'),
        'envelope':envelope}
    audit.write(root,result_path,row)
    progress(root,phase,objective,scenario=scenario,seed=seed,variant=variant,
             status='episode_complete',metric=row['objective_metric'])
    return row


def fine_tune(root,objective,document,webots):
    root=Path(root);identity={'contract_sha256':digest(document),'objective':objective}
    path=root/'fine_tune'/objective/'fine_tune_report.json'
    trained=checked_record(root/'standalone'/objective/'training_report.json',identity)
    if not trained:raise RuntimeError('standalone training is incomplete')
    identity={**identity,'standalone_checkpoint_sha256':trained['checkpoint_sha256']}
    previous=checked_record(path,identity)
    if previous:return previous
    current=root/'fine_tune'/objective/'initial.json'
    model=SarsaV3Agent.load(trained['checkpoint'],objective=objective)
    model.config=replace(model.config,learning_rate=document['config']['learning_rate']*.1)
    if not current.exists():model.save(current)
    elif SarsaV3Agent.load(current,objective=objective).document()!=model.document():
        raise RuntimeError('fine-tune initialization differs from its frozen standalone parent')
    rounds=[]
    for index,seed in enumerate(document['fine_tune_seeds'],1):
        before=SarsaV3Agent.load(current);rows=[]
        for scenario in SCENARIOS:
            row=physical_episode(root,objective,current,scenario,seed,'fine_tune',document,webots)
            rows.append(row);current=Path(row['checkpoint'])
        summary=summarize_round(rows,before,SarsaV3Agent.load(current),index)
        summary.update({'seed':seed,'checkpoint':str(current),'checkpoint_sha256':audit.sha256(current)})
        rounds.append(summary)
        audit.write(root,root/'fine_tune'/objective/'rounds'/f'{index:02d}.json',summary)
        if converged(rounds,document['fine_tune_min_rounds'],document['convergence']):break
    report={**identity,'status':'complete','checkpoint':str(current),'checkpoint_sha256':audit.sha256(current),
        'balanced_rounds':len(rounds),'episodes':len(rounds)*3,'rounds':rounds,
        'converged':converged(rounds,document['fine_tune_min_rounds'],document['convergence']),
        'completion_reason':'joint_learning_criteria' if converged(rounds,document['fine_tune_min_rounds'],document['convergence']) else 'maximum_rounds_reached',
        'intermediate_nonregression':'skipped_by_user_instruction'}
    audit.write(root,path,report)
    return report


def evaluate(root,objective,document,webots):
    root=Path(root);identity={'contract_sha256':digest(document),'objective':objective}
    selected=checked_record(root/'fine_tune'/objective/'fine_tune_report.json',identity)
    if not selected:raise RuntimeError('physical fine tuning is incomplete')
    new_model=Path(selected['checkpoint'])
    old_source=audit.OLD_RUN/'webots_fine_tune'/f'SARSA_{objective}'/'webots_selected_model.json'
    baseline=root/'baseline/models'/f'{objective}.json'
    if baseline.exists():
        if audit.sha256(baseline)!=audit.sha256(old_source):raise RuntimeError('preserved baseline differs from original')
    else:
        baseline.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old_source,baseline)
    pairs=[]
    for scenario in SCENARIOS:
        for seed in document['evaluation_seeds']:
            old=physical_episode(root,objective,baseline,scenario,seed,'evaluation',document,webots,legacy=True)
            new=physical_episode(root,objective,new_model,scenario,seed,'evaluation',document,webots)
            pairs.append({'scenario':scenario,'seed':seed,'baseline':old['objective_metric'],
                'v3':new['objective_metric'],'difference':new['objective_metric']-old['objective_metric'],
                'baseline_completed':old['completed_tasks'],'v3_completed':new['completed_tasks'],
                'baseline_utility':old['utility'],'v3_utility':new['utility'],
                'baseline_fallback_commits':old['fallback_commits'],'v3_fallback_commits':new['fallback_commits'],
                'baseline_collisions':old['physical_collisions'],'v3_collisions':new['physical_collisions']})
    by_seed=[statistics.fmean(p['difference'] for p in pairs if p['seed']==seed) for seed in document['evaluation_seeds']]
    rng=np.random.default_rng(61000)
    bootstrap=np.mean(rng.choice(by_seed,size=(10000,len(by_seed)),replace=True),axis=1)
    report={**identity,'status':'complete','checkpoint':str(new_model),'checkpoint_sha256':audit.sha256(new_model),
        'pairs':pairs,'seed_count':len(document['evaluation_seeds']),'seeds':document['evaluation_seeds'],
        'v3_episodes':len(pairs),'baseline_episodes':len(pairs),'mean_baseline':statistics.fmean(p['baseline'] for p in pairs),
        'mean_v3':statistics.fmean(p['v3'] for p in pairs),'mean_difference':statistics.fmean(by_seed),
        'seed_cluster_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),
        'comparison':'same runtime/manifests/seeds; baseline fallback commits reported explicitly'}
    audit.write(root,root/'evaluation'/objective/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','amend-execution','standalone','fine-tune','evaluate','status'))
    parser.add_argument('--run-root',type=Path,default=audit.DEFAULT_RUN)
    parser.add_argument('--webots',type=Path,default=DEFAULT_WEBOTS)
    args=parser.parse_args();root=audit.validate_root(args.run_root)
    if args.phase=='status':
        for path in sorted((root/'progress').glob('*.json')):print(json.dumps(audit.read(path)))
        return
    with lock(root):
        amended=(root/'audit/execution_amendment.json').is_file()
        needed={'freeze':5,'amend-execution':6,'standalone':6 if amended else 5,
                'fine-tune':6,'evaluate':7}[args.phase]
        audit.require_reviews(root,needed)
        document=contract(root,amend_execution=args.phase=='amend-execution')
        if args.phase in {'freeze','amend-execution'}:
            print('SARSA run contract frozen:',digest(document));return
        step={'standalone':6,'fine-tune':7,'evaluate':8}[args.phase]
        state=audit.read(root/'workflow.json')
        state['steps'][step-1].pop('error',None)
        state['steps'][step-1].update(status='in_progress',started_at=audit.now())
        audit.write(root,root/'workflow.json',state)
        try:
            if args.phase=='standalone':
                with ProcessPoolExecutor(max_workers=2) as executor:
                    futures=[executor.submit(standalone_worker,root,obj,document) for obj in OBJECTIVES]
                    for future in as_completed(futures):
                        try:future.result()
                        except Exception as exc:
                            state['steps'][step-1].update(status='failed',error=f'{type(exc).__name__}: {exc}')
                            audit.write(root,root/'workflow.json',state)
                            raise
            elif args.phase=='fine-tune':
                for objective in OBJECTIVES:fine_tune(root,objective,document,args.webots)
            else:
                for objective in OBJECTIVES:evaluate(root,objective,document,args.webots)
                preservation=audit.verify_preservation(root)
                if preservation['changed_hashes']:raise RuntimeError('protected original artifacts changed')
        except Exception as exc:
            state['steps'][step-1].update(status='failed',error=f'{type(exc).__name__}: {exc}')
            audit.write(root,root/'workflow.json',state)
            raise
        state['steps'][step-1].update(status='awaiting_three_reviews',completed_at=audit.now())
        audit.write(root,root/'workflow.json',state)
        print(f'{args.phase} complete; perform at least three code/evidence reviews before the next phase.',flush=True)


if __name__=='__main__':main()
