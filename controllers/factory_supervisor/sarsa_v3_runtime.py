"""Event-time reward accounting and commit-aware online SARSA v3 scheduling."""
from __future__ import annotations

from copy import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from config import RobotState
from dual_objective import (CountRewardProfile, UtilityV2RewardProfile,
    REWARD_COMPONENT_NAMES, _reset_reward_cursors, physical_episode_reward_attribution)
from schedulers import BaseScheduler, SchedulerResult, ModelValidationError
from sarsa_features import build_snapshot, WAIT_ACTION, select_candidates
from sarsa_v3_agent import SarsaV3Agent, MODEL_VERSION, REWARD_VERSION
from task_generator import TaskStatus


class EventRewardLedger:
    """Difference of observable cumulative rewards; never mutates runtime tasks.

    A task's reward is emitted when its underlying event becomes observable.
    Unfinished losses occur only at the horizon; no penalty is spread over all
    historical assignments. Repeated snapshots are idempotent.
    """
    def __init__(self, objective, utility_profile=None, horizon=1800.):
        self.objective=objective
        self.horizon=float(horizon)
        if objective == "count":
            self.profile=CountRewardProfile(horizon_seconds=horizon)
        elif objective == "utility_v2" and isinstance(utility_profile,UtilityV2RewardProfile):
            self.profile=replace(utility_profile,safety_observation="webots_physical")
        else:
            raise ValueError("SARSA utility reward profile is required")
        self.total=0.
        self.now=0.
        self.collisions=0
        self.finalized=False
        self.events=[]
        self.commits=[]
        self.latest={}
        self.cache={}
        self.task_rewards={}
        self.components={name:0. for name in REWARD_COMPONENT_NAMES}
        self.extra={"reward_defer":0.,"reward_invalid":0.}

    def note_commit(self,assignment,now):
        assigned=assignment.task.assignment_time
        if assigned is None or float(assigned)>float(now)+1e-8:
            raise ValueError('SARSA commit lacks a current runtime assignment timestamp')
        # Match the runtime's sub-nanosecond arrival tolerance. Never replace
        # its canonical timestamp with a slightly earlier accumulated clock.
        committed_at=max(float(now),float(assigned),float(assignment.task.arrival_time))
        row={"task_id":int(assignment.task.task_id),"committed_at":committed_at,
             "ideal_distance":float(assignment.task.ideal_distance)}
        self.commits.append(row)
        self.latest[row['task_id']]=row

    def action_penalty(self,kind,now):
        if self.finalized or kind not in {"defer","invalid"}:
            raise ValueError("invalid SARSA action penalty")
        value=0. if self.objective=="count" else -getattr(self.profile,"b_"+kind)
        self.extra['reward_'+kind]+=value
        self.total+=value
        self.events.append({"time":float(now),"kind":kind,"reward":value})

    def observe(self,tasks,now,collision_count=0,*,final=False):
        now=float(now)
        if (not math.isfinite(now) or now<self.now or now>self.horizon+1e-8 or
            isinstance(collision_count,bool) or not isinstance(collision_count,int) or
            collision_count<self.collisions or
            (final and not math.isclose(now,self.horizon,abs_tol=1e-8))):
            raise ValueError("invalid SARSA reward observation clock/collision count")
        if self.finalized:
            raise ValueError("SARSA reward ledger already finalized")
        tasks=[t for t in tasks if t.arrival_time<=now+1e-9 and t.arrival_time<self.horizon]
        if len({t.task_id for t in tasks})!=len(tasks):
            raise ValueError("duplicate SARSA reward task IDs")
        if not set(self.task_rewards).issubset({t.task_id for t in tasks}):
            raise ValueError("SARSA reward observation dropped previously observed tasks")
        components={name:0. for name in REWARD_COMPONENT_NAMES}
        for task in tasks:
            if task.completion_time is not None and task.completion_time>now+1e-8:
                raise ValueError("SARSA reward contains a future completion")
            latest=self.latest.get(task.task_id)
            fingerprint=(task.status,task.assignment_time,task.completion_time,
                task.ideal_distance,task.actual_distance,task.candidate_reward_eligible,
                bool(task.deadline_time is not None and now>task.deadline_time),final,
                tuple(latest.values()) if latest else None)
            cached=self.cache.get(task.task_id)
            if cached is None or cached[0]!=fingerprint:
                clone=copy(task)
                if latest:
                    clone.assignment_time=latest['committed_at']
                    clone.ideal_distance=latest['ideal_distance']
                _reset_reward_cursors(clone)
                rows=[]
                if self.objective=='utility_v2' and clone.assignment_time is not None:
                    rows.append(self.profile.transition([clone],[],dispatch_task=clone,
                        horizon_seconds=self.horizon,runtime_mode='webots'))
                events=[]
                if clone.status==TaskStatus.COMPLETED:
                    events.append({'type':'task_completed','task_id':clone.task_id})
                if self.objective=='utility_v2':
                    events.extend(clone.deadline_events(now,horizon_seconds=self.horizon,final=final))
                if final:events.append({'type':'episode_horizon','current_time':self.horizon})
                rows.append(self.profile.transition([clone],events,horizon_seconds=self.horizon,runtime_mode='webots'))
                values={name:sum(float(row.get(name) or 0.) for row in rows) for name in REWARD_COMPONENT_NAMES}
                reward=sum(values.values())
                delta=reward-self.task_rewards.get(task.task_id,0.)
                if delta:self.events.append({'time':now,'kind':'task_event','task_id':task.task_id,'reward':delta})
                self.task_rewards[task.task_id]=reward
                self.cache[task.task_id]=(fingerprint,values)
            for name,value in self.cache[task.task_id][1].items():components[name]+=value
        if self.objective=='utility_v2':
            components['reward_collision']=-self.profile.b_collision*collision_count
            if collision_count>self.collisions:
                self.events.append({'time':now,'kind':'physical_collision',
                    'reward':-self.profile.b_collision*(collision_count-self.collisions)})
        for name,value in self.extra.items():components[name]+=value
        previous=self.total
        self.components=components
        self.total=float(sum(components.values()))
        self.now=now;self.collisions=collision_count;self.finalized=final
        if not math.isclose(sum(e['reward'] for e in self.events),self.total,abs_tol=1e-7):
            raise RuntimeError('SARSA event rewards do not conserve cumulative return')
        if final and self.commits:
            reference=physical_episode_reward_attribution(tasks,self.commits,objective=self.objective,
                utility_profile=self.profile if self.objective=='utility_v2' else None,
                collision_count=collision_count,horizon_seconds=self.horizon)
            expected=reference['reward_audit']['reward_total']+sum(self.extra.values())
            if not math.isclose(self.total,expected,abs_tol=1e-7):
                raise RuntimeError('SARSA event return disagrees with existing objective contract')
        return self.total-previous

    def report(self):
        return {'version':REWARD_VERSION,'reward_profile_hash':self.profile.sha256,
                'reward_total':self.total,'components':self.components,
                'event_count':len(self.events),'events':self.events,
                'task_reward_by_id':{str(k):v for k,v in self.task_rewards.items()}}


class SarsaV3Scheduler(BaseScheduler):
    uses_traffic=True
    sarsa_v3=True
    pure_policy_required=True

    def __init__(self,model_path=None,*,agent=None,seed=42,training=None,physical=True):
        super().__init__('SARSA')
        try:self.agent=agent or SarsaV3Agent.load(model_path)
        except Exception as exc:raise ModelValidationError(f'SARSA v3 checkpoint rejected: {exc}') from exc
        self.training=(os.environ.get('SMART_FACTORY_SARSA_V3_TRAINING','0')=='1'
                       if training is None else bool(training))
        self.physical=physical
        self.input_path=Path(model_path).resolve() if model_path else None
        self.initial_document=self.agent.document()
        self.initial_updates=self.agent.updates
        if self.training:self.agent.start_episode(seed=seed)
        self.ledger=None;self.sources=[];self.now=0.;self.collisions=0
        self.previous=None;self.previous_total=0.;self.proposal=None
        self.initial_reward_total=0.
        self.wait_until=0.;self.consecutive_waits=0
        self.decisions=[];self.td_errors=[];self.latencies=[];self.final_report=None

    def observe(self,tasks,now,collision_count=0,*,utility_profile=None):
        self.sources=list(tasks);self.now=float(now);self.collisions=int(collision_count)
        if self.training:
            if self.ledger is None:self.ledger=EventRewardLedger(self.agent.objective,utility_profile)
            self.ledger.observe(self.sources,self.now,self.collisions)

    def _accept(self,feature,action,kind,robot_id=None,task_id=None):
        reward=None
        if self.training:
            if self.ledger is None:raise RuntimeError('observe tasks before SARSA training decisions')
            if self.previous is not None:
                reward=self.ledger.total-self.previous_total
                td=self.agent.update(self.previous,reward,feature)
                self.td_errors.append(td)
            else:
                self.initial_reward_total=self.ledger.total
            self.previous=feature.copy();self.previous_total=self.ledger.total
        self.decisions.append({'time':self.now,'action':int(action),'kind':kind,
            'robot_id':robot_id,'task_id':task_id,'features':feature.tolist(),
            'reward_for_previous':reward})

    def assign(self,pending_tasks,robot_states,context=None):
        started=time.perf_counter()
        if context is None:raise ValueError('SARSA v3 requires runtime context')
        self.now=float(context.current_time)
        if self.now<self.wait_until:
            return SchedulerResult(is_feasible=True,algorithm_name=self.name,
                diagnostics={'intentional_wait':True,'reason':'sarsa_v3_wait_hold'})
        candidates,pending=select_candidates(pending_tasks,self.now)
        horizon=float(context.configuration.get('episode_end_time',1800.))
        oldest=max((self.now-t.arrival_time for t in pending),default=0.)
        busy=any(s.get('current_task') is not None for s in robot_states.values())
        allow_wait=(busy and self.consecutive_waits<self.agent.config.max_consecutive_waits
                    and oldest<600. and horizon-self.now>self.agent.config.wait_seconds)
        snapshot=build_snapshot(robot_states,pending_tasks,context,allow_wait=allow_wait)
        if not snapshot.mask[:WAIT_ACTION].any():
            self.proposal=None
            return SchedulerResult(is_feasible=True,algorithm_name=self.name,
                computation_time=time.perf_counter()-started,
                diagnostics={'intentional_wait':True,'reason':'sarsa_v3_no_feasible_pair'})
        action=self.agent.select_action(snapshot,training=self.training,physical=self.physical)
        feature=snapshot.features[action].copy()
        assignment=snapshot.assignment(action)
        if action==WAIT_ACTION:
            self._accept(feature,action,'wait')
            if self.training:self.ledger.action_penalty('defer',self.now)
            self.wait_until=self.now+self.agent.config.wait_seconds
            self.consecutive_waits+=1
            self.proposal=None
        else:
            self.proposal=(assignment.robot_id,assignment.task.task_id,feature,action)
        elapsed=time.perf_counter()-started;self.latencies.append(elapsed)
        return SchedulerResult(assignments=[assignment] if assignment else [],
            objective_value=float(self.agent.values(snapshot.features)[action]),
            computation_time=elapsed,is_feasible=True,algorithm_name=self.name,
            diagnostics={'model_version':MODEL_VERSION,'fallback':False,'action':action,
                         'intentional_wait':action==WAIT_ACTION,
                         'reason':'sarsa_v3_wait' if action==WAIT_ACTION else 'valid',
                         'inference_ms':elapsed*1000.,'candidate_count':len(snapshot.tasks),
                         'pending_count':snapshot.pending_count})

    def _consume_proposal(self,assignment,kind):
        if self.proposal is None or assignment is None or self.proposal[:2]!=(assignment.robot_id,assignment.task.task_id):
            raise RuntimeError('SARSA proposed action differs from runtime outcome')
        rid,tid,feature,action=self.proposal
        self._accept(feature,action,kind,rid,tid)
        self.proposal=None

    def on_assignment_committed(self,assignment):
        self._consume_proposal(assignment,'commit')
        self.consecutive_waits=0;self.assignments_made+=1
        if self.training:
            self.ledger.note_commit(assignment,self.now)
            self.ledger.observe(self.sources,self.now,self.collisions)

    def on_assignment_rejected(self,assignment,reason):
        self._consume_proposal(assignment,'rejected')
        if self.training:self.ledger.action_penalty('invalid',self.now)

    def assign_task(self,pending_tasks,robot_states,congestion_map=None):
        raise RuntimeError('SARSA v3 requires the context-aware assign interface')

    def finalize(self,tasks,now,collision_count=0,*,utility_profile=None,output_path=None):
        if self.final_report is not None:return self.final_report
        if not math.isclose(float(now),1800.,abs_tol=1e-8):
            raise RuntimeError('SARSA episode must finish at the registered 1800s horizon')
        terminal_reward=None
        if self.training:
            if self.ledger is None:self.ledger=EventRewardLedger(self.agent.objective,utility_profile)
            self.ledger.observe(tasks,float(now),int(collision_count),final=True)
            if self.previous is None:raise RuntimeError('SARSA episode contains no accepted decisions')
            terminal_reward=self.ledger.total-self.previous_total
            self.td_errors.append(self.agent.update(self.previous,terminal_reward,done=True))
            learned_return=sum(row['reward_for_previous'] or 0. for row in self.decisions)+terminal_reward
            if not math.isclose(learned_return+self.initial_reward_total,self.ledger.total,abs_tol=1e-7):
                raise RuntimeError('SARSA transitions lost or duplicated event rewards')
            self.agent.end_episode()
        elif self.agent.document()!=self.initial_document:
            raise RuntimeError('frozen SARSA evaluation changed model state')
        if output_path is not None:
            if not self.training:raise RuntimeError('evaluation must not export updated weights')
            path=Path(output_path).resolve()
            run_root=Path(os.environ['SMART_FACTORY_SARSA_V3_RUN_ROOT']).resolve()
            if path==self.input_path or not path.is_relative_to(run_root) or path.exists():
                raise RuntimeError('SARSA output would overwrite an existing or external model')
            self.agent.save(path)
        self.final_report={'status':'complete','model_version':MODEL_VERSION,
            'objective':self.agent.objective,'training':self.training,
            'updates':self.agent.updates-self.initial_updates,
            'mean_absolute_td':float(np.mean(np.abs(self.td_errors))) if self.td_errors else 0.,
            'max_absolute_td':float(np.max(np.abs(self.td_errors))) if self.td_errors else 0.,
            'decision_count':len(self.decisions),'decisions':self.decisions,
            'wait_count':sum(row['kind']=='wait' for row in self.decisions),
            'rejected_count':sum(row['kind']=='rejected' for row in self.decisions),
            'max_inference_ms':max(self.latencies,default=0.)*1000.,
            'output_checkpoint':str(output_path) if output_path else None,
            'terminal_reward':terminal_reward,'initial_reward_unassigned':self.initial_reward_total,
            'reward_audit':self.ledger.report() if self.ledger else None}
        return self.final_report
