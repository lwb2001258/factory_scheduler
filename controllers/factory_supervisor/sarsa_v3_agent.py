"""Shared linear, true-online SARSA(lambda) with exact checkpoint continuation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import tempfile
import numpy as np

from config import RL_ENVIRONMENT_VERSION
from sarsa_features import (ACTION_DIM, FEATURE_DIM, FEATURE_NAMES, FEATURE_VERSION,
                            INDEX, WAIT_ACTION)
from traffic_features import TRAFFIC_VERSION

MODEL_VERSION = "sarsa-v3-linear-true-online-lambda"
REWARD_VERSION = "sarsa-v3-event-time-v1"
FEATURE_SCALE = math.sqrt(FEATURE_DIM)


@dataclass(frozen=True)
class SarsaV3Config:
    learning_rate: float = .1
    trace_lambda: float = .9
    gamma: float = 1.
    epsilon_start: float = .3
    epsilon_end: float = .03
    epsilon_decay: float = .985
    physical_epsilon_cap: float = .03
    wait_seconds: float = 2.
    max_consecutive_waits: int = 2

    def __post_init__(self):
        numeric = [getattr(self, name) for name in self.__dataclass_fields__]
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v)
               for v in numeric):
            raise ValueError("nonfinite/invalid SARSA v3 configuration")
        if (not 0 < self.learning_rate <= 1 or not 0 <= self.trace_lambda < 1 or
            self.gamma != 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or
            not 0 < self.epsilon_decay <= 1 or not 0 <= self.physical_epsilon_cap <= 1 or
            not 0 < self.wait_seconds <= 10 or
            not isinstance(self.max_consecutive_waits,int) or not 0 <= self.max_consecutive_waits <= 5):
            raise ValueError("unsupported SARSA v3 configuration")


class SarsaV3Agent:
    def __init__(self, objective, *, config=None, seed=42, initialize_prior=True):
        if objective not in {"count","utility_v2"}:
            raise ValueError("invalid SARSA objective")
        self.objective = objective
        self.config = config or SarsaV3Config()
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros(FEATURE_DIM,np.float64)
        self.trace = np.zeros(FEATURE_DIM,np.float64)
        self.q_old = 0.
        self.epsilon = self.config.epsilon_start
        self.episode = 0
        self.updates = 0
        self.last_td = 0.
        if initialize_prior:
            # Transparent cold-start prior, learned thereafter; no teacher labels
            # or external scheduler substitutes an action during policy execution.
            prior = {"dispatch": .2, "route_distance": -2., "misses_horizon": -1.,
                     "traffic_expected_wait_seconds": -.2, "task_wait": .15,
                     "wait_action": -.1}
            if objective == "utility_v2":
                prior.update({"priority": .25, "deadline_slack": .15})
            for name,value in prior.items():
                self.weights[INDEX[name]] = value * FEATURE_SCALE

    def _vector(self, features):
        features = np.asarray(features,np.float64)
        if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
            raise ValueError("invalid SARSA action features")
        return features / FEATURE_SCALE

    def values(self, features):
        features = np.asarray(features,np.float64)
        if features.shape != (ACTION_DIM,FEATURE_DIM) or not np.isfinite(features).all():
            raise ValueError("invalid SARSA decision features")
        values = features @ (self.weights / FEATURE_SCALE)
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite SARSA action values")
        return values

    def select_action(self, snapshot, *, training=False, physical=False):
        legal = np.flatnonzero(snapshot.mask)
        if not len(legal):
            raise ValueError("SARSA snapshot has no legal action")
        epsilon = min(self.epsilon,self.config.physical_epsilon_cap) if physical else self.epsilon
        if training and self.rng.random() < epsilon:
            return int(self.rng.choice(legal))
        values = self.values(snapshot.features)
        best = np.max(values[legal])
        tied = legal[np.isclose(values[legal],best,rtol=0.,atol=1e-12)]
        if training and len(tied)>1:
            return int(self.rng.choice(tied))
        # Equal-valued physical choices are ordered by actual route distance;
        # task/robot IDs only settle exact geometric ties. Evaluation consumes no RNG.
        def key(action):
            if action == WAIT_ACTION:
                return (math.inf,math.inf,math.inf)
            ri,ti=divmod(int(action),20)
            return (snapshot.route_costs[ri,ti],snapshot.tasks[ti].task_id,snapshot.robot_ids[ri])
        return int(min(tied,key=key))

    def update(self, features, reward, next_features=None, *, done=False):
        """One on-policy transition; next_features belongs to the actual next action."""
        if isinstance(reward,bool) or not isinstance(reward,(int,float,np.number)) or not math.isfinite(reward):
            raise ValueError("nonfinite SARSA reward")
        if not isinstance(done,bool) or (not done and next_features is None):
            raise ValueError("SARSA requires the actual next action or a terminal flag")
        x=self._vector(features)
        next_x=np.zeros(FEATURE_DIM) if done else self._vector(next_features)
        q=float(self.weights @ x)
        q_next=0. if done else float(self.weights @ next_x)
        delta=float(reward)+self.config.gamma*q_next-q
        alpha=self.config.learning_rate
        decay=self.config.gamma*self.config.trace_lambda
        trace=decay*self.trace+(1.-alpha*decay*float(self.trace @ x))*x
        correction=q-self.q_old
        weights=self.weights+alpha*(delta+correction)*trace-alpha*correction*x
        if not np.isfinite(weights).all() or not np.isfinite(trace).all() or np.linalg.norm(weights)>1e8:
            raise FloatingPointError("SARSA update diverged; checkpoint was not modified")
        self.weights=weights
        self.trace=np.zeros(FEATURE_DIM) if done else trace
        self.q_old=0. if done else q_next
        self.updates+=1
        self.last_td=delta
        return delta

    def start_episode(self, *, seed=None):
        self.trace.fill(0.)
        self.q_old=0.
        if seed is not None:
            self.rng=np.random.default_rng(seed)

    def end_episode(self):
        self.trace.fill(0.)
        self.q_old=0.
        self.episode+=1
        self.epsilon=max(self.config.epsilon_end,self.epsilon*self.config.epsilon_decay)

    def document(self):
        return {"algorithm":"SARSA","model_version":MODEL_VERSION,
            "feature_version":FEATURE_VERSION,"feature_names":list(FEATURE_NAMES),
            "environment_version":RL_ENVIRONMENT_VERSION,"traffic_version":TRAFFIC_VERSION,
            "reward_version":REWARD_VERSION,"objective":self.objective,
            "action_dim":ACTION_DIM,"config":asdict(self.config),
            "weights":self.weights.tolist(),"trace":self.trace.tolist(),"q_old":self.q_old,
            "epsilon":self.epsilon,"episode":self.episode,"updates":self.updates,
            "last_td":self.last_td,"rng_state":self.rng.bit_generator.state}

    def save(self, path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        text=json.dumps(self.document(),sort_keys=True,allow_nan=False)
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                prefix=path.name+".",suffix=".tmp",delete=False) as stream:
            temporary=Path(stream.name);stream.write(text)
        temporary.replace(path)

    @classmethod
    def load(cls,path,*,objective=None):
        doc=json.loads(Path(path).read_text(encoding="utf-8"))
        expected={"algorithm":"SARSA","model_version":MODEL_VERSION,
            "feature_version":FEATURE_VERSION,"feature_names":list(FEATURE_NAMES),
            "environment_version":RL_ENVIRONMENT_VERSION,"traffic_version":TRAFFIC_VERSION,
            "reward_version":REWARD_VERSION,"action_dim":ACTION_DIM}
        if any(doc.get(k)!=v for k,v in expected.items()):
            raise ValueError("incompatible SARSA v3 checkpoint")
        if objective is not None and doc.get("objective")!=objective:
            raise ValueError("SARSA checkpoint objective mismatch")
        model=cls(doc["objective"],config=SarsaV3Config(**doc["config"]),initialize_prior=False)
        for name in ("weights","trace"):
            value=np.asarray(doc[name],np.float64)
            if value.shape!=(FEATURE_DIM,) or not np.isfinite(value).all():
                raise ValueError("invalid SARSA checkpoint parameters")
            setattr(model,name,value)
        for name in ("q_old","epsilon","last_td"):
            value=doc[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError("invalid SARSA checkpoint scalar")
            setattr(model,name,float(value))
        for name in ("episode","updates"):
            value=doc[name]
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError("invalid SARSA checkpoint counters")
            setattr(model,name,value)
        if not model.config.epsilon_end<=model.epsilon<=model.config.epsilon_start:
            raise ValueError("invalid SARSA checkpoint epsilon")
        model.rng.bit_generator.state=doc["rng_state"]
        return model
