"""Two-phase transaction state for atomic multi-robot plan activation."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class JointPlanTransaction:
    epoch: int
    plans: Dict[int, list]
    versions: Dict[int, int]
    created_at: float
    deadline: float
    prepared: Set[int] = field(default_factory=set)
    armed: Set[int] = field(default_factory=set)
    activated_members: Set[int] = field(default_factory=set)
    committed_members: Set[int] = field(default_factory=set)
    aborted_members: Set[int] = field(default_factory=set)
    rejected: Set[int] = field(default_factory=set)
    state: str = "preparing"
    failure_reason: Optional[str] = None
    activate_at: Optional[float] = None

    @property
    def members(self):
        return set(self.plans)

    def acknowledge(self, robot_id: int, accepted: bool) -> bool:
        if self.state != "preparing" or robot_id not in self.members:
            return False
        if accepted:
            self.prepared.add(robot_id)
        else:
            self.rejected.add(robot_id)
            self.failure_reason = f"robot_{robot_id}_rejected"
        return True

    def decision(self, now: float) -> Optional[str]:
        if self.state != "preparing":
            return self.state
        if self.rejected:
            self.state = "aborted"
        elif self.prepared == self.members:
            self.state = "ready"
        elif now >= self.deadline:
            self.state = "aborted"
            self.failure_reason = "prepare_timeout"
        return None if self.state == "preparing" else self.state

    def abort(self, reason: str):
        if self.state not in ("activated", "aborted"):
            self.state = "aborted"
            self.failure_reason = reason

    def mark_activated(self):
        if self.state not in ("ready", "armed", "activation_confirmed"):
            raise RuntimeError("transaction is not ready or armed")
        self.state = "activated"

    def begin_arming(self, activate_at: float):
        if self.state != "ready":
            raise RuntimeError("transaction is not ready")
        self.activate_at = float(activate_at)
        self.state = "arming"

    def acknowledge_armed(self, robot_id: int) -> bool:
        if self.state != "arming" or robot_id not in self.members:
            return False
        self.armed.add(robot_id)
        if self.armed == self.members:
            self.state = "armed"
        return True

    def acknowledge_activated(self, robot_id: int) -> bool:
        if self.state != "committed" or robot_id not in self.members:
            return False
        self.activated_members.add(robot_id)
        if self.activated_members == self.members:
            self.state = "activation_confirmed"
        return True

    def begin_commit(self, activate_at: float):
        if self.state != "armed":
            raise RuntimeError("transaction is not armed")
        self.activate_at = float(activate_at)
        self.state = "committing"

    def acknowledge_committed(self, robot_id: int) -> bool:
        if self.state != "committing" or robot_id not in self.members:
            return False
        self.committed_members.add(robot_id)
        if self.committed_members == self.members:
            self.state = "committed"
        return True

    def acknowledge_aborted(self, robot_id: int) -> bool:
        if self.state != "aborted" or robot_id not in self.members:
            return False
        self.aborted_members.add(robot_id)
        return True
