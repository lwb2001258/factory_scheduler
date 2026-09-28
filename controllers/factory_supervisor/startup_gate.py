"""Pure startup handshake state machine, independent of Webots APIs."""

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class StartupGate:
    expected_robot_ids: Set[int]
    ready_robot_ids: Set[int] = field(default_factory=set)
    errors: Dict[int, str] = field(default_factory=dict)
    duplicate_robot_ids: Set[int] = field(default_factory=set)
    system_ready: bool = False

    def mark_ready(self, robot_id: int, robot_name: str, ready: bool,
                   error: str = "") -> str:
        if robot_id not in self.expected_robot_ids:
            return "unknown"
        if not ready:
            self.errors[robot_id] = error or "robot_reported_error"
            return "error"
        if robot_name != f"robot_{robot_id}":
            self.errors[robot_id] = "robot_name_mismatch"
            return "error"
        if robot_id in self.ready_robot_ids:
            self.duplicate_robot_ids.add(robot_id)
            return "duplicate"
        self.ready_robot_ids.add(robot_id)
        if self.ready_robot_ids == self.expected_robot_ids and not self.errors:
            self.system_ready = True
            return "system_ready"
        return "ready"

    @property
    def missing_robot_ids(self):
        return self.expected_robot_ids - self.ready_robot_ids

    def reset(self):
        self.ready_robot_ids.clear()
        self.errors.clear()
        self.duplicate_robot_ids.clear()
        self.system_ready = False
