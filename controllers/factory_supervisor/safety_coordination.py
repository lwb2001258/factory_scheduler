"""Deterministic priority and expiring resource reservations."""

from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Tuple


def priority_key(entered_zone: bool, task_priority: int, wait_age: float,
                 conflict_distance: float, robot_id: int) -> tuple:
    """Stable ascending key; older waiters win before robot-id tie-break."""
    return (
        0 if entered_zone else 1,
        -int(task_priority),
        -max(0.0, float(wait_age)),
        float(conflict_distance),
        int(robot_id),
    )


@dataclass(frozen=True)
class Reservation:
    owner: int
    resource: Hashable
    start: float
    end: float


class ReservationManager:
    """Atomic node/edge/zone reservations with TTL pruning."""

    def __init__(self):
        self._items: list[Reservation] = []

    def prune(self, now: float) -> None:
        self._items = [item for item in self._items if item.end > now]

    @staticmethod
    def _canonical_conflict(resource: Hashable) -> Hashable:
        if (isinstance(resource, tuple) and len(resource) == 3
                and resource[0] in ("edge", "grid_edge")):
            a, b = resource[1], resource[2]
            return (resource[0],) + tuple(sorted((a, b), key=repr))
        return resource

    def owner_at(self, resource: Hashable, start: float, end: float,
                 exclude_owner: Optional[int] = None) -> Optional[int]:
        key = self._canonical_conflict(resource)
        for item in self._items:
            if exclude_owner is not None and item.owner == exclude_owner:
                continue
            if self._canonical_conflict(item.resource) != key:
                continue
            if max(start, item.start) < min(end, item.end):
                return item.owner
        return None

    def reserve_batch(self, owner: int,
                      requests: Iterable[Tuple[Hashable, float, float]]) -> bool:
        pending = []
        for resource, start, end in requests:
            if end <= start:
                raise ValueError("reservation end must be after start")
            if self.owner_at(resource, start, end, exclude_owner=owner) is not None:
                return False
            pending.append(Reservation(owner, resource, start, end))
        self.release(owner)
        self._items.extend(pending)
        return True

    def release(self, owner: int) -> None:
        self._items = [item for item in self._items if item.owner != owner]

    def snapshot(self) -> tuple[Reservation, ...]:
        return tuple(self._items)

    def snapshot_owner(self, owner: int) -> tuple[Reservation, ...]:
        """Return an immutable snapshot of one owner's reservations."""
        return tuple(item for item in self._items if item.owner == owner)

    def restore_owner(self, owner: int,
                      reservations: Iterable[Reservation]) -> None:
        """Atomically replace one owner's entries with a prior snapshot."""
        restored = list(reservations)
        if any(item.owner != owner for item in restored):
            raise ValueError("reservation snapshot owner mismatch")
        self.release(owner)
        self._items.extend(restored)
