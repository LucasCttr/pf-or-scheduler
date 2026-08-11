"""Domain models for the GA + deterministic decoder scheduler."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class Specialty:
    id: str
    name: str
    min_blocks: int = 0
    max_blocks: int = 999


@dataclass
class Surgeon:
    id: str
    name: str
    specialty_ids: Set[str]
    availability_hours: Dict[str, Tuple[int, int]]
    contract_minutes_week: int | None = None

    def __post_init__(self) -> None:
        if self.contract_minutes_week is None:
            self.contract_minutes_week = sum(end - start for start, end in self.availability_hours.values())

    @property
    def available_days(self) -> Set[str]:
        return set(self.availability_hours)

    @property
    def contract_hours_week(self) -> float:
        return (self.contract_minutes_week or 0) / 60

    @property
    def specialty_id(self) -> str:
        return sorted(self.specialty_ids)[0] if self.specialty_ids else ""


@dataclass
class Room:
    id: str
    name: str
    room_type: int
    daily_capacity_minutes: int
    available_days: Set[str] | None = None
    day_start_minute: int = 480

    def is_available(self, day: str) -> bool:
        return self.available_days is None or day in self.available_days


@dataclass
class Procedure:
    id: str
    name: str
    specialty_id: str
    required_room_type: int
    estimated_duration: int = 0


@dataclass
class Patient:
    id: str
    specialty_id: str
    procedure_id: str
    surgeon_id: str
    clinical_priority: float
    estimated_duration: int | None = None
    scheduled: bool = False


@dataclass(frozen=True)
class Block:
    day: str
    room_id: str


@dataclass
class ScheduledSurgery:
    patient_id: str
    block: Block
    surgeon_id: str
    surgeon_name: str
    duration: int
    start_minute: int
    end_minute: int


@dataclass
class Agenda:
    assignments: Dict[Block, List[ScheduledSurgery]]
    used_time: Dict[Block, int]

    def all_surgeries(self) -> List[ScheduledSurgery]:
        return [surgery for surgeries in self.assignments.values() for surgery in surgeries]
