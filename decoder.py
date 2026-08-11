"""Deterministic decoder from specialty blocks to a feasible weekly agenda."""

from collections import defaultdict
from typing import Dict, List, Tuple

from models import Agenda, Block, Patient, Procedure, Room, ScheduledSurgery, Surgeon


def _earliest_start(
    candidate: int,
    duration: int,
    occupied: List[Tuple[int, int]],
    window_end: int,
) -> int | None:
    """Return the earliest non-overlapping start inside the availability window."""
    start = candidate
    for existing_start, existing_end in sorted(occupied):
        if start + duration <= existing_start:
            break
        if start < existing_end and existing_start < start + duration:
            start = existing_end
    return start if start + duration <= window_end else None


def build_agenda(
    chromosome: Dict[Block, str],
    patients: List[Patient],
    procedures: Dict[str, Procedure],
    surgeons: Dict[str, Surgeon],
    rooms: Dict[str, Room],
) -> Agenda:
    for patient in patients:
        patient.scheduled = False

    assignments = {block: [] for block in chromosome}
    used_time = {block: 0 for block in chromosome}
    room_clock = {
        block: rooms[block.room_id].day_start_minute
        for block in chromosome
    }
    surgeon_intervals: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        surgeon.id: defaultdict(list) for surgeon in surgeons.values()
    }
    surgeon_minutes = {surgeon.id: 0 for surgeon in surgeons.values()}

    patients_by_specialty: Dict[str, List[Patient]] = defaultdict(list)
    for patient in patients:
        patients_by_specialty[patient.specialty_id].append(patient)

    for block, specialty_id in chromosome.items():
        room = rooms[block.room_id]
        if not specialty_id or not room.is_available(block.day):
            continue

        block_end = room.day_start_minute + room.daily_capacity_minutes
        candidates = sorted(
            patients_by_specialty.get(specialty_id, []),
            key=lambda patient: (-patient.clinical_priority, patient.id),
        )

        for patient in candidates:
            if patient.scheduled:
                continue
            procedure = procedures.get(patient.procedure_id)
            surgeon = surgeons.get(patient.surgeon_id)
            if procedure is None or surgeon is None:
                continue
            if patient.specialty_id not in surgeon.specialty_ids:
                continue
            if procedure.specialty_id != patient.specialty_id:
                continue
            if procedure.required_room_type > room.room_type:
                continue

            availability = surgeon.availability_hours.get(block.day)
            if availability is None:
                continue
            availability_start, availability_end = availability
            window_start = max(room.day_start_minute, availability_start)
            window_end = min(block_end, availability_end)
            duration = patient.estimated_duration or procedure.estimated_duration
            if duration <= 0 or window_end - window_start < duration:
                continue
            contract_limit = surgeon.contract_minutes_week or 0
            if surgeon_minutes[surgeon.id] + duration > contract_limit:
                continue

            start = _earliest_start(
                max(room_clock[block], window_start),
                duration,
                surgeon_intervals[surgeon.id][block.day],
                window_end,
            )
            if start is None or start + duration > block_end:
                continue
            end = start + duration

            assignments[block].append(
                ScheduledSurgery(
                    patient_id=patient.id,
                    block=block,
                    surgeon_id=surgeon.id,
                    surgeon_name=surgeon.name,
                    duration=duration,
                    start_minute=start,
                    end_minute=end,
                )
            )
            used_time[block] += duration
            room_clock[block] = end
            surgeon_intervals[surgeon.id][block.day].append((start, end))
            surgeon_minutes[surgeon.id] += duration
            patient.scheduled = True

    return Agenda(assignments=assignments, used_time=used_time)
