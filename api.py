from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Literal
from urllib import request as urllib_request
from urllib.error import URLError
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from genetic_algorithm import GeneticAlgorithm
from models import Block, Patient, Procedure, Room, Specialty, Surgeon

JobStatus = Literal["planning", "completed", "failed"]
DAY_IDS = ["lunes", "martes", "miercoles", "jueves", "viernes"]
DAY_LABELS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
ROOM_TYPE_RANK = {"baja_complejidad": 1, "media_complejidad": 2, "alta_complejidad": 3}

# El decoder (decoder.py) emite start_time/end_time como minutos RELATIVOS
# al inicio de la jornada del bloque (0 = apertura del quirófano). Para
# serializar horas de reloj absolutas hay que sumar el inicio de jornada.
# Todos los quirófanos comparten la misma franja horaria (supuesto ya
# implícito en el decoder), por lo que el offset es único.
DAY_START_MINUTE = 540  # 09:00: inicio de la jornada de los quirófanos
SLOT_BASE_MINUTE = 480  # 08:00: base de la cuadrícula de slots (slot 0 = 08:00)


class PendingSurgeryPayload(BaseModel):
    id: int
    specialty_id: int
    procedure_id: int
    estimated_duration: int
    clinical_priority: float
    forced_surgeon_id: int | None = None


class OperatingRoomPayload(BaseModel):
    id: int
    name: str
    or_type: str
    availability: list[list[bool]]


class SpecialtyPayload(BaseModel):
    id: int
    name: str
    compatible_or_types: list[str]
    min_blocks: int = 0
    max_blocks: int = 999


class ProcedurePayload(BaseModel):
    id: int
    name: str
    specialty_id: int
    required_room_type: str


class MedicalStaffPayload(BaseModel):
    id: int
    name: str
    role: str
    enabled_procedures_ids: list[int] = Field(default_factory=list)
    availability_hours: dict[str, list[int]] = Field(default_factory=dict)
    main_specialty_id: int = 0
    specialties_ids: list[int] = Field(default_factory=list)


class SchedulerConfigPayload(BaseModel):
    population_size: int | None = None
    max_generations: int | None = None
    convergence_patience: int | None = None
    mutation_rate: float | None = None
    crossover_rate: float | None = None
    tournament_size: int | None = None
    alpha: float | None = None
    beta: float | None = None
    n_days: int | None = None
    block_duration_min: int | None = None
    slot_size_min: int | None = None


class PlanningRequest(BaseModel):
    week_start: str
    pending_surgeries: list[PendingSurgeryPayload]
    operating_rooms: list[OperatingRoomPayload]
    specialties: list[SpecialtyPayload]
    medical_staff: list[MedicalStaffPayload]
    procedures_by_specialty: dict[str, list[ProcedurePayload]] = Field(default_factory=dict)
    config: SchedulerConfigPayload | None = None
    id_maps: dict[str, Any] | None = None


class PlanningStatusResponse(BaseModel):
    uuid: str
    status: JobStatus


class PlanningJob(BaseModel):
    uuid: str
    status: JobStatus
    error_message: str | None = None
    duration_seconds: float | None = None


app = FastAPI(title="PF OR Scheduler Decoder", version="1.0.0")
_jobs: dict[str, PlanningJob] = {}
_jobs_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1)


@app.post("/planning", response_model=PlanningStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def create_planning(payload: PlanningRequest) -> PlanningStatusResponse:
    job_uuid = str(uuid4())
    with _jobs_lock:
        _jobs[job_uuid] = PlanningJob(uuid=job_uuid, status="planning")
    _executor.submit(_run_job, job_uuid, payload)
    return PlanningStatusResponse(uuid=job_uuid, status="planning")


@app.get("/planning/{job_uuid}", response_model=PlanningStatusResponse)
def get_planning_status(job_uuid: str) -> PlanningStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_uuid)
    if job is None:
        raise HTTPException(status_code=404, detail="Planning job not found")
    return PlanningStatusResponse(uuid=job.uuid, status=job.status)


def _run_job(job_uuid: str, payload: PlanningRequest) -> None:
    started = time.perf_counter()
    try:
        result = run_planning(payload)
    except Exception as exc:  # pragma: no cover - background boundary
        elapsed = round(time.perf_counter() - started, 3)
        _send_callback({
            "uuid": job_uuid,
            "status": "failed",
            "output_payload": None,
            "error_message": str(exc),
            "duration_seconds": elapsed,
        })
        _update_job(job_uuid, "failed", str(exc), elapsed)
        return
    elapsed = round(time.perf_counter() - started, 3)
    result["duracion_segundos"] = elapsed
    _send_callback({
        "uuid": job_uuid,
        "status": "completed",
        "output_payload": result,
        "error_message": None,
        "duration_seconds": elapsed,
    })
    _update_job(job_uuid, "completed", None, elapsed)


def run_planning(payload: PlanningRequest) -> dict[str, Any]:
    random.seed(42)
    config = payload.config or SchedulerConfigPayload()
    days = DAY_IDS[: config.n_days or 5]
    capacity = config.block_duration_min or 300

    rooms = [
        Room(
            id=str(item.id),
            name=item.name,
            room_type=ROOM_TYPE_RANK.get(item.or_type, 2),
            daily_capacity_minutes=capacity,
            # availability[day] es una lista con un unico valor booleano
            # (un solo turno por dia): indica si ese quirofano opera ese
            # dia de la semana. Si la lista viene vacia para un dia, se
            # asume no disponible ese dia.
            available_days={
                DAY_IDS[index]
                for index, day_availability in enumerate(item.availability[:len(DAY_IDS)])
                if day_availability and day_availability[0]
            },
        )
        for item in payload.operating_rooms
    ]
    specialties = [
        Specialty(str(item.id), item.name, item.min_blocks)
        for item in payload.specialties
        if item.id != 0
    ]
    durations_by_procedure = {
        str(item.procedure_id): item.estimated_duration
        for item in payload.pending_surgeries
    }
    procedures = [
        Procedure(
            id=str(item.id),
            name=item.name,
            specialty_id=str(item.specialty_id),
            required_room_type=ROOM_TYPE_RANK.get(item.required_room_type, 2),
            estimated_duration=durations_by_procedure.get(str(item.id), 0),
        )
        for items in payload.procedures_by_specialty.values()
        for item in items
    ]
    surgeons = []
    for item in payload.medical_staff:
        if item.role.lower() != "cirujano":
            continue
        specialties_ids = item.specialties_ids or ([item.main_specialty_id] if item.main_specialty_id else [])

        valid_hours = {
            day: hours
            for day, hours in item.availability_hours.items()
            if day.isdigit() and 0 <= int(day) < len(DAY_IDS) and len(hours) == 2
        }

        available_days = {DAY_IDS[int(day)] for day in valid_hours}

        # Horas contractuales semanales: suma de la franja horaria declarada
        # por el cliente para cada día disponible, no un valor fijo.
        contract_hours_week = sum(
            (hours[1] - hours[0]) / 60
            for hours in valid_hours.values()
        )

        surgeons.append(
            Surgeon(
                id=str(item.id),
                name=item.name,
                specialty_id=str(specialties_ids[0]) if specialties_ids else "",
                available_days=available_days,
                contract_hours_week=contract_hours_week,
            )
        )

    patients = []
    for item in payload.pending_surgeries:
        if item.forced_surgeon_id is None:
            raise ValueError(f"La cirugía {item.id} no tiene cirujano asignado")
        patients.append(
            Patient(
                id=str(item.id),
                specialty_id=str(item.specialty_id),
                procedure_id=str(item.procedure_id),
                surgeon_id=str(item.forced_surgeon_id),
                clinical_priority=item.clinical_priority,
            )
        )

    ga = GeneticAlgorithm(
        days=days,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=patients,
        population_size=config.population_size or 30,
        generations=config.max_generations or 30,
        tournament_size=config.tournament_size or 3,
        crossover_rate=config.crossover_rate or 0.85,
        mutation_rate=config.mutation_rate or 0.10,
        stagnation_limit=config.convergence_patience or 7,
        alpha=config.alpha or 1.0,
        beta=config.beta or 0.3,
    )
    chromosome, fitness, agenda = ga.run()
    return _serialize_result(days, rooms, specialties, surgeons, chromosome, fitness, agenda, patients, config.slot_size_min or 15)


def _serialize_result(days, rooms, specialties, surgeons, chromosome, fitness, agenda, patients, slot_size):
    specialty_names = {item.id: item.name for item in specialties}
    surgeon_names_by_id = {surgeon.id: surgeon.name for surgeon in surgeons}
    patients_by_id = {item.id: item for item in patients}
    scheduled_ids = {item.patient_id for item in agenda.all_surgeries()}
    output_days = []
    for index, day in enumerate(days):
        blocks = []
        for room in rooms:
            block = Block(day, room.id)
            surgeries = agenda.assignments.get(block, [])
            blocks.append({
                "quirofano": room.name,
                "turno": "Jornada Completa",
                "especialidad": specialty_names.get(chromosome.get(block, ""), "Libre"),
                "utilizacion_porcentaje": round(agenda.used_time.get(block, 0) / room.daily_capacity_minutes * 100, 2),
                "cronograma": [
                    {
                        "paciente_id": int(item.patient_id),
                        "medico": surgeon_names_by_id.get(patients_by_id[item.patient_id].surgeon_id, ""),
                        # item.start_time / item.end_time son minutos
                        # relativos al inicio de la jornada (0 = apertura),
                        # calculados por el decoder. Aqui se suma el offset
                        # de inicio de jornada para obtener horas de reloj.
                        "slot_inicio": max(0, (DAY_START_MINUTE + item.start_time - SLOT_BASE_MINUTE) // slot_size),
                        "hora_inicio": _format_minute(DAY_START_MINUTE + item.start_time),
                        "hora_fin": _format_minute(DAY_START_MINUTE + item.end_time),
                        "duracion": item.duration,
                    }
                    for item in surgeries
                ],
            })
        output_days.append({"nombre": DAY_LABELS[index], "bloques": blocks})
    pending = sorted(int(patient.id) for patient in patients if patient.id not in scheduled_ids)
    return {
        "fitness_total": round(fitness, 4),
        "dias": output_days,
        "resumen": {
            "total_pacientes": len(patients),
            "pacientes_programados": len(scheduled_ids),
            "pacientes_pendientes": len(pending),
            "ids_pendientes": pending,
        },
    }


def _format_minute(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def _update_job(job_uuid: str, status_value: JobStatus, error: str | None, duration: float) -> None:
    with _jobs_lock:
        job = _jobs[job_uuid]
        job.status = status_value
        job.error_message = error
        job.duration_seconds = duration


def _send_callback(payload: dict[str, Any]) -> None:
    callback_url = os.getenv("BACK_CALLBACK_URL")
    if not callback_url:
        return
    body = json.dumps(payload).encode("utf-8")
    token = os.getenv("SCHEDULER_CALLBACK_TOKEN", "dev-scheduler-token")
    last_error = None
    for attempt in range(3):
        req = urllib_request.Request(
            callback_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-Scheduler-Token": token},
        )
        try:
            with urllib_request.urlopen(req, timeout=15) as response:
                response.read()
            return
        except URLError as exc:  # pragma: no cover - external boundary
            last_error = exc
            if attempt < 2:
                time.sleep(1)
    print(f"Scheduler callback failed after 3 attempts: {last_error}")