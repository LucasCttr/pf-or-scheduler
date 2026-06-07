from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Lock
from typing import Any, Literal
from urllib import request as urllib_request
from urllib.error import URLError
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from genetic_algorithm import GeneticAlgorithm
from main import default_config, reconstruct_agenda
from models import GAConfig, OperatingRoom, Patient, Specialty, Staff

JobStatus = Literal["planning", "completed", "failed"]


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
    min_blocks: int = 1
    max_blocks: int = 10


class MedicalStaffPayload(BaseModel):
    id: int
    name: str
    role: str
    enabled_procedures_ids: list[int] = Field(default_factory=list)
    availability_hours: dict[str, list[int]] = Field(default_factory=dict)


class SchedulerConfigPayload(BaseModel):
    population_size: int | None = None
    max_generations: int | None = None
    convergence_patience: int | None = None
    mutation_rate: float | None = None
    crossover_rate: float | None = None
    tournament_size: int | None = None
    elite_count: int | None = None
    alpha: float | None = None
    beta: float | None = None
    n_days: int | None = None
    n_shifts: int | None = None
    block_duration_min: int | None = None
    slot_size_min: int | None = None
    penalty_below_min_quota: float | None = None
    penalty_above_max_quota: float | None = None
    parallel_workers: int | None = None


class PlanningRequest(BaseModel):
    week_start: str
    pending_surgeries: list[PendingSurgeryPayload]
    operating_rooms: list[OperatingRoomPayload]
    specialties: list[SpecialtyPayload]
    medical_staff: list[MedicalStaffPayload]
    procedures_by_specialty: dict[str, list[int]] = Field(default_factory=dict)
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


app = FastAPI(title="PF OR Scheduler", version="0.1.0")
_jobs: dict[str, PlanningJob] = {}
_jobs_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def create_app() -> FastAPI:
    return app


@app.post("/planning", response_model=PlanningStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def create_planning(payload: PlanningRequest) -> PlanningStatusResponse:
    job_uuid = str(uuid4())
    job = PlanningJob(uuid=job_uuid, status="planning")
    with _jobs_lock:
        _jobs[job_uuid] = job
    _executor.submit(_run_job, job_uuid, payload)
    return PlanningStatusResponse(uuid=job_uuid, status="planning")


@app.get("/planning/{job_uuid}", response_model=PlanningStatusResponse)
def get_planning_status(job_uuid: str) -> PlanningStatusResponse:
    with _jobs_lock:
        job = _jobs.get(job_uuid)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Planning job not found")
    return PlanningStatusResponse(uuid=job.uuid, status=job.status)


def _run_job(job_uuid: str, payload: PlanningRequest) -> None:
    start = time.perf_counter()
    try:
        result = run_planning(payload)
    except Exception as exc:  # pragma: no cover - defensive background boundary
        elapsed = round(time.perf_counter() - start, 3)
        _update_job(job_uuid, "failed", error_message=str(exc), duration_seconds=elapsed)
        _send_callback(
            {
                "uuid": job_uuid,
                "status": "failed",
                "output_payload": None,
                "error_message": str(exc),
                "duration_seconds": elapsed,
            }
        )
        return

    elapsed = round(time.perf_counter() - start, 3)
    result["duracion_segundos"] = elapsed
    _update_job(job_uuid, "completed", duration_seconds=elapsed)
    _send_callback(
        {
            "uuid": job_uuid,
            "status": "completed",
            "output_payload": result,
            "error_message": None,
            "duration_seconds": elapsed,
        }
    )


def run_planning(payload: PlanningRequest) -> dict[str, Any]:
    random.seed(42)
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(42)

    config = _build_config(payload.config)
    operating_rooms = [
        OperatingRoom(
            id=item.id,
            name=item.name,
            or_type=item.or_type,
            availability=item.availability,
        )
        for item in payload.operating_rooms
    ]
    specialties = [
        Specialty(
            id=item.id,
            name=item.name,
            compatible_or_types=item.compatible_or_types,
            min_blocks=item.min_blocks,
            max_blocks=item.max_blocks,
        )
        for item in payload.specialties
    ]
    staff_list = [
        Staff(
            id=item.id,
            name=item.name,
            role=item.role,
            enabled_procedures_ids=item.enabled_procedures_ids,
            availability_hours={int(day): tuple(hours) for day, hours in item.availability_hours.items()},
        )
        for item in payload.medical_staff
    ]

    patients_by_specialty: dict[int, list[Patient]] = {}
    for item in payload.pending_surgeries:
        patients_by_specialty.setdefault(item.specialty_id, []).append(
            Patient(
                id=item.id,
                specialty_id=item.specialty_id,
                procedure_id=item.procedure_id,
                estimated_duration=item.estimated_duration,
                clinical_priority=item.clinical_priority,
                required_roles=["cirujano"],
                forced_surgeon_id=item.forced_surgeon_id,
            )
        )

    ga = GeneticAlgorithm(config, operating_rooms, specialties, patients_by_specialty, staff_list)
    if payload.procedures_by_specialty:
        ga.procedures_by_specialty = {
            int(specialty_id): procedure_ids
            for specialty_id, procedure_ids in payload.procedures_by_specialty.items()
        }

    best = ga.run()
    agenda, _ = reconstruct_agenda(
        ga,
        best,
        patients_by_specialty,
        specialties,
        operating_rooms,
        staff_list,
        config,
        frontend_mode=False,
    )
    return agenda


def _build_config(config_payload: SchedulerConfigPayload | None) -> GAConfig:
    config = default_config()
    if config_payload is None:
        return config
    values = asdict(config)
    overrides = config_payload.model_dump(exclude_none=True)
    values.update(overrides)
    return GAConfig(**values)


def _update_job(
    job_uuid: str,
    status_value: JobStatus,
    *,
    error_message: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    with _jobs_lock:
        job = _jobs[job_uuid]
        job.status = status_value
        job.error_message = error_message
        job.duration_seconds = duration_seconds


def _send_callback(payload: dict[str, Any]) -> None:
    callback_url = os.getenv("BACK_CALLBACK_URL")
    if not callback_url:
        return

    token = os.getenv("SCHEDULER_CALLBACK_TOKEN", "dev-scheduler-token")
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        req = urllib_request.Request(
            callback_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Scheduler-Token": token,
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=15) as response:
                response.read()
            return
        except URLError as exc:  # pragma: no cover - external callback boundary
            last_error = exc
            if attempt < 3:
                time.sleep(1)

    print(f"Scheduler callback failed after 3 attempts: {last_error}")
