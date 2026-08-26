"""
data_loader.py
Carga de datos de entrada desde archivos CSV.

Se esperan estos archivos dentro de la carpeta data/:
  - specialties.csv : id, name, min_blocks
  - rooms.csv       : id, name, room_type, daily_capacity_minutes
  - procedures.csv  : id, name, specialty_id, required_room_type, estimated_duration
  - surgeons.csv    : id, name, specialty_id, available_days, contract_hours_week
  - patients.csv    : id, specialty_id, procedure_id, surgeon_id, clinical_priority

Las dos primeras etapas del flujo son:
  1) cargar los datos de entrada,
  2) dejarlos listos para que el algoritmo y el decoder los consuman.
"""
import csv
import os
from typing import List, Tuple

from models import Patient, Procedure, Room, Specialty, Surgeon


def _read_csv(path: str) -> List[dict]:
    """Lee un CSV y devuelve una lista de filas en formato diccionario."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_int(value, default=0) -> int:
    """Convierte un valor a entero y devuelve un default si falla."""
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def load_specialties(path: str) -> List[Specialty]:
    rows = _read_csv(path)
    return [
        Specialty(id=r["id"], name=r["name"], min_blocks=int(r["min_blocks"]))
        for r in rows
    ]


def load_rooms(path: str) -> List[Room]:
    rows = _read_csv(path)
    return [
        Room(
            id=r["id"],
            name=r["name"],
            room_type=int(r["room_type"]),
            daily_capacity_minutes=int(r["daily_capacity_minutes"]),
        )
        for r in rows
    ]


def load_procedures(path: str) -> List[Procedure]:
    rows = _read_csv(path)
    procedures: List[Procedure] = []
    for r in rows:
        procedures.append(
            Procedure(
                id=r["id"],
                name=r["name"],
                specialty_id=r["specialty_id"],
                required_room_type=int(r["required_room_type"]),
                estimated_duration=_safe_int(r.get("estimated_duration")),
            )
        )
    return procedures


def load_surgeons(path: str) -> List[Surgeon]:
    rows = _read_csv(path)
    surgeons = []
    for r in rows:
        days = {d.strip() for d in r["available_days"].split(";") if d.strip()}
        surgeons.append(
            Surgeon(
                id=r["id"],
                name=r["name"],
                specialty_id=r["specialty_id"],
                available_days=days,
                contract_hours_week=float(r["contract_hours_week"]),
            )
        )
    return surgeons


def load_patients(path: str) -> List[Patient]:
    rows = _read_csv(path)
    return [
        Patient(
            id=r["id"],
            specialty_id=r["specialty_id"],
            procedure_id=r["procedure_id"],
            surgeon_id=r.get("surgeon_id"),
            clinical_priority=float(r["clinical_priority"]),
        )
        for r in rows
    ]


def load_all(data_dir: str = "data") -> Tuple[List[Specialty], List[Room], List[Procedure], List[Surgeon], List[Patient]]:
    """Carga todas las entidades del problema a partir de los CSV de entrada."""
    specialties = load_specialties(os.path.join(data_dir, "specialties.csv"))
    rooms = load_rooms(os.path.join(data_dir, "rooms.csv"))
    procedures = load_procedures(os.path.join(data_dir, "procedures.csv"))
    surgeons = load_surgeons(os.path.join(data_dir, "surgeons.csv"))
    patients = load_patients(os.path.join(data_dir, "patients.csv"))
    return specialties, rooms, procedures, surgeons, patients
