"""
models.py
Estructuras de datos del problema de programación quirúrgica.

Estas entidades representan la demanda, la oferta y la solución intermedia
que produce el algoritmo antes de convertirla en una agenda final.
"""
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Specialty:
    """Especialidad médica asociada a la agenda quirúrgica."""
    id: str
    name: str
    min_blocks: int = 0  # mínimo de bloques semanales garantizados


@dataclass
class Surgeon:
    """Profesional responsable de las cirugías candidatas."""
    id: str
    name: str
    specialty_id: str
    available_days: Set[str]
    contract_hours_week: float


@dataclass
class Room:
    """Quirófano físico disponible para la programación semanal."""
    id: str
    name: str
    room_type: int  # mayor valor = mayor complejidad soportada
    daily_capacity_minutes: int


@dataclass
class Procedure:
    """Procedimiento quirúrgico asociado a un paciente."""
    id: str
    name: str
    specialty_id: str
    required_room_type: int
    estimated_duration: int  # duración estimada en minutos


@dataclass
class Patient:
    """Paciente pendiente de intervención quirúrgica."""
    id: str
    specialty_id: str
    procedure_id: str
    surgeon_id: str
    clinical_priority: float  # combinación de prioridad clínica y tiempo en espera
    scheduled: bool = False  # estado temporal usado por el decoder


@dataclass(frozen=True)
class Block:
    """
    Bloque quirúrgico: combinación de un día y un quirófano.
    B = (día, quirófano)
    """
    day: str
    room_id: str

    def __repr__(self):
        return f"({self.day}, {self.room_id})"


@dataclass
class ScheduledSurgery:
    """Cirugía efectivamente programada dentro de un bloque."""
    patient_id: str
    block: Block
    duration: int
    start_time: int = 0  # minutos desde el inicio del bloque
    end_time: int = 0    # minutos desde el inicio del bloque (start + duration)


@dataclass
class Agenda:
    """Resultado del decoder: agenda quirúrgica semanal completa."""
    assignments: Dict[Block, List[ScheduledSurgery]]
    used_time: Dict[Block, int]  # minutos utilizados por bloque

    def all_surgeries(self) -> List[ScheduledSurgery]:
        result: List[ScheduledSurgery] = []
        for surgeries in self.assignments.values():
            result.extend(surgeries)
        return result