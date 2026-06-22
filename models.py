"""
models.py
Estructuras de datos del problema de programacion quirurgica.
Basado en la seccion 10.4.2 (Modelado de los datos de entrada) y
10.4.3 (Representacion de las soluciones) del documento.
"""
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Specialty:
    """Especialidad medica. Unidad de asignacion del Algoritmo Genetico."""
    id: str
    name: str
    min_blocks: int = 0  # cantidad minima de bloques semanales garantizados


@dataclass
class Surgeon:
    """Profesional medico responsable de las cirugias."""
    id: str
    name: str
    specialty_id: str
    available_days: Set[str]
    contract_hours_week: float


@dataclass
class Room:
    """Quirofano fisico disponible."""
    id: str
    name: str
    room_type: int  # nivel de complejidad soportado (mayor = mas complejo)
    daily_capacity_minutes: int


@dataclass
class Procedure:
    """Procedimiento quirurgico."""
    id: str
    name: str
    specialty_id: str
    required_room_type: int


@dataclass
class Patient:
    """Paciente pendiente de intervencion quirurgica."""
    id: str
    specialty_id: str
    procedure_id: str
    surgeon_id: str
    estimated_duration: int  # minutos
    clinical_priority: float  # combina prioridad clinica y tiempo en espera
    scheduled: bool = False  # estado utilizado durante la construccion de agenda


@dataclass(frozen=True)
class Block:
    """
    Bloque quirurgico: unidad indivisible de asignacion.
    Definido por la combinacion de un dia y un quirofano: B = (d, q)
    """
    day: str
    room_id: str

    def __repr__(self):
        return f"({self.day}, {self.room_id})"


@dataclass
class ScheduledSurgery:
    """Cirugia efectivamente programada dentro de un bloque."""
    patient_id: str
    block: Block
    duration: int


@dataclass
class Agenda:
    """Resultado del decoder: agenda quirurgica semanal completa."""
    assignments: Dict[Block, List[ScheduledSurgery]]
    used_time: Dict[Block, int]  # minutos utilizados por bloque

    def all_surgeries(self) -> List[ScheduledSurgery]:
        result: List[ScheduledSurgery] = []
        for surgeries in self.assignments.values():
            result.extend(surgeries)
        return result
