"""
models.py — Estructuras de datos del dominio.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class OperatingRoom:
    """Representa un quirófano disponible."""
    id: int
    name: str
    or_type: str  # "alta_complejidad" | "media_complejidad" | "baja_complejidad"
    # availability[dia][turno] = True si el quirófano está habilitado
    availability: List[List[bool]] = field(default_factory=list)

    def __post_init__(self):
        if not self.availability:
            # Por defecto: disponible todos los días, ambos turnos
            self.availability = [[True, True] for _ in range(5)]


@dataclass
class Specialty:
    """Representa una especialidad médica."""
    id: int        # 0 = bloque libre
    name: str
    compatible_or_types: List[str]   # tipos de quirófano que puede usar
    min_blocks: int = 1              # mínimo de bloques semanales garantizados
    max_blocks: int = 10             # máximo de bloques semanales permitidos


@dataclass
class Patient:
    """Representa un paciente en lista de espera (entrada al Nivel 2 y MIP)."""
    id: int
    specialty_id: int
    estimated_duration: int     # duración estimada de la cirugía en minutos
    clinical_priority: float    # peso combinado de urgencia + tiempo de espera
    required_roles: List[str] = field(default_factory=list)  # ["cirujano", "anestesista", ...]   FALTA IMPLEMENTAR


@dataclass
class Staff:
    id: int
    name: str
    role: str  # "cirujano", "anestesista", etc.
    specialty_id: int # Para los cirujanos
    availability: List[List[bool]] # No todos están todos los días


@dataclass
class GAConfig:
    """Todos los parámetros configurables del Algoritmo Genético."""
    population_size: int = 50
    max_generations: int = 300
    convergence_patience: int = 40   # generaciones consecutivas sin mejora para parar
    mutation_rate: float = 0.10
    crossover_rate: float = 0.85
    tournament_size: int = 5
    elite_count: int = 2

    n_days: int = 5       # Lunes a Viernes
    n_shifts: int = 2     # 0 = Mañana, 1 = Tarde
    block_duration_min: int = 480   # minutos disponibles por bloque (8 hs)

    # Penalizaciones en el fitness global
    penalty_below_min_quota: float = 50.0   # por cada bloque faltante bajo el mínimo
    penalty_above_max_quota: float = 20.0   # por cada bloque excedente sobre el máximo
