"""
models.py — Estructuras de datos del dominio.
"""
from dataclasses import dataclass, field
from typing import List, Optional


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
    id: int
    specialty_id: int
    estimated_duration: int
    clinical_priority: float
    required_roles: List[str] = field(default_factory=list)
    # ID del médico asignado (opcional)
    forced_surgeon_id: Optional[int] = None

from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Staff:
    id: int
    name: str
    role: str  # "cirujano", "anestesista", etc.
    specialties_ids: List[int]
    # Disponibilidad: {día_idx: (inicio_minutos, fin_minutos)}
    # Ejemplo: {0: (480, 720)} es Lunes de 08:00 a 12:00
    availability_hours: Dict[int, Tuple[int, int]]

    def get_range_for_block(self, day_idx: int, is_morning: bool) -> Tuple[int, int]:
        """Calcula el solapamiento entre el turno del médico y el bloque del quirófano."""
        if day_idx not in self.availability_hours:
            return (0, 0)
        
        # Bloques estándar: 08:00-12:00 (480-720) y 13:00-17:00 (780-1020)
        b_start, b_end = (480, 720) if is_morning else (780, 1020)
        s_start, s_end = self.availability_hours[day_idx]
        
        # Intersección de rangos
        overlap_start = max(b_start, s_start)
        overlap_end = min(b_end, s_end)
        
        return (overlap_start, overlap_end) if overlap_start < overlap_end else (0, 0)

    def get_available_minutes_in_block(self, day_idx: int, is_morning: bool) -> int:
        start, end = self.get_range_for_block(day_idx, is_morning)
        return end - start


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
    
    alpha: float = 0.7  # Peso para la prioridad clínica
    beta: float = 0.3   # Peso para la utilización de tiempo
    
    n_days: int = 5       # Lunes a Viernes
    n_shifts: int = 2     # 0 = Mañana, 1 = Tarde
    block_duration_min: int = 240   # minutos disponibles por bloque 

    # Penalizaciones en el fitness global
    penalty_below_min_quota: float = 50.0   # por cada bloque faltante bajo el mínimo
    penalty_above_max_quota: float = 30.0   # por cada bloque excedente sobre el máximo
