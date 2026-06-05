"""
models.py — Estructuras de datos del dominio actualizadas.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class OperatingRoom:
    """Representa un quirófano disponible."""
    id: int
    name: str
    or_type: str  # "alta_complejidad" | "media_complejidad" | "baja_complejidad"
    availability: List[List[bool]] = field(default_factory=list)

    def __post_init__(self):
        if not self.availability:
            self.availability = [[True, True] for _ in range(5)]

@dataclass
class Specialty:
    """Representa una especialidad médica."""
    id: int        # 0 = bloque libre
    name: str
    compatible_or_types: List[str]
    min_blocks: int = 1
    max_blocks: int = 10

@dataclass
class Patient:
    id: int
    specialty_id: int
    procedure_id: int          # Código de procedimiento del nomenclador real (ICD/CIE)
    estimated_duration: int
    clinical_priority: float
    required_roles: List[str] = field(default_factory=list)
    forced_surgeon_id: Optional[int] = None

@dataclass
class Staff:
    id: int
    name: str
    role: str
    enabled_procedures_ids: List[int] = field(default_factory=list) # Matriz de competencias reales
    availability_hours: Dict[int, Tuple[int, int]] = field(default_factory=dict) # {dia_idx: (min_inicio, min_fin)}

    def get_range_for_block(self, day_idx: int, is_morning: bool) -> Tuple[int, int]:
        """Calcula la intersección horaria entre el contrato del médico y el bloque físico."""
        if day_idx not in self.availability_hours:
            return (0, 0)
        
        b_start, b_end = (480, 720) if is_morning else (780, 1020)
        s_start, s_end = self.availability_hours[day_idx]
        
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
    max_generations: int = 50
    convergence_patience: int = 7
    mutation_rate: float = 0.10
    crossover_rate: float = 0.85
    tournament_size: int = 10
    elite_count: int = 2
    
    alpha: float = 0.7  # Prioridad clínica
    beta: float = 0.3   # Utilización de tiempo
    
    n_days: int = 5
    n_shifts: int = 2
    block_duration_min: int = 240
    slot_size_min: int = 15
    penalty_below_min_quota: float = 50.0
    penalty_above_max_quota: float = 20.0
    parallel_workers: int = 24