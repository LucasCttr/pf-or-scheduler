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
class Procedure:
    """Representa un procedimiento quirúrgico del nomenclador."""
    id: int
    name: str
    specialty_id: int
    required_room_type: str  # "alta_complejidad" | "media_complejidad" | "baja_complejidad"

@dataclass
class Patient:
    id: int
    specialty_id: int
    procedure_id: int
    estimated_duration: int
    clinical_priority: float
    required_roles: List[str] = field(default_factory=list)
    forced_surgeon_id: Optional[int] = None
    
@dataclass
class Staff:
    id: int
    name: str
    role: str
    main_specialty_id: int = 0
    enabled_procedures_ids: List[int] = field(default_factory=list)
    
    # NUEVOS ATRIBUTOS
    max_minutos_semanales: int = 2400  # Por defecto 40 horas
    minutos_consumidos: int = 0
    dias_disponibles: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    def get_available_minutes_in_block(
        self, day_idx: int, is_morning: bool = True, block_duration: Optional[int] = None
    ) -> int:
        """
        Calcula cuánto tiempo le queda al médico para operar, 
        limitado por su bolsa semanal y si el día es hábil.
        """
        # 1. ¿Está el médico disponible este día?
        if day_idx not in self.dias_disponibles:
            return 0
        
        # 2. ¿Cuánto le queda de su bolsa semanal?
        restante = self.max_minutos_semanales - self.minutos_consumidos
        
        # If a block duration is provided, the doctor cannot use more than that
        if block_duration is not None:
            return max(0, min(restante, block_duration))

        # Retorna el límite de la bolsa o 0 si ya se agotó
        return max(0, restante)

    def consumir_minutos(self, minutos: int):
        """Registra el tiempo tras una asignación exitosa."""
        self.minutos_consumidos += minutos

@dataclass
class GAConfig:
    """Todos los parámetros configurables del Algoritmo Genético."""
    population_size: int = 60
    max_generations: int = 80
    convergence_patience: int = 10
    mutation_rate: float = 0.08
    crossover_rate: float = 0.85
    tournament_size: int = 8
    elite_count: int = 3
    alpha: float = 0.7  
    beta: float = 0.3   
    n_days: int = 5
    n_shifts: int = 2
    block_duration_min: int = 720
    slot_size_min: int = 15
    penalty_below_min_quota: float = 50.0
    penalty_above_max_quota: float = 20.0
    parallel_workers: int = 4