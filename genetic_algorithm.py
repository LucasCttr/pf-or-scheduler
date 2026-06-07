"""
genetic_algorithm.py — Nivel 1: Algoritmo Genético optimizado para Slots y Competencias.
"""
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Tuple, Dict, Optional
import numpy as np

from models import OperatingRoom, Specialty, Procedure, Patient, GAConfig, Staff
from mip import solve_mip_for_shift

# ═══════════════════════════════════════════════════════════════════════
# 1. CLASE INDIVIDUAL
# ═══════════════════════════════════════════════════════════════════════

class Individual:
    def __init__(self, chromosome: np.ndarray):
        self.chromosome: np.ndarray = chromosome.copy()
        self.fitness: float = -np.inf

    def copy(self) -> "Individual":
        ind = Individual(self.chromosome.copy())
        ind.fitness = self.fitness
        return ind

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.4f})"


# ═══════════════════════════════════════════════════════════════════════
# 2. CLASE GENETIC ALGORITHM
# ═══════════════════════════════════════════════════════════════════════

class GeneticAlgorithm:
    DAY_NAMES   = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    SHIFT_NAMES = ["Mañana", "Tarde"]

    def __init__(
        self,
        config: GAConfig,
        operating_rooms: List[OperatingRoom],
        specialties: List[Specialty],
        patients_by_specialty: Dict[int, List[Patient]],
        staff_list: List[Staff],
        procedures_by_specialty: Optional[Dict[int, List[Procedure]]] = None,
    ):
        self.cfg = config
        self.operating_rooms = operating_rooms  
        self.specialties = specialties
        self.patients_by_specialty = patients_by_specialty
        self.staff_list = staff_list 

        self.n_ors    = len(operating_rooms)
        self.n_days   = config.n_days
        self.n_shifts = config.n_shifts

        self._spec_by_id: Dict[int, Specialty]    = {s.id: s for s in specialties}
        self._or_by_idx:  Dict[int, OperatingRoom] = dict(enumerate(operating_rooms))
        self._real_spec_ids: List[int] = [s.id for s in specialties if s.id != 0]

        self.best_individual: Optional[Individual] = None
        self.history: List[float] = []

        self._mip_cache: Dict = {}
        self._cache_hits: int = 0
        self._fitness_cache: Dict[bytes, float] = {}
        self._fitness_cache_hits: int = 0
        self._cache_lock: Lock = Lock()
        
        self.slot_size = config.slot_size_min
        self.procedures_by_specialty = procedures_by_specialty or {}
        self._procedure_by_id: Dict[int, Procedure] = {
            proc.id: proc
            for procs in self.procedures_by_specialty.values()
            for proc in procs
        }

    def _specialty_procedures(self, specialty_id: int) -> List[Procedure]:
        return self.procedures_by_specialty.get(specialty_id, [])

    def _procedure_room_type(self, procedure_id: int) -> Optional[str]:
        procedure = self._procedure_by_id.get(procedure_id)
        return procedure.required_room_type if procedure is not None else None

    def _compatible_procedures_for_room(self, specialty_id: int, or_type: str) -> List[Procedure]:
        return [
            proc
            for proc in self._specialty_procedures(specialty_id)
            if {"baja_complejidad": 0, "media_complejidad": 1, "alta_complejidad": 2}.get(proc.required_room_type, 0) \
            <= {"baja_complejidad": 0, "media_complejidad": 1, "alta_complejidad": 2}.get(or_type, 0)
        ]

    def _specialty_valid_for_or(self, specialty_id: int, or_idx: int, day: int, shift: int) -> bool:
        if specialty_id == 0:
            return True

        or_obj = self._or_by_idx[or_idx]
        if not or_obj.availability[day][shift]:
            return False

        compatible_procs = self._compatible_procedures_for_room(specialty_id, or_obj.or_type)
        if not compatible_procs:
            return False

        is_morning = (shift == 0)
        compatible_proc_ids = {proc.id for proc in compatible_procs}

        for staff in self.staff_list:
            # CAMBIO CRÍTICO: Para validar la especialidad macro en la grilla del AG, 
            # el médico debe tener esta especialidad como principal (main_specialty_id)
            if staff.role == "cirujano" and staff.main_specialty_id == specialty_id:
                has_competence = any(pid in staff.enabled_procedures_ids for pid in compatible_proc_ids)
                if has_competence and staff.get_available_minutes_in_block(day, is_morning) > 0:
                    return True

        return False

    # ─── 2.1 INICIALIZACIÓN ───────────────────────────────────────────────

    def initialize_population(self) -> List[Individual]:
        return [self._create_individual() for _ in range(self.cfg.population_size)]

    def _create_individual(self) -> Individual:
        chrom = np.zeros((self.n_days, self.n_shifts, self.n_ors), dtype=int)
        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    chrom[d, t, q] = self._random_specialty_for(q, d, t)
        return Individual(chrom)

    def _random_specialty_for(self, or_idx: int, day: int, shift: int) -> int:
        options = self._valid_specialties_for(or_idx, day, shift)
        if not options or random.random() < 0.15:
            return 0
        return random.choice(options)

    def _valid_specialties_for(self, or_idx: int, day: int, shift: int) -> List[int]:
        or_obj = self._or_by_idx[or_idx]
        
        if not or_obj.availability[day][shift]:
            return [0]

        valid = [0]
        for specialty in self.specialties:
            if self._specialty_valid_for_or(specialty.id, or_idx, day, shift):
                valid.append(specialty.id)
                
        return valid

    # ─── 2.2 CONSTRUCCIÓN DE BLOQUES POR TURNO ────────────────────────────

    @staticmethod
    def _make_shift_cache_key(blocks: List[Dict], day_idx: int, is_morning: bool,
                               alpha: float, beta: float, slot_size: int) -> tuple:
        block_keys = tuple(
            (
                b["or_idx"],
                b["spec_id"],
                tuple(sorted(s.id for s in b["surgeons"])),
                tuple(sorted(p.id for p in b["patients"])),
            )
            for b in blocks
        )
        return (day_idx, is_morning, alpha, beta, slot_size, block_keys)

    def _build_shift_blocks(
        self,
        chrom: np.ndarray,
        d: int,
        t: int,
        is_morning: bool,
        pacientes_excluidos: set
    ) -> List[Dict]:
        blocks = []
        for q in range(self.n_ors):
            spec_id = int(chrom[d, t, q])
            or_     = self._or_by_idx[q]

            if not or_.availability[d][t] or spec_id == 0:
                blocks.append({"or_idx": q, "spec_id": 0,
                                "patients": [], "surgeons": [], "t_max": 0})
                continue

            compatible_procs = self._compatible_procedures_for_room(spec_id, or_.or_type)
            compatible_proc_ids = {proc.id for proc in compatible_procs}
            if not compatible_proc_ids:
                blocks.append({"or_idx": q, "spec_id": spec_id,
                                "patients": [], "surgeons": [], "t_max": 0})
                continue

            # CAMBIO CRÍTICO: Filtramos cirujanos RESTRINGIENDO estrictamente a los que 
            # tienen asignada esta especialidad como su especialidad principal (main_specialty_id)
            surgeons = [
                s for s in self.staff_list
                if s.role == "cirujano"
                and s.main_specialty_id == spec_id
                and any(pid in s.enabled_procedures_ids for pid in compatible_proc_ids)
                and s.get_available_minutes_in_block(d, is_morning, self.cfg.block_duration_min) > 0
            ]

            if not surgeons:
                blocks.append({"or_idx": q, "spec_id": spec_id,
                                "patients": [], "surgeons": [], "t_max": 0})
                continue

            surgeons_ids = {s.id for s in surgeons}
            patients = [
                p for p in self.patients_by_specialty.get(spec_id, [])
                if p.id not in pacientes_excluidos
                and getattr(p, "procedure_id", None) in compatible_proc_ids
                and (getattr(p, "forced_surgeon_id", None) is None
                     or p.forced_surgeon_id in surgeons_ids)
            ]

            blocks.append({
                "or_idx"  : q,
                "spec_id" : spec_id,
                "patients": patients,
                "surgeons": surgeons,
                "t_max"   : self.cfg.block_duration_min,
            })
        return blocks

    def _get_shift_result(self, blocks: List[Dict], d: int, is_morning: bool) -> Optional[Dict]:
        if not any(b["surgeons"] and b["patients"] for b in blocks):
            return None

        key = self._make_shift_cache_key(blocks, d, is_morning, self.cfg.alpha, self.cfg.beta, self.slot_size)

        with self._cache_lock:
            result = self._mip_cache.get(key)
            if result is not None:
                self._cache_hits += 1
                return result

        result = solve_mip_for_shift(blocks, d, is_morning, self.cfg.alpha, self.cfg.beta, slot_size=self.slot_size)
        with self._cache_lock:
            self._mip_cache[key] = result
        return result

    # ─── 2.3 EVALUACIÓN DE FITNESS ────────────────────────────────────────

    @staticmethod
    def _chromosome_key(chrom: np.ndarray) -> bytes:
        return chrom.tobytes()

    def evaluate_fitness(self, individual: Individual) -> float:
        chrom = individual.chromosome
        chrom_key = self._chromosome_key(chrom)
        
        with self._cache_lock:
            cached_f = self._fitness_cache.get(chrom_key)
            if cached_f is not None:
                self._fitness_cache_hits += 1
                individual.fitness = cached_f
                return cached_f

        total_z = 0.0
        pacientes_operados: set = set()

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                is_morning = (t == 0)
                blocks = self._build_shift_blocks(chrom, d, t, is_morning, pacientes_operados)
                
                result = self._get_shift_result(blocks, d, is_morning)
                if result is not None:
                    total_z += result["fitness"]
                    pacientes_operados.update(result["all_pacientes_ids"])
                
        fitness = total_z - self._global_penalty(chrom)
        individual.fitness = fitness
        with self._cache_lock:
            self._fitness_cache[chrom_key] = fitness
        return fitness

    # ─── 2.4 OPERADORES GENÉTICOS ─────────────────────────────────────────

    def get_schedule_details(self, individual: Individual) -> Dict[Tuple[int, int, int], dict]:
        chrom = individual.chromosome
        cache: Dict[Tuple[int, int, int], dict] = {}
        pacientes_operados: set = set()

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                is_morning = (t == 0)
                blocks = self._build_shift_blocks(chrom, d, t, is_morning, pacientes_operados)
                
                if any(b["surgeons"] and b["patients"] for b in blocks):
                    result = self._get_shift_result(blocks, d, is_morning)
                    if result is None:
                        for q in range(self.n_ors):
                            cache[(d, t, q)] = {"pacientes_ids": [], "asignaciones": [], "uso_tiempo": 0, "utilizacion_porcentaje": 0}
                    else:
                        pacientes_operados.update(result["all_pacientes_ids"])
                        for q in range(self.n_ors):
                            cache[(d, t, q)] = result["per_or"].get(q)
                else:
                    for q in range(self.n_ors):
                        cache[(d, t, q)] = {"pacientes_ids": [], "asignaciones": [], "uso_tiempo": 0, "utilizacion_porcentaje": 0}

        return cache

    def tournament_selection(self, population: List[Individual]) -> Individual:
        contestants = random.sample(population, self.cfg.tournament_size)
        return max(contestants, key=lambda ind: ind.fitness).copy()

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.cfg.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        c1, c2 = parent1.chromosome.copy(), parent2.chromosome.copy()
        cut = random.randint(1, self.n_days - 1)
        child1 = np.concatenate([c1[:cut],  c2[cut:]], axis=0)
        child2 = np.concatenate([c2[:cut],  c1[cut:]], axis=0)
        
        return Individual(child1), Individual(child2)
        
    def mutate(self, individual: Individual) -> Individual:
        chrom = individual.chromosome.copy()
        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    if random.random() < self.cfg.mutation_rate:
                        if random.random() < 0.5:
                            chrom[d, t, q] = self._random_specialty_for(q, d, t)
                        else:
                            d2, t2, q2 = random.randrange(self.n_days), random.randrange(self.n_shifts), random.randrange(self.n_ors)
                            comp1 = self._specialty_valid_for_or(int(chrom[d, t, q]), q2, d2, t2)
                            comp2 = self._specialty_valid_for_or(int(chrom[d2, t2, q2]), q, d, t)
                            
                            if comp1 and comp2:
                                chrom[d, t, q], chrom[d2, t2, q2] = int(chrom[d2, t2, q2]), int(chrom[d, t, q])
        return Individual(chrom)

    def _count_blocks_per_specialty(self, chrom: np.ndarray) -> Dict[int, int]:
        return {sid: int(np.sum(chrom == sid)) for sid in self._real_spec_ids}

    def _evaluate_population(self, population: List[Individual]) -> None:
        if self.cfg.parallel_workers <= 1 or len(population) <= 1:
            for individual in population:
                self.evaluate_fitness(individual)
            return
        with ThreadPoolExecutor(max_workers=self.cfg.parallel_workers) as executor:
            list(executor.map(self.evaluate_fitness, population))

    def _global_penalty(self, chrom: np.ndarray) -> float:
        penalty = 0.0
        counts  = self._count_blocks_per_specialty(chrom)
        for spec in self.specialties:
            if spec.id == 0:
                continue
            assigned = counts.get(spec.id, 0)
            if assigned < spec.min_blocks:
                penalty += self.cfg.penalty_below_min_quota * (spec.min_blocks - assigned)
            if assigned > spec.max_blocks:
                penalty += self.cfg.penalty_above_max_quota * (assigned - spec.max_blocks)
        return penalty

    # ─── 2.5 LOOP PRINCIPAL ───────────────────────────────────────────────
    def run(self) -> Individual:
        self.history = []
        print("▶  Inicializando población...")
        population = self.initialize_population()
        self._evaluate_population(population)

        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0].copy()
        generations_no_improve = 0

        print(f"\n{'Gen':>5}  {'Mejor fitness':>16}  {'Promedio':>10}  {'Sin mejora':>10}")
        print("─" * 50)

        for gen in range(self.cfg.max_generations):
            new_population = [population[i].copy() for i in range(self.cfg.elite_count)]

            while len(new_population) < self.cfg.population_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                c1, c2 = self.crossover(p1, p2)

                children = [self.mutate(child) for child in (c1, c2)]
                
                for child in children:
                    self.evaluate_fitness(child)
                    if len(new_population) < self.cfg.population_size:
                        new_population.append(child)

            population = sorted(new_population, key=lambda x: x.fitness, reverse=True)
            self.history.append(population[0].fitness)

            if population[0].fitness > self.best_individual.fitness:
                self.best_individual = population[0].copy()
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            if gen % 10 == 0:
                avg = sum(ind.fitness for ind in population) / len(population)
                print(f"{gen:>5}  {self.best_individual.fitness:>16.4f}  {avg:>10.4f}  {generations_no_improve:>10}")

            if generations_no_improve >= self.cfg.convergence_patience:
                print(f"\n✔ Convergencia en gen {gen}.")
                break

        total_calls = self._cache_hits + len(self._mip_cache)
        print(
            f"\n  Cache MIP: {self._cache_hits} hits / {total_calls} llamadas "
            f"({100 * self._cache_hits // max(total_calls, 1)}% ahorrado)"
        )
        total_fitness_calls = self._fitness_cache_hits + len(self._fitness_cache)
        print(
            f"  Cache fitness: {self._fitness_cache_hits} hits / {total_fitness_calls} evaluaciones "
            f"({100 * self._fitness_cache_hits // max(total_fitness_calls, 1)}% ahorrado)"
        )
        return self.best_individual

    def print_schedule(self, individual: Individual) -> None:
        spec_names = {s.id: s.name for s in self.specialties}
        spec_names[0] = "─── Libre ───"
        print("\n" + "═" * 70)
        print(f"  AGENDA SEMANAL  │  Fitness: {individual.fitness:.4f}")
        print("═" * 70)
        for d in range(self.n_days):
            print(f"\n  {self.DAY_NAMES[d]}")
            for t in range(self.n_shifts):
                print(f"    {self.SHIFT_NAMES[t]}:")
                for q in range(self.n_ors):
                    sid = int(individual.chromosome[d, t, q])
                    print(f"      {self._or_by_idx[q].name:25s} → {spec_names.get(sid, 'ID='+str(sid))}")