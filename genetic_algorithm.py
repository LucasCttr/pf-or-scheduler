"""
genetic_algorithm.py — Nivel 1: Algoritmo Genético optimizado.
"""
import random
import copy
from typing import List, Tuple, Dict, Optional
import numpy as np

from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
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
        staff_list: List[Staff]
    ):
        self.cfg = config
        self.ors = operating_rooms
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

        # Cache de resultados MIP: evita resolver el mismo bloque de turno
        # más de una vez. La clave codifica exactamente qué pacientes y
        # cirujanos entran al solver, por lo que es seguro reutilizar.
        self._mip_cache: Dict = {}
        self._cache_hits: int = 0

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
        or_ = self._or_by_idx[or_idx]
        if not or_.availability[day][shift]: return []

        is_morning = (shift == 0)
        valid_options = []
        
        for s in self.specialties:
            if s.id == 0: continue
            if or_.or_type in s.compatible_or_types:
                has_staff = any(
                    s.id in staff.specialties_ids and 
                    staff.get_available_minutes_in_block(day, is_morning) > 0
                    for staff in self.staff_list if staff.role == "cirujano"
                )
                if has_staff:
                    valid_options.append(s.id)
        return valid_options

    # ─── 2.2 CONSTRUCCIÓN DE BLOQUES POR TURNO ────────────────────────────

    @staticmethod
    def _make_shift_cache_key(blocks: List[Dict], day_idx: int, is_morning: bool,
                               alpha: float, beta: float) -> tuple:
        """
        Clave de cache para un turno completo.
        Codifica exactamente qué pacientes y cirujanos entran al MIP, por lo
        que dos turnos con la misma clave producirán el mismo resultado óptimo.
        """
        block_keys = tuple(
            (
                b["or_idx"],
                b["spec_id"],
                tuple(sorted(s.id for s in b["surgeons"])),
                tuple(sorted(p.id for p in b["patients"])),
            )
            for b in blocks
        )
        return (day_idx, is_morning, alpha, beta, block_keys)

    def _build_shift_blocks(
        self,
        chrom: np.ndarray,
        d: int,
        t: int,
        is_morning: bool,
        pacientes_excluidos: set
    ) -> List[Dict]:
        """
        Construye la lista de bloques para un turno completo.
        Cada bloque representa un quirófano con su especialidad, cirujanos
        disponibles y pacientes candidatos. El MIP consume esta lista y
        garantiza internamente que ningún cirujano exceda su capacidad total
        sumada sobre todos los quirófanos del turno.
        """
        blocks = []
        for q in range(self.n_ors):
            spec_id = int(chrom[d, t, q])
            or_     = self._or_by_idx[q]

            if not or_.availability[d][t] or spec_id == 0:
                blocks.append({"or_idx": q, "spec_id": 0,
                                "patients": [], "surgeons": [], "t_max": 0})
                continue

            surgeons = [
                s for s in self.staff_list
                if s.role == "cirujano"
                and spec_id in s.specialties_ids
                and s.get_available_minutes_in_block(d, is_morning) > 0
            ]

            if not surgeons:
                blocks.append({"or_idx": q, "spec_id": spec_id,
                                "patients": [], "surgeons": [], "t_max": 0})
                continue

            surgeons_ids = {s.id for s in surgeons}
            patients = [
                p for p in self.patients_by_specialty.get(spec_id, [])
                if p.id not in pacientes_excluidos
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

    # ─── 2.3 EVALUACIÓN DE FITNESS ────────────────────────────────────────

    def _count_blocks_per_specialty(self, chrom: np.ndarray) -> Dict[int, int]:
        return {sid: int(np.sum(chrom == sid)) for sid in self._real_spec_ids}

    def _global_penalty(self, chrom: np.ndarray) -> float:
        penalty = 0.0
        counts  = self._count_blocks_per_specialty(chrom)
        for spec in self.specialties:
            if spec.id == 0: continue
            assigned = counts.get(spec.id, 0)
            if assigned < spec.min_blocks:
                penalty += self.cfg.penalty_below_min_quota * (spec.min_blocks - assigned)
            if assigned > spec.max_blocks:
                penalty += self.cfg.penalty_above_max_quota * (assigned - spec.max_blocks)
        return penalty

    def evaluate_fitness(self, individual: Individual) -> float:
        chrom = individual.chromosome
        total_z = 0.0
        pacientes_operados: set = set()

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                is_morning = (t == 0)
                blocks = self._build_shift_blocks(chrom, d, t, is_morning, pacientes_operados)
                key    = self._make_shift_cache_key(blocks, d, is_morning, self.cfg.alpha, self.cfg.beta)
                if key in self._mip_cache:
                    result = self._mip_cache[key]
                    self._cache_hits += 1
                else:
                    result = solve_mip_for_shift(blocks, d, is_morning, self.cfg.alpha, self.cfg.beta)
                    self._mip_cache[key] = result
                total_z += result["fitness"]
                pacientes_operados.update(result["all_pacientes_ids"])

        fitness = total_z - self._global_penalty(chrom)
        individual.fitness = fitness
        return fitness

    # ─── 2.3 OPERADORES GENÉTICOS ─────────────────────────────────────────

    def get_schedule_details(self, individual: Individual) -> Dict[Tuple[int, int, int], dict]:
        """
        Corre el MIP por turno sobre el mejor individuo y devuelve un dict
        {(day, shift, or_idx): per_or_result} para reutilizar en el reporte
        final sin volver a resolver el problema.
        """
        chrom = individual.chromosome
        cache: Dict[Tuple[int, int, int], dict] = {}
        pacientes_operados: set = set()

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                is_morning = (t == 0)
                blocks = self._build_shift_blocks(chrom, d, t, is_morning, pacientes_operados)
                key    = self._make_shift_cache_key(blocks, d, is_morning, self.cfg.alpha, self.cfg.beta)
                if key in self._mip_cache:
                    result = self._mip_cache[key]
                else:
                    result = solve_mip_for_shift(blocks, d, is_morning, self.cfg.alpha, self.cfg.beta)
                    self._mip_cache[key] = result
                pacientes_operados.update(result["all_pacientes_ids"])
                for q in range(self.n_ors):
                    cache[(d, t, q)] = result["per_or"].get(q)

        return cache

    def tournament_selection(self, population: List[Individual]) -> Individual:
        contestants = random.sample(population, self.cfg.tournament_size)
        return max(contestants, key=lambda ind: ind.fitness).copy()

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.cfg.crossover_rate:
            return parent1.copy(), parent2.copy()
        c1, c2 = parent1.chromosome.copy(), parent2.chromosome.copy()
        if random.random() < 0.5:
            cut = random.randint(1, self.n_days - 1)
            child1 = np.concatenate([c1[:cut],  c2[cut:]], axis=0)
            child2 = np.concatenate([c2[:cut],  c1[cut:]], axis=0)
        else:
            cut = random.randint(1, self.n_ors - 1)
            child1 = np.concatenate([c1[:, :, :cut], c2[:, :, cut:]], axis=2)
            child2 = np.concatenate([c2[:, :, :cut], c1[:, :, cut:]], axis=2)
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
                            chrom[d, t, q], chrom[d2, t2, q2] = int(chrom[d2, t2, q2]), int(chrom[d, t, q])
        return Individual(chrom)

    def repair(self, individual: Individual) -> Individual:
        chrom = individual.chromosome.copy()
        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    spec_id = int(chrom[d, t, q])
                    or_ = self._or_by_idx[q]
                    if not or_.availability[d][t]:
                        chrom[d, t, q] = 0
                        continue
                    if spec_id == 0: continue
                    spec = self._spec_by_id.get(spec_id)
                    if spec is None or or_.or_type not in spec.compatible_or_types:
                        chrom[d, t, q] = self._random_specialty_for(q, d, t)

        counts = self._count_blocks_per_specialty(chrom)
        for spec in self.specialties:
            if spec.id == 0: continue
            deficit = spec.min_blocks - counts.get(spec.id, 0)
            if deficit > 0:
                candidates = [(d, t, q) for d in range(self.n_days) for t in range(self.n_shifts) for q in range(self.n_ors)
                              if self._or_by_idx[q].availability[d][t] and self._or_by_idx[q].or_type in spec.compatible_or_types
                              and (chrom[d, t, q] == 0 or counts.get(int(chrom[d, t, q]), 0) > self._spec_by_id[int(chrom[d, t, q])].min_blocks)]
                random.shuffle(candidates)
                for i in range(min(deficit, len(candidates))):
                    d, t, q = candidates[i]
                    old_sid = int(chrom[d, t, q])
                    chrom[d, t, q] = spec.id
                    counts[spec.id] = counts.get(spec.id, 0) + 1
                    if old_sid != 0: counts[old_sid] -= 1
        return Individual(chrom)

    # ─── 2.4 LOOP PRINCIPAL ───────────────────────────────────────────────

    def run(self) -> Individual:
        print("▶  Inicializando población...")
        population = self.initialize_population()
        for ind in population: self.evaluate_fitness(ind)
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0].copy()
        generations_no_improve = 0

        print(f"\n{'Gen':>5}  {'Mejor fitness':>16}  {'Promedio':>10}  {'Sin mejora':>10}")
        print("─" * 50)

        for gen in range(self.cfg.max_generations):
            new_population = [population[i].copy() for i in range(self.cfg.elite_count)]
            while len(new_population) < self.cfg.population_size:
                p1, p2 = self.tournament_selection(population), self.tournament_selection(population)
                c1, c2 = self.crossover(p1, p2)
                for child in [c1, c2]:
                    child = self.mutate(child)
                    child = self.repair(child)
                    self.evaluate_fitness(child)
                    if len(new_population) < self.cfg.population_size:
                        new_population.append(child)

            population = sorted(new_population, key=lambda x: x.fitness, reverse=True)
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
        print(f"\n  Cache MIP: {self._cache_hits} hits / {total_calls} llamadas "
              f"({100 * self._cache_hits // max(total_calls, 1)}% ahorrado)")
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