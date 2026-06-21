"""
genetic_algorithm.py — Nivel 1: Algoritmo Genético optimizado con Decoder Heurístico.
"""

import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
import numpy as np

from models import OperatingRoom, Specialty, Procedure, Patient, GAConfig, Staff
from decoder import build_shift_schedule


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


class GeneticAlgorithm:
    DAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
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

        self.n_ors = len(operating_rooms)
        self.n_days = config.n_days
        self.n_shifts = config.n_shifts

        self._spec_by_id: Dict[int, Specialty] = {s.id: s for s in specialties}
        self._or_by_idx: Dict[int, OperatingRoom] = dict(enumerate(operating_rooms))
        self._real_spec_ids: List[int] = [s.id for s in specialties if s.id != 0]

        self.best_individual: Optional[Individual] = None
        self.history: List[float] = []

        self._fitness_cache: Dict[bytes, float] = {}
        self._fitness_cache_hits: int = 0

        self.slot_size = config.slot_size_min
        self.procedures_by_specialty = procedures_by_specialty or {}

    def _specialty_procedures(self, specialty_id: int) -> List[Procedure]:
        return self.procedures_by_specialty.get(specialty_id, [])

    def _compatible_procedures_for_room(
        self, specialty_id: int, or_type: str
    ) -> List[Procedure]:
        return [
            proc
            for proc in self._specialty_procedures(specialty_id)
            if {
                "baja_complejidad": 0,
                "media_complejidad": 1,
                "alta_complejidad": 2,
            }.get(proc.required_room_type, 0)
            <= {
                "baja_complejidad": 0,
                "media_complejidad": 1,
                "alta_complejidad": 2,
            }.get(or_type, 0)
        ]

    def _specialty_valid_for_or(
        self, specialty_id: int, or_idx: int, day: int, shift: int
    ) -> bool:
        if specialty_id == 0:
            return True

        or_obj = self._or_by_idx[or_idx]
        if not or_obj.availability[day][shift]:
            return False

        compatible_procs = self._compatible_procedures_for_room(
            specialty_id, or_obj.or_type
        )
        if not compatible_procs:
            return False

        is_morning = shift == 0
        compatible_proc_ids = {proc.id for proc in compatible_procs}

        for staff in self.staff_list:
            # Ensure has_competence is always defined for this iteration
            has_competence = False
            if staff.role == "cirujano" and staff.main_specialty_id == specialty_id:
                has_competence = any(
                    pid in staff.enabled_procedures_ids for pid in compatible_proc_ids
                )
            if has_competence and staff.get_available_minutes_in_block(day) > 0:
                return True
        return False

    def initialize_population(self) -> List[Individual]:
        return [self._create_individual() for _ in range(self.cfg.population_size)]

    def _create_individual(self) -> Individual:
        # (días, turnos, quirófanos, 2)
        # [d, t, q, 0] = Specialty ID
        # [d, t, q, 1] = Staff ID (Cirujano)
        chrom = np.zeros((self.n_days, self.n_shifts, self.n_ors, 2), dtype=int)

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    # 1. Elegir especialidad
                    spec_id = self._random_specialty_for(q, d, t)
                    chrom[d, t, q, 0] = spec_id

                    # 2. Elegir un cirujano válido para esa especialidad
                    if spec_id != 0:
                        cirujanos_validos = self._get_surgeons_for(spec_id, d, t == 0)
                        if cirujanos_validos:
                            chrom[d, t, q, 1] = random.choice(cirujanos_validos).id
                        else:
                            chrom[d, t, q, 0] = (
                                0  # Si no hay cirujanos, el quirófano queda libre
                            )
        return Individual(chrom)

    def _get_surgeons_for(
        self, specialty_id: int, day: int, is_morning: bool
    ) -> List[Staff]:
        return [
            s
            for s in self.staff_list
            if s.main_specialty_id == specialty_id
            and s.get_available_minutes_in_block(day) > 0
        ]

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

    def _build_shift_blocks(
        self,
        chrom: np.ndarray,
        d: int,
        t: int,
        is_morning: bool,
        pacientes_excluidos: set,
    ) -> List[Dict]:
        blocks = []
        for q in range(self.n_ors):
            spec_id = int(chrom[d, t, q, 0])
            staff_id = int(chrom[d, t, q, 1])
            or_ = self._or_by_idx[q]

            # 1. Validación básica de disponibilidad del Quirófano y elección del AG
            if not or_.availability[d][t] or spec_id == 0 or staff_id == 0:
                blocks.append(
                    {
                        "or_idx": q,
                        "spec_id": 0,
                        "patients": [],
                        "surgeons": [],
                        "t_max": 0,
                    }
                )
                continue

            # 2. Obtenemos al cirujano
            cirujano = next((s for s in self.staff_list if s.id == staff_id), None)

            # 3. CAMBIO CLAVE: Validar contra BOLSA DE HORAS, no contra turnos rígidos
            # Ya no importa 'is_morning', ahora importa si tiene minutos en su bolsa semanal
            if not cirujano or cirujano.get_available_minutes_in_block(d) <= 0:
                blocks.append(
                    {
                        "or_idx": q,
                        "spec_id": spec_id,
                        "patients": [],
                        "surgeons": [],
                        "t_max": 0,
                    }
                )
                continue

            # Filtrar pacientes: especialidad, procedimiento y cirujano forzado
            compatible_procs = self._compatible_procedures_for_room(
                spec_id, or_.or_type
            )
            compatible_proc_ids = {proc.id for proc in compatible_procs}
            procedimientos_viables = (
                set(cirujano.enabled_procedures_ids) & compatible_proc_ids
            )

            patients = [
                p
                for p in self.patients_by_specialty.get(spec_id, [])
                if p.id not in pacientes_excluidos
                and getattr(p, "procedure_id", None) in procedimientos_viables
                and (
                    getattr(p, "forced_surgeon_id", None) is None
                    or p.forced_surgeon_id == staff_id
                )
            ]

            # 4. Asignamos el bloque
            blocks.append(
                {
                    "or_idx": q,
                    "spec_id": spec_id,
                    "patients": patients,
                    "surgeons": [cirujano],
                    "t_max": self.cfg.block_duration_min,
                }
            )
        return blocks

    @staticmethod
    def _chromosome_key(chrom: np.ndarray) -> bytes:
        return chrom.tobytes()

    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Calcula el fitness exacto utilizando el Decodificador Heurístico.
        """
        # Reset staff consumption so each individual's evaluation starts fresh
        for s in self.staff_list:
            s.minutos_consumidos = 0

        chrom = individual.chromosome
        chrom_key = self._chromosome_key(chrom)

        cached_f = self._fitness_cache.get(chrom_key)
        if cached_f is not None:
            self._fitness_cache_hits += 1
            individual.fitness = cached_f
            return cached_f

        total_score = 0.0
        pacientes_operados_global = set()

        for d in range(self.n_days):
            # Estado acumulado del cirujano durante el día — se propaga entre turnos
            surg_clock_dia: Dict[int, int] = {}
            consumed_dia:   Dict[int, int] = {}

            for t in range(self.n_shifts):
                blocks_turno = self._build_shift_blocks(
                    chrom, d, t, (t == 0), pacientes_operados_global
                )
                capacity_params = {
                    "block_start":              480 if (t == 0) else 780,
                    "block_duration":           self.cfg.block_duration_min,
                    "surg_clock_previo":        surg_clock_dia,
                    "remaining_minutes_previo": consumed_dia,
                }
                resultado = build_shift_schedule(blocks_turno, d, capacity_params)

                # Propagar estado al siguiente turno del mismo día
                surg_clock_dia = resultado.get("surg_clock_final", surg_clock_dia)
                consumed_dia   = resultado.get("consumed_minutes", consumed_dia)

                total_score += resultado["fitness"]
                pacientes_operados_global.update(resultado["all_pacientes_ids"])

                for data_or in resultado["per_or"].values():
                    if data_or["utilizacion_porcentaje"] > 80.0:
                        total_score += 15.0

        fitness = total_score - self._global_penalty(chrom)
        individual.fitness = fitness
        self._fitness_cache[chrom_key] = fitness
        return fitness

    def tournament_selection(self, population: List[Individual]) -> Individual:
        contestants = random.sample(population, self.cfg.tournament_size)
        return max(contestants, key=lambda ind: ind.fitness).copy()

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        if random.random() > self.cfg.crossover_rate:
            return parent1.copy(), parent2.copy()

        c1, c2 = parent1.chromosome.copy(), parent2.chromosome.copy()
        cut = random.randint(1, self.n_days - 1)
        child1 = np.concatenate([c1[:cut], c2[cut:]], axis=0)
        child2 = np.concatenate([c2[:cut], c1[cut:]], axis=0)
        return Individual(child1), Individual(child2)

    def mutate(self, individual: Individual) -> Individual:
        chrom = individual.chromosome.copy()
        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    if random.random() < self.cfg.mutation_rate:
                        if random.random() < 0.5:
                            # CORRECCIÓN: Accede al índice [0] (especialidad)
                            # Y luego debes asignar tanto la especialidad como el cirujano
                            new_spec = self._random_specialty_for(q, d, t)
                            chrom[d, t, q, 0] = new_spec
                            # Si cambia la especialidad, debes reasignar un cirujano válido
                            if new_spec != 0:
                                cirujanos = self._get_surgeons_for(new_spec, d, t == 0)
                                chrom[d, t, q, 1] = (
                                    random.choice(cirujanos).id if cirujanos else 0
                                )
                            else:
                                chrom[d, t, q, 1] = 0
                        else:
                            # Mutación de intercambio (swap)
                            d2, t2, q2 = (
                                random.randrange(self.n_days),
                                random.randrange(self.n_shifts),
                                random.randrange(self.n_ors),
                            )

                            # Intercambiar la celda completa (Especialidad y Cirujano)
                            chrom[d, t, q, :], chrom[d2, t2, q2, :] = (
                                chrom[d2, t2, q2, :].copy(),
                                chrom[d, t, q, :].copy(),
                            )
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
        counts = self._count_blocks_per_specialty(chrom)
        for spec in self.specialties:
            if spec.id == 0:
                continue
            assigned = counts.get(spec.id, 0)
            if assigned < spec.min_blocks:
                penalty += self.cfg.penalty_below_min_quota * (
                    spec.min_blocks - assigned
                )
            if assigned > spec.max_blocks:
                penalty += self.cfg.penalty_above_max_quota * (
                    assigned - spec.max_blocks
                )
        return penalty

    def run(self) -> List[Individual]:
        self.history = []
        print("▶  Inicializando población...")
        population = self.initialize_population()
        self._evaluate_population(population)

        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0].copy()
        generations_no_improve = 0

        print(
            f"\n{'Gen':>5}  {'Mejor fitness':>16}  {'Promedio':>10}  {'Sin mejora':>10}"
        )
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
                print(
                    f"{gen:>5}  {self.best_individual.fitness:>16.4f}  {avg:>10.4f}  {generations_no_improve:>10}"
                )

            if generations_no_improve >= self.cfg.convergence_patience:
                print(f"\n✔ Convergencia en gen {gen}.")
                break

        total_fitness_calls = self._fitness_cache_hits + len(self._fitness_cache)
        print(
            f"  Cache fitness: {self._fitness_cache_hits} hits / {total_fitness_calls} evaluaciones "
            f"({100 * self._fitness_cache_hits // max(total_fitness_calls, 1)}% ahorrado)"
        )

        poblacion_final_ordenada = sorted(
            population, key=lambda x: x.fitness, reverse=True
        )
        mejores_unicos = []
        vistos = set()

        for ind in poblacion_final_ordenada:
            key = self._chromosome_key(ind.chromosome)
            if key not in vistos:
                vistos.add(key)
                mejores_unicos.append(ind.copy())
            if len(mejores_unicos) >= 5:
                break

        return mejores_unicos

    def print_schedule(self, individual: Individual) -> None:
        spec_names = {s.id: s.name for s in self.specialties}
        spec_names[0] = "─── Libre ───"
        print("\n" + "═" * 70)
        print(f"  AGENDA SEMANAL  │  Fitness Exacto: {individual.fitness:.4f}")
        print("═" * 70)
        for d in range(self.n_days):
            print(f"\n  {self.DAY_NAMES[d]}")
            for t in range(self.n_shifts):
                print(f"    {self.SHIFT_NAMES[t]}:")
                for q in range(self.n_ors):
                    # CORRECCIÓN: Accede al índice [0] para obtener la especialidad
                    sid = int(individual.chromosome[d, t, q, 0])
                    # OPCIONAL: Si quieres imprimir también el cirujano
                    cid = int(individual.chromosome[d, t, q, 1])
                    cirujano = next(
                        (s.name for s in self.staff_list if s.id == cid), "Sin asignar"
                    )

                    print(
                        f"      {self._or_by_idx[q].name:25s} → {spec_names.get(sid, 'ID=' + str(sid))} (Dr. {cirujano})"
                    )
