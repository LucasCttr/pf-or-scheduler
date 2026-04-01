"""
genetic_algorithm.py — Nivel 1: Algoritmo Genético para planificación semanal de quirófanos.

Decide qué especialidad ocupa cada bloque (día × turno × quirófano).
El fitness de cada bloque se obtiene del MIP (Nivel 3) a través de mip_stub.

Cromosoma: numpy array 3D de shape (n_days, n_shifts, n_ors)
           Cada celda contiene un specialty_id (int). 0 = bloque libre.
"""
import random
import copy
from typing import List, Tuple, Dict, Optional

import numpy as np

from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from mip import solve_mip_for_block


# ---------------------------------------------------------------------------
# Clase Individual
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Clase GeneticAlgorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    DAY_NAMES   = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    SHIFT_NAMES = ["Mañana", "Tarde"]

    def __init__(
        self,
        config: GAConfig,
        operating_rooms: List[OperatingRoom],
        specialties: List[Specialty],
        patients_by_specialty: Dict[int, List[Patient]],
        staff_list: List[Staff] # Lista de cirujanos con sus horarios
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

    # ═══════════════════════════════════════════════════════════════════════
    # 1. INICIALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def _valid_specialties_for(self, or_idx: int, day: int, shift: int) -> List[int]:
        """Retorna especialidades compatibles con el quirófano Y con médicos disponibles."""
        or_ = self._or_by_idx[or_idx]
        if not or_.availability[day][shift]:
            return []

        is_morning = (shift == 0)
        valid_options = []
        
        for s in self.specialties:
            if s.id == 0: continue
            
            # Chequeo de Quirófano
            if or_.or_type in s.compatible_or_types:
                # Chequeo de Staff: ¿Hay al menos un cirujano de esta especialidad en este bloque?
                has_staff = any(
                    staff.specialty_id == s.id and 
                    staff.get_available_minutes_in_block(day, is_morning) > 0
                    for staff in self.staff_list if staff.role == "cirujano"
                )
                if has_staff:
                    valid_options.append(s.id)
        
        return valid_options

    def _create_individual(self) -> Individual:
        # (Lógica similar a la anterior, pero usando el nuevo _valid_specialties_for)
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

    def initialize_population(self) -> List[Individual]:
        """Crea la población inicial completa."""
        return [self._create_individual() for _ in range(self.cfg.population_size)]

    # ═══════════════════════════════════════════════════════════════════════
    # 2. EVALUACIÓN DE FITNESS
    # ═══════════════════════════════════════════════════════════════════════

    def _count_blocks_per_specialty(self, chrom: np.ndarray) -> Dict[int, int]:
        return {sid: int(np.sum(chrom == sid)) for sid in self._real_spec_ids}

    def _global_penalty(self, chrom: np.ndarray) -> float:
        """Penaliza incumplimiento de cuotas mínimas y exceso sobre las máximas."""
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
        pacientes_operados = set()

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                is_morning = (t == 0)
                
                # --- 1. BOLSA DE TIEMPO DEL TURNO ---
                # Usamos self.staff_list que ya tenés en el __init__
                staff_capacity = {
                    s.id: s.get_available_minutes_in_block(d, is_morning)
                    for s in self.staff_list if s.role == "cirujano"
                }

                for q in range(self.n_ors):
                    spec_id = int(chrom[d, t, q])
                    if spec_id == 0: 
                        continue

                    # 2. Filtrar cirujanos con tiempo remanente (Nivel 2)
                    surgeons_available = [
                        s for s in self.staff_list 
                        if s.role == "cirujano" and s.specialty_id == spec_id
                        and staff_capacity.get(s.id, 0) > 0
                    ]

                    if not surgeons_available: 
                        continue

                    # 3. Filtrar pacientes pendientes
                    all_patients = self.patients_by_specialty.get(spec_id, [])
                    candidatos = [p for p in all_patients if p.id not in pacientes_operados]

                    if not candidatos: 
                        continue

                    # 4. LLAMADA AL MIP (Nivel 3)
                    # USAMOS self.cfg
                    z, ids_elegidos, tiempo_usado = solve_mip_for_block(
                        specialty_id=spec_id,
                        patients=candidatos,
                        surgeons=surgeons_available,
                        day_idx=d,
                        is_morning=is_morning,
                        alpha=self.cfg.alpha,
                        beta=self.cfg.beta,
                        custom_capacities=staff_capacity
                    )

                    # 5. ACTUALIZAR CONSUMO REAL
                    for s_id, minutos in tiempo_usado.items():
                        staff_capacity[s_id] -= minutos
                    
                    total_z += z
                    pacientes_operados.update(ids_elegidos)

        # 6. PENALIZACIONES Y RESULTADO FINAL
        fitness = total_z - self._global_penalty(chrom)
        individual.fitness = fitness
        return fitness

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SELECCIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Selección por torneo: elige el mejor entre k candidatos aleatorios."""
        contestants = random.sample(population, self.cfg.tournament_size)
        return max(contestants, key=lambda ind: ind.fitness).copy()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. CRUCE
    # ═══════════════════════════════════════════════════════════════════════

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Cruce por intercambio de planos 3D.
        - Cruce por día   (eje D): corte en el eje de días
        - Cruce por quirófano (eje Q): corte en el eje de quirófanos
        Se elige aleatoriamente cuál aplicar.
        """
        if random.random() > self.cfg.crossover_rate:
            return parent1.copy(), parent2.copy()

        c1 = parent1.chromosome.copy()
        c2 = parent2.chromosome.copy()

        if random.random() < 0.5:
            # Cruce por día
            cut = random.randint(1, self.n_days - 1)
            child1 = np.concatenate([c1[:cut],  c2[cut:]], axis=0)
            child2 = np.concatenate([c2[:cut],  c1[cut:]], axis=0)
        else:
            # Cruce por quirófano
            cut = random.randint(1, self.n_ors - 1)
            child1 = np.concatenate([c1[:, :, :cut], c2[:, :, cut:]], axis=2)
            child2 = np.concatenate([c2[:, :, :cut], c1[:, :, cut:]], axis=2)

        return Individual(child1), Individual(child2)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. MUTACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def mutate(self, individual: Individual) -> Individual:
        """
        Aplica dos tipos de mutación sobre cada gen con probabilidad mutation_rate:
        - Mutación de reemplazo: asigna una especialidad válida nueva al bloque.
        - Mutación de intercambio: intercambia dos celdas del cromosoma.
        """
        chrom = individual.chromosome.copy()

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    if random.random() >= self.cfg.mutation_rate:
                        continue

                    if random.random() < 0.5:
                        # Reemplazo por especialidad válida aleatoria
                        chrom[d, t, q] = self._random_specialty_for(q, d, t)
                    else:
                        # Intercambio con otra celda aleatoria
                        d2 = random.randrange(self.n_days)
                        t2 = random.randrange(self.n_shifts)
                        q2 = random.randrange(self.n_ors)
                        chrom[d, t, q], chrom[d2, t2, q2] = (
                            int(chrom[d2, t2, q2]),
                            int(chrom[d, t, q]),
                        )

        return Individual(chrom)

    # ═══════════════════════════════════════════════════════════════════════
    # 6. OPERADOR DE REPARACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def repair(self, individual: Individual) -> Individual:
        """
        Corrige el cromosoma aplicando tres reglas:
        R1. Libera bloques en horarios en que el quirófano no está disponible.
        R2. Reemplaza especialidades incompatibles con el tipo de quirófano.
        R3. Reasigna bloques para cubrir cuotas mínimas deficitarias.
        """
        chrom = individual.chromosome.copy()

        # ── R1 y R2: incompatibilidades y disponibilidad ──────────────────
        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    spec_id = int(chrom[d, t, q])
                    or_     = self._or_by_idx[q]

                    # R1: quirófano no disponible → liberar
                    if not or_.availability[d][t]:
                        chrom[d, t, q] = 0
                        continue

                    if spec_id == 0:
                        continue

                    spec = self._spec_by_id.get(spec_id)
                    if spec is None:
                        chrom[d, t, q] = 0
                        continue

                    # R2: especialidad incompatible con el quirófano → reasignar
                    if or_.or_type not in spec.compatible_or_types:
                        chrom[d, t, q] = self._random_specialty_for(q, d, t)

        # ── R3: cubrir cuotas mínimas deficitarias ────────────────────────
        counts = self._count_blocks_per_specialty(chrom)

        for spec in self.specialties:
            if spec.id == 0:
                continue

            deficit = spec.min_blocks - counts.get(spec.id, 0)
            if deficit <= 0:
                continue

            # Candidatos: bloques libres o de especialidades sobre su cuota mínima
            candidates = [
                (d, t, q)
                for d in range(self.n_days)
                for t in range(self.n_shifts)
                for q in range(self.n_ors)
                if self._or_by_idx[q].availability[d][t]
                and spec.id != 0
                and self._or_by_idx[q].or_type in spec.compatible_or_types
                and (
                    chrom[d, t, q] == 0
                    or counts.get(int(chrom[d, t, q]), 0)
                    > self._spec_by_id.get(int(chrom[d, t, q]), Specialty(0,"",[])).min_blocks
                )
            ]

            random.shuffle(candidates)
            repaired = 0

            for d, t, q in candidates:
                if repaired >= deficit:
                    break
                old_sid = int(chrom[d, t, q])
                chrom[d, t, q] = spec.id
                counts[spec.id] = counts.get(spec.id, 0) + 1
                if old_sid != 0:
                    counts[old_sid] = max(0, counts.get(old_sid, 0) - 1)
                repaired += 1

        return Individual(chrom)

    # ═══════════════════════════════════════════════════════════════════════
    # 7. LOOP PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> Individual:
        """
        Ejecuta el AG completo y devuelve el mejor individuo encontrado.

        Criterios de parada:
        - Se alcanzó max_generations.
        - Sin mejora durante convergence_patience generaciones consecutivas.
        """
        print("▶  Inicializando población...")
        population = self.initialize_population()

        print("▶  Evaluando fitness inicial...")
        for ind in population:
            self.evaluate_fitness(ind)

        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0].copy()
        generations_no_improve = 0

        print(f"\n{'Gen':>5}  {'Mejor fitness':>16}  {'Promedio':>10}  {'Sin mejora':>10}")
        print("─" * 50)

        for gen in range(self.cfg.max_generations):
            new_population: List[Individual] = []

            # ── Elitismo ──────────────────────────────────────────────────
            for i in range(self.cfg.elite_count):
                new_population.append(population[i].copy())

            # ── Generar el resto ──────────────────────────────────────────
            while len(new_population) < self.cfg.population_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)

                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                c1 = self.repair(c1)
                c2 = self.repair(c2)

                self.evaluate_fitness(c1)
                self.evaluate_fitness(c2)

                new_population.append(c1)
                if len(new_population) < self.cfg.population_size:
                    new_population.append(c2)

            population = sorted(new_population, key=lambda x: x.fitness, reverse=True)
            gen_best   = population[0]
            avg        = sum(ind.fitness for ind in population) / len(population)

            self.history.append(gen_best.fitness)

            if gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best.copy()
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            if gen % 10 == 0 or gen == self.cfg.max_generations - 1:
                print(f"{gen:>5}  {self.best_individual.fitness:>16.4f}  {avg:>10.4f}  {generations_no_improve:>10}")

            if generations_no_improve >= self.cfg.convergence_patience:
                print(f"\n✔  Convergencia alcanzada en la generación {gen}.")
                break

        return self.best_individual

    # ═══════════════════════════════════════════════════════════════════════
    # 8. UTILIDADES DE VISUALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def print_schedule(self, individual: Individual) -> None:
        """Imprime la agenda semanal de forma legible en consola."""
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
                    spec_id  = int(individual.chromosome[d, t, q])
                    or_name  = self._or_by_idx[q].name
                    spec_name = spec_names.get(spec_id, f"ID={spec_id}")
                    print(f"      {or_name:25s} → {spec_name}")

        print("\n" + "─" * 70)
        print("  BLOQUES POR ESPECIALIDAD:")
        counts = self._count_blocks_per_specialty(individual.chromosome)
        for spec in self.specialties:
            if spec.id == 0:
                continue
            assigned = counts.get(spec.id, 0)
            ok = "✓" if assigned >= spec.min_blocks else "✗ DÉFICIT"
            print(f"    {spec.name:22s}: {assigned:2d} bloques  (mín {spec.min_blocks} / máx {spec.max_blocks})  {ok}")

        print("═" * 70)
