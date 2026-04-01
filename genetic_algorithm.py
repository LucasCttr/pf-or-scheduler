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

from models import OperatingRoom, Specialty, Patient, GAConfig
from mip import solve_mip_for_block


# ---------------------------------------------------------------------------
# Clase Individual
# ---------------------------------------------------------------------------

class Individual:
    """Una agenda semanal completa: un individuo de la población."""

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
    """
    Algoritmo Genético para la planificación semanal de quirófanos.

    Uso básico:
        ga = GeneticAlgorithm(config, operating_rooms, specialties, patients_by_specialty)
        best = ga.run()
        ga.print_schedule(best)
    """

    DAY_NAMES   = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    SHIFT_NAMES = ["Mañana", "Tarde"]

    def __init__(
        self,
        config: GAConfig,
        operating_rooms: List[OperatingRoom],
        specialties: List[Specialty],
        patients_by_specialty: Dict[int, List[Patient]],
    ):
        self.cfg = config
        self.ors = operating_rooms
        self.specialties = specialties
        self.patients_by_specialty = patients_by_specialty

        self.n_ors    = len(operating_rooms)
        self.n_days   = config.n_days
        self.n_shifts = config.n_shifts

        # Índices rápidos
        self._spec_by_id: Dict[int, Specialty]    = {s.id: s for s in specialties}
        self._or_by_idx:  Dict[int, OperatingRoom] = dict(enumerate(operating_rooms))

        # IDs de especialidades reales (excluye id=0 = libre)
        self._real_spec_ids: List[int] = [s.id for s in specialties if s.id != 0]

        # Estado del algoritmo
        self.best_individual: Optional[Individual] = None
        self.history: List[float] = []  # mejor fitness por generación

    # ═══════════════════════════════════════════════════════════════════════
    # 1. INICIALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def _valid_specialties_for(self, or_idx: int, day: int, shift: int) -> List[int]:
        """
        Retorna la lista de specialty_ids compatibles con un bloque dado.
        Tiene en cuenta el tipo de quirófano y su disponibilidad horaria.
        """
        or_ = self._or_by_idx[or_idx]
        if not or_.availability[day][shift]:
            return []
        return [
            s.id for s in self.specialties
            if s.id != 0 and or_.or_type in s.compatible_or_types
        ]

    def _random_specialty_for(self, or_idx: int, day: int, shift: int) -> int:
        """
        Elige aleatoriamente una especialidad válida para el bloque.
        Con un 15 % de probabilidad deja el bloque libre (id=0).
        """
        options = self._valid_specialties_for(or_idx, day, shift)
        if not options or random.random() < 0.15:
            return 0
        return random.choice(options)

    def _create_individual(self) -> Individual:
        """
        Crea un individuo con inicialización semi-inteligente:
        1. Cubre las cuotas mínimas de cada especialidad.
        2. Rellena los bloques sobrantes de forma aleatoria válida.
        """
        chrom = np.zeros((self.n_days, self.n_shifts, self.n_ors), dtype=int)
        counts: Dict[int, int] = {sid: 0 for sid in self._real_spec_ids}

        # Paso 1 — cubrir cuotas mínimas
        for spec_id in self._real_spec_ids:
            spec = self._spec_by_id[spec_id]
            needed  = spec.min_blocks
            filled  = 0
            attempts = 0

            while filled < needed and attempts < 300:
                d = random.randrange(self.n_days)
                t = random.randrange(self.n_shifts)
                q = random.randrange(self.n_ors)
                or_ = self._or_by_idx[q]

                if (chrom[d, t, q] == 0
                        and or_.availability[d][t]
                        and or_.or_type in spec.compatible_or_types):
                    chrom[d, t, q] = spec_id
                    counts[spec_id] += 1
                    filled += 1

                attempts += 1

        # Paso 2 — rellenar bloques vacíos
        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    if chrom[d, t, q] == 0:
                        chrom[d, t, q] = self._random_specialty_for(q, d, t)

        return Individual(chrom)

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
            if spec.id == 0:
                continue
            assigned = counts.get(spec.id, 0)

            if assigned < spec.min_blocks:
                penalty += self.cfg.penalty_below_min_quota * (spec.min_blocks - assigned)

            if assigned > spec.max_blocks:
                penalty += self.cfg.penalty_above_max_quota * (assigned - spec.max_blocks)

        return penalty

    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Fitness Global = Σ Z_bloque(d,t,q)  −  Penalizaciones_globales

        Para cada bloque con especialidad asignada llama al MIP (Nivel 3)
        con los pacientes de esa especialidad como entrada.
        """
        chrom   = individual.chromosome
        total_z = 0.0

        for d in range(self.n_days):
            for t in range(self.n_shifts):
                for q in range(self.n_ors):
                    spec_id = int(chrom[d, t, q])
                    if spec_id == 0:
                        continue

                    patients = self.patients_by_specialty.get(spec_id, [])

                    z = solve_mip_for_block(
                        specialty_id=spec_id,
                        patients=patients,
                        block_duration_min=self.cfg.block_duration_min,
                    )
                    total_z += z

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
