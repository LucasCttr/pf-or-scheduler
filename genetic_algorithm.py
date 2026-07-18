import copy
import random
from typing import Dict, List, Tuple

from models import Agenda, Block, Patient, Procedure, Room, Specialty, Surgeon
from decoder import build_agenda

class GeneticAlgorithm:
    def __init__(
        self,
        days: List[str],
        rooms: List[Room],
        specialties: List[Specialty],
        surgeons: List[Surgeon],
        procedures: List[Procedure],
        patients: List[Patient],
        population_size: int = 50,
        generations: int = 200,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        elitism_rate: float = 0.1,
        stagnation_limit: int = 30,
        alpha: float = 1.0,
        beta: float = 0.3,
    ):
        self.days = days
        self.rooms = rooms
        self.specialties = specialties
        self.surgeons: Dict[str, Surgeon] = {s.id: s for s in surgeons}
        self.procedures: Dict[str, Procedure] = {pr.id: pr for pr in procedures}
        self.patients = patients

        self.blocks: List[Block] = [Block(d, r.id) for d in days for r in rooms]
        self.specialty_ids: List[str] = [s.id for s in specialties]
        self.min_blocks: Dict[str, int] = {s.id: s.min_blocks for s in specialties}
        self.rooms_by_id: Dict[str, Room] = {r.id: r for r in rooms}

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(elitism_rate * population_size))
        self.stagnation_limit = stagnation_limit

        self.alpha = alpha
        self.beta = beta

        self.total_available_time = sum(r.daily_capacity_minutes for r in rooms) * len(days)

        # Cota Superior Ideal para normalizar la suma de prioridades (uso de p^2)
        sorted_patients = sorted(self.patients, key=lambda p: p.clinical_priority, reverse=True)
        self.max_achievable_priority_sq = 0
        time_accumulated = 0
        for p in sorted_patients:
            proc = self.procedures.get(p.procedure_id)
            if proc and (time_accumulated + proc.estimated_duration <= self.total_available_time):
                self.max_achievable_priority_sq += (p.clinical_priority ** 2)
                
                # CORRECCIÓN: Aquí es donde estaba fallando en la línea 60
                time_accumulated += proc.estimated_duration 
            else:
                break
        self.max_achievable_priority_sq = max(1, self.max_achievable_priority_sq)

        self.best_individual: Dict[Block, str] = None
        self.best_fitness: float = float("-inf")
        self.history: List[float] = []

    def _random_chromosome(self) -> Dict[Block, str]:
        return {b: random.choice(self.specialty_ids) for b in self.blocks}

    def _initial_population(self) -> List[Dict[Block, str]]:
        return [self._random_chromosome() for _ in range(self.population_size)]

    def _repair(self, chromosome: Dict[Block, str]) -> Dict[Block, str]:
        chromosome = dict(chromosome)
        counts = {sid: 0 for sid in self.specialty_ids}
        for sid in chromosome.values():
            counts[sid] += 1

        deficit = {sid: max(0, self.min_blocks.get(sid, 0) - counts[sid]) for sid in self.specialty_ids}

        for sid, need in deficit.items():
            if need <= 0: continue
            donors = [b for b, s in chromosome.items() if counts[s] > self.min_blocks.get(s, 0)]
            random.shuffle(donors)
            for b in donors:
                if need <= 0: break
                old_sid = chromosome[b]
                chromosome[b] = sid
                counts[old_sid] -= 1
                counts[sid] += 1
                need -= 1
        return chromosome

    def _evaluate(self, chromosome: Dict[Block, str]) -> Tuple[float, Agenda]:
        agenda = build_agenda(chromosome, self.patients, self.procedures, self.surgeons, self.rooms_by_id)
        surgeries = agenda.all_surgeries()
        patients_by_id = {p.id: p for p in self.patients}

        # Suma de prioridades al cuadrado normalizada
        if surgeries:
            total_scheduled_priority_sq = sum((patients_by_id[s.patient_id].clinical_priority ** 2) for s in surgeries)
            priority_score = min(1.0, total_scheduled_priority_sq / self.max_achievable_priority_sq)
        else:
            priority_score = 0.0

        used_time = sum(agenda.used_time.values())
        or_utilization = (used_time / self.total_available_time) if self.total_available_time > 0 else 0.0

        fitness = (self.alpha * priority_score) + (self.beta * or_utilization)
        return fitness, agenda

    def _tournament_selection(self, population: List[Dict[Block, str]], fitnesses: List[float]) -> Dict[Block, str]:
        contenders = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(contenders, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _crossover(self, parent1: Dict[Block, str], parent2: Dict[Block, str]) -> Tuple[Dict[Block, str], Dict[Block, str]]:
        if random.random() > self.crossover_rate:
            return dict(parent1), dict(parent2)
        cut = random.randint(1, len(self.blocks) - 1)
        child1, child2 = {}, {}
        for i, b in enumerate(self.blocks):
            if i < cut:
                child1[b] = parent1[b]; child2[b] = parent2[b]
            else:
                child1[b] = parent2[b]; child2[b] = parent1[b]
        return child1, child2

    def _mutate(self, chromosome: Dict[Block, str]) -> Dict[Block, str]:
        chromosome = dict(chromosome)
        for b in self.blocks:
            if random.random() < self.mutation_rate:
                chromosome[b] = random.choice(self.specialty_ids)
        return chromosome

    def run(self) -> Tuple[Dict[Block, str], float, Agenda]:
        population = [self._repair(c) for c in self._initial_population()]
        evaluations = [self._evaluate(c) for c in population]
        fitnesses = [e[0] for e in evaluations]

        best_fitness = max(fitnesses)
        best_idx = fitnesses.index(best_fitness)
        best_individual = population[best_idx]
        best_agenda = evaluations[best_idx][1]
        stagnation = 0

        for generation in range(self.generations):
            ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            new_population = [copy.deepcopy(population[i]) for i in ranked[: self.elitism_count]]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._repair(self._mutate(child1))
                child2 = self._repair(self._mutate(child2))
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            evaluations = [self._evaluate(c) for c in population]
            fitnesses = [e[0] for e in evaluations]

            gen_best_fitness = max(fitnesses)
            gen_best_idx = fitnesses.index(gen_best_fitness)
            self.history.append(gen_best_fitness)

            if gen_best_fitness > best_fitness + 1e-6:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx]
                best_agenda = evaluations[gen_best_idx][1]
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.stagnation_limit:
                print(f"Convergencia alcanzada en la generacion {generation + 1}.")
                break

        return best_individual, best_fitness, best_agenda