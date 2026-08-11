"""Genetic allocation of specialties followed by the root greedy decoder."""

import random
from typing import Dict, List, Tuple

from decoder import build_agenda
from models import Agenda, Block, Patient, Procedure, Room, Specialty, Surgeon


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
        stagnation_limit: int = 30,
        alpha: float = 1.0,
        beta: float = 0.3,
    ):
        self.days = days
        self.rooms = rooms
        self.specialties = specialties
        self.surgeons = {surgeon.id: surgeon for surgeon in surgeons}
        self.procedures = {procedure.id: procedure for procedure in procedures}
        self.patients = patients
        self.blocks = [Block(day, room.id) for day in days for room in rooms]
        self.specialty_ids = [specialty.id for specialty in specialties]
        self.min_blocks = {specialty.id: specialty.min_blocks for specialty in specialties}
        self.max_blocks = {specialty.id: specialty.max_blocks for specialty in specialties}
        self.rooms_by_id = {room.id: room for room in rooms}
        self.population_size = max(2, population_size)
        self.generations = max(1, generations)
        self.tournament_size = max(1, min(tournament_size, self.population_size))
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.stagnation_limit = max(1, stagnation_limit)
        self.alpha = alpha
        self.beta = beta
        self.history: List[float] = []

        self._procedures_by_specialty: Dict[str, List[Procedure]] = {}
        for procedure in procedures:
            self._procedures_by_specialty.setdefault(procedure.specialty_id, []).append(procedure)
        self.total_available_time = sum(
            room.daily_capacity_minutes
            for day in days
            for room in rooms
            if room.is_available(day)
        )
        sorted_patients = sorted(patients, key=lambda patient: (-patient.clinical_priority, patient.id))
        self.max_achievable_priority = max(
            1.0,
            sum(patient.clinical_priority ** 2 for patient in sorted_patients),
        )

    def _valid_specialties(self, block: Block) -> List[str]:
        room = self.rooms_by_id[block.room_id]
        if not room.is_available(block.day):
            return [""]
        valid = [""]
        for specialty_id in self.specialty_ids:
            if any(
                procedure.required_room_type <= room.room_type
                for procedure in self._procedures_by_specialty.get(specialty_id, [])
            ):
                valid.append(specialty_id)
        return valid

    def _random_chromosome(self) -> Dict[Block, str]:
        return {block: random.choice(self._valid_specialties(block)) for block in self.blocks}

    def _initial_population(self) -> List[Dict[Block, str]]:
        return [self._random_chromosome() for _ in range(self.population_size)]

    def _repair(self, chromosome: Dict[Block, str]) -> Dict[Block, str]:
        repaired = dict(chromosome)
        for block in self.blocks:
            valid = self._valid_specialties(block)
            if repaired.get(block) not in valid:
                repaired[block] = random.choice(valid)

        counts = {specialty_id: 0 for specialty_id in self.specialty_ids}
        for specialty_id in repaired.values():
            if specialty_id in counts:
                counts[specialty_id] += 1

        # Reduce maxima first while preserving donors' minima.
        for specialty_id in self.specialty_ids:
            excess = counts[specialty_id] - self.max_blocks[specialty_id]
            if excess <= 0:
                continue
            candidates = [block for block, value in repaired.items() if value == specialty_id]
            random.shuffle(candidates)
            for block in candidates:
                if excess <= 0:
                    break
                replacements = [
                    value for value in self._valid_specialties(block)
                    if value != specialty_id
                    and (value == "" or counts.get(value, 0) < self.max_blocks.get(value, 0))
                ]
                if not replacements:
                    continue
                replacement = min(replacements, key=lambda value: (counts.get(value, 0), value))
                repaired[block] = replacement
                counts[specialty_id] -= 1
                if replacement in counts:
                    counts[replacement] += 1
                excess -= 1

        for specialty_id in self.specialty_ids:
            needed = self.min_blocks[specialty_id] - counts[specialty_id]
            if needed <= 0:
                continue
            candidates = list(self.blocks)
            random.shuffle(candidates)
            for block in candidates:
                if needed <= 0:
                    break
                donor = repaired[block]
                if specialty_id not in self._valid_specialties(block) or donor == specialty_id:
                    continue
                if donor in counts and counts[donor] <= self.min_blocks[donor]:
                    continue
                repaired[block] = specialty_id
                if donor in counts:
                    counts[donor] -= 1
                counts[specialty_id] += 1
                needed -= 1
        return repaired

    def _evaluate(self, chromosome: Dict[Block, str]) -> Tuple[float, Agenda]:
        agenda = build_agenda(chromosome, self.patients, self.procedures, self.surgeons, self.rooms_by_id)
        patients_by_id = {patient.id: patient for patient in self.patients}
        scheduled = agenda.all_surgeries()
        priority = sum(patients_by_id[item.patient_id].clinical_priority ** 2 for item in scheduled)
        utilization = sum(agenda.used_time.values()) / self.total_available_time if self.total_available_time else 0.0
        return self.alpha * (priority / self.max_achievable_priority) + self.beta * utilization, agenda

    def _tournament_selection(self, population, fitnesses):
        contenders = random.sample(range(len(population)), self.tournament_size)
        return population[max(contenders, key=lambda index: fitnesses[index])]

    def _crossover(self, parent1, parent2):
        if len(self.blocks) < 2 or random.random() > self.crossover_rate:
            return dict(parent1), dict(parent2)
        cut = random.randint(1, len(self.blocks) - 1)
        child1, child2 = {}, {}
        for index, block in enumerate(self.blocks):
            child1[block] = parent1[block] if index < cut else parent2[block]
            child2[block] = parent2[block] if index < cut else parent1[block]
        return child1, child2

    def _mutate(self, chromosome):
        mutated = dict(chromosome)
        for block in self.blocks:
            if random.random() < self.mutation_rate:
                mutated[block] = random.choice(self._valid_specialties(block))
        return mutated

    def run(self) -> Tuple[Dict[Block, str], float, Agenda]:
        population = [self._repair(chromosome) for chromosome in self._initial_population()]
        evaluations = [self._evaluate(chromosome) for chromosome in population]
        fitnesses = [fitness for fitness, _ in evaluations]
        best_index = max(range(len(population)), key=lambda index: fitnesses[index])
        best, best_fitness, best_agenda = population[best_index], fitnesses[best_index], evaluations[best_index][1]
        stagnation = 0

        for _ in range(self.generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._repair(self._mutate(child1)))
                if len(new_population) < self.population_size:
                    new_population.append(self._repair(self._mutate(child2)))
            population = new_population
            evaluations = [self._evaluate(chromosome) for chromosome in population]
            fitnesses = [fitness for fitness, _ in evaluations]
            generation_index = max(range(len(population)), key=lambda index: fitnesses[index])
            generation_best = fitnesses[generation_index]
            self.history.append(generation_best)
            if generation_best > best_fitness + 1e-9:
                best, best_fitness, best_agenda = (
                    population[generation_index], generation_best, evaluations[generation_index][1]
                )
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= self.stagnation_limit:
                break
        return best, best_fitness, best_agenda
