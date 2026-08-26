import random
from typing import Dict, List, Tuple

from models import Agenda, Block, Patient, Procedure, Room, Specialty, Surgeon
from decoder import build_agenda


class GeneticAlgorithm:
    """Algoritmo genético para asignar especialidades a bloques quirúrgicos semanales."""

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
        self.surgeons: Dict[str, Surgeon] = {s.id: s for s in surgeons}
        self.procedures: Dict[str, Procedure] = {pr.id: pr for pr in procedures}
        self.patients = patients

        self.blocks: List[Block] = [Block(day, room.id) for day in days for room in rooms]
        self.specialty_ids: List[str] = [s.id for s in specialties]
        self.min_blocks: Dict[str, int] = {s.id: s.min_blocks for s in specialties}
        self.rooms_by_id: Dict[str, Room] = {room.id: room for room in rooms}

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.stagnation_limit = stagnation_limit

        self.alpha = alpha
        self.beta = beta

        self.total_available_time = sum(room.daily_capacity_minutes for room in rooms) * len(days)
        self.max_achievable_priority = self._compute_max_achievable_priority()

        self.best_individual: Dict[Block, str] = None
        self.best_fitness: float = float("-inf")
        self.history: List[float] = []

    def _compute_max_achievable_priority(self) -> float:
        """Calcula una cota superior para normalizar la suma de prioridades al cuadrado.

        Se recorren los pacientes desde la prioridad más alta y se agregan sus
        duraciones mientras quepan en el tiempo total disponible. El resultado
        no pretende ser una agenda válida, sino una referencia optimista para
        que el componente de prioridad del fitness quede aproximadamente entre
        0 y 1. (Puede darse el caso de que sea mayor a 1)
        """
        sorted_patients = sorted(self.patients, key=lambda p: p.clinical_priority, reverse=True)
        max_priority = 0.0
        time_accumulated = 0

        for patient in sorted_patients:
            procedure = self.procedures.get(patient.procedure_id)
            if not procedure:
                break
            if time_accumulated + procedure.estimated_duration <= self.total_available_time:
                max_priority += patient.clinical_priority ** 2
                time_accumulated += procedure.estimated_duration
            else:
                break

        return max(1.0, max_priority)  # Evita dividir por cero si no hay pacientes.

    def _random_chromosome(self) -> Dict[Block, str]:
        """Crea una solución asignando una especialidad aleatoria a cada bloque."""
        return {block: random.choice(self.specialty_ids) for block in self.blocks}

    def _initial_population(self) -> List[Dict[Block, str]]:
        """Genera la población inicial que explorará el algoritmo."""
        return [self._random_chromosome() for _ in range(self.population_size)]

    def _repair(self, chromosome: Dict[Block, str]) -> Dict[Block, str]:
        """Ajusta el cromosoma para respetar los mínimos de bloques semanales.

        La reparación cambia bloques de especialidades que tienen excedente,
        evitando quitarles bloques que ya necesitan para cumplir su mínimo.
        Así, crossover y mutación pueden trabajar libremente y la solución se
        corrige antes de ser evaluada.
        """
        chromosome = dict(chromosome)
        counts = {specialty_id: 0 for specialty_id in self.specialty_ids}
        for specialty_id in chromosome.values():
            counts[specialty_id] += 1

        for specialty_id in self.specialty_ids:
            need = self.min_blocks.get(specialty_id, 0) - counts[specialty_id]
            if need <= 0:
                continue

            candidate_blocks = list(chromosome.keys())
            random.shuffle(candidate_blocks)

            for block in candidate_blocks:
                if need <= 0:
                    break
                current_specialty = chromosome[block]
                if current_specialty == specialty_id:
                    continue

                if counts[current_specialty] > self.min_blocks.get(current_specialty, 0):
                    chromosome[block] = specialty_id
                    counts[current_specialty] -= 1
                    counts[specialty_id] += 1
                    need -= 1

        return chromosome

    def _evaluate(self, chromosome: Dict[Block, str]) -> Tuple[float, Agenda]:
        """Construye la agenda representada por un cromosoma y calcula su fitness.

        El decoder aplica las restricciones de pacientes, procedimientos,
        quirófanos y cirujanos. Después se combinan la prioridad clínica
        atendida y la utilización del tiempo según ``alpha`` y ``beta``.
        """
        agenda = build_agenda(chromosome, self.patients, self.procedures, self.surgeons, self.rooms_by_id)
        surgeries = agenda.all_surgeries()
        patients_by_id = {patient.id: patient for patient in self.patients}

        if surgeries:
            total_scheduled_priority_sq = sum(
                (patients_by_id[surgery.patient_id].clinical_priority ** 2) for surgery in surgeries
            )
            priority_score = total_scheduled_priority_sq / self.max_achievable_priority
        else:
            priority_score = 0.0

        used_time = sum(agenda.used_time.values())
        utilization = (used_time / self.total_available_time) if self.total_available_time > 0 else 0.0
        fitness = (self.alpha * priority_score) + (self.beta * utilization)
        return fitness, agenda

    def _tournament_selection(self, population: List[Dict[Block, str]], fitnesses: List[float]) -> Dict[Block, str]:
        """Selecciona un individuo para reproducirse mediante un torneo.

        Se eligen varios individuos al azar, se comparan sus valores de
        fitness y se devuelve el que obtuvo la mejor puntuación. De esta
        manera, las soluciones con mejor rendimiento tienen más posibilidades
        de generar nuevos individuos.
        """
        contenders = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(contenders, key=lambda index: fitnesses[index])
        return population[best_idx]

    def _crossover(self, parent1: Dict[Block, str], parent2: Dict[Block, str]) -> Tuple[Dict[Block, str], Dict[Block, str]]:
        """Combina dos individuos usando un punto de corte sobre los bloques."""
        if random.random() > self.crossover_rate:
            return dict(parent1), dict(parent2)

        cut = random.randint(1, len(self.blocks) - 1)
        child1, child2 = {}, {}
        for index, block in enumerate(self.blocks):
            if index < cut:
                child1[block] = parent1[block]
                child2[block] = parent2[block]
            else:
                child1[block] = parent2[block]
                child2[block] = parent1[block]
        return child1, child2

    def _mutate(self, chromosome: Dict[Block, str]) -> Dict[Block, str]:
        """Introduce variación cambiando algunas especialidades al azar."""
        chromosome = dict(chromosome)
        for block in self.blocks:
            if random.random() < self.mutation_rate:
                chromosome[block] = random.choice(self.specialty_ids)
        return chromosome

    def run(self) -> Tuple[Dict[Block, str], float, Agenda]:
        """Ejecuta la evolución y devuelve la mejor solución encontrada.

        Cada generación selecciona individuos, crea descendientes mediante
        crossover y mutación, repara sus mínimos y los evalúa con el decoder.
        Se conserva el mejor individuo global porque una nueva generación no
        necesariamente mejora a la anterior.
        """
        # La reparación inicial evita comenzar evaluando cromosomas inválidos.
        population = [self._repair(chromosome) for chromosome in self._initial_population()]
        evaluations = [self._evaluate(chromosome) for chromosome in population]
        fitnesses = [score for score, _ in evaluations]

        best_fitness = max(fitnesses)
        best_idx = fitnesses.index(best_fitness)
        best_individual = population[best_idx]
        best_agenda = evaluations[best_idx][1]
        stagnation = 0

        for generation in range(self.generations):
            new_population = []

            # Se generan descendientes hasta recuperar el tamaño de población.
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
            evaluations = [self._evaluate(chromosome) for chromosome in population]
            fitnesses = [score for score, _ in evaluations]

            gen_best_fitness = max(fitnesses)
            gen_best_idx = fitnesses.index(gen_best_fitness)
            self.history.append(gen_best_fitness)

            # Solo se reemplaza el mejor global si la mejora es significativa.
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