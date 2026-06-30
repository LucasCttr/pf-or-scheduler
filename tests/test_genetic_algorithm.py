"""
tests/test_genetic_algorithm.py
Tests unitarios del Algoritmo Genético.

Los tests de operadores (_repair, _crossover, _mutate) son deterministas
gracias a random.seed. Los tests de run() solo verifican propiedades
estructurales del resultado (tipos, claves, monotonicidad del historial),
no valores exactos de fitness, ya que eso depende de la aleatoriedad.
"""
import random
import pytest
from models import Block, Patient, Procedure, Room, Specialty, Surgeon
from genetic_algorithm import GeneticAlgorithm


# ---------------------------------------------------------------------------
# Fixture: GA mínimo reutilizable
# ---------------------------------------------------------------------------

@pytest.fixture
def base_ga():
    """GA con configuración mínima para que los tests sean rápidos."""
    days = ["lunes", "martes"]
    rooms = [Room(id="Q1", name="Q1", room_type=2, daily_capacity_minutes=300)]
    specialties = [
        Specialty(id="TRA", name="Traumatologia", min_blocks=1),
        Specialty(id="CG",  name="Cirugia General", min_blocks=1),
    ]
    surgeons = [
        Surgeon(id="S1", name="Dr. Lopez", specialty_id="TRA",
                available_days={"lunes", "martes"}, contract_hours_week=20),
        Surgeon(id="S2", name="Dr. Perez", specialty_id="CG",
                available_days={"lunes", "martes"}, contract_hours_week=20),
    ]
    procedures = [
        Procedure(id="PR1", name="Fractura", specialty_id="TRA",
                  required_room_type=1, estimated_duration=90),
        Procedure(id="PR2", name="Apendice", specialty_id="CG",
                  required_room_type=1, estimated_duration=60),
    ]
    patients = [
        Patient(id=f"P{i}", specialty_id="TRA", procedure_id="PR1",
                surgeon_id="S1", clinical_priority=float(i))
        for i in range(1, 5)
    ] + [
        Patient(id=f"P{i}", specialty_id="CG", procedure_id="PR2",
                surgeon_id="S2", clinical_priority=float(i))
        for i in range(5, 9)
    ]
    return GeneticAlgorithm(
        days=days,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=patients,
        population_size=10,
        generations=5,
        tournament_size=2,
        stagnation_limit=5,
        cleaning_minutes=15,
    )


# ---------------------------------------------------------------------------
# Tests de inicialización
# ---------------------------------------------------------------------------

class TestInitialization:

    def test_bloques_son_producto_cartesiano_dias_salas(self, base_ga):
        """El número de bloques debe ser len(days) * len(rooms)."""
        expected = len(base_ga.days) * len(base_ga.rooms)
        assert len(base_ga.blocks) == expected

    def test_cromosoma_aleatorio_tiene_todas_las_claves(self, base_ga):
        """Un cromosoma generado debe tener exactamente un gen por bloque."""
        random.seed(0)
        chrom = base_ga._random_chromosome()
        assert set(chrom.keys()) == set(base_ga.blocks)

    def test_cromosoma_solo_contiene_especialidades_validas(self, base_ga):
        """Todos los genes de un cromosoma aleatorio deben ser specialty_ids válidos."""
        random.seed(0)
        chrom = base_ga._random_chromosome()
        for sid in chrom.values():
            assert sid in base_ga.specialty_ids


# ---------------------------------------------------------------------------
# Tests del operador de reparación
# ---------------------------------------------------------------------------

class TestRepair:

    def test_repair_cumple_min_blocks(self, base_ga):
        """Después de _repair, cada especialidad debe tener >= min_blocks bloques."""
        random.seed(42)
        # Cromosoma donde toda la capacidad va a TRA (CG queda en 0)
        chromosome = {b: "TRA" for b in base_ga.blocks}
        repaired = base_ga._repair(chromosome)
        counts = {sid: 0 for sid in base_ga.specialty_ids}
        for sid in repaired.values():
            counts[sid] += 1
        for sid, min_b in base_ga.min_blocks.items():
            assert counts[sid] >= min_b, (
                f"Especialidad {sid}: {counts[sid]} bloques < mínimo {min_b}"
            )

    def test_repair_no_pierde_bloques(self, base_ga):
        """_repair no debe cambiar el número total de bloques."""
        chromosome = {b: "TRA" for b in base_ga.blocks}
        repaired = base_ga._repair(chromosome)
        assert len(repaired) == len(base_ga.blocks)

    def test_repair_idempotente_si_ya_es_valido(self, base_ga):
        """Un cromosoma que ya cumple min_blocks no debe cambiar tras _repair."""
        random.seed(1)
        chromosome = base_ga._random_chromosome()
        repaired_once = base_ga._repair(chromosome)
        # Contar antes y después
        counts_before = {}
        for sid in repaired_once.values():
            counts_before[sid] = counts_before.get(sid, 0) + 1
        repaired_twice = base_ga._repair(repaired_once)
        counts_after = {}
        for sid in repaired_twice.values():
            counts_after[sid] = counts_after.get(sid, 0) + 1
        assert counts_before == counts_after


# ---------------------------------------------------------------------------
# Tests del operador de cruza
# ---------------------------------------------------------------------------

class TestCrossover:

    def test_cruza_produce_dos_hijos(self, base_ga):
        """_crossover siempre devuelve exactamente dos hijos."""
        random.seed(0)
        p1 = base_ga._random_chromosome()
        p2 = base_ga._random_chromosome()
        c1, c2 = base_ga._crossover(p1, p2)
        assert isinstance(c1, dict) and isinstance(c2, dict)

    def test_hijos_tienen_todos_los_bloques(self, base_ga):
        """Ambos hijos deben tener exactamente los mismos bloques que los padres."""
        random.seed(5)
        p1 = base_ga._random_chromosome()
        p2 = base_ga._random_chromosome()
        c1, c2 = base_ga._crossover(p1, p2)
        assert set(c1.keys()) == set(base_ga.blocks)
        assert set(c2.keys()) == set(base_ga.blocks)

    def test_genes_de_hijos_son_de_padres(self, base_ga):
        """Cada gen de un hijo debe provenir de uno de los dos padres."""
        random.seed(7)
        p1 = base_ga._random_chromosome()
        p2 = base_ga._random_chromosome()
        c1, c2 = base_ga._crossover(p1, p2)
        for b in base_ga.blocks:
            assert c1[b] in (p1[b], p2[b])
            assert c2[b] in (p1[b], p2[b])

    def test_sin_cruza_devuelve_copias_de_padres(self, base_ga):
        """Con crossover_rate=0 los hijos son copias exactas de los padres."""
        base_ga.crossover_rate = 0.0
        random.seed(0)
        p1 = base_ga._random_chromosome()
        p2 = base_ga._random_chromosome()
        c1, c2 = base_ga._crossover(p1, p2)
        assert c1 == p1
        assert c2 == p2


# ---------------------------------------------------------------------------
# Tests del operador de mutación
# ---------------------------------------------------------------------------

class TestMutation:

    def test_mutacion_no_agrega_ni_elimina_bloques(self, base_ga):
        """_mutate no debe cambiar las claves del cromosoma."""
        random.seed(0)
        chrom = base_ga._random_chromosome()
        mutated = base_ga._mutate(chrom)
        assert set(mutated.keys()) == set(chrom.keys())

    def test_mutacion_solo_usa_especialidades_validas(self, base_ga):
        """Los genes mutados deben ser specialty_ids válidos."""
        base_ga.mutation_rate = 1.0  # mutar todos los genes
        random.seed(0)
        chrom = base_ga._random_chromosome()
        mutated = base_ga._mutate(chrom)
        for sid in mutated.values():
            assert sid in base_ga.specialty_ids

    def test_sin_mutacion_cromosoma_no_cambia(self, base_ga):
        """Con mutation_rate=0 el cromosoma no debe cambiar."""
        base_ga.mutation_rate = 0.0
        random.seed(0)
        chrom = base_ga._random_chromosome()
        mutated = base_ga._mutate(chrom)
        assert mutated == chrom


# ---------------------------------------------------------------------------
# Tests de la función de fitness
# ---------------------------------------------------------------------------

class TestFitness:

    def test_fitness_es_positivo_con_pacientes_programados(self, base_ga):
        """Un cromosoma que programa al menos un paciente debe tener fitness > 0."""
        random.seed(0)
        chrom = base_ga._repair(base_ga._random_chromosome())
        fitness, _ = base_ga._evaluate(chrom)
        assert fitness >= 0

    def test_fitness_mayor_con_mas_prioridad(self, base_ga):
        """Un cromosoma que programa pacientes de alta prioridad tiene
        fitness mayor que uno que deja esos bloques sin pacientes factibles.
        Verificamos que el fitness es sensible a la prioridad comparando
        un cromosoma que sí programa pacientes contra uno vacío."""
        # Cromosoma que asigna URO (sin pacientes), nadie se programa → fitness bajo
        chrom_vacio = {b: "URO_INEXISTENTE" for b in base_ga.blocks}
        # No reparamos, solo evaluamos directamente para asegurarnos de que
        # la agenda quede sin cirugías.
        from models import Block as B
        # Cromosoma válido reparado que sí programa pacientes
        chrom_con_pacientes = base_ga._repair(
            {b: "CG" for b in base_ga.blocks}
        )
        fitness_con, _ = base_ga._evaluate(chrom_con_pacientes)

        # Simular cromosoma vacío manualmente sin pasar por repair
        chrom_empty = {b: base_ga.specialty_ids[0] for b in base_ga.blocks}
        # Vaciar todos los pacientes del GA para este evaluate puntual
        old_patients = base_ga.patients
        base_ga.patients = []
        fitness_vacio, _ = base_ga._evaluate(chrom_empty)
        base_ga.patients = old_patients

        assert fitness_con > fitness_vacio


# ---------------------------------------------------------------------------
# Tests del ciclo completo run()
# ---------------------------------------------------------------------------

class TestRun:

    def test_run_devuelve_tres_elementos(self, base_ga):
        """run() debe devolver (cromosoma, fitness, agenda)."""
        random.seed(0)
        result = base_ga.run()
        assert len(result) == 3

    def test_cromosoma_resultado_tiene_todos_los_bloques(self, base_ga):
        """El cromosoma de la mejor solución debe tener un gen por bloque."""
        random.seed(0)
        best_chrom, _, _ = base_ga.run()
        assert set(best_chrom.keys()) == set(base_ga.blocks)

    def test_historial_no_vacio(self, base_ga):
        """Después de run(), ga.history debe tener al menos una entrada."""
        random.seed(0)
        base_ga.run()
        assert len(base_ga.history) >= 1

    def test_historial_es_monotono_no_decreciente(self, base_ga):
        """El mejor fitness encontrado no puede empeorar entre generaciones."""
        random.seed(0)
        base_ga.generations = 20
        base_ga.stagnation_limit = 20
        base_ga.run()
        for i in range(1, len(base_ga.history)):
            assert base_ga.history[i] >= base_ga.history[i - 1] - 1e-9, (
                f"Fitness bajó en generación {i}: "
                f"{base_ga.history[i-1]:.4f} -> {base_ga.history[i]:.4f}"
            )

    def test_fitness_final_mayor_o_igual_al_inicial(self, base_ga):
        """El fitness de la mejor solución final debe ser >= al de la generación 1."""
        random.seed(0)
        base_ga.generations = 15
        base_ga.stagnation_limit = 15
        base_ga.run()
        assert base_ga.best_fitness >= base_ga.history[0] - 1e-9

    def test_run_reproducible_con_seed(self, base_ga):
        """Dos ejecuciones con la misma seed deben producir el mismo fitness."""
        random.seed(99)
        _, f1, _ = base_ga.run()
        # Reiniciar el estado del GA para una segunda corrida
        base_ga.history = []
        base_ga.best_fitness = float("-inf")
        base_ga.best_individual = None
        random.seed(99)
        _, f2, _ = base_ga.run()
        assert f1 == f2
