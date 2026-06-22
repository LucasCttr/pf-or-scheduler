import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pulp")

from back.genetic_algorithm import GeneticAlgorithm, Individual
from back.models import GAConfig, OperatingRoom, Specialty, Staff


FREE_SPECIALTY_ID = 0
TRAUMA_SPECIALTY_ID = 1
GENERAL_SPECIALTY_ID = 2


def _make_ga_for_repair():
    # Este builder arma un escenario mínimo para probar repair() en aislamiento,
    # sin depender del loop completo del AG ni del solver.
    #
    # La idea es tener:
    # - 2 días
    # - 1 turno
    # - 2 quirófanos de distinto tipo
    # - una especialidad que solo entra en OR alta
    # - otra que solo entra en OR media
    #
    # Además, el OR alta queda cerrado el día 1 para poder chequear
    # que repair() limpie bloques en quirófanos no disponibles.
    config = GAConfig(
        population_size=4,
        max_generations=3,
        convergence_patience=2,
        mutation_rate=0.1,
        crossover_rate=0.8,
        tournament_size=2,
        elite_count=1,
        n_days=2,
        n_shifts=1,
    )

    operating_rooms = [
        OperatingRoom(
            id=0,
            name="OR Alta",
            or_type="alta_complejidad",
            availability=[[True], [False]],
        ),
        OperatingRoom(
            id=1,
            name="OR Media",
            or_type="media_complejidad",
            availability=[[True], [True]],
        ),
    ]

    # Trauma ya arranca con 1 bloque asignado. Por eso su mínimo es 1.
    # Así el test queda enfocado en verificar cómo repair() completa General.
    specialties = [
        Specialty(id=FREE_SPECIALTY_ID, name="Libre", compatible_or_types=[], min_blocks=0, max_blocks=99),
        Specialty(id=TRAUMA_SPECIALTY_ID, name="Trauma", compatible_or_types=["alta_complejidad"], min_blocks=1, max_blocks=2),
        Specialty(id=GENERAL_SPECIALTY_ID, name="General", compatible_or_types=["media_complejidad"], min_blocks=2, max_blocks=2),
    ]

    staff_list = [
        Staff(
            id=10,
            name="Dr Trauma",
            role="cirujano",
            specialties_ids=[TRAUMA_SPECIALTY_ID],
            availability_hours={0: (480, 720)},
        ),
        Staff(
            id=20,
            name="Dr General",
            role="cirujano",
            specialties_ids=[GENERAL_SPECIALTY_ID],
            availability_hours={0: (480, 720), 1: (480, 720)},
        ),
    ]

    return GeneticAlgorithm(
        config=config,
        operating_rooms=operating_rooms,
        specialties=specialties,
        patients_by_specialty={},
        staff_list=staff_list,
    )


def test_repair_replaces_incompatible_specialties_with_valid_ones():
    # Este test verifica la primera responsabilidad de repair():
    # si una especialidad quedó puesta en un quirófano incompatible,
    # debe reemplazarla por una opción válida para ese bloque.
    #
    # Forzamos _random_specialty_for para que el resultado sea determinístico
    # y así no depender del azar en la corrección.
    ga = _make_ga_for_repair()
    ga._random_specialty_for = lambda or_idx, day, shift: TRAUMA_SPECIALTY_ID if or_idx == 0 else GENERAL_SPECIALTY_ID

    # Reminder: El cromosoma se arma como [día][turno][quirófano].
    # En este escenario: posición 0 = OR alta, posición 1 = OR media.
    individual = Individual(
        np.array(
            [
                # día 0:
                # - OR alta tiene General, que es incompatible y debería corregirse a Trauma
                # - OR media tiene Trauma, que es incompatible y debería corregirse a General
                [[GENERAL_SPECIALTY_ID, TRAUMA_SPECIALTY_ID]],
                # día 1:
                # - ambos bloques quedan libres; no forman parte del caso principal de este test
                [[FREE_SPECIALTY_ID, FREE_SPECIALTY_ID]],
            ],
            dtype=int,
        )
    )

    repaired = ga.repair(individual)

    # día 0, turno 0, OR alta: General era incompatible y debe corregirse a Trauma.
    assert repaired.chromosome[0, 0, 0] == TRAUMA_SPECIALTY_ID
    # día 0, turno 0, OR media: Trauma era incompatible y debe corregirse a General.
    assert repaired.chromosome[0, 0, 1] == GENERAL_SPECIALTY_ID


def test_repair_clears_blocks_for_unavailable_operating_rooms():
    # Acá probamos otra regla base de repair():
    # si el quirófano no está disponible en ese día/turno, el bloque tiene que quedar libre sí o sí.
    #
    # En este escenario, el OR alta está cerrado el día 1.
    ga = _make_ga_for_repair()

    # Igual que en el test anterior, el shape es [día][turno][quirófano].
    individual = Individual(
        np.array(
            [
                # día 0:
                # - ambos bloques están bien asignados y deberían conservarse
                [[TRAUMA_SPECIALTY_ID, GENERAL_SPECIALTY_ID]],
                # día 1:
                # - OR alta tiene Trauma, pero ese quirófano está cerrado ese día
                #   así que repair() debe dejarlo libre
                # - OR media sigue disponible y General es compatible, así que debe quedar igual
                [[TRAUMA_SPECIALTY_ID, GENERAL_SPECIALTY_ID]],
            ],
            dtype=int,
        )
    )

    repaired = ga.repair(individual)

    # día 1, turno 0, OR alta: como el quirófano está cerrado, el bloque debe quedar libre.
    assert repaired.chromosome[1, 0, 0] == FREE_SPECIALTY_ID
    # día 1, turno 0, OR media: sigue disponible y la asignación válida debe mantenerse.
    assert repaired.chromosome[1, 0, 1] == GENERAL_SPECIALTY_ID


def test_repair_fills_minimum_quotas_when_valid_slots_exist():
    # Este test apunta a la parte de cuotas mínimas.
    #
    # Dejamos a General por debajo de su mínimo y con dos slots válidos libres en el OR media. 
    # repair() debería usar esos huecos para completar exactamente la cantidad de bloques faltantes.
    ga = _make_ga_for_repair()
    random.seed(42)

    # Otra vez: [día][turno][quirófano], con OR alta en la posición 0 y OR media en la 1.
    individual = Individual(
        np.array(
            [
                # día 0:
                # - OR alta ya tiene Trauma, con eso Trauma cumple su mínimo
                # - OR media queda libre y puede usarse para completar General
                [[TRAUMA_SPECIALTY_ID, FREE_SPECIALTY_ID]],
                # día 1:
                # - OR alta queda libre pero no disponible, así que no sirve como candidato
                # - OR media queda libre y también puede usarse para completar General
                [[FREE_SPECIALTY_ID, FREE_SPECIALTY_ID]],
            ],
            dtype=int,
        )
    )

    repaired = ga.repair(individual)
    counts = ga._count_blocks_per_specialty(repaired.chromosome)

    assert counts[TRAUMA_SPECIALTY_ID] == 1
    assert counts[GENERAL_SPECIALTY_ID] == 2
    # día 0, turno 0, OR media: uno de los bloques libres se usa para completar General.
    assert repaired.chromosome[0, 0, 1] == GENERAL_SPECIALTY_ID
    # día 1, turno 0, OR media: el segundo bloque válido libre también se asigna a General.
    assert repaired.chromosome[1, 0, 1] == GENERAL_SPECIALTY_ID
