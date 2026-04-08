import math
import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pulp")

from genetic_algorithm import GeneticAlgorithm
from tests.builders import deterministic_end_to_end_scenario


def _run_scenario(seed):
    scenario = deterministic_end_to_end_scenario()
    random.seed(seed)
    np.random.seed(seed)

    # Acá armamos el AG con el escenario del builder.
    # La seed se fija antes para que, si el algoritmo realmente es determinístico,
    # dos corridas iguales terminen en el mismo resultado.
    ga = GeneticAlgorithm(
        config=scenario["config"],
        operating_rooms=scenario["operating_rooms"],
        specialties=scenario["specialties"],
        patients_by_specialty=scenario["patients_by_specialty"],
        staff_list=scenario["staff_list"],
    )

    best = ga.run()
    details = ga.get_schedule_details(best)
    return scenario, ga, best, details


def test_end_to_end_finds_the_unique_expected_schedule(fixed_seed):
    # Este es el test principal de punta a punta.
    # La idea es: con este caso artificial, chico y bien controlado,
    # ya sabemos de antemano cuál debería ser la agenda final.
    scenario, ga, best, details = _run_scenario(fixed_seed)

    # Primero chequeamos la grilla global del AG: qué especialidad puso en cada bloque de la semana.
    assert best.chromosome.tolist() == scenario["expected_chromosome"]
    assert math.isfinite(best.fitness)
    assert ga.history

    # Después bajamos un nivel más:
    # no alcanza con que la especialidad esté bien asignada,
    # también queremos ver que dentro de cada bloque haya elegido
    # exactamente a los pacientes que esperábamos.
    #
    # Como cada bloque dura 240 minutos y cada paciente dura 120,
    # el bloque debería quedar lleno con 2 pacientes y 100% de utilización.
    for key, expected_patients in scenario["expected_schedule"].items():
        per_or = details[key]
        assert per_or["pacientes_ids"] == expected_patients
        assert per_or["uso_tiempo"] == 240
        assert per_or["utilizacion_porcentaje"] == 100.0

    # Por último, juntamos todos los pacientes operados en la semana y verificamos dos cosas:
    # 1. que estén todos los que esperábamos
    # 2. que ninguno aparezca repetido en más de un bloque
    all_patients = []
    for key in scenario["expected_schedule"]:
        all_patients.extend(details[key]["pacientes_ids"])
    assert sorted(all_patients) == scenario["expected_all_patients"]
    assert len(all_patients) == len(set(all_patients))


def test_end_to_end_is_reproducible_for_the_same_seed(fixed_seed):
    # Este segundo test no mira si la solución es "correcta", eso ya lo cubre el test de arriba.
    #
    # Acá lo que queremos probar es otra cosa:
    # si corrés dos veces el mismo escenario con la misma seed, el algoritmo no debería cambiar de resultado entre corridas.
    #
    # Misma asignación de especialidades, mismo fitness y mismos detalles por bloque.
    _, _, best_a, details_a = _run_scenario(fixed_seed)
    _, _, best_b, details_b = _run_scenario(fixed_seed)

    assert best_a.chromosome.tolist() == best_b.chromosome.tolist()
    assert best_a.fitness == best_b.fitness
    assert details_a == details_b
