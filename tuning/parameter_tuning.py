"""
parameter_tuning.py

Grid Search para encontrar una buena configuración de parámetros
del Algoritmo Genético.

Mejoras respecto a la version anterior:
  1. Los datos se cargan una sola vez (no en cada repeticion) y se
     copian profundamente antes de cada corrida, ya que el decoder
     muta el estado `scheduled` de los pacientes.
  2. Se registra `avg_generations_used` (generaciones reales hasta
     convergencia, segun len(ga.history)) para poder verificar si
     `GENERATIONS` esta efectivamente limitando algo o si el
     `stagnation_limit` corta antes en la practica.
  3. El ranking final usa un score robusto (avg - std) en lugar de
     ordenar solo por avg_fitness, para no premiar configuraciones
     que ganaron "por suerte" en pocas repeticiones.
  4. Se fija una semilla derivada por repeticion para que el
     experimento completo sea reproducible.
  5. Soporte opcional de paralelizacion con multiprocessing, dado
     que cada combinacion es independiente entre si.
"""

import copy
import csv
import os
import random
import sys

from itertools import product
from multiprocessing import Pool, cpu_count
from statistics import mean, stdev

# ============================================================
# Configuración de rutas
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__),
    "grid_search_results.csv"
)

# ============================================================
# Imports del proyecto
# ============================================================

from data_loader import load_all
from genetic_algorithm import GeneticAlgorithm

# ============================================================
# Configuración general
# ============================================================

DAYS = [
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
]

# ============================================================
# Parámetros a evaluar
# ============================================================

POPULATION_SIZES = [80, 120]
GENERATIONS = [150]              # fijo, no se barre
TOURNAMENT_SIZES = [2, 3]
CROSSOVER_RATES = [0.70, 0.80, 0.90]
MUTATION_RATES = [0.02, 0.05]
REPETITIONS = 5     

# Semilla base para que el experimento completo sea reproducible.
# Cada repeticion individual deriva su propia semilla a partir de esta.
BASE_SEED = 123

# Usar multiprocessing para acelerar el grid search.
# Cada combinacion de parametros se evalua en un proceso separado.
USE_MULTIPROCESSING = True
N_WORKERS = max(1, cpu_count() - 1)

# ============================================================
# Carga de datos (una sola vez)
# ============================================================

# Estos datos son inmutables salvo `Patient.scheduled`, que el
# decoder modifica en cada corrida. Por eso se copian profundamente
# antes de cada ejecucion del AG en lugar de releer los CSV.
_SPECIALTIES, _ROOMS, _PROCEDURES, _SURGEONS, _PATIENTS_BASE = load_all(
    DATA_DIR
)

# ============================================================
# Evaluación de una configuración
# ============================================================


def _run_single(args):
    """
    Ejecuta una unica corrida del AG con una semilla especifica.
    Pensada para poder usarse tanto secuencialmente como dentro
    de un Pool de multiprocessing.
    """
    (
        population_size,
        generations,
        tournament_size,
        crossover_rate,
        mutation_rate,
        seed,
    ) = args

    random.seed(seed)

    # Copia profunda: cada corrida necesita pacientes "frescos"
    # (scheduled=False) e independientes de otras corridas paralelas.
    patients = copy.deepcopy(_PATIENTS_BASE)

    ga = GeneticAlgorithm(
        days=DAYS,
        rooms=_ROOMS,
        specialties=_SPECIALTIES,
        surgeons=_SURGEONS,
        procedures=_PROCEDURES,
        patients=patients,
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_rate=0.10,
        stagnation_limit=30,
        alpha=1.0,
        beta=0.3,
    )

    _, best_fitness, _ = ga.run()

    generations_used = len(ga.history)

    return best_fitness, generations_used


def evaluate_configuration(
    population_size,
    generations,
    tournament_size,
    crossover_rate,
    mutation_rate,
):
    """
    Ejecuta varias veces el AG utilizando la misma
    configuración y devuelve estadísticas agregadas.
    """

    run_args = [
        (
            population_size,
            generations,
            tournament_size,
            crossover_rate,
            mutation_rate,
            BASE_SEED + rep,
        )
        for rep in range(REPETITIONS)
    ]

    results = [_run_single(args) for args in run_args]

    fitness_values = [r[0] for r in results]
    generations_used_values = [r[1] for r in results]

    return {
        "avg_fitness": mean(fitness_values),
        "std_fitness": (
            stdev(fitness_values)
            if len(fitness_values) > 1
            else 0
        ),
        "max_fitness": max(fitness_values),
        "min_fitness": min(fitness_values),
        "avg_generations_used": mean(generations_used_values),
        # score robusto: penaliza configuraciones inestables
        # (alto std) en lugar de mirar unicamente el promedio.
        "robust_score": (
            mean(fitness_values)
            - (stdev(fitness_values) if len(fitness_values) > 1 else 0)
        ),
    }


def _evaluate_combination(combo):
    """Wrapper para poder mapear combinaciones en un Pool."""
    (
        population_size,
        generations,
        tournament_size,
        crossover_rate,
        mutation_rate,
    ) = combo

    metrics = evaluate_configuration(
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
    )

    return {
        "population_size": population_size,
        "generations": generations,
        "tournament_size": tournament_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        **metrics,
    }


# ============================================================
# Main
# ============================================================


def main():

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(
        f"Dataset: {len(_PATIENTS_BASE)} pacientes, "
        f"{len(_ROOMS)} quirofanos, {len(_SPECIALTIES)} especialidades"
    )

    combinations = list(
        product(
            POPULATION_SIZES,
            GENERATIONS,
            TOURNAMENT_SIZES,
            CROSSOVER_RATES,
            MUTATION_RATES,
        )
    )

    total_runs = len(combinations) * REPETITIONS

    print(
        f"\nProbando {len(combinations)} configuraciones "
        f"x {REPETITIONS} repeticiones = {total_runs} corridas totales..."
    )

    results = []

    if USE_MULTIPROCESSING and N_WORKERS > 1:

        print(f"Usando multiprocessing con {N_WORKERS} workers.\n")

        with Pool(N_WORKERS) as pool:
            for i, result in enumerate(
                pool.imap(_evaluate_combination, combinations), start=1
            ):
                results.append(result)
                print(
                    f"[{i}/{len(combinations)}] "
                    f"pop={result['population_size']} "
                    f"gen={result['generations']} "
                    f"tour={result['tournament_size']} "
                    f"cross={result['crossover_rate']} "
                    f"mut={result['mutation_rate']} "
                    f"-> avg={result['avg_fitness']:.2f} "
                    f"std={result['std_fitness']:.2f} "
                    f"gens_usadas={result['avg_generations_used']:.1f}"
                )

    else:

        for i, combo in enumerate(combinations, start=1):
            print(
                f"[{i}/{len(combinations)}] "
                f"pop={combo[0]} gen={combo[1]} tour={combo[2]} "
                f"cross={combo[3]} mut={combo[4]}"
            )
            results.append(_evaluate_combination(combo))

    # Ordenar por score robusto (avg - std), no solo por avg_fitness.
    # Esto evita elegir como "mejor" una config que gano por varianza
    # favorable en pocas repeticiones.
    results.sort(
        key=lambda r: r["robust_score"],
        reverse=True,
    )

    # Exportar CSV
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    best = results[0]
    best_by_avg = max(results, key=lambda r: r["avg_fitness"])

    print("\n===================================")
    print("MEJOR CONFIGURACIÓN (por robust_score = avg - std)")
    print("===================================")

    for k, v in best.items():
        print(f"{k}: {v}")

    if best is not best_by_avg:
        print("\n--- Nota ---")
        print(
            "La config con mayor avg_fitness puro fue distinta:"
        )
        for k, v in best_by_avg.items():
            print(f"{k}: {v}")

    print(
        f"\nResultados exportados a:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()