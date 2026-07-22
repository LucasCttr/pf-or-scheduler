"""
parameter_tuning.py

Grid Search para encontrar una buena configuración de parámetros
del Algoritmo Genético.
"""

import copy
import csv
import os
import random
import sys
from collections import Counter
from itertools import product
from multiprocessing import Pool, cpu_count
from statistics import mean

# ============================================================
# Configuración de rutas
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "grid_search_results.csv")

# ============================================================
# Imports del proyecto
# ============================================================
from data_loader import load_all
from genetic_algorithm import GeneticAlgorithm

# ============================================================
# Configuración general
# ============================================================
DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes"]

POPULATION_SIZES = [80, 120]
GENERATIONS = [150]
TOURNAMENT_SIZES = [2, 3]
CROSSOVER_RATES = [0.70, 0.80, 0.90]
MUTATION_RATES = [0.02, 0.05]
REPETITIONS = 15     

BASE_SEED = 123
USE_MULTIPROCESSING = True
N_WORKERS = max(1, cpu_count() - 1)

# Carga única de datos
_SPECIALTIES, _ROOMS, _PROCEDURES, _SURGEONS, _PATIENTS_BASE = load_all(DATA_DIR)
_SPECIALTY_IDS = [s.id for s in _SPECIALTIES]


def _count_patients_by_specialty(agenda, patients):
    """
    Cuenta cuántos pacientes fueron efectivamente programados en la
    agenda final, agrupados por especialidad.
    """
    patient_specialty = {p.id: p.specialty_id for p in patients}

    counts = Counter(
        patient_specialty[surgery.patient_id]
        for surgery in agenda.all_surgeries()
    )

    return tuple(counts.get(sid, 0) for sid in _SPECIALTY_IDS)


def _run_single(args):
    """Ejecuta una única corrida del AG."""
    (population_size, generations, tournament_size, crossover_rate, mutation_rate, seed) = args
    random.seed(seed)
    patients = copy.deepcopy(_PATIENTS_BASE)

    ga = GeneticAlgorithm(
        days=DAYS, rooms=_ROOMS, specialties=_SPECIALTIES,
        surgeons=_SURGEONS, procedures=_PROCEDURES, patients=patients,
        population_size=population_size, generations=generations,
        tournament_size=tournament_size, crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        stagnation_limit=30, alpha=1.0, beta=0.3,
    )

    best_chromosome, best_fitness, best_agenda = ga.run()
    specialty_vector = _count_patients_by_specialty(best_agenda, patients)

    return best_fitness, len(ga.history), specialty_vector

def evaluate_configuration(population_size, generations, tournament_size, crossover_rate, mutation_rate):
    """Calcula métricas agregadas (sin std)."""
    run_args = [(population_size, generations, tournament_size, crossover_rate, mutation_rate, BASE_SEED + rep)
                for rep in range(REPETITIONS)]

    results = [_run_single(args) for args in run_args]
    fitness_values = [r[0] for r in results]
    gens_used = [r[1] for r in results]
    vectors = [r[2] for r in results]

    # Vector de pacientes por especialidad de la corrida con mejor fitness
    best_index = fitness_values.index(max(fitness_values))
    best_vector = vectors[best_index]

    metrics = {
        "avg_fitness": mean(fitness_values),
        "max_fitness": max(fitness_values),
        "min_fitness": min(fitness_values),
        "avg_generations_used": mean(gens_used),
    }

    for sid, count in zip(_SPECIALTY_IDS, best_vector):
        metrics[f"assigned_{sid}"] = count

    return metrics

def _evaluate_combination(combo):
    """Wrapper para multiprocessing."""
    metrics = evaluate_configuration(*combo)
    return {
        "population_size": combo[0], "generations": combo[1],
        "tournament_size": combo[2], "crossover_rate": combo[3],
        "mutation_rate": combo[4], **metrics
    }

def main():
    combinations = list(product(POPULATION_SIZES, GENERATIONS, TOURNAMENT_SIZES, CROSSOVER_RATES, MUTATION_RATES))
    
    print(f"Probando {len(combinations)} configuraciones x {REPETITIONS} repeticiones...")
    results = []

    if USE_MULTIPROCESSING and N_WORKERS > 1:
        with Pool(N_WORKERS) as pool:
            for i, result in enumerate(pool.imap(_evaluate_combination, combinations), start=1):
                results.append(result)
                print(f"[{i}/{len(combinations)}] Config evaluada -> avg={result['avg_fitness']:.4f}")
    else:
        for i, combo in enumerate(combinations, start=1):
            results.append(_evaluate_combination(combo))
            print(f"[{i}/{len(combinations)}] Config evaluada -> avg={results[-1]['avg_fitness']:.4f}")

    # Ordenar por fitness promedio
    results.sort(key=lambda r: r["avg_fitness"], reverse=True)

    # Exportar CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n===================================")
    print("MEJOR CONFIGURACIÓN (por avg_fitness)")
    print("===================================")
    for k, v in results[0].items():
        print(f"{k}: {v}")
    print(f"\nResultados guardados en: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()