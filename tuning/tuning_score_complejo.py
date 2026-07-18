"""
diagnostico_convergencia.py

Script de diagnostico para verificar la hipotesis de que el AG converge
casi siempre a la misma distribucion de bloques por especialidad,
independientemente de la configuracion de hiperparametros o la semilla.

Colocar este archivo en la misma carpeta que parameter_tuning.py
(por ejemplo dentro de la carpeta tuning_ag), ya que reutiliza la
misma logica de rutas e imports.

Que hace:
  1. Corre el AG N veces con semillas distintas (y opcionalmente
     configuraciones de hiperparametros distintas).
  2. Para cada corrida, extrae el vector {specialty_id: cantidad_de_bloques}
     del mejor cromosoma encontrado.
  3. Compara esos vectores entre corridas: si son casi identicos pese a
     usar semillas/configs distintas, confirma que el espacio de
     soluciones "efectivo" es mucho mas chico que el nominal, y que el
     decoder (por su logica de asignacion greedy por prioridad) domina
     el resultado por sobre la busqueda genetica.
  4. Reporta tambien la variacion del fitness para contrastar.
"""

import copy
import os
import random
import sys
from collections import Counter
from statistics import mean, pstdev

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Si el script esta en la raiz del proyecto (existe carpeta "data" al lado),
# usamos esa carpeta directamente. Si esta en una subcarpeta (ej. tuning_ag),
# subimos un nivel para llegar a la raiz del proyecto.
if os.path.isdir(os.path.join(_SCRIPT_DIR, "data")):
    BASE_DIR = _SCRIPT_DIR
else:
    BASE_DIR = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

from data_loader import load_all
from genetic_algorithm import GeneticAlgorithm

DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes"]

# ============================================================
# Configuracion del diagnostico
# ============================================================

N_RUNS = 12          # cantidad de corridas independientes a comparar
BASE_SEED = 500

# Configuraciones fijas "razonables" (podes variar estos valores entre
# corridas si queres ademas comparar el efecto de la config, no solo
# de la semilla)
RUN_CONFIGS = [
    dict(population_size=80, generations=150, tournament_size=3,
         crossover_rate=0.8, mutation_rate=0.02, elitism_rate=0.10,
         stagnation_limit=30, alpha=1.0, beta=0.3),
    dict(population_size=120, generations=150, tournament_size=2,
         crossover_rate=0.7, mutation_rate=0.05, elitism_rate=0.10,
         stagnation_limit=30, alpha=1.0, beta=0.3),
]


def specialty_block_vector(chromosome, specialty_ids):
    """Cuenta cuantos bloques quedaron asignados a cada especialidad."""
    counts = Counter(chromosome.values())
    return tuple(counts.get(sid, 0) for sid in specialty_ids)


def vector_distance(v1, v2):
    """Distancia L1 simple entre dos vectores de conteo por especialidad."""
    return sum(abs(a - b) for a, b in zip(v1, v2))


def main():
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}\n")

    specialties, rooms, procedures, surgeons, patients_base = load_all(DATA_DIR)
    specialty_ids = [s.id for s in specialties]

    print(f"Dataset: {len(patients_base)} pacientes, {len(rooms)} quirofanos, "
          f"{len(specialties)} especialidades\n")
    print(f"Corriendo {N_RUNS} ejecuciones independientes "
          f"(alternando entre {len(RUN_CONFIGS)} configs distintas)...\n")

    fitness_values = []
    vectors = []
    run_info = []

    for i in range(N_RUNS):
        seed = BASE_SEED + i
        cfg = RUN_CONFIGS[i % len(RUN_CONFIGS)]

        random.seed(seed)
        patients = copy.deepcopy(patients_base)

        ga = GeneticAlgorithm(
            days=DAYS,
            rooms=rooms,
            specialties=specialties,
            surgeons=surgeons,
            procedures=procedures,
            patients=patients,
            **cfg,
        )

        best_chromosome, best_fitness, best_agenda = ga.run()

        vec = specialty_block_vector(best_chromosome, specialty_ids)
        fitness_values.append(best_fitness)
        vectors.append(vec)
        run_info.append((seed, cfg, vec, best_fitness))

        print(f"[{i+1:>2}/{N_RUNS}] seed={seed} pop={cfg['population_size']} "
              f"tour={cfg['tournament_size']} cross={cfg['crossover_rate']} "
              f"mut={cfg['mutation_rate']} -> fitness={best_fitness:.4f} "
              f"vector={vec}")

    # ============================================================
    # Analisis de resultados
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALISIS DE CONVERGENCIA ESTRUCTURAL")
    print("=" * 70)

    print(f"\nEspecialidades (orden del vector): {specialty_ids}")

    # Fitness reportado solo con media, min y max
    print(f"\nFitness: mean={mean(fitness_values):.4f}  "
          f"min={min(fitness_values):.4f}  max={max(fitness_values):.4f}")

    unique_vectors = set(vectors)
    print(f"\nVectores de bloques-por-especialidad unicos encontrados: "
          f"{len(unique_vectors)} de {N_RUNS} corridas")

    if len(unique_vectors) == 1:
        print(">> TODAS las corridas convergieron EXACTAMENTE a la misma "
              "distribucion de bloques por especialidad.")
        print(">> Esto confirma que el decoder (asignacion greedy por "
              "prioridad clinica) domina el resultado, y que el espacio "
              "de soluciones efectivo es mucho mas chico que el nominal.")
    else:
        # Distancia promedio entre todos los pares de vectores
        distances = []
        for a in range(len(vectors)):
            for b in range(a + 1, len(vectors)):
                distances.append(vector_distance(vectors[a], vectors[b]))
        total_blocks = len(DAYS) * len(rooms)
        print(f">> Distancia L1 promedio entre vectores de distintas "
              f"corridas: {mean(distances):.2f} "
              f"(sobre un total de {total_blocks} bloques)")
        print(f">> Distancia L1 maxima observada: {max(distances)}")
        if mean(distances) < 0.05 * total_blocks:
            print(">> La distancia promedio es muy baja en relacion al "
                  "total de bloques: las corridas convergen a "
                  "distribuciones CASI identicas pese a variar semilla "
                  "y configuracion.")
        else:
            print(">> Hay variacion estructural real entre corridas: "
                  "el resultado NO depende solo del decoder.")

    print("\nDetalle de vectores por corrida:")
    for seed, cfg, vec, fit in run_info:
        print(f"  seed={seed:<5} pop={cfg['population_size']:<4} "
              f"tour={cfg['tournament_size']} -> fitness={fit:.4f}  "
              f"vector={vec}")  


if __name__ == "__main__":
    main()