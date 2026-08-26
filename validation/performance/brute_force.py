"""
brute_force.py
Búsqueda exhaustiva del espacio de cromosomas para comparar
la solución exacta con el algoritmo genético.

Se reutiliza la misma lógica del decoder y del fitness para asegurar
que la comparación sea justa: el mismo `build_agenda` y el mismo
`GeneticAlgorithm._evaluate`, pero con un método de búsqueda distinto.

Uso:
    python brute_force.py

Si la estimación de tiempo resultara demasiado alta, conviene reducir
la cantidad de días en `DAYS` antes de lanzar la búsqueda completa.
"""
import itertools
import os
import random
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)

from data_loader import load_all
from models import Block
from genetic_algorithm import GeneticAlgorithm

DATA_DIR = os.path.join(BASE_DIR, "data")

# Semilla fija para que la corrida del GA sea reproducible y comparable
# entre distintas ejecuciones de este script.
SEED = 42

# --- Escenario reducido para la comparacion exacta ---
# DAYS = ["lunes", "martes", "miercoles", "jueves"]
DAYS = ["lunes", "martes", "miercoles"]

# Umbral a partir del cual se avisa que puede tardar demasiado.
WARNING_THRESHOLD = 20_000_000


def filter_patients_for_days(patients, surgeons_by_id, days):
    """
    Filtra pacientes cuyo cirujano tiene disponibilidad en al menos uno
    de los dias del escenario reducido. No es estrictamente necesario
    (el decoder ya descarta lo que no aplica), pero evita que build_agenda
    recorra en cada bloque pacientes que nunca podrian asignarse aqui.
    """
    days_set = set(days)
    filtered = []
    for p in patients:
        surgeon = surgeons_by_id.get(p.surgeon_id)
        if surgeon and surgeon.available_days & days_set:
            filtered.append(p)
    return filtered


def main():
    specialties, rooms, procedures, surgeons, patients = load_all(DATA_DIR)
    surgeons_by_id = {s.id: s for s in surgeons}

    reduced_patients = filter_patients_for_days(patients, surgeons_by_id, DAYS)

    print(f"Pacientes totales: {len(patients)} | "
          f"Pacientes relevantes para {DAYS}: {len(reduced_patients)}")

    blocks = [Block(d, r.id) for d in DAYS for r in rooms]
    specialty_ids = [s.id for s in specialties]

    total_combinations = len(specialty_ids) ** len(blocks)
    print(f"Bloques totales: {len(blocks)}  |  Especialidades: {len(specialty_ids)}")
    print(f"Combinaciones a evaluar: {total_combinations:,}")

    if total_combinations > WARNING_THRESHOLD:
        print("\n[ADVERTENCIA] Mas de 20 millones de combinaciones.")


    # Se instancia SOLO para reusar la normalizacion y _evaluate.
    # No se llama a .run(), asi que no ejecuta ningun GA todavia.
    ga_helper = GeneticAlgorithm(
        days=DAYS,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=reduced_patients,
        alpha=1.0,
        beta=0.3,
    )

    best_fitness = float("-inf")
    best_chromosome = None
    best_agenda = None

    t0 = time.time()
    evaluated = 0

    for combo in itertools.product(specialty_ids, repeat=len(blocks)):
        chromosome = dict(zip(blocks, combo))

        fitness, agenda = ga_helper._evaluate(chromosome)

        if fitness > best_fitness:
            best_fitness = fitness
            best_chromosome = chromosome
            best_agenda = agenda

        evaluated += 1
        if evaluated % 200_000 == 0:
            elapsed = time.time() - t0
            rate = evaluated / elapsed
            remaining = (total_combinations - evaluated) / rate
            print(f"  {evaluated:,}/{total_combinations:,} evaluadas | "
                  f"{elapsed:.1f}s transcurridos | "
                  f"ETA restante: {remaining/60:.1f} min")

    tiempo_bf = time.time() - t0

    bf_surgeries = best_agenda.all_surgeries()
    bf_scheduled_ids = {s.patient_id for s in bf_surgeries}
    patients_by_id = {p.id: p for p in reduced_patients}

    bf_avg_priority = (
        sum(patients_by_id[pid].clinical_priority for pid in bf_scheduled_ids) / len(bf_scheduled_ids)
        if bf_scheduled_ids else 0.0
    )

    print("\n" + "=" * 60)
    print("RESULTADO FUERZA BRUTA (optimo exacto)")
    print("=" * 60)
    print(f"Combinaciones evaluadas: {evaluated:,}")
    print(f"Tiempo total: {tiempo_bf:.2f}s ({tiempo_bf/60:.2f} min)")
    print(f"Mejor fitness: {best_fitness:.4f}")
    print(f"Pacientes programados: {len(bf_scheduled_ids)} de {len(reduced_patients)}")
    print(f"Prioridad promedio asignados: {bf_avg_priority:.2f}")

    # --- GA sobre EXACTAMENTE el mismo escenario reducido ---
    print(f"\nCorriendo GA (seed={SEED}) sobre el mismo escenario reducido...")
    random.seed(SEED)
    ga = GeneticAlgorithm(
        days=DAYS,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=reduced_patients,
        population_size=80,
        generations=200,
        tournament_size=3,
        crossover_rate=0.85,
        mutation_rate=0.04,
        stagnation_limit=30,
        alpha=1.0,
        beta=0.3,
    )

    t0_ga = time.time()
    ga_chromosome, ga_fitness, ga_agenda = ga.run()
    tiempo_ga = time.time() - t0_ga

    ga_surgeries = ga_agenda.all_surgeries()
    ga_scheduled_ids = {s.patient_id for s in ga_surgeries}
    ga_avg_priority = (
        sum(patients_by_id[pid].clinical_priority for pid in ga_scheduled_ids) / len(ga_scheduled_ids)
        if ga_scheduled_ids else 0.0
    )

    gap = ((best_fitness - ga_fitness) / best_fitness * 100) if best_fitness != 0 else 0.0

    print("\n" + "=" * 60)
    print("COMPARACION GA vs FUERZA BRUTA (mismo escenario)")
    print("=" * 60)
    print(f"{'Metrica':<30}{'Fuerza Bruta':<20}{'GA':<20}")
    print(f"{'Fitness':<30}{best_fitness:<20.4f}{ga_fitness:<20.4f}")
    print(f"{'Pacientes programados':<30}{len(bf_scheduled_ids):<20}{len(ga_scheduled_ids):<20}")
    print(f"{'Prioridad promedio':<30}{bf_avg_priority:<20.2f}{ga_avg_priority:<20.2f}")
    print(f"{'Tiempo (s)':<30}{tiempo_bf:<20.2f}{tiempo_ga:<20.2f}")
    print(f"\nGap del GA respecto al optimo exacto: {gap:.2f}%")


if __name__ == "__main__":
    main()