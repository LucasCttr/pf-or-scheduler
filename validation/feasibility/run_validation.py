"""
run_validation.py

Ejecuta varios escenarios de validación sobre el mismo conjunto base de datos.

Escenarios cubiertos:
1. Base
2. Alta demanda
3. Baja demanda
4. Cirujanos limitados
5. Cirugías largas
6. Cirugías cortas
7. Prioridades variadas

Los datos originales no se modifican; cada escenario se genera en memoria.
Cada corrida sigue este flujo:
    datos -> algoritmo genético -> agenda -> validación -> métricas

Al terminar, el script:
    - imprime una comparación entre escenarios,
    - exporta validation_results.csv,
    - exporta validation_priority_by_specialty.csv.
"""

import copy
import csv
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FEASIBILITY_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, FEASIBILITY_DIR)

from data_loader import load_all
from genetic_algorithm import GeneticAlgorithm
from validation import validate_agenda, print_priority_by_specialty


# ============================================================
# CONFIGURACIÓN
# ============================================================

DAYS = [
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
]

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "validation_results.csv"
)

PRIORITY_BY_SPECIALTY_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "validation_priority_by_specialty.csv"
)

RANDOM_SEED = 42


GA_CONFIG = {
    "population_size": 80,
    "generations": 200,
    "tournament_size": 3,
    "crossover_rate": 0.85,
    "mutation_rate": 0.04,
    "stagnation_limit": 30,
    "alpha": 1.0,
    "beta": 0.3,
}


# ============================================================
# UTILIDADES
# ============================================================

def clone_data(specialties, rooms, procedures, surgeons, patients):
    return (
        copy.deepcopy(specialties),
        copy.deepcopy(rooms),
        copy.deepcopy(procedures),
        copy.deepcopy(surgeons),
        copy.deepcopy(patients),
    )


def run_algorithm(specialties, rooms, procedures, surgeons, patients):
    # Semilla fija justo antes de correr el GA: asi el algoritmo genetico
    # (que usa random internamente para poblacion inicial, cruce y
    # mutacion) da resultados reproducibles cada vez que se corre el
    # script completo, en vez de variar entre ejecuciones.
    random.seed(RANDOM_SEED)

    ga = GeneticAlgorithm(
        days=DAYS,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=patients,
        **GA_CONFIG,
    )
    best_chromosome, best_fitness, best_agenda = ga.run()
    return best_chromosome, best_fitness, best_agenda


# ============================================================
# ESCENARIOS (sin cambios respecto a la version original)
# ============================================================

def scenario_base(specialties, rooms, procedures, surgeons, patients):
    return clone_data(specialties, rooms, procedures, surgeons, patients)


def scenario_high_demand(specialties, rooms, procedures, surgeons, patients):
    specialties, rooms, procedures, surgeons, patients = clone_data(
        specialties, rooms, procedures, surgeons, patients
    )
    original_patients = copy.deepcopy(patients)
    existing_ids = {p.id for p in patients}
    next_id = 1
    for patient in original_patients:
        while str(next_id) in existing_ids:
            next_id += 1
        new_patient = copy.deepcopy(patient)
        new_patient.id = str(next_id)
        patients.append(new_patient)
        existing_ids.add(new_patient.id)
        next_id += 1
    return specialties, rooms, procedures, surgeons, patients


def scenario_low_demand(specialties, rooms, procedures, surgeons, patients):
    specialties, rooms, procedures, surgeons, patients = clone_data(
        specialties, rooms, procedures, surgeons, patients
    )
    random.seed(RANDOM_SEED)
    random.shuffle(patients)
    amount = max(1, int(len(patients) * 0.33))
    patients = patients[:amount]
    return specialties, rooms, procedures, surgeons, patients


def scenario_limited_surgeons(specialties, rooms, procedures, surgeons, patients):
    specialties, rooms, procedures, surgeons, patients = clone_data(
        specialties, rooms, procedures, surgeons, patients
    )
    for surgeon in surgeons:
        if len(surgeon.available_days) > 1:
            surgeon.available_days = {next(iter(surgeon.available_days))}
    return specialties, rooms, procedures, surgeons, patients


def scenario_long_surgeries(specialties, rooms, procedures, surgeons, patients):
    specialties, rooms, procedures, surgeons, patients = clone_data(
        specialties, rooms, procedures, surgeons, patients
    )
    for procedure in procedures:
        new_duration = int(procedure.estimated_duration * 1.5)
        new_duration = ((new_duration + 14) // 15) * 15
        procedure.estimated_duration = new_duration
    return specialties, rooms, procedures, surgeons, patients


def scenario_short_surgeries(specialties, rooms, procedures, surgeons, patients):
    """
    Reduce la duracion de los procedimientos a la mitad, manteniendo
    multiplos de 15 minutos y un piso minimo de 15 minutos (para evitar
    procedimientos de duracion 0 o negativa).

    Permite evaluar el comportamiento del algoritmo ante intervenciones
    mas cortas, en contraste con el escenario de cirugias largas.
    """
    specialties, rooms, procedures, surgeons, patients = clone_data(
        specialties, rooms, procedures, surgeons, patients
    )
    for procedure in procedures:
        new_duration = int(procedure.estimated_duration * 0.5)
        new_duration = ((new_duration + 14) // 15) * 15
        new_duration = max(new_duration, 15)
        procedure.estimated_duration = new_duration
    return specialties, rooms, procedures, surgeons, patients


def scenario_varied_priorities(specialties, rooms, procedures, surgeons, patients):
    specialties, rooms, procedures, surgeons, patients = clone_data(
        specialties, rooms, procedures, surgeons, patients
    )
    random.seed(RANDOM_SEED)
    for patient in patients:
        patient.clinical_priority = random.randint(1, 10)
    return specialties, rooms, procedures, surgeons, patients


SCENARIOS = [
    ("Base", scenario_base),
    ("Alta demanda", scenario_high_demand),
    ("Baja demanda", scenario_low_demand),
    ("Cirujanos limitados", scenario_limited_surgeons),
    ("Cirugías largas", scenario_long_surgeries),
    ("Cirugías cortas", scenario_short_surgeries),
    ("Prioridades variadas", scenario_varied_priorities),
]


# ============================================================
# EJECUTAR ESCENARIO
# ============================================================

def execute_scenario(scenario_name, scenario_function, base_data):

    print("\n")
    print("=" * 70)
    print(f"ESCENARIO: {scenario_name}")
    print("=" * 70)

    specialties, rooms, procedures, surgeons, patients = scenario_function(*base_data)

    print(f"Pacientes: {len(patients)}")
    print(f"Cirujanos: {len(surgeons)}")
    print(f"Quirófanos: {len(rooms)}")

    best_chromosome, best_fitness, best_agenda = run_algorithm(
        specialties, rooms, procedures, surgeons, patients
    )

    validation = validate_agenda(
        agenda=best_agenda,
        patients=patients,
        procedures=procedures,
        surgeons=surgeons,
        rooms=rooms,
        specialties=specialties,
        chromosome=best_chromosome,
    )

    validation.print_report()

    # --------------------------------------------------------
    # Desglose de prioridad por especialidad (impresion en consola)
    # --------------------------------------------------------

    priority_breakdown = validation.metrics.get("Prioridad por especialidad", {})

    if priority_breakdown:
        print_priority_by_specialty(priority_breakdown)

    # --------------------------------------------------------
    # Extraer métricas para la tabla resumen (formato ancho, 1 fila)
    # --------------------------------------------------------

    metrics = validation.metrics

    result = {
        "escenario": scenario_name,
        "pacientes": len(patients),
        "cirujanos": len(surgeons),
        "quirofanos": len(rooms),
        "fitness": round(best_fitness, 4),
        "cirugias_asignadas": metrics.get("Cirugías asignadas", 0),
        "cumplimiento_duracion": metrics.get("Cumplimiento de duración (%)", 0),
        "bloques_capacidad": metrics.get("Bloques dentro de capacidad (%)", 0),
        "especialidades_minimo_cumplido": metrics.get("Especialidades con minimo cumplido", 0),
        "especialidades_totales": metrics.get("Especialidades totales", 0),
        "prioridad_asignados": metrics.get("Prioridad promedio asignados", 0),
        "prioridad_no_asignados": metrics.get("Prioridad promedio no asignados", 0),
        "utilizacion": metrics.get("Utilización global (%)", 0),
        "minutos_utilizados": metrics.get("Minutos utilizados", 0),
        "minutos_disponibles": metrics.get("Minutos disponibles", 0),
        "desviacion_carga_horas": metrics.get("Desviación estándar carga (horas)", 0),
        "carga_minima_horas": metrics.get("Carga mínima (horas)", 0),
        "carga_maxima_horas": metrics.get("Carga máxima (horas)", 0),
        "solapamientos_cirujanos": metrics.get("Solapamientos de cirujanos", 0),
        "violaciones": len(validation.violations),
    }

    # --------------------------------------------------------
    # Filas del desglose por especialidad (formato largo, 1 fila
    # por especialidad, para exportar a un CSV separado)
    # --------------------------------------------------------

    priority_rows = []
    for specialty_id, datos in priority_breakdown.items():
        priority_rows.append({
            "escenario": scenario_name,
            "especialidad": specialty_id,
            "prioridad_asignados": datos["prioridad_asignados"],
            "prioridad_no_asignados": datos["prioridad_no_asignados"],
            "n_asignados": datos["n_asignados"],
            "n_no_asignados": datos["n_no_asignados"],
        })

    return result, priority_rows


# ============================================================
# EXPORTAR CSVs
# ============================================================

def save_results(results):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResultados guardados en:\n{OUTPUT_PATH}")


def save_priority_by_specialty(priority_rows):
    if not priority_rows:
        return
    fieldnames = list(priority_rows[0].keys())
    with open(PRIORITY_BY_SPECIALTY_OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(priority_rows)
    print(f"Desglose por especialidad guardado en:\n{PRIORITY_BY_SPECIALTY_OUTPUT_PATH}")


# ============================================================
# TABLA RESUMEN
# ============================================================

def print_summary(results):
    print("\n")
    print("=" * 120)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 120)

    header = (
        f"{'Escenario':<25}"
        f"{'Pacientes':>10}"
        f"{'Asignadas':>11}"
        f"{'Utilización':>13}"
        f"{'Prior. Asig.':>14}"
        f"{'Prior. No Asig.':>16}"
        f"{'σ carga':>10}"
        f"{'Violaciones':>13}"
    )
    print(header)
    print("-" * 120)

    for result in results:
        print(
            f"{result['escenario']:<25}"
            f"{result['pacientes']:>10}"
            f"{result['cirugias_asignadas']:>11}"
            f"{result['utilizacion']:>12.2f}%"
            f"{result['prioridad_asignados']:>14.2f}"
            f"{result['prioridad_no_asignados']:>16.2f}"
            f"{result['desviacion_carga_horas']:>10.2f}"
            f"{result['violaciones']:>13}"
        )

    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("VALIDACIÓN EXPERIMENTAL DEL ALGORITMO")
    print("=" * 70)

    print("\nCargando dataset base...")
    base_data = load_all(DATA_DIR)
    specialties, rooms, procedures, surgeons, patients = base_data

    print(
        f"Dataset base: {len(patients)} pacientes, "
        f"{len(surgeons)} cirujanos, {len(rooms)} quirófanos."
    )

    results = []
    all_priority_rows = []

    for scenario_name, scenario_function in SCENARIOS:
        result, priority_rows = execute_scenario(scenario_name, scenario_function, base_data)
        results.append(result)
        all_priority_rows.extend(priority_rows)

    print_summary(results)
    save_results(results)
    save_priority_by_specialty(all_priority_rows)


if __name__ == "__main__":
    main()