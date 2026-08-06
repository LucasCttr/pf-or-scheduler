
"""
run_validation.py

Ejecuta distintos escenarios de validación sobre el mismo dataset base.

Escenarios:
1. Base
2. Alta demanda
3. Baja demanda
4. Cirujanos limitados
5. Cirugías largas
6. Prioridades variadas

Los datos originales NO se modifican.
Los escenarios se generan en memoria.

Cada escenario:
    datos -> GA -> agenda -> validation -> métricas

Al finalizar:
    - muestra una tabla comparativa
    - genera validation_results.csv
"""

import copy
import csv
import os
import random

from data_loader import load_all
from genetic_algorithm import GeneticAlgorithm
from validation import validate_agenda


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
    os.path.dirname(__file__),
    "data"
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "validation_results.csv"
)

# Semilla para que los escenarios sean reproducibles
RANDOM_SEED = 42


# ============================================================
# CONFIGURACIÓN DEL ALGORITMO
# ============================================================

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

def clone_data(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Genera copias independientes de los datos.

    Esto evita que un escenario modifique
    accidentalmente el siguiente.
    """

    return (
        copy.deepcopy(specialties),
        copy.deepcopy(rooms),
        copy.deepcopy(procedures),
        copy.deepcopy(surgeons),
        copy.deepcopy(patients),
    )


def run_algorithm(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Ejecuta el algoritmo genético y devuelve
    el cromosoma, fitness y agenda resultantes.
    """

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

    return (
        best_chromosome,
        best_fitness,
        best_agenda,
    )


# ============================================================
# ESCENARIO 1 - BASE
# ============================================================

def scenario_base(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Dataset original sin modificaciones.
    """

    return clone_data(
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )


# ============================================================
# ESCENARIO 2 - ALTA DEMANDA
# ============================================================

def scenario_high_demand(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Aumenta la demanda utilizando todos los pacientes
    disponibles y duplicando el conjunto mediante copias.

    Esto permite evaluar el comportamiento del algoritmo
    cuando existe una demanda superior a la capacidad.
    """

    (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    ) = clone_data(
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )

    original_patients = copy.deepcopy(patients)

    # Crear pacientes adicionales con nuevos IDs
    existing_ids = {
        p.id
        for p in patients
    }

    next_id = 1

    for patient in original_patients:

        while str(next_id) in existing_ids:
            next_id += 1

        new_patient = copy.deepcopy(patient)

        new_patient.id = str(next_id)

        patients.append(new_patient)

        existing_ids.add(new_patient.id)

        next_id += 1

    return (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )


# ============================================================
# ESCENARIO 3 - BAJA DEMANDA
# ============================================================

def scenario_low_demand(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Reduce la cantidad de pacientes al 20%
    del dataset original.
    """

    (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    ) = clone_data(
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )

    random.seed(RANDOM_SEED)

    random.shuffle(patients)

    amount = max(
        1,
        int(len(patients) * 0.2)
    )

    patients = patients[:amount]

    return (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )


# ============================================================
# ESCENARIO 4 - CIRUJANOS LIMITADOS
# ============================================================

def scenario_limited_surgeons(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Reduce la disponibilidad de los cirujanos.

    Se mantienen los cirujanos existentes,
    pero se reduce su cantidad de días disponibles.

    Esto permite evaluar cómo responde el algoritmo
    ante una restricción mayor de disponibilidad.
    """

    (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    ) = clone_data(
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )

    for surgeon in surgeons:

        if len(surgeon.available_days) > 1:

            surgeon.available_days = {
                next(iter(surgeon.available_days))
            }

    return (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )


# ============================================================
# ESCENARIO 5 - CIRUGÍAS LARGAS
# ============================================================

def scenario_long_surgeries(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Aumenta la duración de los procedimientos.

    Se incrementa un 50%, manteniendo múltiplos
    de 15 minutos.
    """

    (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    ) = clone_data(
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )

    for procedure in procedures:

        new_duration = int(
            procedure.estimated_duration * 1.5
        )

        # Redondear al múltiplo de 15 superior
        new_duration = (
            ((new_duration + 14) // 15) * 15
        )

        procedure.estimated_duration = new_duration

    return (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )


# ============================================================
# ESCENARIO 6 - PRIORIDADES VARIADAS
# ============================================================

def scenario_varied_priorities(
    specialties,
    rooms,
    procedures,
    surgeons,
    patients,
):
    """
    Genera una distribución más variada de prioridades.

    Se asignan prioridades entre 1 y 10.
    """

    (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    ) = clone_data(
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )

    random.seed(RANDOM_SEED)

    for patient in patients:

        patient.clinical_priority = random.randint(
            1,
            10
        )

    return (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )


# ============================================================
# DEFINICIÓN DE ESCENARIOS
# ============================================================

SCENARIOS = [
    (
        "Base",
        scenario_base,
    ),
    (
        "Alta demanda",
        scenario_high_demand,
    ),
    (
        "Baja demanda",
        scenario_low_demand,
    ),
    (
        "Cirujanos limitados",
        scenario_limited_surgeons,
    ),
    (
        "Cirugías largas",
        scenario_long_surgeries,
    ),
    (
        "Prioridades variadas",
        scenario_varied_priorities,
    ),
]


# ============================================================
# EJECUTAR ESCENARIO
# ============================================================

def execute_scenario(
    scenario_name,
    scenario_function,
    base_data,
):

    print("\n")
    print("=" * 70)
    print(f"ESCENARIO: {scenario_name}")
    print("=" * 70)

    (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    ) = scenario_function(*base_data)

    print(
        f"Pacientes: {len(patients)}"
    )

    print(
        f"Cirujanos: {len(surgeons)}"
    )

    print(
        f"Quirófanos: {len(rooms)}"
    )

    # --------------------------------------------------------
    # Ejecutar GA
    # --------------------------------------------------------

    (
        best_chromosome,
        best_fitness,
        best_agenda,
    ) = run_algorithm(
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    )

    # --------------------------------------------------------
    # Validar agenda
    # --------------------------------------------------------

    validation = validate_agenda(
        agenda=best_agenda,
        patients=patients,
        procedures=procedures,
        surgeons=surgeons,
        rooms=rooms,
    )

    validation.print_report()

    # --------------------------------------------------------
    # Extraer métricas
    # --------------------------------------------------------

    metrics = validation.metrics

    result = {
        "escenario": scenario_name,
        "pacientes": len(patients),
        "cirujanos": len(surgeons),
        "quirofanos": len(rooms),
        "fitness": round(
            best_fitness,
            4
        ),
        "cirugias_asignadas": metrics.get(
            "Cirugías asignadas",
            0
        ),
        "cumplimiento_duracion": metrics.get(
            "Cumplimiento de duración (%)",
            0
        ),
        "bloques_capacidad": metrics.get(
            "Bloques dentro de capacidad (%)",
            0
        ),
        "prioridad_asignados": metrics.get(
            "Prioridad promedio asignados",
            0
        ),
        "prioridad_no_asignados": metrics.get(
            "Prioridad promedio no asignados",
            0
        ),
        "utilizacion": metrics.get(
            "Utilización global (%)",
            0
        ),
        "minutos_utilizados": metrics.get(
            "Minutos utilizados",
            0
        ),
        "minutos_disponibles": metrics.get(
            "Minutos disponibles",
            0
        ),
        "desviacion_carga_horas": metrics.get(
            "Desviación estándar carga (horas)",
            0
        ),
        "carga_minima_horas": metrics.get(
            "Carga mínima (horas)",
            0
        ),
        "carga_maxima_horas": metrics.get(
            "Carga máxima (horas)",
            0
        ),
        "violaciones": len(
            validation.violations
        ),
    }

    return result


# ============================================================
# EXPORTAR CSV
# ============================================================

def save_results(results):

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\nResultados guardados en:"
        f"\n{OUTPUT_PATH}"
    )


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

    (
        specialties,
        rooms,
        procedures,
        surgeons,
        patients,
    ) = base_data

    print(
        f"Dataset base: "
        f"{len(patients)} pacientes, "
        f"{len(surgeons)} cirujanos, "
        f"{len(rooms)} quirófanos."
    )

    results = []

    # --------------------------------------------------------
    # Ejecutar todos los escenarios
    # --------------------------------------------------------

    for (
        scenario_name,
        scenario_function,
    ) in SCENARIOS:

        result = execute_scenario(
            scenario_name,
            scenario_function,
            base_data,
        )

        results.append(result)

    # --------------------------------------------------------
    # Mostrar resumen
    # --------------------------------------------------------

    print_summary(results)

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    save_results(results)


if __name__ == "__main__":
    main()

