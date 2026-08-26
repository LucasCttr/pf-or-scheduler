"""
main.py
Punto de entrada del sistema.

Flujo principal:
1. Carga los CSV de entrada.
2. Ejecuta el algoritmo genético para decidir la especialidad de cada bloque.
3. Construye la agenda final con el decoder.
4. Exporta la salida a agenda_resultado.json.
"""
import json
import os
import time

from data_loader import load_all
from genetic_algorithm import GeneticAlgorithm
from models import Block

DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "agenda_resultado.json")


def build_distribution_map(days, rooms, chromosome):
    """Devuelve la distribución semanal de especialidades por quirófano."""
    return {
        day: {room.id: chromosome[Block(day, room.id)] for room in rooms}
        for day in days
    }


def build_block_result(room, block, chromosome, agenda, patients_by_id):
    """Genera el JSON de detalle para un bloque concreto."""
    surgeries = agenda.assignments.get(block, [])
    return {
        "especialidad": chromosome[block],
        "minutos_utilizados": agenda.used_time.get(block, 0),
        "minutos_disponibles": room.daily_capacity_minutes,
        "cirugias": [
            {
                "paciente_id": surgery.patient_id,
                "especialidad": patients_by_id[surgery.patient_id].specialty_id,
                "duracion_min": surgery.duration,
                "hora_inicio_min": surgery.start_time,
                "hora_fin_min": surgery.end_time,
                "prioridad_clinica": patients_by_id[surgery.patient_id].clinical_priority,
            }
            for surgery in surgeries
        ],
    }


def build_result_dict(days, rooms, best_chromosome, best_fitness, best_agenda, patients, ga, execution_time_seconds):
    """Compone el JSON final con la distribución semanal y la agenda detallada."""
    patients_by_id = {patient.id: patient for patient in patients}
    distribucion = build_distribution_map(days, rooms, best_chromosome)

    agenda_detalle = {}
    for day in days:
        agenda_detalle[day] = {}
        for room in rooms:
            block = Block(day, room.id)
            agenda_detalle[day][room.id] = build_block_result(
                room, block, best_chromosome, best_agenda, patients_by_id
            )

    scheduled_ids = {surgery.patient_id for surgery in best_agenda.all_surgeries()}
    pendientes = [patient.id for patient in patients if patient.id not in scheduled_ids]

    return {
        "fitness": round(best_fitness, 4),
        "generaciones_ejecutadas": len(ga.history),
        "tiempo_ejecucion_segundos": round(execution_time_seconds, 4),
        "distribucion_semanal_especialidades": distribucion,
        "agenda": agenda_detalle,
        "resumen": {
            "total_pacientes": len(patients),
            "pacientes_programados": len(scheduled_ids),
            "pacientes_pendientes": len(pendientes),
            "ids_pacientes_pendientes": pendientes,
        },
        "historial_fitness": [round(score, 4) for score in ga.history],
    }


def main():
    specialties, rooms, procedures, surgeons, patients = load_all(DATA_DIR)

    print(
        f"Datos cargados desde CSV: {len(specialties)} especialidades, "
        f"{len(rooms)} quirófanos, {len(surgeons)} cirujanos, "
        f"{len(procedures)} procedimientos, {len(patients)} pacientes."
    )

    ga = GeneticAlgorithm(
        days=DAYS,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=patients,
        population_size=120,
        generations=150,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.05,
        stagnation_limit=30,
        alpha=1.0,
        beta=0.3,
    )

    start_time = time.time()
    best_chromosome, best_fitness, best_agenda = ga.run()
    execution_time = time.time() - start_time

    result = build_result_dict(
        DAYS,
        rooms,
        best_chromosome,
        best_fitness,
        best_agenda,
        patients,
        ga,
        execution_time,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nMejor fitness encontrado: {best_fitness:.2f}")
    print(f"Tiempo de ejecucion: {execution_time:.2f}s")
    print(
        f"Pacientes programados: {result['resumen']['pacientes_programados']} "
        f"de {result['resumen']['total_pacientes']}"
    )
    print(f"Resultado exportado a: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()