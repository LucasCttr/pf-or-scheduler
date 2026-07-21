"""
main.py
Punto de entrada del sistema.

1. Carga los datos de entrada (especialidades, quirofanos, procedimientos,
   cirujanos y pacientes) desde archivos CSV ubicados en la carpeta `data/`.
2. Ejecuta el Algoritmo Genetico para obtener la mejor distribucion semanal
   de especialidades sobre los quirofanos.
3. Construye la agenda final mediante el decoder.
4. Exporta el resultado completo a un archivo JSON.
"""
import json
import os

from data_loader import load_all
from models import Block
from genetic_algorithm import GeneticAlgorithm

DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "agenda_resultado.json")


def build_result_dict(days, rooms, best_chromosome, best_fitness, best_agenda,
                       patients, ga):
    patients_by_id = {p.id: p for p in patients}

    # Distribucion semanal de especialidades (cromosoma) en formato legible
    distribucion = {
        day: {room.id: best_chromosome[Block(day, room.id)] for room in rooms}
        for day in days
    }

    # Agenda detallada por dia / quirofano
    agenda_detalle = {}
    for day in days:
        agenda_detalle[day] = {}
        for room in rooms:
            block = Block(day, room.id)
            surgeries = best_agenda.assignments.get(block, [])
            agenda_detalle[day][room.id] = {
                "especialidad": best_chromosome[block],
                "minutos_utilizados": best_agenda.used_time.get(block, 0),
                "minutos_disponibles": room.daily_capacity_minutes,
                "cirugias": [
                    {
                        "paciente_id": s.patient_id,
                        "especialidad": patients_by_id[s.patient_id].specialty_id,
                        "duracion_min": s.duration,
                        "prioridad_clinica": patients_by_id[s.patient_id].clinical_priority,
                    }
                    for s in surgeries
                ],
            }

    scheduled_ids = {s.patient_id for s in best_agenda.all_surgeries()}
    pendientes = [p.id for p in patients if p.id not in scheduled_ids]

    return {
        "fitness": round(best_fitness, 4),
        "generaciones_ejecutadas": len(ga.history),
        "distribucion_semanal_especialidades": distribucion,
        "agenda": agenda_detalle,
        "resumen": {
            "total_pacientes": len(patients),
            "pacientes_programados": len(scheduled_ids),
            "pacientes_pendientes": len(pendientes),
            "ids_pacientes_pendientes": pendientes,
        },
        "historial_fitness": [round(f, 4) for f in ga.history],
    }


def main():
    specialties, rooms, procedures, surgeons, patients = load_all(DATA_DIR)

    print(f"Datos cargados desde CSV: {len(specialties)} especialidades, "
          f"{len(rooms)} quirofanos, {len(surgeons)} cirujanos, "
          f"{len(procedures)} procedimientos, {len(patients)} pacientes.")

    ga = GeneticAlgorithm(
        days=DAYS,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=patients,
        population_size=80,
        generations=200,
        tournament_size=3,
        crossover_rate=0.85,
        mutation_rate=0.04,
        stagnation_limit=30,
        alpha=1.0,
        beta=0.3,
    )

    best_chromosome, best_fitness, best_agenda = ga.run()

    result = build_result_dict(DAYS, rooms, best_chromosome, best_fitness,
                                best_agenda, patients, ga)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nMejor fitness encontrado: {best_fitness:.2f}")
    print(f"Pacientes programados: {result['resumen']['pacientes_programados']} "
          f"de {result['resumen']['total_pacientes']}")
    print(f"Resultado exportado a: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()