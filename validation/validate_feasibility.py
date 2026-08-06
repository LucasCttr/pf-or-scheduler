"""
validar_factibilidad.py

Validador de factibilidad INDEPENDIENTE del decoder. La idea es no confiar
en que "el codigo que construye la agenda tambien la valida correctamente":
se recorre la agenda ya construida y se re-chequean todas las restricciones
duras desde cero, con logica separada de decoder.py.

Chequea:
  1. min_blocks: cada especialidad tiene al menos sus bloques minimos.
  2. Capacidad de quirofano: tiempo usado <= capacidad diaria de la sala.
  3. Compatibilidad de sala: procedure.required_room_type <= room.room_type.
  4. Disponibilidad del cirujano: la cirugia esta en un dia disponible.
  5. Horas contractuales: el cirujano no supera sus horas semanales.
  6. Unicidad de paciente: ningun paciente programado mas de una vez.
  7. Doble reserva de cirujano: que un cirujano NO quede asignado a mas de
     un quirofano distinto el mismo dia (el decoder no chequea esto -
     solo lleva la cuenta de horas, no de solapamiento fisico real).

Uso:
    python validar_factibilidad.py --data-dir data
    python validar_factibilidad.py --data-dir data --from-json agenda_resultado.json
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

from data_loader import load_all
from models import Block
from decoder import build_agenda
from genetic_algorithm import GeneticAlgorithm

DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes"]
SHIFT_HOURS = 5  # debe coincidir con el valor usado en decoder.py


def chromosome_from_json(path: str) -> Dict[Block, str]:
    """Reconstruye el cromosoma {Block: specialty_id} desde el JSON exportado por main.py."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    distribucion = data["distribucion_semanal_especialidades"]
    chromosome = {}
    for day, rooms in distribucion.items():
        for room_id, specialty_id in rooms.items():
            chromosome[Block(day, room_id)] = specialty_id
    return chromosome


def validate(chromosome, agenda, specialties, rooms, procedures, surgeons, patients, label=""):
    rooms_by_id = {r.id: r for r in rooms}
    procedures_by_id = {p.id: p for p in procedures}
    surgeons_by_id = {s.id: s for s in surgeons}
    patients_by_id = {p.id: p for p in patients}
    min_blocks = {s.id: s.min_blocks for s in specialties}

    violations = defaultdict(list)

    # --- 1. min_blocks ---
    counts = defaultdict(int)
    for sid in chromosome.values():
        counts[sid] += 1
    for sid, minimo in min_blocks.items():
        if counts[sid] < minimo:
            violations["min_blocks"].append(
                f"Especialidad {sid}: tiene {counts[sid]} bloques, requiere minimo {minimo}"
            )

    # --- 2. Capacidad de quirofano ---
    for block, used in agenda.used_time.items():
        capacity = rooms_by_id[block.room_id].daily_capacity_minutes
        if used > capacity:
            violations["capacidad_sala"].append(
                f"Bloque {block}: usa {used} min, capacidad {capacity} min"
            )

    # --- Recorrido de todas las cirugias programadas (para 3, 4, 5, 6, 7) ---
    all_surgeries = agenda.all_surgeries()

    # 6. Unicidad de paciente
    patient_appearances = defaultdict(list)
    for s in all_surgeries:
        patient_appearances[s.patient_id].append(s.block)
    for pid, blocks in patient_appearances.items():
        if len(blocks) > 1:
            violations["paciente_duplicado"].append(
                f"Paciente {pid} programado {len(blocks)} veces en bloques {blocks}"
            )

    # Estructuras para 4, 5 y 7
    surgeon_days: Dict[str, set] = defaultdict(set)          # surgeon_id -> dias trabajados
    surgeon_day_rooms: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))  # surgeon_id -> dia -> {room_id}

    for s in all_surgeries:
        patient = patients_by_id.get(s.patient_id)
        if patient is None:
            violations["paciente_no_encontrado"].append(f"Cirugia referencia paciente inexistente: {s.patient_id}")
            continue

        surgeon = surgeons_by_id.get(patient.surgeon_id)
        procedure = procedures_by_id.get(patient.procedure_id)
        room = rooms_by_id[s.block.room_id]

        # 3. Compatibilidad de sala
        if procedure and procedure.required_room_type > room.room_type:
            violations["compatibilidad_sala"].append(
                f"Paciente {s.patient_id}: procedimiento requiere room_type "
                f"{procedure.required_room_type}, sala {room.id} es tipo {room.room_type}"
            )

        if surgeon is None:
            violations["cirujano_no_encontrado"].append(
                f"Paciente {s.patient_id}: cirujano {patient.surgeon_id} no existe"
            )
            continue

        # 4. Disponibilidad del cirujano
        if s.block.day not in surgeon.available_days:
            violations["disponibilidad_cirujano"].append(
                f"Cirujano {surgeon.id}: programado el {s.block.day}, "
                f"disponible solo {sorted(surgeon.available_days)}"
            )

        surgeon_days[surgeon.id].add(s.block.day)
        surgeon_day_rooms[surgeon.id][s.block.day].add(s.block.room_id)

    # 5. Horas contractuales (misma logica que el decoder: SHIFT_HOURS por dia trabajado)
    for surgeon_id, days_worked in surgeon_days.items():
        surgeon = surgeons_by_id[surgeon_id]
        horas_totales = len(days_worked) * SHIFT_HOURS
        if horas_totales > surgeon.contract_hours_week:
            violations["horas_contractuales"].append(
                f"Cirujano {surgeon_id}: {horas_totales}hs trabajadas "
                f"({len(days_worked)} dias x {SHIFT_HOURS}hs), contrato {surgeon.contract_hours_week}hs/semana"
            )

    # 7. Doble reserva de cirujano (mismo dia, mas de un quirofano)
    for surgeon_id, days_rooms in surgeon_day_rooms.items():
        for day, rooms_used in days_rooms.items():
            if len(rooms_used) > 1:
                violations["doble_reserva_cirujano"].append(
                    f"Cirujano {surgeon_id}: el {day} queda asignado a {len(rooms_used)} "
                    f"quirofanos distintos simultaneamente: {sorted(rooms_used)}"
                )

    # --- Reporte ---
    total_checks = 7
    categorias = [
        ("min_blocks", "Minimos de especialidad"),
        ("capacidad_sala", "Capacidad de quirofano"),
        ("compatibilidad_sala", "Compatibilidad de sala"),
        ("disponibilidad_cirujano", "Disponibilidad de cirujano"),
        ("horas_contractuales", "Horas contractuales"),
        ("paciente_duplicado", "Unicidad de paciente"),
        ("doble_reserva_cirujano", "Doble reserva de cirujano"),
    ]

    print(f"\n{'=' * 60}")
    print(f"REPORTE DE FACTIBILIDAD {('- ' + label) if label else ''}")
    print(f"{'=' * 60}")
    total_violaciones = 0
    for key, nombre in categorias:
        viol = violations.get(key, [])
        total_violaciones += len(viol)
        estado = "OK" if not viol else f"{len(viol)} violacion(es)"
        print(f"  [{'OK' if not viol else 'FALLA':<5}] {nombre:<30} {estado}")
        for v in viol[:5]:
            print(f"           - {v}")
        if len(viol) > 5:
            print(f"           ... y {len(viol) - 5} mas")

    ok_categorias = sum(1 for key, _ in categorias if not violations.get(key))
    print(f"\n  Categorias sin violaciones: {ok_categorias}/{total_checks}")
    print(f"  Total de violaciones individuales encontradas: {total_violaciones}")
    if total_violaciones == 0:
        print("  -> La solucion es 100% factible respecto a las restricciones duras evaluadas.")
    print(f"{'=' * 60}\n")

    return violations


def main():
    parser = argparse.ArgumentParser(description="Valida factibilidad de una agenda quirurgica.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--from-json", default=None,
                         help="Ruta a un agenda_resultado.json ya exportado por main.py. "
                              "Si no se pasa, corre el GA una vez con los parametros default.")
    args = parser.parse_args()

    specialties, rooms, procedures, surgeons, patients = load_all(args.data_dir)
    procedures_by_id = {p.id: p for p in procedures}
    surgeons_by_id = {s.id: s for s in surgeons}
    rooms_by_id = {r.id: r for r in rooms}

    if args.from_json:
        chromosome = chromosome_from_json(args.from_json)
        agenda = build_agenda(chromosome, patients, procedures_by_id, surgeons_by_id, rooms_by_id)
        label = f"GA (desde {os.path.basename(args.from_json)})"
    else:
        print("No se paso --from-json: corriendo el GA una vez con parametros default...")
        ga = GeneticAlgorithm(
            days=DAYS, rooms=rooms, specialties=specialties, surgeons=surgeons,
            procedures=procedures, patients=patients,
            population_size=80, generations=200, tournament_size=3,
            crossover_rate=0.85, mutation_rate=0.04, stagnation_limit=30,
            alpha=1.0, beta=0.3,
        )
        chromosome, _, agenda = ga.run()
        label = "GA (corrida nueva)"

    validate(chromosome, agenda, specialties, rooms, procedures, surgeons, patients, label=label)


if __name__ == "__main__":
    main()