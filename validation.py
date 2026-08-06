"""
validation.py

Validador independiente de agendas quirúrgicas.

Evalúa:
1. Factibilidad de la agenda.
2. Disponibilidad de cirujanos.
3. Duración de cirugías.
4. Capacidad de quirófanos.
5. Priorización de pacientes.
6. Utilización de quirófanos.
7. Balance de carga entre cirujanos.

IMPORTANTE:
Este módulo NO modifica la agenda ni ejecuta el algoritmo.
Solo analiza la solución generada.
"""

from typing import Dict, List
from collections import defaultdict
from statistics import mean, pstdev

from models import (
    Agenda,
    Block,
    Patient,
    Procedure,
    Surgeon,
    Room,
)


SHIFT_HOURS = 5


def get_room(rooms, room_id):
    for room in rooms:
        if room.id == room_id:
            return room

    return None


def get_by_id(collection, item_id):
    """
    Obtiene un objeto por ID independientemente de si
    collection es una lista o un diccionario.
    """

    if isinstance(collection, dict):
        return collection.get(item_id)

    for item in collection:
        if item.id == item_id:
            return item

    return None


# ============================================================
# RESULTADO
# ============================================================


class ValidationResult:
    def __init__(self):
        self.violations = []
        self.warnings = []
        self.metrics = {}

    @property
    def valid(self):
        return len(self.violations) == 0

    def add_violation(self, message):
        self.violations.append(message)

    def add_warning(self, message):
        self.warnings.append(message)

    def print_report(self):

        print("\n")
        print("=" * 70)
        print("VALIDACIÓN DE AGENDA QUIRÚRGICA")
        print("=" * 70)

        print("\n--- RESTRICCIONES DURAS ---")

        if self.valid:
            print("✓ No se detectaron violaciones.")
        else:
            print(f"✗ Se detectaron {len(self.violations)} violaciones:")

            for violation in self.violations:
                print(f"  - {violation}")

        print("\n--- MÉTRICAS ---")

        for name, value in self.metrics.items():
            if isinstance(value, float):
                print(f"{name}: {value:.2f}")
            else:
                print(f"{name}: {value}")

        if self.warnings:
            print("\n--- ADVERTENCIAS ---")

            for warning in self.warnings:
                print(f"  - {warning}")

        print("\n" + "=" * 70)


# ============================================================
# 1. FACTIBILIDAD GENERAL
# ============================================================


def validate_general_feasibility(
    agenda: Agenda,
    patients: List[Patient],
    procedures: Dict[str, Procedure],
    rooms: Dict[str, Room],
    surgeons: Dict[str, Surgeon],
) -> ValidationResult:

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    total_surgeries = 0

    # --------------------------------------------------------
    # Recorrer todos los bloques de la agenda
    # --------------------------------------------------------

    for block, surgeries in agenda.assignments.items():
        room = get_room(rooms, block.room_id)

        if room is None:
            result.add_violation(f"El quirófano {block.room_id} no existe.")

            continue

        used = 0

        for surgery in surgeries:
            total_surgeries += 1

            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                result.add_violation(f"Paciente {surgery.patient_id} no existe.")

                continue

            procedure = get_by_id(procedures, patient.procedure_id)

            if procedure is None:
                result.add_violation(
                    f"No existe el procedimiento "
                    f"{patient.procedure_id} del paciente "
                    f"{patient.id}."
                )

                continue

            # ------------------------------------------------
            # Duración
            # ------------------------------------------------

            if surgery.duration != procedure.estimated_duration:
                result.add_violation(
                    f"Paciente {patient.id}: duración "
                    f"asignada {surgery.duration} != "
                    f"duración requerida "
                    f"{procedure.estimated_duration}."
                )

            used += surgery.duration

            # ------------------------------------------------
            # Compatibilidad de sala
            # ------------------------------------------------

            if procedure.required_room_type > room.room_type:
                result.add_violation(
                    f"Paciente {patient.id}: el procedimiento "
                    f"requiere sala tipo "
                    f"{procedure.required_room_type}, "
                    f"pero fue asignado a sala tipo "
                    f"{room.room_type}."
                )

        # ----------------------------------------------------
        # Capacidad
        # ----------------------------------------------------

        if used > room.daily_capacity_minutes:
            result.add_violation(
                f"{block.day} - {block.room_id}: "
                f"se utilizaron {used} minutos, "
                f"superando la capacidad de "
                f"{room.daily_capacity_minutes}."
            )

    result.metrics["Cirugías asignadas"] = total_surgeries

    return result


# ============================================================
# 2. DISPONIBILIDAD DE CIRUJANOS
# ============================================================


def validate_surgeon_availability(
    agenda: Agenda,
    patients: List[Patient],
    surgeons: Dict[str, Surgeon],
) -> ValidationResult:

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    surgeon_days = defaultdict(set)

    for block, surgeries in agenda.assignments.items():
        for surgery in surgeries:
            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                continue

            surgeon = get_by_id(surgeons, patient.surgeon_id)

            if surgeon is None:
                result.add_violation(
                    f"El cirujano {patient.surgeon_id} "
                    f"del paciente {patient.id} no existe."
                )

                continue

            # ------------------------------------------------
            # Día disponible
            # ------------------------------------------------

            if block.day not in surgeon.available_days:
                result.add_violation(
                    f"Paciente {patient.id}: cirujano "
                    f"{surgeon.id} no está disponible "
                    f"el {block.day}."
                )

            surgeon_days[surgeon.id].add(block.day)

    # --------------------------------------------------------
    # Horas contractuales
    #
    # El decoder considera una jornada de 5 horas por día.
    # --------------------------------------------------------

    for surgeon_id, days in surgeon_days.items():
        surgeon = get_by_id(surgeons, surgeon_id)
        hours = len(days) * SHIFT_HOURS

        if hours > surgeon.contract_hours_week:
            result.add_violation(
                f"Cirujano {surgeon_id}: "
                f"{hours} horas de presencia asignadas, "
                f"superando contrato de "
                f"{surgeon.contract_hours_week} horas."
            )

    result.metrics["Cirujanos utilizados"] = len(surgeon_days)

    result.metrics["Días-cirujano utilizados"] = sum(
        len(days) for days in surgeon_days.values()
    )

    return result

# ============================================================
# 2.5. SOLAPAMIENTO DE CIRUJANOS
# ============================================================

def validate_surgeon_overlaps(
    agenda: Agenda,
    patients: List[Patient],
) -> ValidationResult:

    result = ValidationResult()

    patient_by_id = {
        p.id: p
        for p in patients
    }

    # Cirugías agrupadas por cirujano
    surgeries_by_surgeon = defaultdict(list)

    # --------------------------------------------------------
    # Reconstruir los intervalos de tiempo de cada cirugía
    # --------------------------------------------------------

    for block, surgeries in agenda.assignments.items():

        current_minute = 0

        for surgery in surgeries:

            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                continue

            start = current_minute
            end = current_minute + surgery.duration

            surgeries_by_surgeon[
                patient.surgeon_id
            ].append({
                "day": block.day,
                "room_id": block.room_id,
                "patient_id": patient.id,
                "start": start,
                "end": end,
            })

            current_minute = end

    # --------------------------------------------------------
    # Buscar solapamientos
    # --------------------------------------------------------

    for surgeon_id, surgeries in surgeries_by_surgeon.items():

        for i in range(len(surgeries)):

            surgery_a = surgeries[i]

            for j in range(i + 1, len(surgeries)):

                surgery_b = surgeries[j]

                # Si son días diferentes, no pueden solaparse
                if surgery_a["day"] != surgery_b["day"]:
                    continue

                # Condición de solapamiento:
                #
                # A: |---------|
                # B:     |---------|
                #
                if (
                    surgery_a["start"] < surgery_b["end"]
                    and
                    surgery_b["start"] < surgery_a["end"]
                ):

                    result.add_violation(
                        f"Cirujano {surgeon_id}: "
                        f"solapamiento el {surgery_a['day']} "
                        f"entre Q{surgery_a['room_id']} "
                        f"(paciente {surgery_a['patient_id']}, "
                        f"{surgery_a['start']}-{surgery_a['end']} min) "
                        f"y Q{surgery_b['room_id']} "
                        f"(paciente {surgery_b['patient_id']}, "
                        f"{surgery_b['start']}-{surgery_b['end']} min)."
                    )

    result.metrics[
        "Solapamientos de cirujanos"
    ] = len(result.violations)

    return result

# ============================================================
# 3. DURACIÓN DE CIRUGÍAS
# ============================================================


def validate_durations(
    agenda: Agenda,
    patients: List[Patient],
    procedures: Dict[str, Procedure],
) -> ValidationResult:

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    total = 0
    correct = 0

    for surgeries in agenda.assignments.values():
        for surgery in surgeries:
            total += 1

            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                continue

            procedure = get_by_id(procedures, patient.procedure_id)

            if procedure is None:
                continue

            if surgery.duration == procedure.estimated_duration:
                correct += 1

            else:
                result.add_violation(f"Paciente {patient.id}: duración incorrecta.")

    percentage = correct / total * 100 if total > 0 else 0

    result.metrics["Cumplimiento de duración (%)"] = percentage

    return result


# ============================================================
# 4. CAPACIDAD DE QUIRÓFANOS
# ============================================================


def validate_room_capacity(
    agenda: Agenda,
    rooms: Dict[str, Room],
) -> ValidationResult:

    result = ValidationResult()

    total_blocks = 0
    valid_blocks = 0

    total_used = 0
    total_capacity = 0

    for block, surgeries in agenda.assignments.items():
        room = get_by_id(rooms, block.room_id)

        if room is None:
            continue

        total_blocks += 1

        used = sum(surgery.duration for surgery in surgeries)

        capacity = room.daily_capacity_minutes

        total_used += used
        total_capacity += capacity

        if used <= capacity:
            valid_blocks += 1

        else:
            result.add_violation(
                f"{block.day} - {block.room_id}: {used}/{capacity} minutos."
            )

    result.metrics["Bloques dentro de capacidad (%)"] = (
        valid_blocks / total_blocks * 100 if total_blocks > 0 else 0
    )

    return result


# ============================================================
# 5. PRIORIZACIÓN
# ============================================================


def validate_priority(
    agenda: Agenda,
    patients: List[Patient],
) -> ValidationResult:

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    assigned_priorities = []
    unassigned_priorities = []

    assigned_ids = set()

    for surgeries in agenda.assignments.values():
        for surgery in surgeries:
            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                continue

            assigned_ids.add(patient.id)

            assigned_priorities.append(patient.clinical_priority)

    for patient in patients:
        if patient.id not in assigned_ids:
            unassigned_priorities.append(patient.clinical_priority)

    # --------------------------------------------------------
    # Prioridad promedio
    # --------------------------------------------------------

    avg_assigned = mean(assigned_priorities) if assigned_priorities else 0

    avg_unassigned = mean(unassigned_priorities) if unassigned_priorities else 0

    result.metrics["Prioridad promedio asignados"] = avg_assigned

    result.metrics["Prioridad promedio no asignados"] = avg_unassigned

    # --------------------------------------------------------
    # Pacientes de alta prioridad
    # --------------------------------------------------------

    HIGH_PRIORITY = 8

    high_priority = [p for p in patients if p.clinical_priority >= HIGH_PRIORITY]

    high_priority_assigned = [p for p in high_priority if p.id in assigned_ids]

    percentage = (
        len(high_priority_assigned) / len(high_priority) * 100 if high_priority else 0
    )

    result.metrics["Pacientes alta prioridad asignados (%)"] = percentage

    return result


# ============================================================
# 6. UTILIZACIÓN
# ============================================================


def calculate_utilization(
    agenda: Agenda,
    rooms: Dict[str, Room],
) -> ValidationResult:

    result = ValidationResult()

    total_used = 0
    total_capacity = 0

    room_utilization = {}

    for block, surgeries in agenda.assignments.items():
        room = get_by_id(rooms, block.room_id)

        if room is None:
            continue

        used = sum(surgery.duration for surgery in surgeries)

        capacity = room.daily_capacity_minutes

        total_used += used
        total_capacity += capacity

        key = f"{block.day}-{block.room_id}"

        room_utilization[key] = used / capacity * 100 if capacity > 0 else 0

    global_utilization = total_used / total_capacity * 100 if total_capacity > 0 else 0

    result.metrics["Utilización global (%)"] = global_utilization

    result.metrics["Minutos utilizados"] = total_used

    result.metrics["Minutos disponibles"] = total_capacity

    # Guardamos también el detalle
    result.metrics["Utilización por bloque"] = room_utilization

    return result


# ============================================================
# 7. BALANCE DE CARGA
# ============================================================


def calculate_surgeon_balance(
    agenda: Agenda,
    patients: List[Patient],
) -> ValidationResult:

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    surgeon_minutes = defaultdict(int)

    for surgeries in agenda.assignments.values():
        for surgery in surgeries:
            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                continue

            surgeon_minutes[patient.surgeon_id] += surgery.duration

    values = list(surgeon_minutes.values())

    if values:
        average = mean(values)

        deviation = pstdev(values) if len(values) > 1 else 0

        minimum = min(values)
        maximum = max(values)

    else:
        average = 0
        deviation = 0
        minimum = 0
        maximum = 0

    result.metrics["Horas promedio por cirujano"] = average / 60

    result.metrics["Desviación estándar carga (horas)"] = deviation / 60

    result.metrics["Carga mínima (horas)"] = minimum / 60

    result.metrics["Carga máxima (horas)"] = maximum / 60

    return result


# ============================================================
# VALIDACIÓN COMPLETA
# ============================================================


def validate_agenda(
    agenda: Agenda,
    patients: List[Patient],
    procedures: Dict[str, Procedure],
    surgeons: Dict[str, Surgeon],
    rooms: Dict[str, Room],
):

    # Ejecutar validaciones individuales

    validations = [
        validate_general_feasibility(
            agenda,
            patients,
            procedures,
            rooms,
            surgeons,
        ),

        validate_surgeon_availability(
            agenda,
            patients,
            surgeons,
        ),

        validate_surgeon_overlaps(
            agenda,
            patients,
        ),

        validate_durations(
            agenda,
            patients,
            procedures,
        ),

        validate_room_capacity(
            agenda,
            rooms,
        ),

        validate_priority(
            agenda,
            patients,
        ),

        calculate_utilization(
            agenda,
            rooms,
        ),

        calculate_surgeon_balance(
            agenda,
            patients,
        ),
    ]

    # --------------------------------------------------------
    # Unificar resultados
    # --------------------------------------------------------

    final = ValidationResult()

    for validation in validations:
        final.violations.extend(validation.violations)

        final.warnings.extend(validation.warnings)

        final.metrics.update(validation.metrics)

    return final


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    print("Este módulo debe ejecutarse pasando una Agenda generada por el algoritmo.")
