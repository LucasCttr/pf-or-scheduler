"""
validation.py

Validador independiente de agendas quirúrgicas.

Este módulo solo analiza la solución generada para verificar que cumple con las
restricciones del problema y resumir métricas operativas.

Se evalúan estas restricciones relevantes:
1. Existencia de referencias.
2. Duración de las cirugías.
3. Compatibilidad de sala.
4. Capacidad de quirófano.
5. Disponibilidad de cirujanos.
6. Horas contractuales.
7. Solapamiento de cirujanos.
8. Cuotas mínimas de especialidad.
9. Unicidad de paciente.

Además calcula métricas complementarias (no restricciones duras):
priorización, priorización por especialidad, utilización de quirófanos
y balance de carga entre cirujanos.
"""

from typing import Dict, List
from collections import defaultdict
from statistics import mean, pstdev

from models import (
    Agenda,
    Block,
    Patient,
    Procedure,
    Specialty,
    Surgeon,
    Room,
)


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
# 1. EXISTENCIA DE REFERENCIAS
# ============================================================


def validate_general_feasibility(
    agenda: Agenda,
    patients: List[Patient],
    procedures: Dict[str, Procedure],
    rooms: Dict[str, Room],
) -> ValidationResult:
    """
    Verifica que cada quirofano, paciente y procedimiento referenciado
    por una cirugia programada exista realmente en los datos de entrada.

    Las restricciones de duracion, compatibilidad de sala y capacidad
    de quirofano se validan por separado en sus propias funciones
    (validate_durations, validate_room_compatibility, validate_room_capacity)
    para evitar reportar la misma violacion dos veces.
    """

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

    result.metrics["Cirugías asignadas"] = total_surgeries

    return result


# ============================================================
# 1.5. COMPATIBILIDAD DE SALA
# ============================================================


def validate_room_compatibility(
    agenda: Agenda,
    patients: List[Patient],
    procedures: Dict[str, Procedure],
    rooms: Dict[str, Room],
) -> ValidationResult:
    """
    Verifica que el tipo de sala asignada a cada cirugia tenga, como
    minimo, el nivel de equipamiento que el procedimiento requiere.
    """

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    for block, surgeries in agenda.assignments.items():
        room = get_room(rooms, block.room_id)

        if room is None:
            continue

        for surgery in surgeries:
            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                continue

            procedure = get_by_id(procedures, patient.procedure_id)

            if procedure is None:
                continue

            if procedure.required_room_type > room.room_type:
                result.add_violation(
                    f"Paciente {patient.id}: el procedimiento "
                    f"requiere sala tipo "
                    f"{procedure.required_room_type}, "
                    f"pero fue asignado a sala tipo "
                    f"{room.room_type}."
                )

    return result


# ============================================================
# 2. DISPONIBILIDAD DE CIRUJANOS
# ============================================================


def validate_surgeon_availability(
    agenda: Agenda,
    patients: List[Patient],
    surgeons: Dict[str, Surgeon],
    rooms: Dict[str, Room],
) -> ValidationResult:

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    surgeon_days = defaultdict(set)

    # Sala en la que el cirujano tuvo su primera cirugia cada dia. Se usa
    # para calcular la jornada consumida con la misma logica que
    # decoder.py (capacidad real de esa sala, no un valor fijo).
    surgeon_day_room = defaultdict(dict)

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
            surgeon_day_room[surgeon.id].setdefault(block.day, block.room_id)

    # --------------------------------------------------------
    # Horas contractuales
    #
    # La jornada consumida por dia equivale a la capacidad real del
    # quirofano donde el cirujano tuvo su primera cirugia ese dia,
    # convertida a horas (misma logica que decoder.py).
    # --------------------------------------------------------

    for surgeon_id, days in surgeon_days.items():
        surgeon = get_by_id(surgeons, surgeon_id)

        hours = 0.0
        for day in days:
            room_id = surgeon_day_room[surgeon_id][day]
            room = get_by_id(rooms, room_id)
            hours += (room.daily_capacity_minutes / 60) if room else 0.0

        if hours > surgeon.contract_hours_week:
            result.add_violation(
                f"Cirujano {surgeon_id}: "
                f"{hours:.2f} horas de presencia asignadas, "
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
# 5.5. PRIORIZACIÓN POR ESPECIALIDAD
# ============================================================


def validate_priority_by_specialty(
    agenda: Agenda,
    patients: List[Patient],
) -> ValidationResult:
    """
    Repite el mismo analisis de validate_priority (prioridad promedio
    de asignados vs. no asignados), pero desglosado por especialidad.

    Esto permite detectar si el criterio de priorizacion funciona bien
    DENTRO de cada especialidad, y no solo en el agregado global. Un
    promedio global alto podria estar oculto por una sola especialidad
    con pacientes de alta prioridad, sin que el algoritmo este realmente
    discriminando bien dentro de las demas especialidades.
    """

    result = ValidationResult()

    patient_by_id = {p.id: p for p in patients}

    assigned_ids = set()

    for surgeries in agenda.assignments.values():
        for surgery in surgeries:
            patient = patient_by_id.get(surgery.patient_id)

            if patient is None:
                continue

            assigned_ids.add(patient.id)

    # Agrupar prioridades por especialidad, separando asignados de no asignados
    assigned_by_specialty = defaultdict(list)
    unassigned_by_specialty = defaultdict(list)

    for patient in patients:
        if patient.id in assigned_ids:
            assigned_by_specialty[patient.specialty_id].append(patient.clinical_priority)
        else:
            unassigned_by_specialty[patient.specialty_id].append(patient.clinical_priority)

    all_specialty_ids = set(assigned_by_specialty) | set(unassigned_by_specialty)

    breakdown = {}

    for specialty_id in sorted(all_specialty_ids):
        asignados = assigned_by_specialty.get(specialty_id, [])
        no_asignados = unassigned_by_specialty.get(specialty_id, [])

        breakdown[specialty_id] = {
            "prioridad_asignados": mean(asignados) if asignados else None,
            "prioridad_no_asignados": mean(no_asignados) if no_asignados else None,
            "n_asignados": len(asignados),
            "n_no_asignados": len(no_asignados),
        }

        # Advertencia (no violacion dura): si dentro de esta especialidad
        # los no asignados tienen, en promedio, MAYOR prioridad que los
        # asignados, es una señal de que la priorizacion no esta
        # funcionando bien para esa especialidad en particular.
        if asignados and no_asignados:
            if mean(no_asignados) > mean(asignados):
                result.add_warning(
                    f"Especialidad {specialty_id}: la prioridad promedio de "
                    f"los NO asignados ({mean(no_asignados):.2f}) supera a "
                    f"la de los asignados ({mean(asignados):.2f})."
                )

    result.metrics["Prioridad por especialidad"] = breakdown

    return result


def print_priority_by_specialty(breakdown: Dict[str, dict]):
    """
    Imprime en formato de tabla el desglose de prioridad por especialidad
    generado por validate_priority_by_specialty. Se ofrece como funcion
    separada porque ValidationResult.print_report() no tiene formato de
    tabla para metricas anidadas.
    """

    print("\n--- PRIORIZACIÓN POR ESPECIALIDAD ---")

    header = (
        f"{'Especialidad':<15}"
        f"{'Prior. Asig.':>14}"
        f"{'Prior. No Asig.':>17}"
        f"{'N Asig.':>10}"
        f"{'N No Asig.':>12}"
    )

    print(header)
    print("-" * len(header))

    for specialty_id, datos in breakdown.items():
        prior_asig = datos["prioridad_asignados"]
        prior_no_asig = datos["prioridad_no_asignados"]

        prior_asig_str = f"{prior_asig:.2f}" if prior_asig is not None else "N/A"
        prior_no_asig_str = f"{prior_no_asig:.2f}" if prior_no_asig is not None else "N/A"

        print(
            f"{specialty_id:<15}"
            f"{prior_asig_str:>14}"
            f"{prior_no_asig_str:>17}"
            f"{datos['n_asignados']:>10}"
            f"{datos['n_no_asignados']:>12}"
        )


# ============================================================
# 5.6. MÍNIMOS DE ESPECIALIDAD (min_blocks)
# ============================================================


def validate_specialty_minimums(
    chromosome: Dict[Block, str],
    specialties: List[Specialty],
) -> ValidationResult:
    """
    Verifica que cada especialidad tenga asignada, en el cromosoma, al
    menos la cantidad minima de bloques semanales garantizados
    (specialty.min_blocks). A diferencia del resto de las validaciones,
    esta no depende de la Agenda (pacientes agendados) sino directamente
    del cromosoma (asignacion especialidad -> bloque), ya que min_blocks
    es una restriccion sobre esa asignacion, no sobre los pacientes.
    """

    result = ValidationResult()

    counts = defaultdict(int)

    for specialty_id in chromosome.values():
        counts[specialty_id] += 1

    for specialty in specialties:
        assigned = counts.get(specialty.id, 0)

        if assigned < specialty.min_blocks:
            result.add_violation(
                f"Especialidad {specialty.id}: tiene {assigned} bloques "
                f"asignados, requiere un minimo de {specialty.min_blocks}."
            )

    result.metrics["Especialidades con minimo cumplido"] = sum(
        1 for s in specialties if counts.get(s.id, 0) >= s.min_blocks
    )

    result.metrics["Especialidades totales"] = len(specialties)

    return result


# ============================================================
# 5.7. UNICIDAD DE PACIENTE
# ============================================================


def validate_patient_uniqueness(
    agenda: Agenda,
) -> ValidationResult:
    """
    Verifica que ningun paciente aparezca programado mas de una vez en
    la agenda generada.
    """

    result = ValidationResult()

    appearances = defaultdict(list)

    for block, surgeries in agenda.assignments.items():
        for surgery in surgeries:
            appearances[surgery.patient_id].append(block)

    duplicated = 0

    for patient_id, blocks in appearances.items():
        if len(blocks) > 1:
            duplicated += 1
            result.add_violation(
                f"Paciente {patient_id} programado {len(blocks)} veces "
                f"en bloques {blocks}."
            )

    result.metrics["Pacientes duplicados"] = duplicated

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
    specialties: List[Specialty],
    chromosome: Dict[Block, str],
):

    # Ejecutar validaciones individuales. El orden sigue las 9
    # restricciones documentadas; las ultimas cuatro son metricas
    # complementarias que no constituyen restricciones duras.

    validations = [
        # 1. Existencia de referencias
        validate_general_feasibility(
            agenda,
            patients,
            procedures,
            rooms,
        ),

        # 2. Duracion de las cirugias
        validate_durations(
            agenda,
            patients,
            procedures,
        ),

        # 3. Compatibilidad de sala
        validate_room_compatibility(
            agenda,
            patients,
            procedures,
            rooms,
        ),

        # 4. Capacidad de quirofano
        validate_room_capacity(
            agenda,
            rooms,
        ),

        # 5 y 6. Disponibilidad de cirujano y horas contractuales
        validate_surgeon_availability(
            agenda,
            patients,
            surgeons,
            rooms,
        ),

        # 7. Solapamiento de cirujanos
        validate_surgeon_overlaps(
            agenda,
            patients,
        ),

        # 8. Cuotas minimas de especialidad
        validate_specialty_minimums(
            chromosome,
            specialties,
        ),

        # 9. Unicidad de paciente
        validate_patient_uniqueness(
            agenda,
        ),

        # --- Metricas complementarias (no restricciones duras) ---

        validate_priority(
            agenda,
            patients,
        ),

        validate_priority_by_specialty(
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