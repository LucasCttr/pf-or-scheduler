"""
decoder.py
Mecanismo determinista (decoder) que transforma un cromosoma -es decir,
una asignacion de especialidades a bloques quirurgicos- en una agenda
quirurgica factible y evaluable.

Implementa la seccion 10.4.4 del documento:
  1. Identificacion de pacientes candidatos por especialidad/bloque.
  2. Filtrado por disponibilidad del cirujano y compatibilidad de sala.
  3. Priorizacion por prioridad clinica.
  4. Asignacion secuencial respetando la capacidad temporal del quirofano.

Incorpora ademas una ventana de limpieza parametrizable (`cleaning_minutes`)
que se consume entre cirugias consecutivas dentro de un mismo bloque. Este
tiempo se resta de la capacidad disponible del quirofano -no se contabiliza
como tiempo ocioso- ya que representa una actividad real (higienizacion y
preparacion de sala) que impide programar otra cirugia en ese lapso.
"""

from typing import Dict, List, Tuple

from models import (
    Agenda,
    Block,
    Patient,
    Procedure,
    Room,
    ScheduledSurgery,
    Surgeon
)

DEFAULT_CLEANING_MINUTES = 15


def build_agenda(
    chromosome: Dict[Block, str],
    patients: List[Patient],
    procedures: Dict[str, Procedure],
    surgeons: Dict[str, Surgeon],
    rooms: Dict[str, Room],
    cleaning_minutes: int = DEFAULT_CLEANING_MINUTES
) -> Agenda:
    """
    Construye una agenda quirurgica factible a partir de un cromosoma.

    Parametros
    ----------
    chromosome : dict {Block: specialty_id}
        Asignacion de especialidad medica para cada bloque (dia, quirofano).

    patients : lista de pacientes pendientes
        Se asume p.scheduled = False antes de invocar esta funcion.

    procedures, surgeons, rooms :
        Diccionarios indexados por id.

    cleaning_minutes : duracion de la ventana de limpieza que debe
        respetarse entre el final de una cirugia y el inicio de la
        siguiente dentro del mismo bloque. No se aplica antes de la
        primera cirugia del bloque (el quirofano ya esta preparado al
        inicio de la jornada).
    """

    SHIFT_HOURS = 5

    # Reiniciar estado de programacion
    for p in patients:
        p.scheduled = False

    assignments: Dict[Block, List[ScheduledSurgery]] = {
        b: []
        for b in chromosome
    }

    used_time: Dict[Block, int] = {
        b: 0
        for b in chromosome
    }

    cleaning_time: Dict[Block, int] = {
        b: 0
        for b in chromosome
    }

    # Horas de presencia acumuladas por cirujano
    surgeon_presence_hours = {
        s.id: 0
        for s in surgeons.values()
    }

    # Dias ya contabilizados para cada cirujano
    surgeon_days_used = {
        s.id: set()
        for s in surgeons.values()
    }

    # Agrupar pacientes pendientes por especialidad (P_E)
    patients_by_specialty: Dict[str, List[Patient]] = {}

    for p in patients:
        patients_by_specialty.setdefault(
            p.specialty_id,
            []
        ).append(p)

    # Procesar cada bloque del cromosoma
    for block, specialty_id in chromosome.items():

        room = rooms[block.room_id]
        capacity = room.daily_capacity_minutes

        candidates = patients_by_specialty.get(
            specialty_id,
            []
        )

        # -----------------------------
        # Conjunto de pacientes factibles
        # (se arrastra el procedimiento junto al paciente
        #  para no tener que volver a buscarlo despues)
        # -----------------------------
        feasible: List[Tuple[Patient, Procedure]] = []

        for p in candidates:

            if p.scheduled:
                continue

            surgeon = surgeons.get(p.surgeon_id)

            if surgeon is None:
                continue

            # Disponibilidad del cirujano
            if block.day not in surgeon.available_days:
                continue

            # Restriccion de horas contractuales
            # Solo se consume una jornada por dia.
            if block.day not in surgeon_days_used[surgeon.id]:

                projected_hours = (
                    surgeon_presence_hours[surgeon.id]
                    + SHIFT_HOURS
                )

                if projected_hours > surgeon.contract_hours_week:
                    continue

            procedure = procedures.get(p.procedure_id)

            if procedure is None:
                continue

            # Compatibilidad de complejidad de sala
            if procedure.required_room_type > room.room_type:
                continue

            feasible.append((p, procedure))

        # -----------------------------
        # Priorizacion
        # -----------------------------
        feasible.sort(
            key=lambda pair: pair[0].clinical_priority,
            reverse=True
        )

        # -----------------------------
        # Asignacion secuencial (respetando capacidad + limpieza)
        # -----------------------------
        remaining_capacity = capacity
        block_cleaning_used = 0
        is_first_surgery = True

        for p, procedure in feasible:

            duration = procedure.estimated_duration

            # A partir de la segunda cirugia del bloque, se debe reservar
            # ademas la ventana de limpieza posterior a la cirugia previa.
            required = duration if is_first_surgery else (
                cleaning_minutes + duration
            )

            if required > remaining_capacity:
                continue

            assignments[block].append(
                ScheduledSurgery(
                    patient_id=p.id,
                    block=block,
                    duration=duration
                )
            )

            remaining_capacity -= required

            if not is_first_surgery:
                block_cleaning_used += cleaning_minutes

            is_first_surgery = False
            p.scheduled = True

            surgeon_id = p.surgeon_id

            # Primera participacion del cirujano
            # en este dia -> consume la jornada.
            if block.day not in surgeon_days_used[surgeon_id]:

                surgeon_days_used[surgeon_id].add(
                    block.day
                )

                surgeon_presence_hours[surgeon_id] += (
                    SHIFT_HOURS
                )

        cleaning_time[block] = block_cleaning_used

        used_time[block] = (
            capacity - remaining_capacity
        )

    return Agenda(
        assignments=assignments,
        used_time=used_time,
        cleaning_time=cleaning_time
    )