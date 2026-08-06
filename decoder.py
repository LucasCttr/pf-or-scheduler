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

Restriccion adicional (fisica, no solo de horas contratadas):
  Un cirujano no puede operar en dos quirofanos distintos al mismo tiempo,
  aunque en teoria le queden horas de contrato disponibles. Para modelar
  esto de forma precisa (y no simplemente prohibirle usar una segunda sala
  el mismo dia), se calcula el intervalo de tiempo real que ocuparia cada
  cirugia dentro de su bloque (a partir de cuanto tiempo ya se consumio en
  ese bloque), y se verifica que no se solape con ningun otro intervalo que
  el mismo cirujano ya tenga ocupado ese dia en cualquier otra sala. Si no
  hay solapamiento, el cirujano puede operar en ambos quirofanos el mismo
  dia sin problema. Se asume que todos los quirofanos comparten la misma
  franja horaria de apertura (supuesto ya implicito en SHIFT_HOURS, que es
  un valor fijo independiente de la sala).
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


def build_agenda(
    chromosome: Dict[Block, str],
    patients: List[Patient],
    procedures: Dict[str, Procedure],
    surgeons: Dict[str, Surgeon],
    rooms: Dict[str, Room]
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

    # Intervalos de tiempo que cada cirujano ya tiene ocupados, por dia,
    # sin importar en que sala. surgeon_day_intervals[surgeon_id][day] es
    # una lista de tuplas (inicio, fin) en minutos relativos al inicio de
    # la jornada. Se usa para impedir que un cirujano quede agendado en
    # dos quirofanos distintos en el mismo momento del dia.
    surgeon_day_intervals: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        s.id: {}
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
        # Asignacion secuencial
        # -----------------------------
        remaining_capacity = capacity

        for p, procedure in feasible:

            duration = procedure.estimated_duration

            if duration > remaining_capacity:
                continue

            surgeon_id = p.surgeon_id

            # Intervalo tentativo que ocuparia esta cirugia dentro del
            # bloque actual (relativo al inicio de la jornada del dia).
            start_time = capacity - remaining_capacity
            end_time = start_time + duration

            # Restriccion fisica: el cirujano no puede estar operando en
            # dos quirofanos al mismo tiempo. Se verifica que este
            # intervalo no se solape con ningun otro que el cirujano ya
            # tenga ocupado este mismo dia, en cualquier sala (incluido
            # este mismo bloque, por si ya se le asigno otro paciente aca).
            ocupados_hoy = surgeon_day_intervals[surgeon_id].setdefault(block.day, [])
            solapa = any(
                start_time < fin_existente and inicio_existente < end_time
                for inicio_existente, fin_existente in ocupados_hoy
            )
            if solapa:
                continue

            assignments[block].append(
                ScheduledSurgery(
                    patient_id=p.id,
                    block=block,
                    duration=duration
                )
            )

            remaining_capacity -= duration
            p.scheduled = True

            ocupados_hoy.append((start_time, end_time))

            # Primera participacion del cirujano
            # en este dia -> consume la jornada.
            if block.day not in surgeon_days_used[surgeon_id]:

                surgeon_days_used[surgeon_id].add(
                    block.day
                )

                surgeon_presence_hours[surgeon_id] += (
                    SHIFT_HOURS
                )

        used_time[block] = (
            capacity - remaining_capacity
        )

    return Agenda(
        assignments=assignments,
        used_time=used_time
    )