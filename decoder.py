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
"""
from typing import Dict, List

from models import Agenda, Block, Patient, Procedure, Room, ScheduledSurgery, Surgeon


def build_agenda(chromosome: Dict[Block, str],
                  patients: List[Patient],
                  procedures: Dict[str, Procedure],
                  surgeons: Dict[str, Surgeon],
                  rooms: Dict[str, Room]) -> Agenda:
    """
    Construye una agenda quirurgica factible a partir de un cromosoma.

    Parametros
    ----------
    chromosome : dict {Block: specialty_id}
        Asignacion de especialidad medica para cada bloque (dia, quirofano).
    patients : lista de pacientes pendientes (se asume p.scheduled = False
        antes de invocar esta funcion).
    procedures, surgeons, rooms : diccionarios indexados por id.
    """
    # Reiniciar estado de "programado" antes de decodificar este individuo
    for p in patients:
        p.scheduled = False

    assignments: Dict[Block, List[ScheduledSurgery]] = {b: [] for b in chromosome}
    used_time: Dict[Block, int] = {b: 0 for b in chromosome}

    # Agrupar pacientes pendientes por especialidad (conjunto P_E)
    patients_by_specialty: Dict[str, List[Patient]] = {}
    for p in patients:
        patients_by_specialty.setdefault(p.specialty_id, []).append(p)

    for block, specialty_id in chromosome.items():
        room = rooms[block.room_id]
        capacity = room.daily_capacity_minutes
        candidates = patients_by_specialty.get(specialty_id, [])

        # --- Conjunto de pacientes factibles P_F,d,q ---
        feasible = []
        for p in candidates:
            if p.scheduled:
                continue
            surgeon = surgeons.get(p.surgeon_id)
            if surgeon is None or block.day not in surgeon.available_days:
                continue  # cirujano no disponible ese dia
            procedure = procedures.get(p.procedure_id)
            if procedure is None or procedure.required_room_type > room.room_type:
                continue  # complejidad de sala insuficiente
            feasible.append(p)

        # --- Priorizacion (mayor prioridad clinica primero) ---
        feasible.sort(key=lambda p: p.clinical_priority, reverse=True)

        # --- Asignacion secuencial respetando la capacidad del bloque ---
        remaining_capacity = capacity
        for p in feasible:
            if p.estimated_duration <= remaining_capacity:
                assignments[block].append(
                    ScheduledSurgery(patient_id=p.id, block=block, duration=p.estimated_duration)
                )
                remaining_capacity -= p.estimated_duration
                p.scheduled = True
            # si no entra, el paciente queda pendiente para futuras programaciones

        used_time[block] = capacity - remaining_capacity

    return Agenda(assignments=assignments, used_time=used_time)
