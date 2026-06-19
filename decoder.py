"""
decoder.py — Decodificador Heurístico de Inserción
"""
from typing import Dict, List, Any
import math

def format_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

def is_overlapping(new_start: int, new_end: int, schedule: List[tuple]) -> bool:
    """Verifica si un intervalo de tiempo choca con la agenda existente."""
    for s, e in schedule:
        if new_start < e and new_end > s:
            return True
    return False

def build_shift_schedule(
    blocks: List[Dict], day_idx: int, is_morning: bool, slot_size: int = 15
) -> Dict[str, Any]:
    """
    Decodificador Goloso (Greedy Decoder) con Densidad de Valor y Turnaround Time.
    """
    block_start = 480 if is_morning else 780
    TIEMPO_LIMPIEZA = 30  # Buffer obligatorio entre cirugías

    active_blocks = [
        b for b in blocks if b["spec_id"] > 0 and b["surgeons"] and b["patients"]
    ]
    per_or = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}

    if not active_blocks:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": per_or}

    reloj_quirofano = {b["or_idx"]: 0 for b in active_blocks}
    agenda_cirujanos = {s.id: [] for b in active_blocks for s in b["surgeons"]}
    minutos_asignados_cirujano = {s.id: 0 for b in active_blocks for s in b["surgeons"]}

    # Consolidar y ordenar: Densidad de Valor (Prioridad / Duración)
    todos_los_pacientes = []
    paciente_block_map = {}
    for b in active_blocks:
        for p in b["patients"]:
            todos_los_pacientes.append(p)
            paciente_block_map[p.id] = b

    todos_los_pacientes.sort(
        key=lambda x: (x.clinical_priority / max(x.estimated_duration, 1)), 
        reverse=True
    )

    fitness_acumulado = 0.0
    all_pacientes_ids = []

    for p in todos_los_pacientes:
        b = paciente_block_map[p.id]
        q_idx = b["or_idx"]
        duracion = p.estimated_duration
        duracion_ajustada = math.ceil(duracion / slot_size) * slot_size
        
        asignado = False
        cirujanos_candidatos = [
            s for s in b["surgeons"]
            if p.procedure_id in s.enabled_procedures_ids
            and (p.forced_surgeon_id is None or p.forced_surgeon_id == s.id)
        ]

        for s in cirujanos_candidatos:
            if asignado: break

            # LÓGICA DE TIEMPO CON BUFFER (Turnaround Time)
            # Si el quirófano ya tiene asignaciones, sumamos el buffer de limpieza
            tiempo_base = reloj_quirofano[q_idx]
            minuto_inicio_propuesto = tiempo_base + (TIEMPO_LIMPIEZA if per_or[q_idx]["pacientes_ids"] else 0)
            minuto_fin_propuesto = minuto_inicio_propuesto + duracion_ajustada

            # REGLAS DE VALIDACIÓN
            if minuto_fin_propuesto > b["t_max"]: continue
            
            contrato_max = s.get_available_minutes_in_block(day_idx, is_morning, b["t_max"])
            if minutos_asignados_cirujano[s.id] + duracion_ajustada > contrato_max: continue
                
            if is_overlapping(minuto_inicio_propuesto, minuto_fin_propuesto, agenda_cirujanos[s.id]):
                continue

            # ASIGNACIÓN EXITOSA
            reloj_quirofano[q_idx] = minuto_fin_propuesto
            agenda_cirujanos[s.id].append((minuto_inicio_propuesto, minuto_fin_propuesto))
            minutos_asignados_cirujano[s.id] += duracion_ajustada
            
            per_or[q_idx]["asignaciones"].append({
                "p": p.id,
                "doc": s.name,
                "hora_inicio": format_time(block_start + minuto_inicio_propuesto),
                "hora_fin": format_time(block_start + minuto_fin_propuesto),
                "duracion": duracion
            })
            per_or[q_idx]["pacientes_ids"].append(p.id)
            per_or[q_idx]["uso_tiempo"] += duracion_ajustada
            per_or[q_idx]["t_max"] = b["t_max"]
            
            all_pacientes_ids.append(p.id)
            fitness_acumulado += p.clinical_priority
            asignado = True

    # Cálculo final de utilización
    for q_idx in per_or:
        if per_or[q_idx]["t_max"] > 0:
            per_or[q_idx]["utilizacion_porcentaje"] = round(
                (per_or[q_idx]["uso_tiempo"] / per_or[q_idx]["t_max"]) * 100, 2
            )

    return {"fitness": fitness_acumulado, "all_pacientes_ids": all_pacientes_ids, "per_or": per_or}

def _empty_or(or_idx: int) -> Dict:
    return {"or_idx": or_idx, "pacientes_ids": [], "asignaciones": [], "t_max": 0, "uso_tiempo": 0, "utilizacion_porcentaje": 0.0}