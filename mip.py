"""
mip.py — MIP con slots de tiempo discretos unificado por turno completo.
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, PULP_CBC_CMD

def _min_to_slot(minutes: int, block_start: int, slot_size: int) -> int:
    return math.ceil((minutes - block_start) / slot_size)

def _slot_to_min(slot: int, block_start: int, slot_size: int) -> int:
    return block_start + slot * slot_size

def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

def _dur_slots(duration_min: int, slot_size: int) -> int:
    return math.ceil(duration_min / slot_size)


def solve_mip_for_shift(
    blocks:      List[Dict],
    day_idx:     int,
    is_morning:  bool,
    alpha:       float = 0.7,
    beta:        float = 0.3,
    slot_size:   int   = 15,     
) -> Dict[str, Any]:
    """
    Resuelve la ventana operativa completa para UN turno (d, t), consolidando todos los ORs.
    """
    block_start = 480 if is_morning else 780   
    _active_t_max = next((b["t_max"] for b in blocks if b["t_max"] > 0), 0)
    n_slots       = _active_t_max // slot_size if _active_t_max else 0

    # 1. Filtrar bloques activos asignados por el AG en este turno
    active = [
        b for b in blocks
        if b["spec_id"] > 0 and b["surgeons"] and b["patients"] and b["t_max"] > 0
    ]
    empty_per_or = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}

    if not active or n_slots == 0:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # 2. Pre-calcular ventanas de cada cirujano disponible en el turno
    surgeon_window: Dict[int, Tuple[int, int]] = {}
    for b in active:
        for s in b["surgeons"]:
            if s.id in surgeon_window:
                continue
            s_start, s_end = s.get_range_for_block(day_idx, is_morning)
            if s_start == s_end == 0:
                surgeon_window[s.id] = (0, 0)
            else:
                ws = max(0, _min_to_slot(s_start, block_start, slot_size))
                we = min(n_slots, _min_to_slot(s_end, block_start, slot_size))
                surgeon_window[s.id] = (ws, we)

    # 3. Duración en slots por paciente
    dur_s: Dict[int, int] = {}
    for b in active:
        for p in b["patients"]:
            if p.id not in dur_s:
                dur_s[p.id] = _dur_slots(p.estimated_duration, slot_size)

    # 4. Instanciar el Problema Lineal
    label = f"MIPSlots_Turno_{day_idx}_{'M' if is_morning else 'T'}"
    prob  = LpProblem(label, LpMaximize)

    # 5. Variables de Decisión x[p_id, s_id, q, k] (k = slot de inicio)
    x: Dict[Tuple[int, int, int, int], LpVariable] = {}

    for b in active:
        q = b["or_idx"]
        for p in b["patients"]:
            dp = dur_s[p.id]
            forced = getattr(p, "forced_surgeon_id", None)

            for s in b["surgeons"]:
                if forced is not None and s.id != forced:
                    continue

                ws, we = surgeon_window.get(s.id, (0, 0))
                if we <= ws:
                    continue

                max_k = min(we - dp, n_slots - dp)
                for k in range(ws, max_k + 1):
                    x[(p.id, s.id, q, k)] = LpVariable(
                        f"x_p{p.id}_s{s.id}_q{q}_k{k}", cat="Binary"
                    )

    if not x:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # 6. Función Objetivo Limpia (Sin términos de dispersión espacial)
    t_max_total = sum(b["t_max"] for b in active)
    x_items = list(x.items())

    patient_priority: Dict[int, float] = {}
    patient_duration: Dict[int, int] = {}
    for b in active:
        for p in b["patients"]:
            patient_priority[p.id] = p.clinical_priority
            patient_duration[p.id] = p.estimated_duration

    x_by_patient: Dict[int, List[LpVariable]] = defaultdict(list)
    for (pid, sid, q, _), var in x_items:
        x_by_patient[pid].append(var)

    obj_prio = lpSum(patient_priority[pid] * var for (pid, _, _, _), var in x_items)
    obj_util = lpSum(patient_duration[pid] * var for (pid, _, _, _), var in x_items) / t_max_total

    # Maximización pura de Salud + Eficiencia Hospitalaria
    prob += (alpha * obj_prio) + (beta * obj_util)

    # 7. Restricciones del Modelo

    # R1 — Unicidad: Cada paciente se opera como máximo una vez en el turno
    all_pids = {p.id for b in active for p in b["patients"]}
    for p_id in all_pids:
        terms = x_by_patient.get(p_id, [])
        if terms:
            prob += lpSum(terms) <= 1

    # Mapeos inversos para indexación de slots activos K_p,k'
    x_by_or_slot: Dict[Tuple[int, int], List[LpVariable]] = defaultdict(list)
    x_by_surgeon_slot: Dict[Tuple[int, int], List[LpVariable]] = defaultdict(list)
    
    for (pid, sid, q, start_k), var in x.items():
        end_k = min(start_k + dur_s[pid], n_slots)
        for k in range(start_k, end_k):
            x_by_or_slot[(q, k)].append(var)
            x_by_surgeon_slot[(sid, k)].append(var)

    # R2 — No solapamiento en Quirófanos (Capacidad de Sala)
    for b in active:
        q = b["or_idx"]
        for k in range(n_slots):
            terms = x_by_or_slot.get((q, k), [])
            if terms:
                prob += lpSum(terms) <= 1

    # R3 — Sincronización del Staff: Un cirujano no puede duplicarse en el mismo slot k
    all_sids = {s.id for b in active for s in b["surgeons"]}
    for s_id in all_sids:
        for k in range(n_slots):
            terms = x_by_surgeon_slot.get((s_id, k), [])
            if terms:
                prob += lpSum(terms) <= 1

    # 8. Resolver
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10))

    # 9. Extraer Cronograma Detallado
    z_final  = value(prob.objective) or 0.0
    all_ids: List[int] = []
    per_or   = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}

    pac_map = {p.id: p for b in active for p in b["patients"]}
    staff_name = {s.id: s.name for b in active for s in b["surgeons"]}

    for b in active:
        q = b["or_idx"]
        ids_or = []
        uso = 0
        scheduled = []
        
        for (pid, sid, bq, t), v in x_items:
            if bq == q and (value(v) or 0) > 0.5:
                p_obj = pac_map[pid]
                inicio_min = _slot_to_min(t, block_start, slot_size)
                fin_min    = inicio_min + p_obj.estimated_duration
                scheduled.append({
                    "p"          : pid,
                    "doc"        : staff_name[sid],
                    "slot_inicio": t,
                    "hora_inicio": _fmt(inicio_min),
                    "hora_fin"   : _fmt(fin_min),
                    "duracion"   : p_obj.estimated_duration,
                })
                ids_or.append(pid)
                all_ids.append(pid)
                uso += p_obj.estimated_duration

        scheduled.sort(key=lambda a: a["slot_inicio"])

        per_or[q] = {
            "or_idx"                : q,
            "pacientes_ids"         : ids_or,
            "asignaciones"          : scheduled,
            "t_max"                 : b["t_max"],
            "uso_tiempo"            : uso,
            "utilizacion_porcentaje": round((uso / b["t_max"]) * 100, 2) if b["t_max"] > 0 else 0.0,
        }

    return {"fitness": z_final, "all_pacientes_ids": all_ids, "per_or": per_or}

def _empty_or(or_idx: int) -> Dict:
    return {
        "or_idx"                : or_idx,
        "pacientes_ids"         : [],
        "asignaciones"          : [],
        "t_max"                 : 0,
        "uso_tiempo"            : 0,
        "utilizacion_porcentaje": 0.0,
    }