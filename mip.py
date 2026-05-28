"""
mip_slots.py — MIP con slots de tiempo discretos.

Reemplaza mip.py + sequence.py. El MIP asigna cada paciente a un
slot de inicio exacto dentro del bloque, produciendo un cronograma
completo sin necesidad de secuenciación posterior.

Slot: unidad mínima de tiempo configurable (default 15 min).
  Bloque mañana  08:00-12:00 → 16 slots de 15 min (slots 0..15)
  Bloque tarde   13:00-17:00 → 16 slots de 15 min (slots 0..15)

Variable principal:
    x[p_id, s_id, q, t] = 1 si el paciente p es operado por el
    cirujano s en el OR q comenzando en el slot t.

Ventajas sobre el MIP continuo + secuenciador:
  • Sin brecha entre asignación y cronograma.
  • El orden es parte de la solución óptima.
  • Se elimina sequence.py por completo.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Any, Tuple

from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, PULP_CBC_CMD


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de conversión minutos ↔ slots
# ─────────────────────────────────────────────────────────────────────────────

def _min_to_slot(minutes: int, block_start: int, slot_size: int) -> int:
    """Convierte minutos absolutos a índice de slot (redondeado hacia arriba)."""
    return math.ceil((minutes - block_start) / slot_size)


def _slot_to_min(slot: int, block_start: int, slot_size: int) -> int:
    """Convierte índice de slot a minutos absolutos."""
    return block_start + slot * slot_size


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _dur_slots(duration_min: int, slot_size: int) -> int:
    """Duración en slots (redondeo hacia arriba)."""
    return math.ceil(duration_min / slot_size)


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def solve_mip_for_shift(
    blocks:      List[Dict],
    day_idx:     int,
    is_morning:  bool,
    alpha:       float = 0.7,
    beta:        float = 0.3,
    delta:       float = 5.0,
    slot_size:   int   = 15,     # minutos por slot
) -> Dict[str, Any]:
    """
    Parámetros
    ----------
    blocks : lista de dicts, uno por quirófano en el turno:
        {
            "or_idx"   : int,
            "spec_id"  : int,
            "patients" : List[Patient],
            "surgeons" : List[Staff],
            "t_max"    : int,  # minutos totales del bloque
        }
    slot_size : granularidad en minutos (default 15).

    Retorna
    -------
    {
        "fitness"          : float,
        "all_pacientes_ids": List[int],
        "per_or": {
            or_idx: {
                "pacientes_ids"         : List[int],
                "asignaciones"          : List[{p, doc, slot_inicio,
                                                hora_inicio, hora_fin, duracion}],
                "t_max"                 : int,
                "uso_tiempo"            : int,
                "utilizacion_porcentaje": float,
            }
        }
    }
    """
    block_start = 480 if is_morning else 780   # minutos desde medianoche
    # Usar el t_max del primer bloque con tiempo asignado.
    # blocks[0] puede ser un OR libre (t_max=0), lo que daría n_slots=0
    # y mataría todo el turno silenciosamente.
    _active_t_max = next((b["t_max"] for b in blocks if b["t_max"] > 0), 0)
    n_slots       = _active_t_max // slot_size if _active_t_max else 0

    # ── 1. Filtrar bloques activos ────────────────────────────────────────
    active = [
        b for b in blocks
        if b["spec_id"] > 0 and b["surgeons"] and b["patients"] and b["t_max"] > 0
    ]
    empty_per_or = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}

    if not active or n_slots == 0:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # ── 2. Pre-calcular ventanas de cada cirujano en slots ────────────────
    # surgeon_window[s_id] = (start_slot, end_slot) dentro del bloque
    # start_slot: primer slot en que el cirujano puede empezar
    # end_slot:   último slot en que puede TERMINAR (exclusivo)
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

    # ── 3. Calcular slots de duración por paciente ─────────────────────────
    dur_s: Dict[int, int] = {}
    for b in active:
        for p in b["patients"]:
            if p.id not in dur_s:
                dur_s[p.id] = _dur_slots(p.estimated_duration, slot_size)

    # ── 4. Problema ───────────────────────────────────────────────────────
    label = f"MIPSlots_{day_idx}_{'M' if is_morning else 'T'}"
    prob  = LpProblem(label, LpMaximize)

    # ── 5. Variables x[p_id, s_id, q, t] ─────────────────────────────────
    # Solo creamos la variable si la asignación es factible:
    #   • El cirujano tiene ventana > 0 en este bloque.
    #   • La cirugía cabe dentro de la ventana del cirujano.
    #   • El slot t es válido para ese cirujano.
    x: Dict[Tuple[int, int, int, int], LpVariable] = {}

    for b in active:
        q = b["or_idx"]
        for p in b["patients"]:
            dp = dur_s[p.id]
            forced = getattr(p, "forced_surgeon_id", None)

            for s in b["surgeons"]:
                # Respetar médico forzado
                if forced is not None and s.id != forced:
                    continue

                ws, we = surgeon_window.get(s.id, (0, 0))
                if we <= ws:
                    continue

                # t puede ir de ws hasta el último slot donde la cirugía termina a tiempo
                max_t = min(we - dp, n_slots - dp)
                for t in range(ws, max_t + 1):
                    x[(p.id, s.id, q, t)] = LpVariable(
                        f"x_p{p.id}_s{s.id}_q{q}_t{t}", cat="Binary"
                    )

    if not x:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # Variables de dispersión
    # c[s_id, q] = 1 si el cirujano s opera en OR q
    all_sids = {s.id for b in active for s in b["surgeons"]}
    all_qidx = {b["or_idx"] for b in active}

    c: Dict[Tuple[int, int], LpVariable] = {}
    for s_id in all_sids:
        for q in all_qidx:
            c[(s_id, q)] = LpVariable(f"c_s{s_id}_q{q}", cat="Binary")

    # y[s_id] = 1 si el cirujano opera en más de un OR
    y: Dict[int, LpVariable] = {
        s_id: LpVariable(f"y_s{s_id}", cat="Binary")
        for s_id in all_sids
    }

    # ── 6. Objetivo ───────────────────────────────────────────────────────
    t_max_total = sum(b["t_max"] for b in active)
    x_items = list(x.items())

    patient_priority: Dict[int, float] = {}
    patient_duration: Dict[int, int] = {}
    for b in active:
        for p in b["patients"]:
            patient_priority[p.id] = p.clinical_priority
            patient_duration[p.id] = p.estimated_duration

    x_by_patient: Dict[int, List[LpVariable]] = defaultdict(list)
    x_by_sid_q: Dict[Tuple[int, int], List[LpVariable]] = defaultdict(list)

    for (pid, sid, q, _), var in x_items:
        x_by_patient[pid].append(var)
        x_by_sid_q[(sid, q)].append(var)

    obj_prio = lpSum(patient_priority[pid] * var for (pid, _, _, _), var in x_items)

    obj_util = lpSum(patient_duration[pid] * var for (pid, _, _, _), var in x_items) / t_max_total

    obj_disp = lpSum(y[s_id] for s_id in all_sids)

    prob += (alpha * obj_prio) + (beta * obj_util) - (delta * obj_disp)

    # ── 7. Restricciones ─────────────────────────────────────────────────

    # R1: cada paciente se opera como máximo una vez en todo el turno
    all_pids = {p.id for b in active for p in b["patients"]}
    for p_id in all_pids:
        terms = x_by_patient.get(p_id, [])
        if terms:
            prob += lpSum(terms) <= 1

    # R2: no-solapamiento en cada OR
    # En cada slot k, la suma de cirugías "activas" en el OR q es ≤ 1.
    # Una cirugía (p,s,q,t) está activa en el slot k si t <= k < t+dur_s[p].
    x_by_or_time: Dict[Tuple[int, int], List[LpVariable]] = defaultdict(list)
    x_by_surgeon_time: Dict[Tuple[int, int], List[LpVariable]] = defaultdict(list)
    for (pid, sid, q, t), var in x.items():
        end_t = min(t + dur_s[pid], n_slots)
        for k in range(t, end_t):
            x_by_or_time[(q, k)].append(var)
            x_by_surgeon_time[(sid, k)].append(var)

    for b in active:
        q = b["or_idx"]
        for k in range(n_slots):
            terms = x_by_or_time.get((q, k), [])
            if terms:
                prob += lpSum(terms) <= 1

    # R3: no-solapamiento por cirujano (no puede estar en dos ORs a la vez)
    # En cada slot k, el cirujano s tiene como máximo una cirugía activa.
    for s_id in all_sids:
        for k in range(n_slots):
            terms = x_by_surgeon_time.get((s_id, k), [])
            if terms:
                prob += lpSum(terms) <= 1

    # R4: médico forzado ya se aplicó al crear variables (solo se crean
    # variables para el médico correcto), no necesita restricción extra.

    # R5: vinculación c[s,q] con asignaciones x
    for s_id in all_sids:
        for q in all_qidx:
            terms_sq = x_by_sid_q.get((s_id, q), [])
            if not terms_sq:
                continue
            n_sq = len(terms_sq)
            # c=0 si nadie asignado
            prob += lpSum(terms_sq) >= c[(s_id, q)]
            # c=1 si alguien asignado
            prob += n_sq * c[(s_id, q)] >= lpSum(terms_sq)

    # R6: vinculación y[s] con c[s,q]
    # y=1 cuando el cirujano opera en 2+ ORs
    for s_id in all_sids:
        c_vars = [c[(s_id, q)] for q in all_qidx if (s_id, q) in c]
        if len(c_vars) > 1:
            prob += lpSum(c_vars) - 1 <= (len(c_vars) - 1) * y[s_id]

    # ── 8. Resolver ───────────────────────────────────────────────────────
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=10))

    # ── 9. Extraer cronograma ─────────────────────────────────────────────
    z_final  = value(prob.objective) or 0.0
    all_ids: List[int] = []
    per_or   = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}

    # Índice inverso: paciente → objeto Patient
    pac_map = {p.id: p for b in active for p in b["patients"]}
    # Índice inverso: staff_id → nombre
    staff_name = {s.id: s.name for b in active for s in b["surgeons"]}

    for b in active:
        q    = b["or_idx"]
        ids_or = []
        uso   = 0

        # Recolectar asignaciones activas y ordenarlas por slot de inicio
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

        # Ordenar por slot de inicio para el cronograma
        scheduled.sort(key=lambda a: a["slot_inicio"])

        per_or[q] = {
            "or_idx"                : q,
            "pacientes_ids"         : ids_or,
            "asignaciones"          : scheduled,
            "t_max"                 : b["t_max"],
            "uso_tiempo"            : uso,
            "utilizacion_porcentaje": round((uso / b["t_max"]) * 100, 2)
                                      if b["t_max"] > 0 else 0.0,
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