"""
mip.py — Nivel 3: MIP por turno completo.

solve_mip_for_shift resuelve un único problema de optimización por turno
que cubre todos los quirófanos activos. La restricción de capacidad del
cirujano aplica sobre la suma de todos los ORs del turno, eliminando la
necesidad de rastrear un monedero externo.
"""
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, PULP_CBC_CMD
from typing import List, Dict, Any


def solve_mip_for_shift(
    blocks: List[Dict],
    day_idx: int,
    is_morning: bool,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> Dict[str, Any]:
    """
    Parámetros
    ----------
    blocks : lista de dicts, uno por quirófano en el turno:
        {
            "or_idx"   : int,
            "spec_id"  : int,
            "patients" : List[Patient],
            "surgeons" : List[Staff],   # ya filtrados: specialty_id correcto y cap > 0
            "t_max"    : int,           # minutos físicos del quirófano en este turno
        }

    Retorna
    -------
    {
        "fitness"          : float,
        "all_pacientes_ids": List[int],
        "per_or"           : { or_idx: { pacientes_ids, asignaciones,
                                          consumo_medicos, t_max,
                                          uso_tiempo, utilizacion_porcentaje } }
    }
    """
    # ── 1. Filtrar bloques activos ────────────────────────────────────────
    active = [
        b for b in blocks
        if b["spec_id"] > 0 and b["surgeons"] and b["patients"] and b["t_max"] > 0
    ]

    empty_per_or = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}

    if not active:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # ── 2. Capacidad por cirujano (global sobre el turno completo) ─────────
    surgeon_cap: Dict[int, int] = {}
    for b in active:
        for s in b["surgeons"]:
            if s.id not in surgeon_cap:
                surgeon_cap[s.id] = s.get_available_minutes_in_block(day_idx, is_morning)

    t_max_total = sum(b["t_max"] for b in active)

    # ── 3. Problema ───────────────────────────────────────────────────────
    label = f"MIP_shift_{day_idx}_{'M' if is_morning else 'T'}"
    prob  = LpProblem(label, LpMaximize)

    # ── 4. Variables: x[p_id, s_id, q] ───────────────────────────────────
    x: Dict = {}
    for b in active:
        q = b["or_idx"]
        for p in b["patients"]:
            for s in b["surgeons"]:
                x[(p.id, s.id, q)] = LpVariable(f"x_p{p.id}_s{s.id}_q{q}", cat="Binary")

    if not x:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # ── 5. Objetivo ───────────────────────────────────────────────────────
    all_combos = [(p, s, b["or_idx"]) for b in active for p in b["patients"] for s in b["surgeons"]]

    obj_prio = lpSum(p.clinical_priority  * x[(p.id, s.id, q)] for p, s, q in all_combos)
    obj_util = lpSum(p.estimated_duration * x[(p.id, s.id, q)] for p, s, q in all_combos) / t_max_total

    prob += (alpha * obj_prio) + (beta * obj_util)

    # ── 6. Restricciones ─────────────────────────────────────────────────

    # R1: cada paciente se opera como máximo una vez en el turno (en cualquier OR)
    all_patient_ids = {p.id for b in active for p in b["patients"]}
    for p_id in all_patient_ids:
        terms = [v for (pid, sid, q), v in x.items() if pid == p_id]
        if terms:
            prob += lpSum(terms) <= 1

    # R2: cada cirujano no supera su capacidad total en el turno
    #     la restricción cruza todos los ORs donde ese cirujano aparece
    for s_id, cap in surgeon_cap.items():
        terms = [
            p.estimated_duration * x[(p.id, s_id, b["or_idx"])]
            for b in active
            for p in b["patients"]
            if (p.id, s_id, b["or_idx"]) in x
        ]
        if terms:
            prob += lpSum(terms) <= cap

    # R3: cada quirófano no supera su capacidad física
    for b in active:
        q     = b["or_idx"]
        terms = [
            p.estimated_duration * x[(p.id, s.id, q)]
            for p in b["patients"]
            for s in b["surgeons"]
        ]
        if terms:
            prob += lpSum(terms) <= b["t_max"]

    # R4: modelo híbrido — si el paciente tiene médico forzado, nadie más lo opera
    for b in active:
        q = b["or_idx"]
        for p in b["patients"]:
            forced = getattr(p, "forced_surgeon_id", None)
            if forced is not None:
                for s in b["surgeons"]:
                    if s.id != forced and (p.id, s.id, q) in x:
                        prob += x[(p.id, s.id, q)] == 0

    # ── 7. Resolver ───────────────────────────────────────────────────────
    prob.solve(PULP_CBC_CMD(msg=0))

    # ── 8. Procesar resultados ────────────────────────────────────────────
    z_final = value(prob.objective) or 0.0
    all_ids: List[int] = []
    per_or  = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}

    for b in active:
        q       = b["or_idx"]
        consumo = {s.id: 0 for s in b["surgeons"]}
        asigs   = []
        ids_or  = []
        uso     = 0

        for p in b["patients"]:
            for s in b["surgeons"]:
                key = (p.id, s.id, q)
                if key in x and (value(x[key]) or 0) > 0.5:
                    consumo[s.id] += p.estimated_duration
                    uso           += p.estimated_duration
                    ids_or.append(p.id)
                    all_ids.append(p.id)
                    asigs.append({"p": p.id, "doc": s.name})

        per_or[q] = {
            "or_idx"                : q,
            "pacientes_ids"         : ids_or,
            "asignaciones"          : asigs,
            "consumo_medicos"       : consumo,
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
        "consumo_medicos"       : {},
        "t_max"                 : 0,
        "uso_tiempo"            : 0,
        "utilizacion_porcentaje": 0.0,
    }