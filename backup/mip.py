from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, PULP_CBC_CMD
from typing import List, Dict, Any

def solve_mip_for_shift(
    blocks: List[Dict],
    day_idx: int,
    is_morning: bool,
    alpha: float = 0.7,
    beta: float = 0.3,
    delta: float = 5,
    slot_size: int = 30
) -> Dict[str, Any]:
    # 1. Filtrar bloques con cirujanos y pacientes[cite: 1]
    active = [b for b in blocks if b["spec_id"] > 0 and b["surgeons"] and b["patients"] and b["t_max"] > 0]
    empty_per_or = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}
    if not active:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # 2. Configuración de tiempo[cite: 2, 5]
    start_shift = 480 if is_morning else 780
    t_max_turno = max(b["t_max"] for b in active)
    slots = list(range(start_shift, start_shift + t_max_turno, slot_size))

    prob = LpProblem(f"MIP_Slots_{day_idx}_{'M' if is_morning else 'T'}", LpMaximize)

    # 3. Variables de Decisión[cite: 1]
    x = {}
    for b in active:
        q = b["or_idx"]
        for p in b["patients"]:
            for s in b["surgeons"]:
                s_start, s_end = s.get_range_for_block(day_idx, is_morning)
                for t in slots:
                    # La cirugía debe terminar antes del fin del turno del médico
                    if t >= s_start and (t + p.estimated_duration) <= s_end:
                        x[(p.id, s.id, q, t)] = LpVariable(f"x_p{p.id}_s{s.id}_q{q}_t{t}", cat="Binary")

    if not x:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": empty_per_or}

    # Variables de dispersión solo para cirujanos con variables x factibles[cite: 1]
    s_ids_active = {k[1] for k in x.keys()}
    sid_q_pairs = {(sid, q) for (_, sid, q, _) in x.keys()}
    c = {(sid, q): LpVariable(f"c_s{sid}_q{q}", cat="Binary") for (sid, q) in sid_q_pairs}
    y = {sid: LpVariable(f"y_s{sid}", cat="Binary") for sid in s_ids_active}

    # Precalculate patient data to avoid repeated lookups
    patient_duration: Dict[int, int] = {}
    patient_priority: Dict[int, float] = {}
    patient_by_id: Dict[int, Any] = {}
    surgeon_by_id: Dict[int, Any] = {}
    for b in active:
        for p in b["patients"]:
            patient_duration[p.id] = p.estimated_duration
            patient_priority[p.id] = p.clinical_priority
            patient_by_id[p.id] = p
        for s in b["surgeons"]:
            surgeon_by_id[s.id] = s

    # 4. Función Objetivo (optimized)
    obj_prio = lpSum(patient_priority[k[0]] * x[k] for k in x.keys())
    obj_util = (lpSum(patient_duration[k[0]] * x[k] for k in x.keys())) / (t_max_turno * len(active)) if (t_max_turno * len(active)) > 0 else 0
    prob += (alpha * obj_prio) + (beta * obj_util) - (delta * lpSum(y.values()))

    # 5. Restricciones[cite: 1, 2]
    # R1: Un solo procedimiento por paciente
    for p_id in {k[0] for k in x.keys()}:
        prob += lpSum(v for k, v in x.items() if k[0] == p_id) <= 1

    # R2: No solapamiento en Quirófano[cite: 1]
    x_by_or_time: Dict[tuple, list] = {}
    for (p_id, s_id, or_idx, t_start), var in x.items():
        p_duration = patient_duration[p_id]
        for slot in slots:
            if t_start <= slot < t_start + p_duration:
                key = (or_idx, slot)
                if key not in x_by_or_time:
                    x_by_or_time[key] = []
                x_by_or_time[key].append(var)
    
    for conflicting_vars in x_by_or_time.values():
        if conflicting_vars:
            prob += lpSum(conflicting_vars) <= 1

    # R3: No solapamiento por Médico[cite: 1]
    x_by_surgeon_time: Dict[tuple, list] = {}
    for (p_id, s_id, or_idx, t_start), var in x.items():
        p_duration = patient_duration[p_id]
        for slot in slots:
            if t_start <= slot < t_start + p_duration:
                key = (s_id, slot)
                if key not in x_by_surgeon_time:
                    x_by_surgeon_time[key] = []
                x_by_surgeon_time[key].append(var)
    
    for conflicting_vars in x_by_surgeon_time.values():
        if conflicting_vars:
            prob += lpSum(conflicting_vars) <= 1

    # R4: Lógica de Dispersión[cite: 1]
    for sid in s_ids_active:
        qs_sid = {k[2] for k in x.keys() if k[1] == sid}
        for q_idx in qs_sid:
            asigs = [v for k, v in x.items() if k[1] == sid and k[2] == q_idx]
            if asigs:
                prob += lpSum(asigs) >= c[(sid, q_idx)]
                prob += len(asigs) * c[(sid, q_idx)] >= lpSum(asigs)
        
        c_vars = [c[(sid, q_idx)] for q_idx in qs_sid]
        if len(c_vars) > 1:
            prob += lpSum(c_vars) - 1 <= (len(c_vars) - 1) * y[sid]

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=0.5))

    # 6. Resultados
    res_per_or = {b["or_idx"]: _empty_or(b["or_idx"]) for b in blocks}
    all_pids = []
    if value(prob.objective):
        for (pid, sid, q, t), var in x.items():
            if value(var) > 0.5:
                p = patient_by_id[pid]
                s_obj = surgeon_by_id[sid]
                t_fin = t + p.estimated_duration
                all_pids.append(pid)
                res_per_or[q]["pacientes_ids"].append(pid)
                res_per_or[q]["uso_tiempo"] += p.estimated_duration
                res_per_or[q]["asignaciones"].append({
                    "p": pid,
                    "doc": s_obj.name,
                    "t_inicio": t,
                    "t_fin": t_fin,
                    "slot_inicio": t // slot_size,
                    "hora_inicio": f"{t // 60:02d}:{t % 60:02d}",
                    "hora_fin": f"{t_fin // 60:02d}:{t_fin % 60:02d}",
                    "duracion": p.estimated_duration,
                })

    for q, data in res_per_or.items():
        t_m = next((b["t_max"] for b in blocks if b["or_idx"] == q), 0)
        data["t_max"] = t_m
        data["utilizacion_porcentaje"] = round((data["uso_tiempo"] / t_m * 100), 2) if t_m > 0 else 0
    
    return {"fitness": value(prob.objective) or 0.0, "all_pacientes_ids": all_pids, "per_or": res_per_or}

def _empty_or(idx):
    return {"or_idx": idx, "pacientes_ids": [], "asignaciones": [], "uso_tiempo": 0, "utilizacion_porcentaje": 0.0}