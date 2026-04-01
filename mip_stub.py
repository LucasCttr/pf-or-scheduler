"""
mip_stub.py — Stub del modelo MIP (Nivel 3).

Devuelve un valor de fitness Z simulado para cada bloque quirúrgico.
Reemplazar esta función con la implementación real usando PuLP, OR-Tools o Gurobi.

Estructura del MIP real:
    Max Z = sum(alpha * w_i * x_i) + beta * (sum(d_i * x_i) / T_max)
    s.t.   sum(d_i * x_i) <= T_max
           x_i in {0, 1}

Donde:
    w_i  = prioridad clínica del paciente i
    d_i  = duración estimada de la cirugía i (minutos)
    x_i  = 1 si el paciente i es seleccionado, 0 si no
    T_max = tiempo total disponible en el bloque (minutos)
"""
from typing import List
from models import Patient


def solve_mip_for_block(
    specialty_id: int,
    patients: List[Patient],
    block_duration_min: int,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """
    Resuelve el MIP para un bloque y retorna el valor Z.

    Parámetros
    ----------
    specialty_id       : ID de la especialidad asignada al bloque por el AG
    patients           : Lista de pacientes elegibles P' (ya filtrados por Nivel 2)
    block_duration_min : Tiempo disponible del bloque en minutos
    alpha, beta        : Pesos; alpha > beta para priorizar urgencia sobre utilización

    Retorna
    -------
    float : Valor Z de la función objetivo del MIP

    ---
    TODO: Reemplazar la heurística greedy por el solver exacto.

    Ejemplo con PuLP:
        from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, PULP_CBC_CMD
        prob = LpProblem("block_mip", LpMaximize)
        x = {p.id: LpVariable(f"x_{p.id}", cat="Binary") for p in patients}
        prob += (alpha * lpSum(p.clinical_priority * x[p.id] for p in patients)
                 + beta * lpSum(p.estimated_duration * x[p.id] for p in patients) / block_duration_min)
        prob += lpSum(p.estimated_duration * x[p.id] for p in patients) <= block_duration_min
        prob.solve(PULP_CBC_CMD(msg=0))
        return value(prob.objective)
    """
    if specialty_id == 0 or not patients:
        return 0.0

    # ── Heurística greedy como placeholder ─────────────────────────────────
    # Ordena pacientes por prioridad clínica descendente y los va seleccionando
    # hasta agotar el tiempo del bloque. No garantiza óptimo, pero es válido
    # para evaluar el AG mientras el MIP real no esté integrado.

    sorted_patients = sorted(patients, key=lambda p: p.clinical_priority, reverse=True)

    time_used = 0
    priority_sum = 0.0

    for p in sorted_patients:
        if time_used + p.estimated_duration <= block_duration_min:
            time_used += p.estimated_duration
            priority_sum += p.clinical_priority

    utilization = time_used / block_duration_min if block_duration_min > 0 else 0.0
    z = alpha * priority_sum + beta * utilization

    return z
