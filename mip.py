"""
mip.py — modelo MIP (Nivel 3).

Devuelve un valor de fitness Z simulado para cada bloque quirúrgico.

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


from typing import List
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, PULP_CBC_CMD
from models import Patient

def solve_mip_for_block(
    specialty_id: int,
    patients: List[Patient],
    block_duration_min: int,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """
    Resuelve el problema de optimización exacto para un bloque quirúrgico.
    """
    # 1. Validaciones iniciales
    if specialty_id == 0 or not patients:
        return 0.0

    # 2. Definir el Problema
    # Queremos maximizar la función objetivo (LpMaximize)
    prob = LpProblem(f"Block_Optimization_Spec_{specialty_id}", LpMaximize)

    # 3. Variables de Decisión
    # x_i es 1 si el paciente i es seleccionado, 0 si no (Binary)
    x = {p.id: LpVariable(f"x_{p.id}", cat="Binary") for p in patients}

    # 4. Función Objetivo
    # Max Z = alpha * sum(w_i * x_i) + beta * (sum(d_i * x_i) / T_max)
    term_priority = lpSum(p.clinical_priority * x[p.id] for p in patients)
    term_utilization = lpSum(p.estimated_duration * x[p.id] for p in patients) / block_duration_min
    
    prob += (alpha * term_priority) + (beta * term_utilization)

    # 5. Restricción de Capacidad
    # La suma de las duraciones no puede exceder el tiempo del bloque
    prob += lpSum(p.estimated_duration * x[p.id] for p in patients) <= block_duration_min

    # 6. Ejecutar el Solver
    # msg=0 desactiva los logs de la consola para no saturar el algoritmo genético
    prob.solve(PULP_CBC_CMD(msg=0))

    # 7. Retornar el valor óptimo de Z
    # Si el solver no encuentra solución, value() devuelve None, por eso el "or 0.0"
    return value(prob.objective) or 0.0
