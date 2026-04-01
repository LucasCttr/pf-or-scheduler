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
    return_details: bool = False  # Si True, retorna un dict con detalles adicionales además del fitness (se usa al final del AG para mostrar pacientes seleccionados y uso de tiempo)
):
    """
    Resuelve el problema de optimización exacto para un bloque quirúrgico.
    """
    # 1. Validaciones iniciales
    if specialty_id == 0 or not patients:
        return {"fitness": 0.0, "pacientes_ids": [], "uso_tiempo": 0} if return_details else 0.0

    # 2. Definición del Problema
    prob = LpProblem(f"Block_Optimization_Spec_{specialty_id}", LpMaximize)

    # 3. Variables de Decisión
    # x[p.id] es 1 si el paciente i es seleccionado, 0 si no
    x = {p.id: LpVariable(f"x_{p.id}", cat="Binary") for p in patients}

    # 4. Función Objetivo
    # Max Z = alpha * sum(w_i * x_i) + beta * (sum(d_i * x_i) / T_max)
    term_priority = lpSum(p.clinical_priority * x[p.id] for p in patients)
    term_utilization = lpSum(p.estimated_duration * x[p.id] for p in patients) / block_duration_min
    
    prob += (alpha * term_priority) + (beta * term_utilization)

    # 5. Restricción de Capacidad
    prob += lpSum(p.estimated_duration * x[p.id] for p in patients) <= block_duration_min

    # 6. Ejecutar el Solver (silencioso)
    prob.solve(PULP_CBC_CMD(msg=0))

    # 7. Obtener el Fitness
    z_final = value(prob.objective) or 0.0

    # 8. Retorno condicional
    if return_details:
        # Filtramos los pacientes cuyo valor en el solver sea 1 (o muy cercano a 1 por precisión de coma flotante)
        pacientes_seleccionados = [p.id for p in patients if value(x[p.id]) > 0.5]
        
        # Calculamos el tiempo total de los seleccionados
        tiempo_total = sum(p.estimated_duration for p in patients if value(x[p.id]) > 0.5)

        return {
            "fitness": z_final,
            "pacientes_ids": pacientes_seleccionados,
            "uso_tiempo": tiempo_total,
            "utilizacion_porcentaje": round((tiempo_total / block_duration_min) * 100, 2)
        }
    
    return z_final