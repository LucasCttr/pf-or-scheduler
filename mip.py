from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, PULP_CBC_CMD

def solve_mip_for_block(
    specialty_id: int,
    patients: list,
    surgeons: list,
    day_idx: int,
    is_morning: bool,
    alpha: float = 0.7,
    beta: float = 0.3,
    custom_capacities: dict = None,
    return_details: bool = False
):
    if not patients or not surgeons:
        return (0.0, [], {}) if not return_details else {"fitness": 0.0, "pacientes_ids": [], "asignaciones": []}

    # 1. Definir capacidades individuales reales (las que quedan del turno)
    # Si no viene custom_capacities (ej. primera llamada), usamos el total del médico 
    capacidades_actuales = {}
    for s in surgeons:
        if custom_capacities and s.id in custom_capacities:
            capacidades_actuales[s.id] = custom_capacities[s.id]
        else:
            capacidades_actuales[s.id] = s.get_available_minutes_in_block(day_idx, is_morning)

    # 2. Capacidad Reloj (Unión de rangos de los cirujanos con tiempo disponible)
    minutos_reloj = set()
    for s in surgeons:
        if capacidades_actuales[s.id] > 0:
            start, end = s.get_range_for_block(day_idx, is_morning)
            for m in range(start, end):
                minutos_reloj.add(m)
    
    t_max_quirofano = len(minutos_reloj)
    if t_max_quirofano == 0:
        return (0.0, [], {}) if not return_details else {"fitness": 0.0, "pacientes_ids": []}

    # 3. Problema y Variables
    prob = LpProblem(f"MIP_Block_{day_idx}_{is_morning}", LpMaximize)
    x = {p.id: {s.id: LpVariable(f"x_p{p.id}_s{s.id}", cat="Binary") for s in surgeons} for p in patients}

    # 4. Objetivo
    prioridad = lpSum(p.clinical_priority * x[p.id][s.id] for p in patients for s in surgeons)
    utilizacion = lpSum(p.estimated_duration * x[p.id][s.id] for p in patients for s in surgeons) / t_max_quirofano
    prob += (alpha * prioridad) + (beta * utilizacion)

    # 5. Restricciones
    # Cada paciente se asigna a lo sumo a un cirujano
    for p in patients:
        prob += lpSum(x[p.id][s.id] for s in surgeons) <= 1

    # Cada cirujano no puede exceder su capacidad REMANENTE
    for s in surgeons:
        # Aquí usamos la capacidad REMANENTE que nos pasó el AG
        prob += lpSum(p.estimated_duration * x[p.id][s.id] for p in patients) <= capacidades_actuales[s.id]

    # El total de tiempo asignado no puede exceder el tiempo del quirófano (reloj)
    prob += lpSum(p.estimated_duration * x[p.id][s.id] for p in patients for s in surgeons) <= t_max_quirofano

    # 6. Resolver
    prob.solve(PULP_CBC_CMD(msg=0))

    # 7. Calcular tiempo consumido por cada médico en ESTE quirófano
    # Esto es lo que el AG usará para restar del turno
    consumo_medicos = {}
    for s in surgeons:
        minutos_usados = sum(p.estimated_duration * value(x[p.id][s.id]) for p in patients)
        consumo_medicos[s.id] = minutos_usados

    ids_elegidos = [p.id for p in patients if any(value(x[p.id][s.id]) > 0.5 for s in surgeons)]
    z_final = value(prob.objective) or 0.0

    if return_details:
        # Calculamos el uso real en minutos
        uso_tiempo_minutos = sum(p.estimated_duration for p in patients if any(value(x[p.id][s.id]) > 0.5 for s in surgeons))
        return {
            "fitness": z_final,
            "pacientes_ids": ids_elegidos,
            "consumo_medicos": consumo_medicos,
            "t_max_real": t_max_quirofano,
            "uso_tiempo": uso_tiempo_minutos, # <--- Nuevo
            "utilizacion_porcentaje": round((uso_tiempo_minutos / t_max_quirofano) * 100, 2) if t_max_quirofano > 0 else 0, # <--- Nuevo
            "asignaciones": [{"p": p.id, "doc": s.name} for p in patients for s in surgeons if value(x[p.id][s.id]) > 0.5]
        }
    
    # Retornamos los 3 valores que espera el AG
    return z_final, ids_elegidos, consumo_medicos