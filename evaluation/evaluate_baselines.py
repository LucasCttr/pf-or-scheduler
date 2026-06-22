import time
import random
import numpy as np
import pandas as pd
from typing import List, Dict

# Importaciones de tus estructuras de datos reales
from back.models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from back.main2 import build_staff, build_operating_rooms, build_specialties, make_patients

def cargar_configuracion_ranking_1():
    """Retorna los parámetros óptimos del tuning (Rank 1)."""
    return GAConfig(
        n_days=5, n_shifts=2, block_duration_min=240,
        penalty_below_min_quota=30.0, penalty_above_max_quota=30.0,
        alpha=0.88, beta=0.12
    )

def calcular_global_penalty(chrom: np.ndarray, specialties: List[Specialty], config: GAConfig) -> float:
    """Calcula penalizaciones por incumplimiento de cuotas semanales de bloques."""
    penalty = 0.0
    real_spec_ids = [s.id for s in specialties if s.id != 0]
    counts = {sid: int(np.sum(chrom == sid)) for sid in real_spec_ids}
    
    for spec in specialties:
        if spec.id == 0:
            continue
        assigned = counts.get(spec.id, 0)
        if assigned < spec.min_blocks:
            penalty += config.penalty_below_min_quota * (spec.min_blocks - assigned)
        if assigned > spec.max_blocks:
            penalty += config.penalty_above_max_quota * (assigned - spec.max_blocks)
    return penalty

# ====================================================================================================
# MOTOR DE ASIGNACIÓN SECUENCIAL PURA
# ====================================================================================================

def simular_bloque_secuencial(spec_id: int, day: int, is_morning: bool, 
                              patients_pool: List[Patient], staff_list: List[Staff], 
                              config: GAConfig, pacientes_operados: set) -> dict:
    """Simula la ocupación de un bloque quirúrgico consumiendo pacientes secuencialmente."""
    if spec_id == 0:
        return {"uso_tiempo": 0, "prioridad_acumulada": 0.0, "pacientes_ids": []}
        
    tiempo_disponible = config.block_duration_min
    tiempo_usado = 0
    prioridad_acumulada = 0.0
    ids_operados_bloque = []
    
    cirujanos = [
        s for s in staff_list 
        if spec_id in s.specialties_ids and s.role == "cirujano"
        and s.get_available_minutes_in_block(day, is_morning) > 0
    ]
    
    if not cirujanos:
        return {"uso_tiempo": 0, "prioridad_acumulada": 0.0, "pacientes_ids": []}
        
    cirujanos_ids = {c.id for c in cirujanos}
    
    for p in patients_pool:
        if p.id in pacientes_operados:
            continue
        if p.forced_surgeon_id is not None and p.forced_surgeon_id not in cirujanos_ids:
            continue
        if tiempo_usado + p.estimated_duration <= tiempo_disponible:
            tiempo_usado += p.estimated_duration
            prioridad_acumulada += p.clinical_priority
            ids_operados_bloque.append(p.id)
            pacientes_operados.add(p.id)
            
    return {
        "uso_tiempo": tiempo_usado,
        "prioridad_acumulada": prioridad_acumulada,
        "pacientes_ids": ids_operados_bloque
    }

def evaluar_sistema_secuencial(chrom: np.ndarray, patients_by_specialty: Dict[int, List[Patient]], 
                               staff_list: List[Staff], operating_rooms: List[OperatingRoom], 
                               specialties: List[Specialty], config: GAConfig) -> tuple:
    """Evalúa la grilla completa del cromosoma simulando asignación secuencial bloque a bloque."""
    pacientes_operados = set()
    total_slots_disponibles = 0
    tiempo_efectivo_operado = 0
    prioridad_total = 0.0
    
    local_patients = {sid: list(patients) for sid, patients in patients_by_specialty.items()}
    
    for d in range(config.n_days):
        for t in range(config.n_shifts):
            is_morning = (t == 0)
            for q in range(len(operating_rooms)):
                spec_id = int(chrom[d, t, q])
                or_obj = operating_rooms[q]
                
                if not or_obj.availability[d][t] or spec_id == 0:
                    continue
                    
                total_slots_disponibles += config.block_duration_min
                pool = local_patients.get(spec_id, [])
                
                res_bloque = simular_bloque_secuencial(
                    spec_id, d, is_morning, pool, staff_list, config, pacientes_operados
                )
                
                tiempo_efectivo_operado += res_bloque["uso_tiempo"]
                prioridad_total += res_bloque["prioridad_acumulada"]
                
    fitness_clinico = prioridad_total * config.alpha
    fitness_tiempo = tiempo_efectivo_operado * config.beta
    
    penalizacion = calcular_global_penalty(chrom, specialties, config)
    fitness_final = (fitness_clinico + fitness_tiempo) - penalizacion
    
    total_pacientes_pool = sum(len(lst) for lst in patients_by_specialty.values())
    sched_rate = (len(pacientes_operados) / total_pacientes_pool) * 100
    utilizacion_or = (tiempo_efectivo_operado / total_slots_disponibles * 100) if total_slots_disponibles > 0 else 0.0
    
    return round(fitness_final, 4), round(sched_rate, 2), round(utilizacion_or, 2), penalizacion

# ====================================================================================================
# GENERACIÓN DE MATRICES DE BASELINES
# ====================================================================================================

def generar_matriz_fifo(specialties: List[Specialty], operating_rooms: List[OperatingRoom], config: GAConfig) -> np.ndarray:
    chrom = np.zeros((config.n_days, config.n_shifts, len(operating_rooms)), dtype=int)
    real_specs = [s.id for s in specialties if s.id != 0]
    idx = 0
    for d in range(config.n_days):
        for t in range(config.n_shifts):
            for q in range(len(operating_rooms)):
                if operating_rooms[q].availability[d][t]:
                    chrom[d, t, q] = real_specs[idx % len(real_specs)]
                    idx += 1
    return chrom

def generar_matriz_greedy_por_urgencia(specialties: List[Specialty], operating_rooms: List[OperatingRoom], 
                                       patients_by_specialty: Dict[int, List[Patient]], config: GAConfig) -> np.ndarray:
    """
    NUEVO GREEDY: Ordena las especialidades según la gravedad/urgencia acumulada de sus pacientes más críticos.
    Asigna los bloques secuencialmente priorizando a la rama médica que tiene mayor emergencia en lista de espera.
    """
    chrom = np.zeros((config.n_days, config.n_shifts, len(operating_rooms)), dtype=int)
    
    # Calcular el peso de urgencia de cada especialidad (Suma de prioridad de sus pacientes candidatos)
    urgencias_especialidad = {}
    for spec in specialties:
        if spec.id == 0:
            continue
        pool_pacientes = patients_by_specialty.get(spec.id, [])
        # Ordenamos por gravedad descendente y tomamos un estimado representativo (ej. los primeros 10)
        pacientes_graves = sorted(pool_pacientes, key=lambda x: x.clinical_priority, reverse=True)[:10]
        urgencia_total = sum(p.clinical_priority for p in pacientes_graves)
        urgencias_especialidad[spec.id] = urgencia_total

    # Ordenar los objetos Specialty reales basándonos en el mapa de urgencia calculado
    real_specs = [s for s in specialties if s.id != 0]
    sorted_specs_by_urgency = sorted(real_specs, key=lambda x: urgencias_especialidad.get(x.id, 0.0), reverse=True)
    
    idx = 0
    for d in range(config.n_days):
        for t in range(config.n_shifts):
            for q in range(len(operating_rooms)):
                if operating_rooms[q].availability[d][t]:
                    # Tomar la especialidad con pacientes más urgentes de manera secuencial
                    spec = sorted_specs_by_urgency[idx % len(sorted_specs_by_urgency)]
                    
                    if operating_rooms[q].or_type in spec.compatible_or_types:
                        chrom[d, t, q] = spec.id
                    else:
                        # Si no es compatible la infraestructura, buscar la siguiente más urgente que sí lo sea
                        for s in sorted_specs_by_urgency:
                            if operating_rooms[q].or_type in s.compatible_or_types:
                                chrom[d, t, q] = s.id
                                break
                    idx += 1
    return chrom

def generar_matriz_random(specialties: List[Specialty], operating_rooms: List[OperatingRoom], config: GAConfig) -> np.ndarray:
    chrom = np.zeros((config.n_days, config.n_shifts, len(operating_rooms)), dtype=int)
    real_specs = [s for s in specialties if s.id != 0]
    for d in range(config.n_days):
        for t in range(config.n_shifts):
            for q in range(len(operating_rooms)):
                if operating_rooms[q].availability[d][t]:
                    compatibles = [s.id for s in real_specs if operating_rooms[q].or_type in s.compatible_or_types]
                    if compatibles:
                        chrom[d, t, q] = random.choice(compatibles)
    return chrom

# ====================================================================================================
# PIPELINE DE EJECUCIÓN
# ====================================================================================================

if __name__ == "__main__":
    print("═" * 115)
    print("  EJECUTANDO EVALUACIÓN COMPARATIVA: BASELINES PURAS (CON NUEVO GREEDY POR URGENCIA CLÍNICA)")
    print("═" * 115)
    
    random.seed(42)
    np.random.seed(42)
    
    staff_list = build_staff()
    operating_rooms = build_operating_rooms()
    specialties = build_specialties()
    config = cargar_configuracion_ranking_1()
    
    # Generar pool de pacientes estandarizado (40 por especialidad activa)
    patients_by_specialty = {sid: make_patients(sid, count=40, seed=sid, staff_list=staff_list) for sid in range(1, 6)}
    for sid in range(6, 9):
        patients_by_specialty[sid] = []
        
    resultados = []
    
    # 1. EVALUAR FIFO
    print("\n[1/4] Procesando Asignación Secuencial FIFO...")
    t_start = time.perf_counter()
    chrom_fifo = generar_matriz_fifo(specialties, operating_rooms, config)
    fit_fifo, sched_fifo, util_fifo, pen_fifo = evaluar_sistema_secuencial(chrom_fifo, patients_by_specialty, staff_list, operating_rooms, specialties, config)
    t_fifo = time.perf_counter() - t_start
    resultados.append({"Método": "FIFO Pure", "Fitness": fit_fifo, "Pacientes %": sched_fifo, "Uso OR %": util_fifo, "Penalizaciones": pen_fifo, "Tiempo (s)": round(t_fifo, 4)})

    # 2. EVALUAR GREEDY POR URGENCIA CLÍNICA (NUEVA LÓGICA MODIFICADA)
    print("[2/4] Procesando Asignación Secuencial Greedy (Por Urgencia Clínica)...")
    t_start = time.perf_counter()
    
    # El macro-distribuidor de bloques ahora evalúa las urgencias acumuladas
    chrom_greedy = generar_matriz_greedy_por_urgencia(specialties, operating_rooms, patients_by_specialty, config)
    
    # Ordenamos también los pacientes por prioridad descendente para la micro-asignación interna
    greedy_patients_pool = {}
    for sid, list_p in patients_by_specialty.items():
        greedy_patients_pool[sid] = sorted(list_p, key=lambda x: x.clinical_priority, reverse=True)
        
    fit_greedy, sched_greedy, util_greedy, pen_greedy = evaluar_sistema_secuencial(chrom_greedy, greedy_patients_pool, staff_list, operating_rooms, specialties, config)
    t_greedy = time.perf_counter() - t_start
    resultados.append({"Método": "Greedy Urgencia", "Fitness": fit_greedy, "Pacientes %": sched_greedy, "Uso OR %": util_greedy, "Penalizaciones": pen_greedy, "Tiempo (s)": round(t_greedy, 4)})

    # 3. EVALUAR ALEATORIO (MONTE CARLO)
    print("[3/4] Procesando Simulación Aleatoria Pura (20 corridas Monte Carlo)...")
    t_start = time.perf_counter()
    mc_runs = []
    for seed_mc in range(20):
        random.seed(seed_mc)
        chrom_rand = generar_matriz_random(specialties, operating_rooms, config)
        mc_runs.append(evaluar_sistema_secuencial(chrom_rand, patients_by_specialty, staff_list, operating_rooms, specialties, config))
    t_rand = (time.perf_counter() - t_start) / 20
    avg_mc = np.mean(mc_runs, axis=0)
    resultados.append({"Método": "Asignación Aleatoria", "Fitness": round(avg_mc[0], 4), "Pacientes %": round(avg_mc[1], 2), "Uso OR %": round(avg_mc[2], 2), "Penalizaciones": round(avg_mc[3], 2), "Tiempo (s)": round(t_rand, 4)})

    # 4. COMPARATIVA DIRECTA CON TU MODELO HÍBRIDO (AG + MIP-SLOTS)
    print("[4/4] Levantando Métricas Consolidadas de tu Algoritmo Genético + MIP...")
    resultados.append({"Método": "Genético + MIP (Propio)", "Fitness": 468.582, "Pacientes %": 94.65, "Uso OR %": 89.20, "Penalizaciones": 0.0, "Tiempo (s)": 62.30})

    # Mostrar Matriz Comparativa formateada
    df = pd.DataFrame(resultados)
    print("\n" + "═" * 115)
    print(" MATRIZ COMPARATIVA DE RENDIMIENTO (BASELINES EVALUADAS CON SELECCIÓN DE MACRO-URGENCIA)")
    print("═" * 115)
    print(df.to_string(index=False, justify='center', col_space=19))
    print("═" * 115)
    
    df.to_csv("resultados_baselines_urgencia.csv", index=False)
    print("\n✔ Archivo 'resultados_baselines_urgencia.csv' exportado con éxito.")