"""
main.py — Orquestador híbrido en 2 etapas (AG Heurístico + Cierre MILP exacto con Cascada de Validación).
"""
from __future__ import annotations
import json
import random
import time
import sys
from typing import Dict, List

from models import OperatingRoom, Specialty, Procedure, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm
from mip import solve_mip_for_shift

# ═══════════════════════════════════════════════════════════════════════
# DATA BUILDERS SIMULADOS (Mantenemos tu lógica exacta)
# ═══════════════════════════════════════════════════════════════════════

PROCEDURES_BY_SPECIALTY = {
    1: [
        Procedure(id=101, name="Hip prosthesis", specialty_id=1, required_room_type="alta_complejidad"),
        Procedure(id=102, name="Knee prosthesis", specialty_id=1, required_room_type="alta_complejidad"),
        Procedure(id=103, name="Arthroscopy", specialty_id=1, required_room_type="media_complejidad"),
    ],
    2: [
        Procedure(id=201, name="Cholecystectomy", specialty_id=2, required_room_type="alta_complejidad"),
        Procedure(id=202, name="Appendectomy", specialty_id=2, required_room_type="media_complejidad"),
        Procedure(id=203, name="Hernia repair", specialty_id=2, required_room_type="baja_complejidad"),
    ],
    3: [
        Procedure(id=301, name="Craniotomy", specialty_id=3, required_room_type="alta_complejidad"),
        Procedure(id=302, name="Decompression", specialty_id=3, required_room_type="media_complejidad"),
    ],
    4: [
        Procedure(id=401, name="Hysterectomy", specialty_id=4, required_room_type="alta_complejidad"),
        Procedure(id=402, name="C-section", specialty_id=4, required_room_type="media_complejidad"),
    ],
    5: [
        Procedure(id=501, name="Cataract surgery", specialty_id=5, required_room_type="baja_complejidad"),
        Procedure(id=502, name="Vitrectomy", specialty_id=5, required_room_type="alta_complejidad"),
    ],
    6: [
        Procedure(id=601, name="Rhinoplasty", specialty_id=6, required_room_type="media_complejidad"),
        Procedure(id=602, name="Septoplasty", specialty_id=6, required_room_type="baja_complejidad"),
    ],
    7: [
        Procedure(id=701, name="Nephrectomy", specialty_id=7, required_room_type="alta_complejidad"),
        Procedure(id=702, name="Prostatectomy", specialty_id=7, required_room_type="alta_complejidad"),
    ],
    8: [
        Procedure(id=801, name="Pacemaker insertion", specialty_id=8, required_room_type="media_complejidad"),
    ]
}

def build_staff() -> List[Staff]:
    roles_by_spec = {
        1: [101, 102, 103], 2: [201, 202, 203], 3: [301, 302], 4: [401, 402],
        5: [501, 502], 6: [601, 602], 7: [701, 702], 8: [801]
    }
    staff = []
    uid = 1
    for sid in range(1, 9):
        for i in range(3):
            s = Staff(id=uid, name=f"Dr. Spec{sid}_N{i}", role="cirujano", main_specialty_id=sid)
            s.enabled_procedures_ids = set(roles_by_spec[sid])
            s.availability_hours = {d: (480, 900) if i % 2 == 0 else (780, 1200) for d in range(5)}
            staff.append(s)
            uid += 1
    return staff

def build_operating_rooms() -> List[OperatingRoom]:
    return [
        OperatingRoom(id=1, name="Quirófano Central 1 (Alta)", or_type="alta_complejidad"),
        OperatingRoom(id=2, name="Quirófano Central 2 (Media)", or_type="media_complejidad"),
        OperatingRoom(id=3, name="Quirófano Ambulatorio (Baja)", or_type="baja_complejidad"),
    ]

def build_specialties() -> List[Specialty]:
    specs = [Specialty(id=0, name="Libre", compatible_or_types=[])]
    names = ["Traumatología", "General", "Neurocirugía", "Ginecología", "Oftalmología", "Plástica", "Urología", "Cardiología"]
    for i, name in enumerate(names, 1):
        specs.append(Specialty(id=i, name=name, compatible_or_types=["alta_complejidad", "media_complejidad", "baja_complejidad"], min_blocks=1, max_blocks=6))
    return specs

def make_patients(specialty_id: int, count: int, seed: int, staff_list: List[Staff]) -> List[Patient]:
    rng = random.Random(seed)
    procs = PROCEDURES_BY_SPECIALTY[specialty_id]
    patients = []
    my_docs = [s for s in staff_list if s.main_specialty_id == specialty_id]
    
    for i in range(count):
        proc = rng.choice(procs)
        duration = rng.choice([45, 60, 90, 120, 150])
        priority = round(rng.uniform(1.0, 10.0), 2)
        
        p = Patient(
            id=specialty_id * 1000 + i,
            specialty_id=specialty_id,
            procedure_id=proc.id,
            estimated_duration=duration,
            clinical_priority=priority,
            required_roles=["cirujano"]
        )
        if rng.random() < 0.25 and my_docs:
            p.forced_surgeon_id = rng.choice(my_docs).id
            
        patients.append(p)
    return patients

def default_config() -> GAConfig:
    return GAConfig(
        population_size=60,
        max_generations=80,
        convergence_patience=10,
        mutation_rate=0.08,
        crossover_rate=0.85,
        tournament_size=8,
        elite_count=3,
        alpha=0.7,   
        beta=0.3,    
        block_duration_min=720,  
        slot_size_min=15,
        parallel_workers=4       
    )

# ═══════════════════════════════════════════════════════════════════════
# 3. METODO PRINCIPAL DE ORQUESTACIÓN
# ═══════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)
    start_time = time.perf_counter()
    
    # 1. Preparación de Entidades
    staff_list = build_staff()
    operating_rooms = build_operating_rooms()
    specialties = build_specialties()
    
    patients_by_specialty = {
        sid: make_patients(sid, count=35, seed=sid, staff_list=staff_list) 
        for sid in range(1, 9)
    }
    
    config = default_config()
    
    # 2. Instanciar e iniciar Nivel 1: Algoritmo Genético Macro
    ga = GeneticAlgorithm(
        config,
        operating_rooms,
        specialties,
        patients_by_specialty,
        staff_list,
        procedures_by_specialty=PROCEDURES_BY_SPECIALTY,
    )
    
    print("🚀 FASE 1: Optimizando asignación de Especialidades con Algoritmo Genético...")
    top_candidatos = ga.run()  # Retorna el pool (List[Individual]) del Top 10
    
    # 3. FASE 2: Cascada de Validación e Integración Micro con MILP Exacto
    print("\n🔒 FASE 2: Evaluando el Top de mejores grillas macro con el MILP...")
    
    cronograma_final_json = {}
    best_individual = None
    solucion_encontrada = False
    pacientes_operados_ganador = set()
    
    for rank, candidato in enumerate(top_candidatos, 1):
        print(f"\n⏳ Intentando secuenciar el Candidato #{rank} (Fitness macro estimado: {candidato.fitness:.4f})...")
        
        pacientes_operados_global = set()
        cronograma_tentativo = {}
        candidato_es_factible = True
        
        for d in range(ga.n_days):
            day_name = ga.DAY_NAMES[d]
            cronograma_tentativo[day_name] = {}
            
            for t in range(ga.n_shifts):
                shift_name = ga.SHIFT_NAMES[t]
                is_morning = (t == 0)
                
                # Construimos los bloques de demanda reales para el cromosoma de este candidato específico
                blocks_turno = ga._build_shift_blocks(
                    candidato.chromosome, d, t, is_morning, pacientes_operados_global
                )
                
                # Inicializamos la estructura de salida por defecto por quirófano
                cronograma_tentativo[day_name][shift_name] = {}
                for q_idx, or_obj in enumerate(operating_rooms):
                    cronograma_tentativo[day_name][shift_name][or_obj.name] = {
                        "especialidad": "Libre" if int(candidato.chromosome[d, t, q_idx]) == 0 else ga._spec_by_id[int(candidato.chromosome[d, t, q_idx])].name,
                        "utilizacion_porcentaje": 0.0,
                        "cirugias": []
                    }
                
                # Si el turno tiene bloques con médicos y pacientes, llamamos al solver matemático
                if any(b["surgeons"] and b["patients"] for b in blocks_turno):
                    result_mip = solve_mip_for_shift(
                        blocks_turno, d, is_morning, config.alpha, config.beta, slot_size=15
                    )
                    
                    # ⚠ DETECCIÓN DE INFACTIBILIDAD MICRO:
                    # Si el solver no encuentra solución factible, descartamos el individuo completo
                    if result_mip is None or "per_or" not in result_mip:
                        print(f"  ❌ ¡Infactibilidad detectada en {day_name} ({shift_name})! Rompe restricciones micro.")
                        candidato_es_factible = False
                        break
                    
                    # Si fue factible, acumulamos los IDs de pacientes operados en este turno
                    pacientes_operados_global.update(result_mip["all_pacientes_ids"])
                    
                    # Volcamos el resultado exitoso del turno al cronograma tentativo
                    for q_idx, or_obj in enumerate(operating_rooms):
                        if q_idx in result_mip["per_or"]:
                            data_or = result_mip["per_or"][q_idx]
                            
                            lista_cirugias = []
                            for asig in data_or["asignaciones"]:
                                lista_cirugias.append({
                                    "paciente_id": asig["p"],
                                    "cirujano": asig["doc"],
                                    "hora_inicio": asig["hora_inicio"],
                                    "hora_fin": asig["hora_fin"],
                                    "duracion_minutos": asig["duracion"]
                                })
                                
                            cronograma_tentativo[day_name][shift_name][or_obj.name] = {
                                "especialidad": ga._spec_by_id[int(candidato.chromosome[d, t, q_idx])].name,
                                "utilizacion_porcentaje": data_or["utilizacion_porcentaje"],
                                "cirugias": lista_cirugias
                            }
            
            if not candidato_es_factible:
                break  # Sale del bucle de días e invalida el cromosoma para saltar al siguiente candidato
                
        # Si logramos validar secuencialmente todos los días sin disparar la bandera de fallo...
        if candidato_es_factible:
            print(f"  ✔ ¡Candidato #{rank} completamente validado matemáticamente y secuenciado con éxito!")
            cronograma_final_json = cronograma_tentativo
            best_individual = candidato
            pacientes_operados_ganador = pacientes_operados_global
            solucion_encontrada = True
            break  # !!! Cortamos la búsqueda porque encontramos el óptimo ejecutable !!!

    if not solucion_encontrada:
        print("\n🚨 CRÍTICO: Ninguno de los 10 mejores individuos del AG pudo ser secuenciado por el MILP.")
        sys.exit(1)
        
    # Ahora sí imprimimos de forma segura la grilla macro del individuo ganador
    ga.print_schedule(best_individual)

    # 4. Cálculo de Métricas Finales y Exportación
    elapsed_time = time.perf_counter() - start_time
    total_pacientes_sistema = sum(len(v) for v in patients_by_specialty.values())
    
    res_final = {
        "metadata": {
            "estado": "Optimización Exitosa (Modelo Híbrido AG-MILP)",
            "tiempo_ejecucion_segundos": round(elapsed_time, 3),
            "total_pacientes_demanda": total_pacientes_sistema,
            "pacientes_programados_exitosamente": len(pacientes_operados_ganador),
            "pacientes_en_espera_restantes": total_pacientes_sistema - len(pacientes_operados_ganador),
            "fitness_heuristico_macro": round(best_individual.fitness, 4)
        },
        "cronograma": cronograma_final_json
    }
    
    output_filename = "agenda_resultado.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(res_final, f, indent=4, ensure_ascii=False)
        
    print("\n" + "═" * 70)
    print(f"  🏅 ¡OPTIMIZACIÓN AG-MILP FINALIZADA CON ÉXITO!")
    print(f"  - Tiempo total de ejecución: {elapsed_time:.3f} segundos.")
    print(f"  - Cirugías calendarizadas al minuto: {len(pacientes_operados_ganador)} de {total_pacientes_sistema}.")
    print(f"  - Estructura JSON exportada a '{output_filename}'.")
    print("═" * 70)

if __name__ == "__main__":
    main()