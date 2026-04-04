"""
main.py — Sistema de Planificación Quirúrgica (Modelo Híbrido)
Hospital Centenario, Gualeguaychú.
"""
import json
import random
import time
from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm

def make_patients(specialty_id: int, count: int, seed: int = 0, staff_list: list = None) -> list:
    """
    Genera pacientes de prueba. 
    Implementa el MODELO HÍBRIDO: algunos pacientes tienen médico asignado y otros no.
    """
    rng = random.Random(seed)
    duraciones_permitidas = [30, 45, 60, 90, 120]
    
    # Obtener IDs de cirujanos de esta especialidad para asignaciones aleatorias
    cirujanos_ids = []
    if staff_list:
        cirujanos_ids = [s.id for s in staff_list if specialty_id in s.specialties_ids]
    
    patients = []
    
    # --- 1. CASO CRÍTICO (Paciente 2000) ---
    if specialty_id == 1:
        urgencia = Patient(
            id=2000,
            specialty_id=1,
            estimated_duration=200, # 3.55 hs
            clinical_priority=99.0, # Urgencia máxima
            required_roles=["cirujano", "anestesista", "instrumentador"],
            forced_surgeon_id=1    # Forzado al Dr. Pérez (ID 1)
        )
        patients.append(urgencia)
        
    # --- 2. RESTO DE PACIENTES ---
    for i in range(count):
        # 20% de probabilidad de tener un médico asignado (Modelo Híbrido)
        asignado_id = None
        if cirujanos_ids and rng.random() < 0.20:
            asignado_id = rng.choice(cirujanos_ids)

        p = Patient(
            id=specialty_id * 100 + i,
            specialty_id=specialty_id,
            estimated_duration=rng.choice(duraciones_permitidas),
            clinical_priority=round(rng.uniform(1.0, 10.0), 2),
            required_roles=["cirujano", "anestesista", "instrumentador"],
            forced_surgeon_id=asignado_id
        )
        patients.append(p)
        
    return patients

def main():
    random.seed(42)
    start_time = time.perf_counter()

    # ── 1. DEFINICIÓN DE STAFF (Médicos) ──────────────────────────────────
    staff_list = [
        # Dr. Pérez: Traumatología (1) y Cirugía General (2)
        Staff(id=1, name="Dr. Pérez", role="cirujano", specialties_ids=[1, 2], 
              availability_hours={0: (480, 620), 1: (780, 1020)}), 
        Staff(id=2, name="Dra. Sosa", role="cirujano", specialties_ids=[1], 
              availability_hours={0: (480, 1020), 2: (480, 720)}), 
        Staff(id=3, name="Dra. Carter", role="cirujano", specialties_ids=[1], 
              availability_hours={0: (620, 1020), 2: (480, 720)}), 
        # Dr. Gomez: Cirugía General (2) y Urología (4)
        Staff(id=4, name="Dr. Gomez", role="cirujano", specialties_ids=[2, 4], 
              availability_hours={0: (480, 720), 1: (480, 720)}), 
        Staff(id=5, name="Dra. Ruiz", role="cirujano", specialties_ids=[2], 
              availability_hours={1: (780, 1020), 3: (780, 1020)}), 
        Staff(id=6, name="Dr. Martinez", role="cirujano", specialties_ids=[2], 
              availability_hours={2: (480, 600), 4: (480, 720)}), 
        Staff(id=7, name="Dra. Blanco", role="cirujano", specialties_ids=[3], 
              availability_hours={3: (480, 720), 4: (780, 1020)}),
        Staff(id=8, name="Dr. Lopez", role="cirujano", specialties_ids=[3], 
              availability_hours={0: (780, 1020), 2: (780, 1020)}), 
        Staff(id=9, name="Dra. García", role="cirujano", specialties_ids=[4, 5], 
              availability_hours={1: (480, 720), 3: (480, 720)}), 
        Staff(id=10, name="Dr. Rodríguez", role="cirujano", specialties_ids=[4, 5],
              availability_hours={2: (780, 1020), 4: (780, 1020)})
    ]

    # ── 2. QUIRÓFANOS Y ESPECIALIDADES ────────────────────────────────────
    operating_rooms = [
        OperatingRoom(id=0, name="Quirófano 1 (Alta)", or_type="alta_complejidad", availability=[[True, True]]*5),
        OperatingRoom(id=1, name="Quirófano 2 (Media)", or_type="media_complejidad", availability=[[True, True]]*5),
        OperatingRoom(id=2, name="Quirófano 3 (Baja)", or_type="baja_complejidad", availability=[[True, False]]*5),
    ]

    specialties = [
        Specialty(id=0, name="Libre", compatible_or_types=[], min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología", compatible_or_types=["alta_complejidad", "media_complejidad"], min_blocks=3, max_blocks=6),
        Specialty(id=2, name="Cirugía General", compatible_or_types=["alta_complejidad", "media_complejidad", "baja_complejidad"], min_blocks=4, max_blocks=8),
        Specialty(id=3, name="Neurología", compatible_or_types=["alta_complejidad"], min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología", compatible_or_types=["media_complejidad", "baja_complejidad"], min_blocks=2, max_blocks=5),
        Specialty(id=5, name="Ginecología", compatible_or_types=["media_complejidad", "baja_complejidad"], min_blocks=2, max_blocks=5),
    ]

    # ── 3. GENERAR PACIENTES (HÍBRIDO) ────────────────────────────────────
    patients_by_specialty = {
        sid: make_patients(sid, count=40, seed=sid, staff_list=staff_list) 
        for sid in range(1, 6)
    }

    # ── 4. EJECUTAR ALGORITMO GENÉTICO ────────────────────────────────────
    config = GAConfig(
        population_size=50, max_generations=200, convergence_patience=15,
        mutation_rate=0.10, crossover_rate=0.85, tournament_size=5,
        elite_count=2, n_days=5, n_shifts=2, block_duration_min=240,
        penalty_below_min_quota=50.0, penalty_above_max_quota=20.0,
    )

    ga = GeneticAlgorithm(config, operating_rooms, specialties, patients_by_specialty, staff_list)
    best = ga.run()
    ga.print_schedule(best)

    # ── 5. RECONSTRUCCIÓN DE LA AGENDA (HEURÍSTICA DE TRENES) ───────────
    print("\n▶  Generando cronograma con Heurística de Trenes...")
    schedule_cache = ga.get_schedule_details(best)
    
    all_patients_lookup = {p.id: p for lista in patients_by_specialty.values() for p in lista}
    pacientes_asignados_semana = set()
    agenda_final = {"hospital": "Hospital Centenario", "fitness_total": round(best.fitness, 4), "dias": []}

    for d in range(config.n_days):
        dia_dict = {"nombre": ga.DAY_NAMES[d], "bloques": []}
        for t in range(config.n_shifts):
            is_morning = (t == 0)
            
            # Punteros de tiempo (Relojes)
            # El médico empieza cuando abre el bloque o según su disponibilidad personal
            libre_staff = {s.id: s.get_range_for_block(d, is_morning)[0] for s in staff_list}
            # El quirófano empieza exactamente al inicio del turno (08:00 o 13:00)
            libre_q = {or_obj.id: (480 if is_morning else 780) for or_obj in operating_rooms}

            # --- PASO A: Agrupar y Ordenar (La esencia de la Heurística) ---
            asignaciones_por_q = {}
            for q_idx in range(len(operating_rooms)):
                detalles = schedule_cache.get((d, t, q_idx))
                if detalles and detalles["asignaciones"]:
                    # Ordenamos: 1º por Médico (crea el 'tren'), 2º por Prioridad (dentro del tren)
                    asigs_ordenadas = sorted(
                        detalles["asignaciones"], 
                        key=lambda x: (x["doc"], -all_patients_lookup[x["p"]].clinical_priority)
                    )
                    asignaciones_por_q[q_idx] = asigs_ordenadas
                else:
                    asignaciones_por_q[q_idx] = []

            cronogramas_finales = {q: [] for q in range(len(operating_rooms))}
            quirofanos_activos = [q for q, asigs in asignaciones_por_q.items() if asigs]

            # --- PASO B: Simulación de avance de tiempo ---
            # Mientras haya cirugías pendientes en algún quirófano...
            while quirofanos_activos:
                for q_idx in quirofanos_activos[:]:
                    if not asignaciones_por_q[q_idx]:
                        quirofanos_activos.remove(q_idx)
                        continue
                    
                    asig = asignaciones_por_q[q_idx][0]
                    p_obj = all_patients_lookup[asig["p"]]
                    medico = next(s for s in staff_list if s.name == asig["doc"])
                    or_id = operating_rooms[q_idx].id

                    # Lógica de Sincronización:
                    # La cirugía empieza cuando el médico llega Y la sala está vacía
                    inicio = max(libre_staff[medico.id], libre_q[or_id])
                    
                    # Si el inicio es 0 (médico no disponible), algo falló en el MIP,
                    # pero aquí lo manejamos por seguridad:
                    if inicio == 0: 
                        asignaciones_por_q[q_idx].pop(0)
                        continue

                    fin = inicio + p_obj.estimated_duration

                    cronogramas_finales[q_idx].append({
                        "paciente_id": p_obj.id,
                        "medico": medico.name,
                        "hora_inicio": f"{inicio // 60:02d}:{inicio % 60:02d}",
                        "hora_fin": f"{fin // 60:02d}:{fin % 60:02d}",
                        "duracion": p_obj.estimated_duration,
                        "prioridad": p_obj.clinical_priority
                    })

                    # Actualizamos relojes globales
                    libre_staff[medico.id] = fin
                    libre_q[or_id] = fin
                    
                    # Quitamos de la cola de este quirófano
                    asignaciones_por_q[q_idx].pop(0)
                    pacientes_asignados_semana.add(p_obj.id)

            # --- PASO C: Construcción del JSON por Bloque ---
            for q_idx in range(len(operating_rooms)):
                spec_id = int(best.chromosome[d, t, q_idx])
                spec_name = next(s.name for s in specialties if s.id == spec_id)
                
                dia_dict["bloques"].append({
                    "quirofano": operating_rooms[q_idx].name,
                    "turno": ga.SHIFT_NAMES[t],
                    "especialidad": spec_name,
                    "utilizacion_porcentaje": schedule_cache.get((d,t,q_idx), {}).get("utilizacion_porcentaje", 0),
                    "cronograma": cronogramas_finales[q_idx]
                })
        agenda_final["dias"].append(dia_dict)

    # Calcular duración total de ejecución y añadir al reporte
    elapsed = time.perf_counter() - start_time
    agenda_final["duracion_segundos"] = round(elapsed, 3)

    with open("agenda_resultado.json", "w", encoding="utf-8") as f:
        json.dump(agenda_final, f, indent=4, ensure_ascii=False)

    print(f"\n✔ Éxito. Reporte generado. Pacientes totales: {len(pacientes_asignados_semana)}")
    print(f"Tiempo de ejecución: {elapsed:.2f} s")

if __name__ == "__main__":
    main()