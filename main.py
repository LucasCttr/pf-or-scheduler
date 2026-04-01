"""
main.py — Ejemplo de uso del AG con datos similares al Hospital Centenario.

Ejecutar:
    python main.py
"""
import json
import random
from mip import solve_mip_for_block
from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm


def make_patients(specialty_id: int, count: int, seed: int = 0) -> list:
    """Genera una lista de pacientes de prueba con duraciones estandarizadas."""
    rng = random.Random(seed)
    # Lista de duraciones permitidas en minutos
    duraciones_permitidas = [30, 45, 60, 90, 120]
    
    return [
        Patient(
            id=specialty_id * 100 + i,
            specialty_id=specialty_id,
            # Selecciona aleatoriamente de la lista de duraciones estandarizadas
            estimated_duration=rng.choice(duraciones_permitidas),
            clinical_priority=round(rng.uniform(1.0, 10.0), 2),
            required_roles=["cirujano", "anestesista", "instrumentador"],
        )
        for i in range(count)
    ]

def main():
    random.seed(42)

    # ── Quirófanos ────────────────────────────────────────────────────────
    operating_rooms = [
        OperatingRoom(
            id=0, name="Quirófano 1 (Alta)",
            or_type="alta_complejidad",
            availability=[[True, True]] * 5,
        ),
        OperatingRoom(
            id=1, name="Quirófano 2 (Media)",
            or_type="media_complejidad",
            availability=[[True, True]] * 5,
        ),
        OperatingRoom(
            id=2, name="Quirófano 3 (Baja)",
            or_type="baja_complejidad",
            # Solo turno mañana (turno tarde = False)
            availability=[[True, False]] * 5,
        ),
    ]

    # ── Especialidades ────────────────────────────────────────────────────
    # id=0 es el bloque libre (obligatorio, no modificar)
    specialties = [
        Specialty(id=0, name="Libre",            compatible_or_types=[], min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología",    compatible_or_types=["alta_complejidad", "media_complejidad"],                    min_blocks=3, max_blocks=6),
        Specialty(id=2, name="Cirugía General",  compatible_or_types=["alta_complejidad", "media_complejidad", "baja_complejidad"], min_blocks=4, max_blocks=8),
        Specialty(id=3, name="Neurología",       compatible_or_types=["alta_complejidad"],                                         min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología",         compatible_or_types=["media_complejidad", "baja_complejidad"],                    min_blocks=2, max_blocks=5),
        Specialty(id=5, name="Ginecología",      compatible_or_types=["media_complejidad", "baja_complejidad"],                    min_blocks=2, max_blocks=5),
    ]

    # ── Pacientes en lista de espera (Aumentados para cubrir la semana) ────
    patients_by_specialty = {
        1: make_patients(1, count=35, seed=1),   # Traumatología
        2: make_patients(2, count=50, seed=2),   # Cirugía General
        3: make_patients(3, count=25, seed=3),   # Neurología
        4: make_patients(4, count=30, seed=4),   # Urología
        5: make_patients(5, count=30, seed=5),   # Ginecología
    }
    
    # ── Staff Médico (Cirujanos) ─────────────────────────────────────────────
    staff_list = [
    # --- TRAUMATOLOGÍA (ID: 1) ---
    Staff(id=1, name="Dr. Pérez", role="cirujano", specialty_id=1,
          availability_hours={0: (480, 660), 1: (780, 1020)}), # Lun Mañana, Mar Tarde
    
    Staff(id=2, name="Dra. Sosa", role="cirujano", specialty_id=1,
          availability_hours={0: (600, 1220), 2: (480, 720)}), # Lun (solapa con Pérez), Mié Mañana completo

    # --- CIRUGÍA GENERAL (ID: 2) ---
    Staff(id=3, name="Dr. Gomez", role="cirujano", specialty_id=2,
          availability_hours={0: (480, 720), 1: (480, 720)}), # Lun y Mar Mañana
    
    Staff(id=4, name="Dra. Ruiz", role="cirujano", specialty_id=2,
          availability_hours={1: (780, 1020), 3: (780, 1020)}), # Mar y Jue Tarde
    
    Staff(id=5, name="Dr. Martinez", role="cirujano", specialty_id=2,
          availability_hours={2: (480, 600), 4: (480, 720)}), # Mié (solo 2hs), Vie Mañana

    # --- NEUROLOGÍA (ID: 3) ---
    Staff(id=6, name="Dra. Blanco", role="cirujano", specialty_id=3,
          availability_hours={3: (480, 720), 4: (780, 1020),}), # Jue Mañana, Vie Tarde y se agrego martes (donde habia un bloque vacio para verificar que el AG lo use)
    
    Staff(id=7, name="Dr. Lopez", role="cirujano", specialty_id=3,
          availability_hours={0: (780, 1020), 2: (780, 1020)}), # Lun y Mié Tarde
    ]

    # ── Configuración del AG ──────────────────────────────────────────────
    config = GAConfig(
        population_size=50,
        max_generations=250,
        convergence_patience=12,
        mutation_rate=0.10,
        crossover_rate=0.85,
        tournament_size=5,
        elite_count=2, 
        n_days=5,
        n_shifts=2,                       # mañana y tarde
        block_duration_min=240,           # 8 horas por bloque
        penalty_below_min_quota=50.0,
        penalty_above_max_quota=20.0,
    )

    # ── Ejecutar ──────────────────────────────────────────────────────────
    ga = GeneticAlgorithm(
        config=config,
        operating_rooms=operating_rooms,
        specialties=specialties,
        patients_by_specialty=patients_by_specialty,
        staff_list=staff_list
    )

    print("=" * 70)
    print("  SISTEMA DE PLANIFICACIÓN DE QUIRÓFANOS — Algoritmo Genético")
    print("  Hospital Centenario, Gualeguaychú")
    print("=" * 70 + "\n")

    best = ga.run()
    ga.print_schedule(best)

    # ── Evolución del fitness ─────────────────────────────────────────────
    print("\n  Evolución del fitness (cada 10 generaciones):")
    for i, f in enumerate(ga.history):
        if i % 10 == 0:
            print(f"    Gen {i:4d}: {f:.4f}")

    print(f"\n  Fitness final del mejor individuo: {best.fitness:.4f}")
    print("\n  Agenda lista para ser enviada a revisión de cirujanos.\n")


    # ── RECONSTRUCCIÓN DE LA AGENDA TOTAL ──────────────────
    all_patients_lookup = {p.id: p for lista in patients_by_specialty.values() for p in lista}

    print("\n▶ Generando reporte de grilla completa...")
    
    pacientes_asignados_semana = set()
    agenda_final = {
        "hospital": "Hospital Centenario",
        "fitness_total": round(best.fitness, 4),
        "dias": []
    }

    for d in range(config.n_days):
        dia_dict = {"nombre": GeneticAlgorithm.DAY_NAMES[d], "bloques": []}
        for t in range(config.n_shifts):
            is_morning = (t == 0)
            
            # Capacidad del turno
            staff_capacity_remanente = {
                s.id: s.get_available_minutes_in_block(d, is_morning)
                for s in staff_list if s.role == "cirujano"
            }
            
            for q in range(len(operating_rooms)):
                spec_id = int(best.chromosome[d, t, q])
                spec_name = next(s.name for s in specialties if s.id == spec_id)
                
                # Inicializamos variables por defecto por si falla algo
                detalles = {"pacientes_ids": [], "asignaciones": [], "t_max_real": config.block_duration_min // 2}
                cronograma = []
                utilizacion_quirofano = 0.0
                t_uso = 0

                # Intentamos llenar el bloque SOLO si no es "Libre"
                if spec_id > 0:
                    surgeons_in_block = [
                        s for s in staff_list 
                        if s.role == "cirujano" and s.specialty_id == spec_id
                        and staff_capacity_remanente.get(s.id, 0) > 0
                    ]

                    candidatos = [p for p in patients_by_specialty.get(spec_id, []) 
                                 if p.id not in pacientes_asignados_semana]

                    # Solo llamamos al MIP si hay chances de éxito
                    if surgeons_in_block and candidatos:
                        detalles = solve_mip_for_block(
                            specialty_id=spec_id,
                            patients=candidatos,
                            surgeons=surgeons_in_block,
                            day_idx=d,
                            is_morning=is_morning,
                            alpha=config.alpha,
                            beta=config.beta,
                            custom_capacities=staff_capacity_remanente,
                            return_details=True
                        )
                        
                        # Actualizar datos reales si el MIP devolvió algo
                        pacientes_asignados_semana.update(detalles["pacientes_ids"])
                        consumo = detalles.get("consumo_medicos", {})
                        for s_id, minutos in consumo.items():
                            staff_capacity_remanente[s_id] -= minutos

                        # Generar cronograma
                        asig_por_doc = {}
                        for asig in detalles.get("asignaciones", []):
                            doc_name = asig["doc"]
                            if doc_name not in asig_por_doc: asig_por_doc[doc_name] = []
                            asig_por_doc[doc_name].append(asig["p"])

                        for doc_name, p_ids in asig_por_doc.items():
                            medico = next(s for s in staff_list if s.name == doc_name)
                            curr_min, _ = medico.get_range_for_block(d, is_morning)
                            for p_id in p_ids:
                                p_obj = all_patients_lookup[p_id]
                                cronograma.append({
                                    "paciente_id": p_id,
                                    "medico": doc_name,
                                    "hora_inicio": f"{curr_min // 60:02d}:{curr_min % 60:02d}",
                                    "hora_fin": f"{(curr_min + p_obj.estimated_duration) // 60:02d}:{(curr_min + p_obj.estimated_duration) % 60:02d}",
                                    "duracion": p_obj.estimated_duration
                                })
                                curr_min += p_obj.estimated_duration

                        t_bloque_teorico = config.block_duration_min 
                        t_uso = sum(all_patients_lookup[pid].estimated_duration for pid in detalles["pacientes_ids"])
                        
                        # El porcentaje ahora refleja el uso del ESPACIO FÍSICO
                        utilizacion_quirofano = round((t_uso / t_bloque_teorico * 100), 2)

                # Agregamos el bloque SIEMPRE
                dia_dict["bloques"].append({
                    "quirofano": operating_rooms[q].name,
                    "turno": GeneticAlgorithm.SHIFT_NAMES[t],
                    "especialidad": spec_name,
                    "utilizacion_porcentaje": utilizacion_quirofano,
                    "pacientes_contados": len(detalles["pacientes_ids"]),
                    "cronograma": cronograma
                })
        
        agenda_final["dias"].append(dia_dict)

    with open("agenda_resultado.json", "w", encoding="utf-8") as f:
        json.dump(agenda_final, f, indent=4, ensure_ascii=False)

    print(f"✔ Agenda guardada. Total pacientes programados: {len(pacientes_asignados_semana)}")


if __name__ == "__main__":
    main()
