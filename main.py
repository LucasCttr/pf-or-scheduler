"""main.py — usa mip_slots.py (cronograma directo, sin secuenciador)."""
from __future__ import annotations
import json, random, time
from typing import Dict, List
from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm

# Diccionario auxiliar que emula un Nomenclador Real (Procedimientos por especialidad)
PROCEDURES_BY_SPECIALTY = {
    1: [101, 102, 103], # Traumatología: Prótesis cadera, Rodilla, Artroscopia
    2: [201, 202, 203], # Cirugía General: Colecistectomía, Apendicectomía, Hernia
    3: [301, 302],      # Neurología: Craneotomía, Descompresión
    4: [401, 402],      # Urología: Resección prostática, Nefrectomía
    5: [501, 502],      # Ginecología: Histerectomía, Laparoscopia
    6: [601, 602],      # Cardiología: Bypass, Angioplastia
    7: [701, 702],      # Otorrinolaringología: Amígdalas, Septoplastia
    8: [801, 802]       # Oftalmología: Cataratas, Vitrectomía
}

def make_patients(specialty_id: int, count: int, seed: int = 0, staff_list: List[Staff] = None) -> List[Patient]:
    rng = random.Random(seed)
    duraciones = [30, 45, 60, 90, 120]
    proc_pool = PROCEDURES_BY_SPECIALTY.get(specialty_id, [specialty_id * 100])
    
    # Filtrar médicos capaces de realizar al menos un procedimiento de esta especialidad
    cirujanos_ids = [
        s.id for s in (staff_list or []) 
        if any(pid in s.enabled_procedures_ids for pid in proc_pool)
    ]
    
    patients = []
    # Paciente de prueba crítico forzado para simular restricciones
    if specialty_id == 1:
        patients.append(Patient(id=2000, specialty_id=1, procedure_id=101, estimated_duration=60,
                                clinical_priority=99.0, required_roles=["cirujano"],
                                forced_surgeon_id=1))
                                
    for i in range(count):
        forced = None
        chosen_proc = rng.choice(proc_pool)
        
        # Filtrar cirujanos que tengan asignado este procedimiento específico en su matriz
        capable_surgeons = [s.id for s in (staff_list or []) if chosen_proc in s.enabled_procedures_ids]
        
        if capable_surgeons and rng.random() < 0.20:
            forced = rng.choice(capable_surgeons)
            
        patients.append(Patient(id=specialty_id*100+i, specialty_id=specialty_id,
                                procedure_id=chosen_proc,
                                estimated_duration=rng.choice(duraciones),
                                clinical_priority=round(rng.uniform(1.0, 10.0), 2),
                                required_roles=["cirujano"], forced_surgeon_id=forced))
    return patients

def build_staff() -> List[Staff]:
    """Instancia el Staff Médico asignando códigos de procedimientos reales (Matriz de competencias)."""
    return [
        Staff(id=1,  name="Dr. Pérez",     role="cirujano", enabled_procedures_ids=[101,102,103,201,202], availability_hours={0:(480,620),  1:(780,1020)}),
        Staff(id=2,  name="Dra. Sosa",     role="cirujano", enabled_procedures_ids=[101,102,103],         availability_hours={0:(480,1020), 2:(480,720)}),
        Staff(id=3,  name="Dra. Carter",   role="cirujano", enabled_procedures_ids=[101,102],             availability_hours={0:(620,1020), 2:(480,720)}),
        Staff(id=4,  name="Dr. Gomez",     role="cirujano", enabled_procedures_ids=[201,202,203,401,402], availability_hours={0:(480,720),  1:(480,720)}),
        Staff(id=5,  name="Dra. Ruiz",     role="cirujano", enabled_procedures_ids=[201,202,203],         availability_hours={1:(780,1020), 3:(780,1020)}),
        Staff(id=6,  name="Dr. Martinez",  role="cirujano", enabled_procedures_ids=[201,202],             availability_hours={2:(480,600),  4:(480,720)}),
        Staff(id=7,  name="Dra. Blanco",   role="cirujano", enabled_procedures_ids=[301,302],             availability_hours={3:(480,720),  4:(780,1020)}),
        Staff(id=8,  name="Dr. Lopez",     role="cirujano", enabled_procedures_ids=[301,302],             availability_hours={0:(780,1020), 2:(780,1020)}),
        Staff(id=9,  name="Dra. García",   role="cirujano", enabled_procedures_ids=[401,402,501,502],     availability_hours={1:(480,720),  3:(480,720)}),
        Staff(id=10, name="Dr. Rodríguez", role="cirujano", enabled_procedures_ids=[401,501,502],         availability_hours={2:(780,1020), 4:(780,1020)}),
        
        # --- Nuevos integrantes del Staff (Especialidades 6, 7 y 8) ---
        Staff(id=11, name="Dr. Morales",   role="cirujano", enabled_procedures_ids=[601,602],             availability_hours={0:(480,720),  3:(480,1020)}),
        Staff(id=12, name="Dra. Herrera",  role="cirujano", enabled_procedures_ids=[601,701,702],         availability_hours={1:(480,720),  4:(480,720)}),
        Staff(id=13, name="Dr. Castro",    role="cirujano", enabled_procedures_ids=[701,702],             availability_hours={2:(780,1020), 3:(780,1020)}),
        Staff(id=14, name="Dra. Mendez",   role="cirujano", enabled_procedures_ids=[801,802],             availability_hours={0:(480,720),  4:(480,1020)}),
        Staff(id=15, name="Dr. Silva",     role="cirujano", enabled_procedures_ids=[201,202,801],         availability_hours={1:(780,1020), 2:(480,720)}),
        Staff(id=16, name="Dra. Flores",   role="cirujano", enabled_procedures_ids=[101,301,302],         availability_hours={0:(780,1020), 4:(480,720)})
    ]

def build_operating_rooms() -> List[OperatingRoom]:
    return [
        OperatingRoom(id=0, name="Quirófano 1 (Alta)",  or_type="alta_complejidad",  availability=[[True,True]]*5),
        OperatingRoom(id=1, name="Quirófano 2 (Media)", or_type="media_complejidad", availability=[[True,True]]*5),
        OperatingRoom(id=2, name="Quirófano 3 (Baja)",  or_type="baja_complejidad",  availability=[[True,False]]*5),
    ]

def build_specialties() -> List[Specialty]:
    return [
        Specialty(id=0, name="Libre",           compatible_or_types=[],                                                     min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología",   compatible_or_types=["alta_complejidad","media_complejidad"],               min_blocks=3, max_blocks=6),
        Specialty(id=2, name="Cirugía General", compatible_or_types=["alta_complejidad","media_complejidad","baja_complejidad"], min_blocks=4, max_blocks=8),
        Specialty(id=3, name="Neurología",      compatible_or_types=["alta_complejidad"],                                   min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología",        compatible_or_types=["media_complejidad","baja_complejidad"],               min_blocks=2, max_blocks=5),
        Specialty(id=5, name="Ginecología",     compatible_or_types=["media_complejidad","baja_complejidad"],               min_blocks=2, max_blocks=5),
        Specialty(id=6, name="Cardiología",     compatible_or_types=["alta_complejidad","media_complejidad"],               min_blocks=3, max_blocks=6),
        Specialty(id=7, name="Otorrinolaringología", compatible_or_types=["media_complejidad","baja_complejidad"],          min_blocks=2, max_blocks=4),
        Specialty(id=8, name="Oftalmología",    compatible_or_types=["baja_complejidad"],                                   min_blocks=2, max_blocks=6)
    ]

def default_config() -> GAConfig:
    return GAConfig(population_size=50, max_generations=50, convergence_patience=7,
                    mutation_rate=0.10, crossover_rate=0.85, tournament_size=10,
                    elite_count=2, n_days=5, n_shifts=2, block_duration_min=240,
                    slot_size_min=15, penalty_below_min_quota=50.0, penalty_above_max_quota=20.0, parallel_workers=24)

def reconstruct_agenda(ga, best, patients_by_specialty, specialties, operating_rooms, staff_list, config):
    print("\n▶  Leyendo cronograma unificado desde MIP-Slots por Turno...")
    schedule_cache = ga.get_schedule_details(best)
    all_pids = {p.id for lst in patients_by_specialty.values() for p in lst}
    pacientes_asignados = set()
    agenda = {"hospital": "Hospital de Alta Complejidad", "fitness_total": round(best.fitness,4),
               "slot_size_min": config.slot_size_min, "dias": []}

    for d in range(config.n_days):
        dia_dict = {"nombre": ga.DAY_NAMES[d], "bloques": []}
        for t in range(config.n_shifts):
            for q_idx, or_obj in enumerate(operating_rooms):
                spec_id   = int(best.chromosome[d, t, q_idx])
                spec_name = next(s.name for s in specialties if s.id == spec_id)
                per_or    = schedule_cache.get((d, t, q_idx)) or {}
                
                cronograma = [{"paciente_id": a["p"], "medico": a["doc"],
                                "slot_inicio": a["slot_inicio"],
                                "hora_inicio": a["hora_inicio"],
                                "hora_fin":    a["hora_fin"],
                                "duracion":    a["duracion"]}
                               for a in per_or.get("asignaciones", [])]
                               
                for a in per_or.get("asignaciones", []):
                    pacientes_asignados.add(a["p"])
                    
                dia_dict["bloques"].append({
                    "quirofano": or_obj.name, 
                    "turno": ga.SHIFT_NAMES[t],
                    "especialidad": spec_name,
                    "utilizacion_porcentaje": per_or.get("utilizacion_porcentaje", 0),
                    "cronograma": cronograma
                })
        agenda["dias"].append(dia_dict)

    agenda["resumen"] = {"pacientes_programados": len(pacientes_asignados),
                          "pacientes_pendientes":  len(all_pids - pacientes_asignados),
                          "ids_pendientes":        sorted(all_pids - pacientes_asignados)}
    agenda["duracion_segundos"] = None
    return agenda, pacientes_asignados

def main():
    random.seed(42)
    start = time.perf_counter()
    
    staff_list = build_staff()
    operating_rooms = build_operating_rooms()
    specialties = build_specialties()
    
    # CAMBIO CRÍTICO: Genera la demanda completa recorriendo las 8 especialidades simuladas
    patients_by_specialty = {
        sid: make_patients(sid, count=40, seed=sid, staff_list=staff_list) 
        for sid in range(1, 9)
    }
    
    config = default_config()
    
    ga = GeneticAlgorithm(config, operating_rooms, specialties, patients_by_specialty, staff_list)
    best = ga.run()
    ga.print_schedule(best)
    
    agenda, asignados = reconstruct_agenda(ga, best, patients_by_specialty, specialties, operating_rooms, staff_list, config)
    elapsed = time.perf_counter() - start
    agenda["duracion_segundos"] = round(elapsed, 3)
    
    with open("agenda_resultado.json", "w", encoding="utf-8") as f:
        json.dump(agenda, f, indent=4, ensure_ascii=False)
        
    print(f"\n✔  {len(asignados)} pacientes programados con éxito. Tiempo total: {elapsed:.2f}s")

if __name__ == "__main__":
    main()