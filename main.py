"""main.py — usa mip_slots.py (cronograma directo, sin secuenciador)."""
from __future__ import annotations
import json, random, time, sys
from typing import Dict, List
from models import OperatingRoom, Specialty, Procedure, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm

# Diccionario auxiliar que emula un Nomenclador Real (Procedimientos por especialidad)
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
        Procedure(id=302, name="Decompression", specialty_id=3, required_room_type="alta_complejidad"),
    ],
    4: [
        Procedure(id=401, name="Prostate resection", specialty_id=4, required_room_type="media_complejidad"),
        Procedure(id=402, name="Nephrectomy", specialty_id=4, required_room_type="alta_complejidad"),
    ],
    5: [
        Procedure(id=501, name="Hysterectomy", specialty_id=5, required_room_type="media_complejidad"),
        Procedure(id=502, name="Laparoscopy", specialty_id=5, required_room_type="baja_complejidad"),
    ],
    6: [
        Procedure(id=601, name="Bypass", specialty_id=6, required_room_type="alta_complejidad"),
        Procedure(id=602, name="Angioplasty", specialty_id=6, required_room_type="media_complejidad"),
    ],
    7: [
        Procedure(id=701, name="Tonsillectomy", specialty_id=7, required_room_type="media_complejidad"),
        Procedure(id=702, name="Septoplasty", specialty_id=7, required_room_type="baja_complejidad"),
    ],
    8: [
        Procedure(id=801, name="Cataract surgery", specialty_id=8, required_room_type="baja_complejidad"),
        Procedure(id=802, name="Vitrectomy", specialty_id=8, required_room_type="baja_complejidad"),
    ],
}


def _procedures_for_specialty(specialty_id: int) -> List[Procedure]:
    return PROCEDURES_BY_SPECIALTY.get(specialty_id, [
        Procedure(
            id=specialty_id * 100,
            name=f"Procedure {specialty_id * 100}",
            specialty_id=specialty_id,
            required_room_type="media_complejidad",
        )
    ])

def make_patients(specialty_id: int, count: int, seed: int = 0, staff_list: List[Staff] = None) -> List[Patient]:
    rng = random.Random(seed)
    duraciones = [30, 45, 60, 90, 120]
    proc_pool = _procedures_for_specialty(specialty_id)
    
    # Precalculamos qué procedimientos sabe resolver la capacidad instalada real del staff
    valid_proc_ids = []
    if staff_list:
        staff_procs = {pid for s in staff_list for pid in s.enabled_procedures_ids}
        valid_proc_ids = [p.id for p in proc_pool if p.id in staff_procs]
    
    # Salvaguarda si no se inyecta staff o no se encuentran coincidencias preliminares
    if not valid_proc_ids:
        valid_proc_ids = [p.id for p in proc_pool]

    patients = []
    # Paciente de prueba crítico forzado para simular restricciones en Traumatología
    if specialty_id == 1:
        patients.append(Patient(id=2000, specialty_id=1, procedure_id=101, estimated_duration=60,
                                clinical_priority=99.0, required_roles=["cirujano"],
                                forced_surgeon_id=1))
                                
    for i in range(count):
        forced = None
        # Selecciona únicamente códigos válidos que el personal del hospital puede realizar
        chosen_proc = rng.choice(valid_proc_ids)
        
        capable_surgeons = [
            s.id for s in (staff_list or []) 
            if chosen_proc in s.enabled_procedures_ids and s.main_specialty_id == specialty_id
        ]
        
        # Simula asignaciones forzadas o urgencias específicas pre-programadas (20% prob)
        if capable_surgeons and rng.random() < 0.20:
            forced = rng.choice(capable_surgeons)
            
        patients.append(Patient(id=specialty_id*100+i, specialty_id=specialty_id,
                                procedure_id=chosen_proc,
                                estimated_duration=rng.choice(duraciones),
                                clinical_priority=round(rng.uniform(1.0, 10.0), 2),
                                required_roles=["cirujano"], forced_surgeon_id=forced))
    return patients

def build_staff() -> List[Staff]:
    """
    Instancia el Staff Médico alineado a la jornada unificada de 720 minutos (08:00 a 20:00 hs).
    Garantiza cobertura cruzada mañana/tarde en servicios críticos para evitar sub-utilización.
    """
    return [
        # --- ID 1: Traumatología (Especialidad 1) ---
        Staff(id=1,  name="Dr. Pérez",     role="cirujano", enabled_procedures_ids=[101,102,103], availability_hours={0:(480,840), 2:(480,840)}, main_specialty_id=1),  # Lu, Mi - Mañana
        Staff(id=2,  name="Dra. Sosa",     role="cirujano", enabled_procedures_ids=[101,102,103], availability_hours={0:(480,1200), 2:(480,1200)}, main_specialty_id=1), # Lu, Mi - Completo
        Staff(id=3,  name="Dra. Carter",   role="cirujano", enabled_procedures_ids=[101,102,103], availability_hours={0:(840,1200), 2:(840,1200)}, main_specialty_id=1), # Lu, Mi - Tarde

        # --- ID 2: Cirugía General (Especialidad 2) ---
        Staff(id=4,  name="Dr. Gomez",     role="cirujano", enabled_procedures_ids=[201,202,203], availability_hours={0:(480,840), 1:(480,840)}, main_specialty_id=2),  # Lu, Ma - Mañana
        Staff(id=5,  name="Dra. Ruiz",     role="cirujano", enabled_procedures_ids=[201,202,203], availability_hours={1:(840,1200), 3:(840,1200)}, main_specialty_id=2), # Ma, Ju - Tarde
        Staff(id=6,  name="Dr. Martinez",  role="cirujano", enabled_procedures_ids=[201,202,203], availability_hours={2:(480,840), 4:(480,840)}, main_specialty_id=2),  # Mi, Vi - Mañana
        Staff(id=15, name="Dr. Silva",     role="cirujano", enabled_procedures_ids=[201,202,203], availability_hours={1:(480,1200), 2:(480,1200)}, main_specialty_id=2), # Ma, Mi - Completo

        # --- ID 3: Neurología (Especialidad 3) ---
        Staff(id=7,  name="Dra. Blanco",   role="cirujano", enabled_procedures_ids=[301,302],     availability_hours={3:(840,1200), 4:(840,1200)}, main_specialty_id=3), # Ju, Vi - Tarde
        Staff(id=8,  name="Dr. Lopez",     role="cirujano", enabled_procedures_ids=[301,302],     availability_hours={0:(480,1200), 2:(480,1200)}, main_specialty_id=3), # Lu, Mi - Completo
        Staff(id=16, name="Dra. Flores",   role="cirujano", enabled_procedures_ids=[301,302],     availability_hours={0:(480,840), 4:(480,1200)}, main_specialty_id=3),  # Lu (Mañana), Vi (Completo)

        # --- ID 4: Urología (Especialidad 4) ---
        Staff(id=9,  name="Dra. García",   role="cirujano", enabled_procedures_ids=[401,402],     availability_hours={1:(480,840), 3:(480,1200)}, main_specialty_id=4),  # Ma (Mañana), Ju (Completo)
        Staff(id=17, name="Dr. Rossi",     role="cirujano", enabled_procedures_ids=[401,402],     availability_hours={1:(840,1200), 3:(480,840)}, main_specialty_id=4),  # Ma (Tarde), Ju (Mañana)

        # --- ID 5: Ginecología (Especialidad 5) ---
        Staff(id=10, name="Dr. Rodríguez", role="cirujano", enabled_procedures_ids=[501,502],     availability_hours={2:(840,1200), 4:(840,1200)}, main_specialty_id=5), # Mi, Vi - Tarde
        Staff(id=18, name="Dra. Paul",     role="cirujano", enabled_procedures_ids=[501,502],     availability_hours={2:(480,840), 4:(480,840)}, main_specialty_id=5),   # Mi, Vi - Mañana

        # --- ID 6: Cardiología (Especialidad 6) ---
        Staff(id=11, name="Dr. Morales",   role="cirujano", enabled_procedures_ids=[601,602],     availability_hours={0:(480,1200), 3:(480,1200)}, main_specialty_id=6), # Lu, Ju - Completo
        Staff(id=12, name="Dra. Herrera",  role="cirujano", enabled_procedures_ids=[601,602],     availability_hours={1:(480,840), 4:(480,840)}, main_specialty_id=6),  # Ma, Vi - Mañana

        # --- ID 7: Otorrinolaringología (Especialidad 7) ---
        Staff(id=13, name="Dr. Castro",    role="cirujano", enabled_procedures_ids=[701,702],     availability_hours={2:(840,1200), 3:(840,1200)}, main_specialty_id=7), # Mi, Ju - Tarde
        Staff(id=19, name="Dra. Velez",    role="cirujano", enabled_procedures_ids=[701,702],     availability_hours={2:(480,840), 3:(480,840)}, main_specialty_id=7),   # Mi, Ju - Mañana

        # --- ID 8: Oftalmología (Especialidad 8) ---
        Staff(id=14, name="Dra. Mendez",   role="cirujano", enabled_procedures_ids=[801,802],     availability_hours={0:(480,1200), 4:(480,1200)}, main_specialty_id=8), # Lu, Vi - Completo
    ]

def build_operating_rooms() -> List[OperatingRoom]:
    """
    Instancia los quirófanos habilitando la disponibilidad total diaria.
    Dado que n_shifts = 1, la lista interna representa la disponibilidad del bloque de 12hs por día.
    """
    return [
        OperatingRoom(id=0, name="Quirófano 1 (Alta)",  or_type="alta_complejidad",  availability=[[True]]*5),
        OperatingRoom(id=1, name="Quirófano 2 (Media)", or_type="media_complejidad", availability=[[True]]*5),
        OperatingRoom(id=2, name="Quirófano 3 (Baja)",  or_type="baja_complejidad",  availability=[[True]]*5),
    ]

def build_specialties() -> List[Specialty]:
    """
    Configura las cuotas máximas semanales basándose en el volumen real de staff.
    Oftalmología (un solo médico) se acota estratégicamente para mitigar desperdicios de capacidad física.
    """
    return [
        Specialty(id=0, name="Libre",           compatible_or_types=[],                                                     min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología",   compatible_or_types=["alta_complejidad","media_complejidad"],               min_blocks=2, max_blocks=5),
        Specialty(id=2, name="Cirugía General", compatible_or_types=["alta_complejidad","media_complejidad","baja_complejidad"], min_blocks=3, max_blocks=6),
        Specialty(id=3, name="Neurología",      compatible_or_types=["alta_complejidad"],                                   min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología",        compatible_or_types=["media_complejidad","baja_complejidad"],               min_blocks=1, max_blocks=3),
        Specialty(id=5, name="Ginecología",     compatible_or_types=["media_complejidad","baja_complejidad"],               min_blocks=1, max_blocks=3),
        Specialty(id=6, name="Cardiología",     compatible_or_types=["alta_complejidad","media_complejidad"],               min_blocks=2, max_blocks=4),
        Specialty(id=7, name="Otorrinolaringología", compatible_or_types=["media_complejidad","baja_complejidad"],          min_blocks=1, max_blocks=3),
        Specialty(id=8, name="Oftalmología",    compatible_or_types=["baja_complejidad"],                                   min_blocks=1, max_blocks=2)
    ]

def default_config() -> GAConfig:
    """Configuración unificada balanceada para bloques de 720 minutos continuos."""
    return GAConfig(
        population_size=30,          # Ajustado para agilizar el procesamiento con el MIP
        max_generations=30, 
        convergence_patience=6,
        mutation_rate=0.12, 
        crossover_rate=0.85, 
        tournament_size=5,
        elite_count=2, 
        n_days=5, 
        n_shifts=1,                  # 1 Solo bloque diario (Jornada completa continua)
        block_duration_min=720,      # 12 Horas de duración (48 slots de 15 min)
        slot_size_min=15, 
        penalty_below_min_quota=50.0, 
        penalty_above_max_quota=20.0, 
        parallel_workers=24         
    )

def reconstruct_agenda(ga, best, patients_by_specialty, specialties, operating_rooms, staff_list, config, frontend_mode: bool = False):
    print("\n▶  Leyendo cronograma unificado desde MIP-Slots continuo...")
    schedule_cache = ga.get_schedule_details(best)
    all_pids = {p.id for lst in patients_by_specialty.values() for p in lst}
    pacientes_asignados = set()
    agenda = {"hospital": "Hospital de Alta Complejidad", "fitness_total": round(best.fitness, 4),
              "slot_size_min": config.slot_size_min, "dias": []}

    for d in range(config.n_days):
        dia_dict = {"nombre": ga.DAY_NAMES[d], "bloques": []}
        for t in range(config.n_shifts):
            for q_idx, or_obj in enumerate(operating_rooms):
                spec_id   = int(best.chromosome[d, t, q_idx])
                spec_name = next(s.name for s in specialties if s.id == spec_id)
                per_or    = schedule_cache.get((d, t, q_idx)) or {}
                
                if frontend_mode:
                    cronograma = [{
                        "paciente_id": a["p"],
                        "procedimiento": None,
                        "medico": a["doc"],
                        "hora_inicio": a["hora_inicio"],
                        "hora_fin": a["hora_fin"]
                    } for a in per_or.get("asignaciones", [])]
                    pid_to_proc = {p.id: p.procedure_id for lst in patients_by_specialty.values() for p in lst}
                    for entry in cronograma:
                        entry["procedimiento"] = pid_to_proc.get(entry["paciente_id"])
                else:
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
                    "turno": "Jornada Completa",
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
    
    # Genera la demanda completa recorriendo las 8 especialidades simuladas
    patients_by_specialty = {
        sid: make_patients(sid, count=40, seed=sid, staff_list=staff_list) 
        for sid in range(1, 9)
    }
    
    config = default_config()
    
    ga = GeneticAlgorithm(
        config,
        operating_rooms,
        specialties,
        patients_by_specialty,
        staff_list,
        procedures_by_specialty=PROCEDURES_BY_SPECIALTY,
    )
    best = ga.run()
    ga.print_schedule(best)
    
    frontend_mode = "--frontend" in sys.argv

    agenda, asignados = reconstruct_agenda(ga, best, patients_by_specialty, specialties, operating_rooms, staff_list, config, frontend_mode=frontend_mode)
    elapsed = time.perf_counter() - start
    agenda["duracion_segundos"] = round(elapsed, 3)
    
    with open("agenda_resultado.json", "w", encoding="utf-8") as f:
        json.dump(agenda, f, indent=4, ensure_ascii=False)
        
    print(f"\n✔  {len(asignados)} pacientes programados con éxito. Tiempo total: {elapsed:.2f}s")

if __name__ == "__main__":
    main()