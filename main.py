"""main.py — usa mip_slots.py (cronograma directo, sin secuenciador)."""
from __future__ import annotations
import json, random, time
from typing import Dict, List
from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm

def make_patients(specialty_id, count, seed=0, staff_list=None):
    rng = random.Random(seed)
    duraciones = [30, 45, 60, 90, 120]
    cirujanos_ids = [s.id for s in (staff_list or []) if specialty_id in s.specialties_ids]
    patients = []
    if specialty_id == 1:
        patients.append(Patient(id=2000, specialty_id=1, estimated_duration=60,
                                clinical_priority=99.0, required_roles=["cirujano"],
                                forced_surgeon_id=1))
    for i in range(count):
        forced = None
        if cirujanos_ids and rng.random() < 0.20:
            forced = rng.choice(cirujanos_ids)
        patients.append(Patient(id=specialty_id*100+i, specialty_id=specialty_id,
                                estimated_duration=rng.choice(duraciones),
                                clinical_priority=round(rng.uniform(1.0,10.0),2),
                                required_roles=["cirujano"], forced_surgeon_id=forced))
    return patients

def build_staff():
    return [
        Staff(id=1,  name="Dr. Pérez",     role="cirujano", specialties_ids=[1,2], availability_hours={0:(480,620),  1:(780,1020)}),
        Staff(id=2,  name="Dra. Sosa",     role="cirujano", specialties_ids=[1],   availability_hours={0:(480,1020), 2:(480,720)}),
        Staff(id=3,  name="Dra. Carter",   role="cirujano", specialties_ids=[1],   availability_hours={0:(620,1020), 2:(480,720)}),
        Staff(id=4,  name="Dr. Gomez",     role="cirujano", specialties_ids=[2,4], availability_hours={0:(480,720),  1:(480,720)}),
        Staff(id=5,  name="Dra. Ruiz",     role="cirujano", specialties_ids=[2],   availability_hours={1:(780,1020), 3:(780,1020)}),
        Staff(id=6,  name="Dr. Martinez",  role="cirujano", specialties_ids=[2],   availability_hours={2:(480,600),  4:(480,720)}),
        Staff(id=7,  name="Dra. Blanco",   role="cirujano", specialties_ids=[3],   availability_hours={3:(480,720),  4:(780,1020)}),
        Staff(id=8,  name="Dr. Lopez",     role="cirujano", specialties_ids=[3],   availability_hours={0:(780,1020), 2:(780,1020)}),
        Staff(id=9,  name="Dra. García",   role="cirujano", specialties_ids=[4,5], availability_hours={1:(480,720),  3:(480,720)}),
        Staff(id=10, name="Dr. Rodríguez", role="cirujano", specialties_ids=[4,5], availability_hours={2:(780,1020), 4:(780,1020)}),
        
        # --- Nuevos integrantes del Staff ---
        Staff(id=11, name="Dr. Morales",   role="cirujano", specialties_ids=[6],   availability_hours={0:(480,720),  3:(480,1020)}),
        Staff(id=12, name="Dra. Herrera",  role="cirujano", specialties_ids=[6,7], availability_hours={1:(480,720),  4:(480,720)}),
        Staff(id=13, name="Dr. Castro",    role="cirujano", specialties_ids=[7],   availability_hours={2:(780,1020), 3:(780,1020)}),
        Staff(id=14, name="Dra. Mendez",   role="cirujano", specialties_ids=[8],   availability_hours={0:(480,720),  4:(480,1020)}),
        Staff(id=15, name="Dr. Silva",     role="cirujano", specialties_ids=[2,8], availability_hours={1:(780,1020), 2:(480,720)}),
        Staff(id=16, name="Dra. Flores",   role="cirujano", specialties_ids=[1,3], availability_hours={0:(780,1020), 4:(480,720)})
    ]

def build_operating_rooms():
    return [
        OperatingRoom(id=0, name="Quirófano 1 (Alta)",  or_type="alta_complejidad",  availability=[[True,True]]*5),
        OperatingRoom(id=1, name="Quirófano 2 (Media)", or_type="media_complejidad", availability=[[True,True]]*5),
        OperatingRoom(id=2, name="Quirófano 3 (Baja)",  or_type="baja_complejidad",  availability=[[True,False]]*5),
    ]

def build_specialties():
    return [
        Specialty(id=0, name="Libre",           compatible_or_types=[],                                                     min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología",   compatible_or_types=["alta_complejidad","media_complejidad"],               min_blocks=3, max_blocks=6),
        Specialty(id=2, name="Cirugía General", compatible_or_types=["alta_complejidad","media_complejidad","baja_complejidad"], min_blocks=4, max_blocks=8),
        Specialty(id=3, name="Neurología",      compatible_or_types=["alta_complejidad"],                                   min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología",        compatible_or_types=["media_complejidad","baja_complejidad"],               min_blocks=2, max_blocks=5),
        Specialty(id=5, name="Ginecología",     compatible_or_types=["media_complejidad","baja_complejidad"],               min_blocks=2, max_blocks=5),
        
        # --- Nuevas Especialidades ---
        Specialty(id=6, name="Cardiología",     compatible_or_types=["alta_complejidad","media_complejidad"],               min_blocks=3, max_blocks=6),
        Specialty(id=7, name="Otorrinolaringología", compatible_or_types=["media_complejidad","baja_complejidad"],          min_blocks=2, max_blocks=4),
        Specialty(id=8, name="Oftalmología",    compatible_or_types=["baja_complejidad"],                                   min_blocks=2, max_blocks=6)
    ]

def default_config():
    return GAConfig(population_size=50, max_generations=50, convergence_patience=7,
                    mutation_rate=0.10, crossover_rate=0.85, tournament_size=10,
                    elite_count=2, n_days=5, n_shifts=2, block_duration_min=240,
                    slot_size_min=15, penalty_below_min_quota=50.0, penalty_above_max_quota=20.0, parallel_workers=24)

def reconstruct_agenda(ga, best, patients_by_specialty, specialties, operating_rooms, staff_list, config):
    print("\n▶  Leyendo cronograma desde MIP-Slots...")
    schedule_cache = ga.get_schedule_details(best)
    all_pids = {p.id for lst in patients_by_specialty.values() for p in lst}
    pacientes_asignados = set()
    agenda = {"hospital": "Hospital Centenario", "fitness_total": round(best.fitness,4),
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
                dia_dict["bloques"].append({"quirofano": or_obj.name, "turno": ga.SHIFT_NAMES[t],
                                            "especialidad": spec_name,
                                            "utilizacion_porcentaje": per_or.get("utilizacion_porcentaje",0),
                                            "cronograma": cronograma})
        agenda["dias"].append(dia_dict)

    agenda["resumen"] = {"pacientes_programados": len(pacientes_asignados),
                          "pacientes_pendientes":  len(all_pids - pacientes_asignados),
                          "ids_pendientes":        sorted(all_pids - pacientes_asignados)}
    agenda["duracion_segundos"] = None
    return agenda, pacientes_asignados

def main():
    random.seed(42)
    start = time.perf_counter()
    staff_list = build_staff(); operating_rooms = build_operating_rooms()
    specialties = build_specialties()
    patients_by_specialty = {sid: make_patients(sid, count=40, seed=sid, staff_list=staff_list) for sid in range(1,6)}
    config = default_config()
    ga   = GeneticAlgorithm(config, operating_rooms, specialties, patients_by_specialty, staff_list)
    best = ga.run(); ga.print_schedule(best)
    agenda, asignados = reconstruct_agenda(ga, best, patients_by_specialty, specialties, operating_rooms, staff_list, config)
    elapsed = time.perf_counter() - start
    agenda["duracion_segundos"] = round(elapsed, 3)
    with open("agenda_resultado.json","w",encoding="utf-8") as f:
        json.dump(agenda, f, indent=4, ensure_ascii=False)
    print(f"\n✔  {len(asignados)} pacientes programados.  Tiempo: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
