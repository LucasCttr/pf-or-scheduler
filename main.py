"""main.py

Script principal para generar una agenda quirúrgica mediante un algoritmo genético
y reconstruir un cronograma horario con el secuenciador óptimo (EDF + knapsack).

Uso:
    python main.py
"""

from __future__ import annotations

import json
import random
import time
from typing import Dict, List

from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm
from sequence import sequence_shift, SequencedOR


def make_patients(specialty_id: int, count: int, seed: int = 0, staff_list: list | None = None) -> List[Patient]:
    """Genera una lista de `Patient` de prueba.

    - Implementa un "modelo híbrido": algunos pacientes vienen con un cirujano
      forzado (`forced_surgeon_id`) y otros no.
    - Añade un caso crítico fijo (id 2000) para la especialidad 1.
    """
    rng = random.Random(seed)
    duraciones_permitidas = [30, 45, 60, 90, 120]

    # IDs de cirujanos que pueden operar esta especialidad
    cirujanos_ids = [s.id for s in (staff_list or []) if specialty_id in s.specialties_ids]

    patients: List[Patient] = []

    # Caso crítico para Traumatología (ejemplo fijo del proyecto)
    if specialty_id == 1:
        patients.append(
            Patient(
                id=2000,
                specialty_id=1,
                estimated_duration=200,
                clinical_priority=99.0,
                required_roles=["cirujano", "anestesista", "instrumentador"],
                forced_surgeon_id=1,
            )
        )

    # Generar pacientes adicionales
    for i in range(count):
        assigned_id = None
        if cirujanos_ids and rng.random() < 0.20:  # 20% tienen médico asignado
            assigned_id = rng.choice(cirujanos_ids)

        patients.append(
            Patient(
                id=specialty_id * 100 + i,
                specialty_id=specialty_id,
                estimated_duration=rng.choice(duraciones_permitidas),
                clinical_priority=round(rng.uniform(1.0, 10.0), 2),
                required_roles=["cirujano", "anestesista", "instrumentador"],
                forced_surgeon_id=assigned_id,
            )
        )

    return patients


def build_staff() -> List[Staff]:
    """Define la plantilla de médicos (staff) utilizada en el ejemplo."""
    return [
        Staff(id=1, name="Dr. Pérez",      role="cirujano", specialties_ids=[1, 2], availability_hours={0: (480, 620),  1: (780, 1020)}),
        Staff(id=2, name="Dra. Sosa",      role="cirujano", specialties_ids=[1],    availability_hours={0: (480, 1020), 2: (480, 720)}),
        Staff(id=3, name="Dra. Carter",    role="cirujano", specialties_ids=[1],    availability_hours={0: (620, 1020), 2: (480, 720)}),
        Staff(id=4, name="Dr. Gomez",      role="cirujano", specialties_ids=[2, 4], availability_hours={0: (480, 720),  1: (480, 720)}),
        Staff(id=5, name="Dra. Ruiz",      role="cirujano", specialties_ids=[2],    availability_hours={1: (780, 1020), 3: (780, 1020)}),
        Staff(id=6, name="Dr. Martinez",   role="cirujano", specialties_ids=[2],    availability_hours={2: (480, 600),  4: (480, 720)}),
        Staff(id=7, name="Dra. Blanco",    role="cirujano", specialties_ids=[3],    availability_hours={3: (480, 720),  4: (780, 1020)}),
        Staff(id=8, name="Dr. Lopez",      role="cirujano", specialties_ids=[3],    availability_hours={0: (780, 1020), 2: (780, 1020)}),
        Staff(id=9, name="Dra. García",    role="cirujano", specialties_ids=[4, 5], availability_hours={1: (480, 720),  3: (480, 720)}),
        Staff(id=10, name="Dr. Rodríguez", role="cirujano", specialties_ids=[4, 5], availability_hours={2: (780, 1020), 4: (780, 1020)}),
    ]


def build_operating_rooms() -> List[OperatingRoom]:
    """Crea la lista de quirófanos y su disponibilidad por día/turno."""
    return [
        OperatingRoom(id=0, name="Quirófano 1 (Alta)",  or_type="alta_complejidad",  availability=[[True, True]]  * 5),
        OperatingRoom(id=1, name="Quirófano 2 (Media)", or_type="media_complejidad", availability=[[True, True]]  * 5),
        OperatingRoom(id=2, name="Quirófano 3 (Baja)",  or_type="baja_complejidad",  availability=[[True, False]] * 5),
    ]


def build_specialties() -> List[Specialty]:
    """Define las especialidades, su compatibilidad con quirófanos y cuotas."""
    return [
        Specialty(id=0, name="Libre",           compatible_or_types=[],                                                    min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología",   compatible_or_types=["alta_complejidad", "media_complejidad"],             min_blocks=3, max_blocks=6),
        Specialty(id=2, name="Cirugía General", compatible_or_types=["alta_complejidad", "media_complejidad", "baja_complejidad"], min_blocks=4, max_blocks=8),
        Specialty(id=3, name="Neurología",      compatible_or_types=["alta_complejidad"],                                  min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología",        compatible_or_types=["media_complejidad", "baja_complejidad"],             min_blocks=2, max_blocks=5),
        Specialty(id=5, name="Ginecología",     compatible_or_types=["media_complejidad", "baja_complejidad"],             min_blocks=2, max_blocks=5),
    ]


def default_config() -> GAConfig:
    """Devuelve la configuración por defecto usada en el ejemplo principal."""
    return GAConfig(
        population_size=50,
        max_generations=200,
        convergence_patience=15,
        mutation_rate=0.10,
        crossover_rate=0.85,
        tournament_size=5,
        elite_count=2,
        n_days=5,
        n_shifts=2,
        block_duration_min=240,
        penalty_below_min_quota=50.0,
        penalty_above_max_quota=20.0,
    )


def reconstruct_agenda(
    ga: GeneticAlgorithm,
    best,
    patients_by_specialty: Dict[int, List[Patient]],
    specialties: List[Specialty],
    operating_rooms: List[OperatingRoom],
    staff_list: List[Staff],
    config: GAConfig,
) -> tuple:
    """Reconstruye el cronograma horario usando el secuenciador óptimo (EDF + knapsack).

    El secuenciador decide qué pacientes entran ANTES de asignar tiempos, por lo
    que el cronograma final no tiene conflictos de horario ni cirugías
    descartadas a último momento.

    Retorna
    -------
    (agenda_final: dict, pacientes_asignados_semana: set)
    """
    print("\n▶  Generando cronograma con Secuenciador Óptimo (EDF + Knapsack)...")

    schedule_cache = ga.get_schedule_details(best)

    # Mapas de acceso rápido
    all_patients_map = {
        p.id: p
        for lista in patients_by_specialty.values()
        for p in lista
    }
    staff_map = {s.name: s for s in staff_list}

    pacientes_asignados_semana:   set = set()
    pacientes_descartados_semana: set = set()

    agenda_final = {
        "hospital":      "Hospital Centenario",
        "fitness_total": round(best.fitness, 4),
        "dias":          [],
    }

    for d in range(config.n_days):
        dia_dict = {"nombre": ga.DAY_NAMES[d], "bloques": []}

        for t in range(config.n_shifts):
            is_morning = (t == 0)

            # Dict {or_idx: per_or_result} para este turno
            shift_cache = {
                q: schedule_cache.get((d, t, q))
                for q in range(len(operating_rooms))
            }

            # ── Secuenciación óptima de todos los ORs del turno ──────────
            sequenced: Dict[int, SequencedOR] = sequence_shift(
                schedule_cache_entry = shift_cache,
                or_indices           = list(range(len(operating_rooms))),
                patients_map         = all_patients_map,
                staff_map            = staff_map,
                day_idx              = d,
                is_morning           = is_morning,
                t_max                = config.block_duration_min,
            )

            # ── Construir bloques del día ─────────────────────────────────
            for q_idx, or_obj in enumerate(operating_rooms):
                spec_id   = int(best.chromosome[d, t, q_idx])
                spec_name = next(s.name for s in specialties if s.id == spec_id)
                seq_or    = sequenced[q_idx]

                # Convertir SurgerySlots al formato JSON del proyecto
                cronograma = [
                    {
                        "paciente_id": slot.patient_id,
                        "medico":      slot.surgeon_name,
                        "hora_inicio": slot.hora_inicio,
                        "hora_fin":    slot.hora_fin,
                        "duracion":    slot.duracion,
                    }
                    for slot in seq_or.slots
                ]

                # Registrar pacientes programados y descartados
                for slot in seq_or.slots:
                    pacientes_asignados_semana.add(slot.patient_id)
                for pid in seq_or.skipped_patients:
                    pacientes_descartados_semana.add(pid)

                # Log informativo de descartados
                if seq_or.skipped_patients:
                    print(
                        f"  [i] {ga.DAY_NAMES[d]} ({ga.SHIFT_NAMES[t]}) | "
                        f"{or_obj.name}: {len(seq_or.skipped_patients)} paciente(s) "
                        f"sin ventana disponible → reprogramar"
                    )

                dia_dict["bloques"].append(
                    {
                        "quirofano":              or_obj.name,
                        "turno":                  ga.SHIFT_NAMES[t],
                        "especialidad":           spec_name,
                        "utilizacion_porcentaje": seq_or.utilizacion_porcentaje,
                        "cronograma":             cronograma,
                        "descartados":            seq_or.skipped_patients,
                    }
                )

        agenda_final["dias"].append(dia_dict)

    agenda_final["duracion_segundos"] = None
    return agenda_final, pacientes_asignados_semana


def main() -> None:
    """Función principal: orquesta la configuración, ejecución del AG y salida JSON."""
    random.seed(42)
    start_time = time.perf_counter()

    staff_list       = build_staff()
    operating_rooms  = build_operating_rooms()
    specialties      = build_specialties()
    patients_by_specialty = {
        sid: make_patients(sid, count=40, seed=sid, staff_list=staff_list)
        for sid in range(1, 6)
    }

    config = default_config()

    ga   = GeneticAlgorithm(config, operating_rooms, specialties, patients_by_specialty, staff_list)
    best = ga.run()
    ga.print_schedule(best)

    agenda_final, pacientes_asignados_semana = reconstruct_agenda(
        ga, best, patients_by_specialty, specialties, operating_rooms, staff_list, config
    )

    elapsed = time.perf_counter() - start_time
    agenda_final["duracion_segundos"] = round(elapsed, 3)

    with open("agenda_resultado.json", "w", encoding="utf-8") as f:
        json.dump(agenda_final, f, indent=4, ensure_ascii=False)

    print(f"\n✔ Éxito. Reporte generado. Pacientes totales: {len(pacientes_asignados_semana)}")
    print(f"Tiempo de ejecución: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
