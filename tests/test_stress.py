import json
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from back.models import Staff, OperatingRoom, Patient, Specialty, GAConfig
from backup.mip import solve_mip_for_shift
from sequence import sequence_shift

def test_escenario_complejo():
    # 1. Definimos 3 Quirófanos
    quirofanos = [
        OperatingRoom(id=1, name="Q-Alfa", or_type="alta_complejidad"),
        OperatingRoom(id=2, name="Q-Beta", or_type="alta_complejidad"),
        OperatingRoom(id=3, name="Q-Gamma", or_type="alta_complejidad"),
    ]

    # 2. Definimos 5 Médicos con horarios muy distintos
    # Bloque mañana: 08:00 (480) a 12:00 (720)
    staff = [
        # Staff(id=1, name="Dr. Temprano", role="cirujano", specialties_ids=[1], availability_hours={day: (480, 600) for day in range(5)}),
        Staff(id=2, name="Dra. Tarde", role="cirujano", specialties_ids=[1], availability_hours={day: (480, 720) for day in range(5)}),
        # Staff(id=3, name="Dr. TodoElDia", role="cirujano", specialties_ids=[1], availability_hours={day: (480, 720) for day in range(5)}),
        Staff(id=4, name="Dra. Corta", role="cirujano", specialties_ids=[1], availability_hours={day: (480, 660) for day in range(5)}),
        Staff(id=5, name="Dr. Rotante", role="cirujano", specialties_ids=[1], availability_hours={day: (480, 720) for day in range(5)}),
        Staff(id=3, name="Dr. asd", role="cirujano", specialties_ids=[1], availability_hours={day: (480, 540) for day in range(5)}),
    ]

    # 3. Especialidad y muchos pacientes para estresar al MIP
    spec = Specialty(id=1, name="Cirugía General", compatible_or_types=["alta_complejidad"])
    pacientes = [Patient(id=i, specialty_id=1, estimated_duration=40, clinical_priority=1.0) for i in range(15)]
    
    patients_map = {p.id: p for p in pacientes}
    staff_map = {s.name: s for s in staff}

    # 4. Preparamos los bloques para el MIP (Turno Mañana)
    blocks = []
    for i in range(3):
        blocks.append({
            "or_idx": i,
            "spec_id": 1,
            "patients": pacientes,
            "surgeons": staff,
            "t_max": 240 # 4 horas
        })

    print("--- EJECUTANDO MIP ---")
    resultado_mip = solve_mip_for_shift(blocks, day_idx=0, is_morning=True)

    print("--- EJECUTANDO SECUENCIADOR ---")
    resultado_seq = sequence_shift(
        schedule_cache_entry = resultado_mip["per_or"],
        or_indices           = [0, 1, 2],
        patients_map         = patients_map,
        staff_map            = staff_map,
        day_idx              = 0,
        is_morning           = True,
        t_max                = 240
    )

    # 5. Visualización de Resultados
    print("\n" + "="*60)
    print("RESULTADO DE LA SECUENCIACIÓN (5 MÉDICOS / 3 SALAS)")
    print("="*60)
    
    for q_idx, res in resultado_seq.items():
        print(f"\n>> {quirofanos[q_idx].name}:")
        if not res.slots:
            print("   (Vacío)")
        for slot in res.slots:
            print(f"   [{slot.hora_inicio} - {slot.hora_fin}] | {slot.surgeon_name} | Paciente {slot.patient_id}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_escenario_complejo()