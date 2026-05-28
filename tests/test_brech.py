"""
test_brecha_mip_seq.py

Reproduce el escenario exacto donde el MIP asigna pacientes que el
secuenciador no puede entregar porque no considera la cola del quirófano.

Escenario:
    Dr. A:  08:00-10:00 (120 min) → 3 pacientes × 40min = 120min
    Dr. B:  08:00-10:30 (150 min) → 3 pacientes × 50min = 150min

MIP: ve 120min para A y 150min para B → asigna 6 pacientes, todo OK.
SEQ: A ocupa el quirófano hasta las 10:00, a B le quedan solo 30min → 0 pacientes de 50min caben.
Brecha esperada: MIP dice 6, secuenciador entrega 3.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import Patient, Staff, OperatingRoom
from backup.mip import solve_mip_for_shift
from sequence import sequence_shift


QUIROFANO = [OperatingRoom(id=0, name="Q-Unico", or_type="alta_complejidad")]

STAFF = [
    # 120 min disponibles — sale a las 10:00
    Staff(id=1, name="Dr. A", role="cirujano", specialties_ids=[1],
          availability_hours={0: (480, 600)}),
    # 150 min disponibles — sale a las 10:30
    # El MIP ve 150min. El secuenciador ve solo 30min (quirófano ocupado hasta 10:00)
    Staff(id=2, name="Dr. B", role="cirujano", specialties_ids=[1],
          availability_hours={0: (480, 630)}),
]
STAFF_MAP = {s.name: s for s in STAFF}
T_MAX = 240


def correr(titulo, pacs_a, dur_a, pacs_b, dur_b):
    pacientes = (
        [Patient(id=i,   specialty_id=1, estimated_duration=dur_a, clinical_priority=5.0) for i in range(pacs_a)] +
        [Patient(id=i+pacs_a, specialty_id=1, estimated_duration=dur_b, clinical_priority=5.0) for i in range(pacs_b)]
    )
    # Forzamos médico para cada paciente para que el MIP no mezcle asignaciones
    for i in range(pacs_a):
        pacientes[i].forced_surgeon_id = 1          # Dr. A
    for i in range(pacs_b):
        pacientes[pacs_a + i].forced_surgeon_id = 2  # Dr. B

    patients_map = {p.id: p for p in pacientes}
    blocks = [{"or_idx": 0, "spec_id": 1, "patients": pacientes,
               "surgeons": STAFF, "t_max": T_MAX}]

    res_mip = solve_mip_for_shift(blocks, day_idx=0, is_morning=True)
    res_seq = sequence_shift(res_mip["per_or"], [0], patients_map, STAFF_MAP, 0, True, T_MAX)

    asig_mip = res_mip["per_or"][0]["asignaciones"]
    slots_seq = res_seq[0].slots
    skip_seq  = res_seq[0].skipped_patients

    print(f"\n{'='*58}")
    print(f"  {titulo}")
    print(f"{'='*58}")
    print(f"\nVentanas:")
    print(f"  Dr. A  08:00-10:00  ({pacs_a} pac × {dur_a}min = {pacs_a*dur_a}min)")
    print(f"  Dr. B  08:00-10:30  ({pacs_b} pac × {dur_b}min = {pacs_b*dur_b}min)")

    print(f"\nMIP asignó     : {len(asig_mip)} pacientes")
    for a in asig_mip:
        print(f"  Pac.{a['p']} → {a['doc']}")

    print(f"\nSecuenciador   : {len(slots_seq)} cirugías programadas")
    for s in slots_seq:
        print(f"  [{s.hora_inicio}-{s.hora_fin}] {s.surgeon_name}  Pac.{s.patient_id}")

    brecha = len(asig_mip) - len(slots_seq)
    if brecha > 0:
        print(f"\n  ⚠  BRECHA: MIP prometió {len(asig_mip)}, secuenciador entregó {len(slots_seq)}")
        print(f"     Pacientes perdidos: {skip_seq}")
        print(f"     Causa: Dr. B llega a las 08:00 pero el quirófano se libera")
        print(f"     a las 10:00 → ventana real = 10:30-10:00 = 30min")
        print(f"     El MIP calculó {pacs_b*dur_b}min disponibles para Dr. B (su ventana personal)")
        print(f"     pero la ventana real en la cola es solo 30min")
    else:
        print(f"\n  ✓  Sin brecha: MIP y secuenciador coinciden")


if __name__ == "__main__":

    print("""
┌─────────────────────────────────────────────────────┐
│  TEST DE BRECHA MIP vs SECUENCIADOR                 │
│                                                     │
│  Un quirófano. Dos médicos con ventanas solapadas.  │
│  Dr. A ocupa el OR hasta las 10:00.                 │
│  Dr. B cree tener 150min pero en realidad           │
│  solo encuentra 30min de quirófano libre.           │
└─────────────────────────────────────────────────────┘
""")

    # CASO 1: pacientes de 40min para B → 1 paciente sí cabe en 30min
    # No hay brecha porque 40min > 30min... espera, 40 > 30: tampoco cabe.
    # En realidad: ventana = 630-600 = 30min. Un paciente de 40min NO cabe.
    correr(
        titulo = "CASO 1 — Dr.B tiene pac de 40min (no caben en 30min de ventana real)",
        pacs_a = 3, dur_a = 40,
        pacs_b = 3, dur_b = 40,
    )

    # CASO 2: pacientes de 50min para B → definitivamente no caben en 30min
    correr(
        titulo = "CASO 2 — Dr.B tiene pac de 50min (MIP los acepta, SEQ los descarta)",
        pacs_a = 3, dur_a = 40,
        pacs_b = 3, dur_b = 50,
    )

    # CASO 3: pacientes de 30min para B → sí caben en 30min de ventana real
    # Aquí NO hay brecha: el único paciente de 30min entra justo
    correr(
        titulo = "CASO 3 — Dr.B tiene pac de 30min (justo caben, sin brecha)",
        pacs_a = 3, dur_a = 40,
        pacs_b = 1, dur_b = 30,
    )

    # CASO 4: Dr. A tiene ventanas más cortas → más espacio para B
    # Dr. A: 2 pac × 40min = 80min → quirófano libre a las 09:20
    # Dr. B: ventana real = 10:30 - 09:20 = 70min → 1 pac de 50min sí cabe
    correr(
        titulo = "CASO 4 — Dr.A más corto, Dr.B recupera 70min reales (sin brecha)",
        pacs_a = 2, dur_a = 40,
        pacs_b = 1, dur_b = 50,
    )