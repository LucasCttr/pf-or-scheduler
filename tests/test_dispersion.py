"""
test_dispersion.py — escasez de médicos para ver dispersión real.

Solo 3 médicos para 3 quirófanos:
  Dr. Mañana   08:00-10:00 (120 min)
  Dra. Tarde   10:00-12:00 (120 min)
  Dr. Completo 08:00-12:00 (240 min) → candidato a dispersarse

Sin dispersión: OR0 y OR1 cubiertos, OR2 vacío → 12 pacientes máx.
Con dispersión de Completo entre OR1 (mañana) y OR2 (tarde): +3 pacientes.
"""
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import Patient, Staff, OperatingRoom
from backup.mip import solve_mip_for_shift
from sequence import sequence_shift

QUIROFANOS = [
    OperatingRoom(id=0, name="Q-Alfa",  or_type="alta_complejidad"),
    OperatingRoom(id=1, name="Q-Beta",  or_type="alta_complejidad"),
    OperatingRoom(id=2, name="Q-Gamma", or_type="alta_complejidad"),
]
STAFF = [
    Staff(id=1, name="Dr. Mañana",   role="cirujano", specialties_ids=[1], availability_hours={0: (480, 600)}),
    Staff(id=2, name="Dra. Tarde",   role="cirujano", specialties_ids=[1], availability_hours={0: (600, 720)}),
    Staff(id=3, name="Dr. Completo", role="cirujano", specialties_ids=[1], availability_hours={0: (480, 720)}),
]
STAFF_MAP = {s.name: s for s in STAFF}
T_MAX = 240
DUR  = 40

def correr(titulo, n, prioridad=5.0, delta=5.0):
    pacs  = [Patient(id=i, specialty_id=1, estimated_duration=DUR, clinical_priority=prioridad) for i in range(n)]
    pmap  = {p.id: p for p in pacs}
    blks  = [{"or_idx": i, "spec_id": 1, "patients": pacs, "surgeons": STAFF, "t_max": T_MAX} for i in range(3)]
    mip   = solve_mip_for_shift(blks, day_idx=0, is_morning=True, delta=delta)
    seq   = sequence_shift(mip["per_or"], [0,1,2], pmap, STAFF_MAP, 0, True, T_MAX)

    print(f"\n{'='*60}\n  {titulo}\n{'='*60}")
    print("MIP:")
    for q, d in mip["per_or"].items():
        docs = {}
        for a in d["asignaciones"]: docs[a["doc"]] = docs.get(a["doc"],0)+1
        print(f"  {QUIROFANOS[q].name}: {len(d['asignaciones'])} pac  {docs}")
    doc_ors = {}
    for q, d in mip["per_or"].items():
        for a in d["asignaciones"]: doc_ors.setdefault(a["doc"], set()).add(q)
    disp = {doc: sorted(ors) for doc, ors in doc_ors.items() if len(ors)>1}
    print("  Dispersión:", disp if disp else "ninguna ✓")

    total = sum(len(r.slots) for r in seq.values())
    print(f"Secuenciador — {total} cirugías:")
    for q, r in seq.items():
        print(f"  {QUIROFANOS[q].name}:", end="")
        if not r.slots: print(" (vacío)")
        else:
            print()
            for s in r.slots:
                print(f"    [{s.hora_inicio}-{s.hora_fin}] {s.surgeon_name}")

if __name__ == "__main__":
    correr("CASO 1 — 9 pac, delta=5.0 | carga baja, sin necesidad de dispersarse", 9, delta=5.0)
    correr("CASO 2 — 12 pac, delta=5.0 | llena 2 ORs exacto, no vale dispersarse", 12, delta=5.0)
    correr("CASO 3 — 15 pac, delta=5.0 | 3 extra solo entran si Completo se dispersa", 15, delta=5.0)
    correr("CASO 4 — 15 pac, delta=12.0 | dispersión cara, 3 pac quedan sin operar", 15, delta=12.0)
    correr("CASO 5 — 15 pac, delta=1.0 | delta bajo, dispersión sin dudarlo", 15, delta=1.0)