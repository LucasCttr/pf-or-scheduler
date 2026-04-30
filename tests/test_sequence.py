"""
test_sequence.py — Tests del secuenciador (sequence.py)

Corre sin GA ni MIP. Cada test usa datos fijos con resultado esperado conocido.
Ejecutar con:
    python test_sequence.py
    pytest test_sequence.py -v        # si tenés pytest instalado
"""

from __future__ import annotations
import sys

from models import Patient, Staff
from sequence import sequence_or, SequencedOR


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_patient(pid: int, duration: int, priority: float, forced: int | None = None) -> Patient:
    return Patient(
        id=pid,
        specialty_id=1,
        estimated_duration=duration,
        clinical_priority=priority,
        required_roles=["cirujano"],
        forced_surgeon_id=forced,
    )


def make_surgeon(sid: int, name: str, day: int, start: int, end: int) -> Staff:
    return Staff(
        id=sid,
        name=name,
        role="cirujano",
        specialties_ids=[1],
        availability_hours={day: (start, end)},
    )


def run_seq(assignments, patients, surgeons, day=0, morning=True, t_max=240):
    """Wrapper que arma los mapas y llama a sequence_or."""
    patients_map = {p.id: p for p in patients}
    staff_map    = {s.name: s for s in surgeons}
    return sequence_or(
        or_idx       = 0,
        assignments  = assignments,
        patients_map = patients_map,
        staff_map    = staff_map,
        day_idx      = day,
        is_morning   = morning,
        t_max        = t_max,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Casos de test
# ═══════════════════════════════════════════════════════════════════════════════

def test_caso_basico_un_cirujano():
    """
    Un solo cirujano, todos los pacientes caben en su ventana.
    Resultado esperado: todos programados en orden de prioridad.

    Ventana: 08:00-12:00 (480-720), 240 min
    Pacientes: 3 x 60min = 180min total → entran todos
    """
    surgeon  = make_surgeon(1, "Dr. A", day=0, start=480, end=720)
    patients = [
        make_patient(1, duration=60, priority=9.0),
        make_patient(2, duration=60, priority=5.0),
        make_patient(3, duration=60, priority=7.0),
    ]
    assignments = [
        {"p": 1, "doc": "Dr. A"},
        {"p": 2, "doc": "Dr. A"},
        {"p": 3, "doc": "Dr. A"},
    ]

    result = run_seq(assignments, patients, [surgeon])

    assert len(result.slots) == 3, f"Esperaba 3 slots, obtuve {len(result.slots)}"
    assert len(result.skipped_patients) == 0

    # Orden: prioridad desc → P1(9.0), P3(7.0), P2(5.0)
    ids_orden = [s.patient_id for s in result.slots]
    assert ids_orden == [1, 3, 2], f"Orden incorrecto: {ids_orden}"

    # Tiempos exactos
    assert result.slots[0].hora_inicio == "08:00"
    assert result.slots[0].hora_fin    == "09:00"
    assert result.slots[1].hora_inicio == "09:00"
    assert result.slots[1].hora_fin    == "10:00"
    assert result.slots[2].hora_inicio == "10:00"
    assert result.slots[2].hora_fin    == "11:00"

    print("✔  test_caso_basico_un_cirujano")


def test_knapsack_descarta_menor_prioridad():
    """
    El cirujano tiene 90 min disponibles pero los pacientes suman 120 min.
    El knapsack debe descartar el de menor prioridad para maximizar el total.

    Ventana: 08:00-09:30 (480-570), 90 min disponibles
    Pacientes:
        P1 → 60min, prioridad 9.0
        P2 → 60min, prioridad 3.0   ← debe descartarse
    Óptimo: solo P1 (60min ≤ 90min, prioridad 9.0 > 3.0)
    """
    surgeon  = make_surgeon(1, "Dr. B", day=0, start=480, end=570)
    patients = [
        make_patient(1, duration=60, priority=9.0),
        make_patient(2, duration=60, priority=3.0),
    ]
    assignments = [{"p": 1, "doc": "Dr. B"}, {"p": 2, "doc": "Dr. B"}]

    result = run_seq(assignments, patients, [surgeon])

    assert len(result.slots) == 1
    assert result.slots[0].patient_id == 1
    assert 2 in result.skipped_patients

    print("✔  test_knapsack_descarta_menor_prioridad")


def test_knapsack_prefiere_combinacion_optima():
    """
    El knapsack elige la combinación de mayor prioridad total,
    no simplemente el paciente más prioritario.

    Ventana: 90 min
    Pacientes:
        P1 → 90min, prioridad 8.0   (entra solo, prioridad total = 8.0)
        P2 → 45min, prioridad 5.0   (juntos: 45+45=90min, total = 5.0+5.5=10.5)
        P3 → 45min, prioridad 5.5
    Óptimo: P2 + P3 (prioridad total 10.5 > 8.0)
    """
    surgeon  = make_surgeon(1, "Dr. C", day=0, start=480, end=570)
    patients = [
        make_patient(1, duration=90, priority=8.0),
        make_patient(2, duration=45, priority=5.0),
        make_patient(3, duration=45, priority=5.5),
    ]
    assignments = [
        {"p": 1, "doc": "Dr. C"},
        {"p": 2, "doc": "Dr. C"},
        {"p": 3, "doc": "Dr. C"},
    ]

    result = run_seq(assignments, patients, [surgeon])

    ids = {s.patient_id for s in result.slots}
    assert ids == {2, 3}, f"Esperaba P2+P3, obtuve {ids}"
    assert 1 in result.skipped_patients

    print("✔  test_knapsack_prefiere_combinacion_optima")


def test_edf_dos_cirujanos_sin_solapamiento():
    """
    Reproduce el caso del JSON de ejemplo: Dr. Pérez sale antes,
    Dra. Sosa empieza cuando el quirófano queda libre.

    Dr. Pérez:  08:00-10:20 (480-620), 140 min
    Dra. Sosa:  08:00-17:00 (480-1020)

    Pérez opera primero (EDF). El reloj del quirófano avanza.
    Sosa empieza cuando Pérez termina (no a las 08:00).
    """
    perez = make_surgeon(1, "Dr. Pérez", day=0, start=480, end=620)
    sosa  = make_surgeon(2, "Dra. Sosa", day=0, start=480, end=1020)

    patients = [
        make_patient(101, duration=45, priority=8.5),   # Pérez
        make_patient(102, duration=30, priority=7.2),   # Pérez
        make_patient(103, duration=30, priority=6.9),   # Pérez
        make_patient(104, duration=30, priority=5.1),   # Pérez  → suma=135 ≤ 140 ✓
        make_patient(201, duration=45, priority=7.0),   # Sosa
        make_patient(202, duration=45, priority=6.0),   # Sosa
    ]
    assignments = [
        {"p": 101, "doc": "Dr. Pérez"},
        {"p": 102, "doc": "Dr. Pérez"},
        {"p": 103, "doc": "Dr. Pérez"},
        {"p": 104, "doc": "Dr. Pérez"},
        {"p": 201, "doc": "Dra. Sosa"},
        {"p": 202, "doc": "Dra. Sosa"},
    ]

    result = run_seq(assignments, patients, [perez, sosa], t_max=240)

    assert len(result.skipped_patients) == 0, f"Descartados: {result.skipped_patients}"

    slots_perez = [s for s in result.slots if s.surgeon_name == "Dr. Pérez"]
    slots_sosa  = [s for s in result.slots if s.surgeon_name == "Dra. Sosa"]

    # Pérez termina antes de su deadline (10:20 = 620 min)
    ultimo_perez = slots_perez[-1]
    fin_perez_min = int(ultimo_perez.hora_fin[:2]) * 60 + int(ultimo_perez.hora_fin[3:])
    assert fin_perez_min <= 620, f"Pérez excede deadline: {ultimo_perez.hora_fin}"

    # Sosa empieza donde terminó Pérez (no a las 08:00)
    primer_sosa = slots_sosa[0]
    assert primer_sosa.hora_inicio != "08:00", \
        f"Sosa debería esperar a que Pérez termine, no empezar a las 08:00"

    # Sosa empieza exactamente cuando termina el último paciente de Pérez
    assert primer_sosa.hora_inicio == ultimo_perez.hora_fin, \
        f"Sosa debería empezar en {ultimo_perez.hora_fin}, empieza en {primer_sosa.hora_inicio}"

    print("✔  test_edf_dos_cirujanos_sin_solapamiento")


def test_cirujano_llega_tarde():
    """
    El cirujano llega después de que el quirófano ya lleva tiempo ocupado.
    Su cursor debe ser max(clock_quirofano, s_start).

    Cirujano A: 08:00-10:00 (480-600), opera 1 paciente de 90min
    Cirujano B: 09:30-12:00 (570-720), llega cuando A ya ocupó 90min

    Después de A: clock = 480 + 90 = 570
    B llega a las 570 también → cursor = max(570, 570) = 570 → empieza a las 09:30
    """
    cir_a = make_surgeon(1, "Dr. A", day=0, start=480, end=600)
    cir_b = make_surgeon(2, "Dr. B", day=0, start=570, end=720)

    patients = [
        make_patient(1, duration=90, priority=8.0),   # Dr. A
        make_patient(2, duration=60, priority=7.0),   # Dr. B
    ]
    assignments = [
        {"p": 1, "doc": "Dr. A"},
        {"p": 2, "doc": "Dr. B"},
    ]

    result = run_seq(assignments, patients, [cir_a, cir_b])

    assert len(result.slots) == 2
    assert result.slots[0].surgeon_name == "Dr. A"
    assert result.slots[0].hora_inicio  == "08:00"
    assert result.slots[0].hora_fin     == "09:30"
    assert result.slots[1].surgeon_name == "Dr. B"
    assert result.slots[1].hora_inicio  == "09:30"   # espera al quirófano, no llega antes
    assert result.slots[1].hora_fin     == "10:30"

    print("✔  test_cirujano_llega_tarde")


def test_cirujano_sin_disponibilidad():
    """
    Un cirujano no tiene disponibilidad el día indicado.
    Sus pacientes deben ir a skipped_patients.
    """
    # Solo tiene disponibilidad el martes (día 1), no el lunes (día 0)
    surgeon = make_surgeon(1, "Dr. X", day=1, start=480, end=720)
    patients = [make_patient(1, duration=60, priority=5.0)]
    assignments = [{"p": 1, "doc": "Dr. X"}]

    result = run_seq(assignments, patients, [surgeon], day=0)

    assert len(result.slots) == 0
    assert 1 in result.skipped_patients

    print("✔  test_cirujano_sin_disponibilidad")


def test_quirofano_agotado_antes_de_segundo_cirujano():
    """
    El primer cirujano consume todo el bloque. El segundo no tiene ventana.

    Cirujano A: 08:00-12:00 (480-720), opera 240min exactos
    Cirujano B: 08:00-12:00 (480-720), llega cuando clock=720 → ventana=0

    Todos los pacientes de B deben ir a skipped.
    """
    cir_a = make_surgeon(1, "Dr. A", day=0, start=480, end=720)
    cir_b = make_surgeon(2, "Dr. B", day=0, start=480, end=720)

    patients = [
        make_patient(1, duration=120, priority=9.0),   # Dr. A
        make_patient(2, duration=120, priority=8.0),   # Dr. A  → suma=240 = t_max
        make_patient(3, duration=60,  priority=7.0),   # Dr. B  → sin espacio
    ]
    assignments = [
        {"p": 1, "doc": "Dr. A"},
        {"p": 2, "doc": "Dr. A"},
        {"p": 3, "doc": "Dr. B"},
    ]

    result = run_seq(assignments, patients, [cir_a, cir_b], t_max=240)

    ids_programados = {s.patient_id for s in result.slots}
    assert ids_programados == {1, 2}
    assert 3 in result.skipped_patients

    print("✔  test_quirofano_agotado_antes_de_segundo_cirujano")


def test_turno_tarde():
    """
    Verifica que el turno tarde (is_morning=False) arranca a las 13:00 (780 min).
    """
    surgeon  = make_surgeon(1, "Dr. T", day=0, start=780, end=1020)
    patients = [make_patient(1, duration=60, priority=5.0)]
    assignments = [{"p": 1, "doc": "Dr. T"}]

    result = run_seq(assignments, patients, [surgeon], morning=False, t_max=240)

    assert len(result.slots) == 1
    assert result.slots[0].hora_inicio == "13:00"
    assert result.slots[0].hora_fin    == "14:00"

    print("✔  test_turno_tarde")


def test_utilizacion_correcta():
    """
    Verifica que utilizacion_porcentaje refleja solo el tiempo efectivamente usado.

    t_max = 240 min, se usan 90 min → 37.5%
    """
    surgeon  = make_surgeon(1, "Dr. U", day=0, start=480, end=720)
    patients = [
        make_patient(1, duration=60, priority=5.0),
        make_patient(2, duration=30, priority=4.0),
    ]
    assignments = [{"p": 1, "doc": "Dr. U"}, {"p": 2, "doc": "Dr. U"}]

    result = run_seq(assignments, patients, [surgeon], t_max=240)

    assert result.utilizacion_porcentaje == 37.5, \
        f"Esperaba 37.5%, obtuve {result.utilizacion_porcentaje}%"

    print("✔  test_utilizacion_correcta")


def test_sin_asignaciones():
    """Quirófano sin asignaciones del MIP → resultado vacío."""
    result = run_seq([], [], [])

    assert result.slots == []
    assert result.skipped_patients == []
    assert result.utilizacion_porcentaje == 0.0

    print("✔  test_sin_asignaciones")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

TESTS = [
    test_caso_basico_un_cirujano,
    test_knapsack_descarta_menor_prioridad,
    test_knapsack_prefiere_combinacion_optima,
    test_edf_dos_cirujanos_sin_solapamiento,
    test_cirujano_llega_tarde,
    test_cirujano_sin_disponibilidad,
    test_quirofano_agotado_antes_de_segundo_cirujano,
    test_turno_tarde,
    test_utilizacion_correcta,
    test_sin_asignaciones,
]


if __name__ == "__main__":
    print(f"Corriendo {len(TESTS)} tests...\n")
    failed = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as e:
            print(f"✘  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✘  {test.__name__}: ERROR inesperado → {e}")
            failed += 1

    print(f"\n{'─' * 40}")
    if failed == 0:
        print(f"✔  Todos los tests pasaron.")
    else:
        print(f"✘  {failed}/{len(TESTS)} tests fallaron.")
        sys.exit(1)
