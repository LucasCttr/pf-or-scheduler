"""
tests/test_e2e.py
Tests de integracion end-to-end.

Verifican el ciclo completo del sistema:
    datos de entrada
        → GeneticAlgorithm.run()
        → mejor cromosoma
        → build_agenda() (ya ejecutado internamente por el AG)
        → agenda final
        → verificacion de todas las invariantes del dominio

A diferencia de test_invariants.py (que llama al decoder directamente),
estos tests ejercen el sistema completo: inicializacion de poblacion,
operadores geneticos, elitismo, criterio de parada y construccion de
agenda final. Si hay un bug en la interaccion entre el AG y el decoder
(por ejemplo, que el AG pase un cromosoma malformado al decoder, o que
el estado de `scheduled` quede contaminado entre evaluaciones), estos
tests lo van a detectar.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest

from models import Block, Patient, Procedure, Room, Specialty, Surgeon
from genetic_algorithm import GeneticAlgorithm

SHIFT_HOURS = 5  # debe coincidir con decoder.py


# ---------------------------------------------------------------------------
# Escenario compartido
# ---------------------------------------------------------------------------

@pytest.fixture
def scenario():
    """Escenario realista: 3 especialidades, 3 quirofanos, 5 dias, 9 cirujanos,
    30 pacientes. Suficientemente complejo para que el AG tenga algo que
    optimizar, suficientemente pequeno para que los tests sean rapidos."""
    days = ["lunes", "martes", "miercoles", "jueves", "viernes"]

    specialties = [
        Specialty(id="TRA", name="Traumatologia",   min_blocks=3),
        Specialty(id="CG",  name="Cirugia General", min_blocks=3),
        Specialty(id="URO", name="Urologia",         min_blocks=2),
    ]

    rooms = [
        Room(id="Q1", name="Quirofano 1", room_type=1, daily_capacity_minutes=300),
        Room(id="Q2", name="Quirofano 2", room_type=2, daily_capacity_minutes=300),
        Room(id="Q3", name="Quirofano 3", room_type=3, daily_capacity_minutes=300),
    ]

    procedures = [
        Procedure(id="PR1", name="Fractura",      specialty_id="TRA", required_room_type=1, estimated_duration=90),
        Procedure(id="PR2", name="Protesis",      specialty_id="TRA", required_room_type=2, estimated_duration=120),
        Procedure(id="PR3", name="Apendicectomia",specialty_id="CG",  required_room_type=1, estimated_duration=60),
        Procedure(id="PR4", name="Colecistectom", specialty_id="CG",  required_room_type=2, estimated_duration=90),
        Procedure(id="PR5", name="Nefrectomia",   specialty_id="URO", required_room_type=3, estimated_duration=150),
    ]

    surgeons = [
        Surgeon(id="S1", name="Dr. Lopez",  specialty_id="TRA",
                available_days={"lunes", "martes", "miercoles"}, contract_hours_week=15),
        Surgeon(id="S2", name="Dra. Diaz",  specialty_id="TRA",
                available_days={"jueves", "viernes"},            contract_hours_week=10),
        Surgeon(id="S3", name="Dr. Vera",   specialty_id="TRA",
                available_days={"lunes", "jueves"},              contract_hours_week=10),
        Surgeon(id="S4", name="Dr. Perez",  specialty_id="CG",
                available_days={"lunes", "miercoles", "viernes"}, contract_hours_week=15),
        Surgeon(id="S5", name="Dra. Sosa",  specialty_id="CG",
                available_days={"martes", "jueves"},             contract_hours_week=10),
        Surgeon(id="S6", name="Dr. Acosta", specialty_id="CG",
                available_days={"lunes", "martes"},              contract_hours_week=10),
        Surgeon(id="S7", name="Dra. Gomez", specialty_id="URO",
                available_days={"martes", "jueves"},             contract_hours_week=10),
        Surgeon(id="S8", name="Dr. Ruiz",   specialty_id="URO",
                available_days={"miercoles", "viernes"},         contract_hours_week=10),
        Surgeon(id="S9", name="Dra. Medina",specialty_id="URO",
                available_days={"lunes", "miercoles", "jueves"}, contract_hours_week=15),
    ]

    random.seed(7)
    proc_by_spec = {
        "TRA": [("PR1", "S1"), ("PR1", "S2"), ("PR2", "S1"), ("PR2", "S3"),
                ("PR1", "S3"), ("PR2", "S2"), ("PR1", "S1"), ("PR1", "S3"),
                ("PR2", "S2"), ("PR1", "S1")],
        "CG":  [("PR3", "S4"), ("PR4", "S4"), ("PR3", "S5"), ("PR4", "S6"),
                ("PR3", "S6"), ("PR4", "S5"), ("PR3", "S4"), ("PR4", "S6"),
                ("PR3", "S5"), ("PR4", "S4")],
        "URO": [("PR5", "S7"), ("PR5", "S8"), ("PR5", "S9"), ("PR5", "S7"),
                ("PR5", "S9"), ("PR5", "S8"), ("PR5", "S7"), ("PR5", "S9"),
                ("PR5", "S8"), ("PR5", "S7")],
    }

    patients = []
    pid = 1
    for spec_id, cases in proc_by_spec.items():
        for proc_id, surgeon_id in cases:
            patients.append(Patient(
                id=f"P{pid:02d}",
                specialty_id=spec_id,
                procedure_id=proc_id,
                surgeon_id=surgeon_id,
                clinical_priority=round(random.uniform(1.0, 10.0), 2),
            ))
            pid += 1

    return dict(
        days=days,
        rooms=rooms,
        specialties=specialties,
        surgeons=surgeons,
        procedures=procedures,
        patients=patients,
    )


@pytest.fixture
def ga_result(scenario):
    """Ejecuta el AG completo y devuelve (cromosoma, fitness, agenda, scenario)."""
    random.seed(42)
    ga = GeneticAlgorithm(
        days=scenario["days"],
        rooms=scenario["rooms"],
        specialties=scenario["specialties"],
        surgeons=scenario["surgeons"],
        procedures=scenario["procedures"],
        patients=scenario["patients"],
        population_size=20,
        generations=30,
        stagnation_limit=15,
        cleaning_minutes=15,
    )
    best_chromosome, best_fitness, best_agenda = ga.run()
    return best_chromosome, best_fitness, best_agenda, scenario, ga


# ---------------------------------------------------------------------------
# Tests estructurales del resultado
# ---------------------------------------------------------------------------

class TestResultStructure:

    def test_run_devuelve_cromosoma_con_todos_los_bloques(self, ga_result):
        best_chromosome, _, _, scenario, _ = ga_result
        expected_blocks = {
            Block(d, r.id)
            for d in scenario["days"]
            for r in scenario["rooms"]
        }
        assert set(best_chromosome.keys()) == expected_blocks

    def test_fitness_es_numero_finito(self, ga_result):
        _, best_fitness, _, _, _ = ga_result
        assert isinstance(best_fitness, float)
        assert best_fitness > float("-inf")

    def test_agenda_tiene_todos_los_bloques(self, ga_result):
        best_chromosome, _, best_agenda, _, _ = ga_result
        assert set(best_agenda.assignments.keys()) == set(best_chromosome.keys())
        assert set(best_agenda.used_time.keys()) == set(best_chromosome.keys())

    def test_al_menos_un_paciente_programado(self, ga_result):
        _, _, best_agenda, _, _ = ga_result
        assert len(best_agenda.all_surgeries()) > 0

    def test_historial_no_vacio_y_monotono(self, ga_result):
        _, _, _, _, ga = ga_result
        assert len(ga.history) >= 1
        for i in range(1, len(ga.history)):
            assert ga.history[i] >= ga.history[i - 1] - 1e-9, (
                f"Fitness bajó en generación {i}: "
                f"{ga.history[i-1]:.4f} → {ga.history[i]:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests de invariantes sobre la agenda final del AG
# ---------------------------------------------------------------------------

class TestE2EInvariants:

    def test_capacidad_quirofano_no_violada(self, ga_result):
        _, _, best_agenda, scenario, _ = ga_result
        rooms_by_id = {r.id: r for r in scenario["rooms"]}
        violations = []
        for block, used in best_agenda.used_time.items():
            cap = rooms_by_id[block.room_id].daily_capacity_minutes
            if used > cap:
                violations.append(f"{block}: usado={used} > cap={cap}")
        assert not violations, "\n".join(violations)

    def test_ningun_paciente_duplicado(self, ga_result):
        _, _, best_agenda, _, _ = ga_result
        ids = [s.patient_id for s in best_agenda.all_surgeries()]
        duplicates = [pid for pid in set(ids) if ids.count(pid) > 1]
        assert not duplicates, f"Pacientes duplicados: {duplicates}"

    def test_especialidad_de_cada_cirugia_coincide_con_bloque(self, ga_result):
        best_chromosome, _, best_agenda, scenario, _ = ga_result
        patients_by_id = {p.id: p for p in scenario["patients"]}
        violations = []
        for block, surgeries in best_agenda.assignments.items():
            expected = best_chromosome[block]
            for s in surgeries:
                actual = patients_by_id[s.patient_id].specialty_id
                if actual != expected:
                    violations.append(
                        f"{block}: esperada={expected}, "
                        f"paciente {s.patient_id} tiene {actual}"
                    )
        assert not violations, "\n".join(violations)

    def test_cirujano_disponible_en_dia_programado(self, ga_result):
        _, _, best_agenda, scenario, _ = ga_result
        surgeons_by_id = {s.id: s for s in scenario["surgeons"]}
        patients_by_id = {p.id: p for p in scenario["patients"]}
        violations = []
        for surgery in best_agenda.all_surgeries():
            sid = patients_by_id[surgery.patient_id].surgeon_id
            day = surgery.block.day
            if day not in surgeons_by_id[sid].available_days:
                violations.append(
                    f"Cirujano {sid} programado el {day} "
                    f"pero no tiene disponibilidad ese dia"
                )
        assert not violations, "\n".join(violations)

    def test_sala_compatible_con_procedimiento(self, ga_result):
        _, _, best_agenda, scenario, _ = ga_result
        rooms_by_id = {r.id: r for r in scenario["rooms"]}
        procedures_by_id = {p.id: p for p in scenario["procedures"]}
        patients_by_id = {p.id: p for p in scenario["patients"]}
        violations = []
        for surgery in best_agenda.all_surgeries():
            p = patients_by_id[surgery.patient_id]
            proc = procedures_by_id[p.procedure_id]
            room = rooms_by_id[surgery.block.room_id]
            if proc.required_room_type > room.room_type:
                violations.append(
                    f"Paciente {p.id}: proc requiere tipo "
                    f"{proc.required_room_type}, sala es tipo {room.room_type}"
                )
        assert not violations, "\n".join(violations)

    def test_horas_contractuales_no_superadas(self, ga_result):
        _, _, best_agenda, scenario, _ = ga_result
        surgeons_by_id = {s.id: s for s in scenario["surgeons"]}
        patients_by_id = {p.id: p for p in scenario["patients"]}

        days_used = {sid: set() for sid in surgeons_by_id}
        for surgery in best_agenda.all_surgeries():
            sid = patients_by_id[surgery.patient_id].surgeon_id
            days_used[sid].add(surgery.block.day)

        violations = []
        for sid, days in days_used.items():
            hours = len(days) * SHIFT_HOURS
            contract = surgeons_by_id[sid].contract_hours_week
            if hours > contract + 1e-9:
                violations.append(
                    f"Cirujano {sid}: {hours}h presencia > contrato {contract}h"
                )
        assert not violations, "\n".join(violations)

    def test_used_time_coherente_con_duraciones_y_limpieza(self, ga_result):
        _, _, best_agenda, _, ga = ga_result
        cleaning = ga.cleaning_minutes
        violations = []
        for block, surgeries in best_agenda.assignments.items():
            n = len(surgeries)
            if n == 0:
                continue
            expected = sum(s.duration for s in surgeries) + (n - 1) * cleaning
            actual = best_agenda.used_time[block]
            if actual != expected:
                violations.append(
                    f"{block}: used_time={actual}, esperado={expected} "
                    f"({n} cirugias, cleaning={cleaning})"
                )
        assert not violations, "\n".join(violations)

    def test_min_blocks_respetado_en_cromosoma(self, ga_result):
        """El cromosoma final debe respetar min_blocks de cada especialidad
        (garantizado por el operador de reparacion del AG)."""
        best_chromosome, _, _, scenario, _ = ga_result
        min_blocks = {s.id: s.min_blocks for s in scenario["specialties"]}
        counts = {}
        for sid in best_chromosome.values():
            counts[sid] = counts.get(sid, 0) + 1
        violations = []
        for sid, min_b in min_blocks.items():
            actual = counts.get(sid, 0)
            if actual < min_b:
                violations.append(
                    f"Especialidad {sid}: {actual} bloques < minimo {min_b}"
                )
        assert not violations, "\n".join(violations)
