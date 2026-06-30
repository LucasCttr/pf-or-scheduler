"""
tests/test_invariants.py
Tests de invariantes del dominio: verifican que la agenda producida por el
decoder nunca viole ninguna restriccion operativa, sin importar los datos
de entrada.

A diferencia de los tests unitarios (que prueban un comportamiento puntual
con datos minimos controlados), estos tests corren el decoder con escenarios
mas realistas y luego inspeccionan la agenda completa buscando cualquier
violacion posible.

Restricciones verificadas
--------------------------
1. Capacidad de quirofano: used_time <= daily_capacity_minutes para todo bloque.
2. Paciente unico: ningun paciente aparece mas de una vez en la agenda.
3. Especialidad correcta: cada cirugia programada en un bloque corresponde
   a la especialidad asignada por el cromosoma a ese bloque.
4. Disponibilidad del cirujano: el cirujano de cada cirugia debe tener
   disponibilidad en el dia del bloque donde fue programado.
5. Compatibilidad de sala: required_room_type del procedimiento <= room_type
   del quirofano donde fue programado.
6. Carga horaria contractual: las horas de presencia acumuladas de cada
   cirujano no superan su contract_hours_week.
7. Ventana de limpieza: el tiempo utilizado de cada bloque es coherente con
   la suma de duraciones mas las ventanas de limpieza entre cirugias.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import Block, Patient, Procedure, Room, Specialty, Surgeon
from decoder import build_agenda

SHIFT_HOURS = 5  # debe coincidir con el valor hardcodeado en decoder.py


# ---------------------------------------------------------------------------
# Fixtures: escenario realista con varios dias, salas, cirujanos y pacientes
# ---------------------------------------------------------------------------

@pytest.fixture
def days():
    return ["lunes", "martes", "miercoles", "jueves", "viernes"]


@pytest.fixture
def rooms():
    return {
        "Q1": Room(id="Q1", name="Quirofano 1", room_type=1, daily_capacity_minutes=300),
        "Q2": Room(id="Q2", name="Quirofano 2", room_type=2, daily_capacity_minutes=240),
        "Q3": Room(id="Q3", name="Quirofano 3", room_type=3, daily_capacity_minutes=300),
    }


@pytest.fixture
def procedures():
    return {
        "PR1": Procedure(id="PR1", name="Fractura",      specialty_id="TRA", required_room_type=1, estimated_duration=90),
        "PR2": Procedure(id="PR2", name="Protesis",      specialty_id="TRA", required_room_type=2, estimated_duration=120),
        "PR3": Procedure(id="PR3", name="Apendice",      specialty_id="CG",  required_room_type=1, estimated_duration=60),
        "PR4": Procedure(id="PR4", name="Colecistectom", specialty_id="CG",  required_room_type=2, estimated_duration=90),
        "PR5": Procedure(id="PR5", name="Nefrectomia",   specialty_id="URO", required_room_type=3, estimated_duration=150),
    }


@pytest.fixture
def surgeons():
    return {
        "S1": Surgeon(id="S1", name="Dr. Lopez",    specialty_id="TRA",
                      available_days={"lunes", "martes", "miercoles"}, contract_hours_week=15),
        "S2": Surgeon(id="S2", name="Dra. Diaz",    specialty_id="TRA",
                      available_days={"jueves", "viernes"},            contract_hours_week=10),
        "S3": Surgeon(id="S3", name="Dr. Perez",    specialty_id="CG",
                      available_days={"lunes", "miercoles", "viernes"}, contract_hours_week=15),
        "S4": Surgeon(id="S4", name="Dra. Gomez",   specialty_id="URO",
                      available_days={"martes", "jueves"},             contract_hours_week=10),
    }


@pytest.fixture
def patients():
    """Lista variada: distintas especialidades, procedimientos y cirujanos."""
    return [
        Patient(id="P01", specialty_id="TRA", procedure_id="PR1", surgeon_id="S1", clinical_priority=9.0),
        Patient(id="P02", specialty_id="TRA", procedure_id="PR1", surgeon_id="S1", clinical_priority=7.5),
        Patient(id="P03", specialty_id="TRA", procedure_id="PR2", surgeon_id="S1", clinical_priority=6.0),
        Patient(id="P04", specialty_id="TRA", procedure_id="PR1", surgeon_id="S2", clinical_priority=8.0),
        Patient(id="P05", specialty_id="TRA", procedure_id="PR2", surgeon_id="S2", clinical_priority=5.0),
        Patient(id="P06", specialty_id="CG",  procedure_id="PR3", surgeon_id="S3", clinical_priority=9.5),
        Patient(id="P07", specialty_id="CG",  procedure_id="PR4", surgeon_id="S3", clinical_priority=7.0),
        Patient(id="P08", specialty_id="CG",  procedure_id="PR3", surgeon_id="S3", clinical_priority=4.0),
        Patient(id="P09", specialty_id="URO", procedure_id="PR5", surgeon_id="S4", clinical_priority=8.5),
        Patient(id="P10", specialty_id="URO", procedure_id="PR5", surgeon_id="S4", clinical_priority=6.5),
    ]


@pytest.fixture
def chromosome_mixed(days, rooms):
    """Cromosoma realista: especialidades distribuidas entre bloques."""
    specialties = ["TRA", "CG", "URO"]
    chrom = {}
    for i, day in enumerate(days):
        for j, room_id in enumerate(rooms):
            chrom[Block(day, room_id)] = specialties[(i + j) % len(specialties)]
    return chrom


# ---------------------------------------------------------------------------
# Helper: calcula horas de presencia reales a partir de la agenda
# ---------------------------------------------------------------------------

def compute_presence_hours(agenda, surgeons, patients_by_id):
    """Reconstruye las horas de presencia por cirujano a partir de la agenda."""
    days_used = {sid: set() for sid in surgeons}
    for surgery in agenda.all_surgeries():
        sid = patients_by_id[surgery.patient_id].surgeon_id
        days_used[sid].add(surgery.block.day)
    return {
        sid: len(days) * SHIFT_HOURS
        for sid, days in days_used.items()
    }


# ---------------------------------------------------------------------------
# Tests de invariantes
# ---------------------------------------------------------------------------

class TestAgendaInvariants:

    def _run(self, chromosome, patients, procedures, surgeons, rooms,
             cleaning_minutes=15):
        """Ejecuta el decoder y devuelve (agenda, patients_by_id)."""
        agenda = build_agenda(
            chromosome, patients, procedures, surgeons, rooms,
            cleaning_minutes=cleaning_minutes,
        )
        patients_by_id = {p.id: p for p in patients}
        return agenda, patients_by_id

    # ------------------------------------------------------------------
    # 1. Capacidad de quirofano
    # ------------------------------------------------------------------

    def test_used_time_nunca_supera_capacidad(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """used_time[block] <= daily_capacity_minutes para todo bloque."""
        agenda, _ = self._run(chromosome_mixed, patients, procedures,
                               surgeons, rooms)
        violations = []
        for block, used in agenda.used_time.items():
            cap = rooms[block.room_id].daily_capacity_minutes
            if used > cap:
                violations.append(
                    f"{block}: usado={used} > capacidad={cap}"
                )
        assert not violations, "Violaciones de capacidad:\n" + "\n".join(violations)

    # ------------------------------------------------------------------
    # 2. Paciente unico en la agenda
    # ------------------------------------------------------------------

    def test_cada_paciente_programado_como_maximo_una_vez(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """Ningun patient_id aparece mas de una vez en la agenda."""
        agenda, _ = self._run(chromosome_mixed, patients, procedures,
                               surgeons, rooms)
        ids = [s.patient_id for s in agenda.all_surgeries()]
        duplicates = [pid for pid in set(ids) if ids.count(pid) > 1]
        assert not duplicates, f"Pacientes duplicados: {duplicates}"

    # ------------------------------------------------------------------
    # 3. Especialidad correcta en cada bloque
    # ------------------------------------------------------------------

    def test_cirugias_corresponden_a_especialidad_del_bloque(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """Cada cirugia programada en un bloque pertenece a la especialidad
        asignada a ese bloque por el cromosoma."""
        agenda, patients_by_id = self._run(chromosome_mixed, patients,
                                            procedures, surgeons, rooms)
        violations = []
        for block, surgeries in agenda.assignments.items():
            expected_specialty = chromosome_mixed[block]
            for s in surgeries:
                actual = patients_by_id[s.patient_id].specialty_id
                if actual != expected_specialty:
                    violations.append(
                        f"{block}: esperada={expected_specialty}, "
                        f"paciente {s.patient_id} tiene {actual}"
                    )
        assert not violations, "Violaciones de especialidad:\n" + "\n".join(violations)

    # ------------------------------------------------------------------
    # 4. Disponibilidad del cirujano
    # ------------------------------------------------------------------

    def test_cirujano_disponible_en_dia_del_bloque(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """El cirujano asignado a cada cirugia debe tener disponibilidad
        declarada en el dia del bloque donde fue programado."""
        agenda, patients_by_id = self._run(chromosome_mixed, patients,
                                            procedures, surgeons, rooms)
        violations = []
        for surgery in agenda.all_surgeries():
            sid = patients_by_id[surgery.patient_id].surgeon_id
            day = surgery.block.day
            if day not in surgeons[sid].available_days:
                violations.append(
                    f"Cirujano {sid} programado el {day} "
                    f"pero no tiene disponibilidad ese dia"
                )
        assert not violations, "Violaciones de disponibilidad:\n" + "\n".join(violations)

    # ------------------------------------------------------------------
    # 5. Compatibilidad de sala
    # ------------------------------------------------------------------

    def test_procedimiento_compatible_con_sala(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """required_room_type del procedimiento <= room_type del quirofano."""
        agenda, patients_by_id = self._run(chromosome_mixed, patients,
                                            procedures, surgeons, rooms)
        violations = []
        for surgery in agenda.all_surgeries():
            p = patients_by_id[surgery.patient_id]
            proc = procedures[p.procedure_id]
            room = rooms[surgery.block.room_id]
            if proc.required_room_type > room.room_type:
                violations.append(
                    f"Paciente {p.id}: procedimiento requiere sala tipo "
                    f"{proc.required_room_type} pero fue asignado a "
                    f"{surgery.block.room_id} (tipo {room.room_type})"
                )
        assert not violations, "Violaciones de compatibilidad de sala:\n" + "\n".join(violations)

    # ------------------------------------------------------------------
    # 6. Carga horaria contractual
    # ------------------------------------------------------------------

    def test_horas_presencia_no_superan_contrato(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """Las horas de presencia acumuladas de cada cirujano no superan
        su contract_hours_week."""
        agenda, patients_by_id = self._run(chromosome_mixed, patients,
                                            procedures, surgeons, rooms)
        presence = compute_presence_hours(agenda, surgeons, patients_by_id)
        violations = []
        for sid, hours in presence.items():
            contract = surgeons[sid].contract_hours_week
            if hours > contract + 1e-9:
                violations.append(
                    f"Cirujano {sid}: {hours}h de presencia > "
                    f"contrato {contract}h"
                )
        assert not violations, "Violaciones contractuales:\n" + "\n".join(violations)

    # ------------------------------------------------------------------
    # 7. Coherencia de used_time con duraciones + limpieza
    # ------------------------------------------------------------------

    def test_used_time_coherente_con_duraciones_y_limpieza(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """Para cada bloque: used_time == sum(duraciones) + (N-1)*cleaning."""
        cleaning = 15
        agenda, _ = self._run(chromosome_mixed, patients, procedures,
                               surgeons, rooms, cleaning_minutes=cleaning)
        violations = []
        for block, surgeries in agenda.assignments.items():
            n = len(surgeries)
            if n == 0:
                continue
            expected = sum(s.duration for s in surgeries) + (n - 1) * cleaning
            actual = agenda.used_time[block]
            if actual != expected:
                violations.append(
                    f"{block}: used_time={actual}, "
                    f"esperado={expected} "
                    f"({n} cirugias, limpieza={cleaning})"
                )
        assert not violations, "Incoherencias en used_time:\n" + "\n".join(violations)

    # ------------------------------------------------------------------
    # Invariantes con cleaning_minutes=0 (caso borde)
    # ------------------------------------------------------------------

    def test_invariantes_sin_limpieza(
        self, chromosome_mixed, patients, procedures, surgeons, rooms
    ):
        """Con cleaning_minutes=0 todas las restricciones siguen vigentes."""
        agenda, patients_by_id = self._run(chromosome_mixed, patients,
                                            procedures, surgeons, rooms,
                                            cleaning_minutes=0)
        # Capacidad
        for block, used in agenda.used_time.items():
            cap = rooms[block.room_id].daily_capacity_minutes
            assert used <= cap, f"{block}: usado={used} > capacidad={cap}"

        # Unicidad
        ids = [s.patient_id for s in agenda.all_surgeries()]
        assert len(ids) == len(set(ids))

        # Carga horaria
        presence = compute_presence_hours(agenda, surgeons, patients_by_id)
        for sid, hours in presence.items():
            assert hours <= surgeons[sid].contract_hours_week + 1e-9, (
                f"Cirujano {sid}: {hours}h > contrato {surgeons[sid].contract_hours_week}h"
            )
