"""
tests/test_decoder.py
Tests unitarios del decoder (build_agenda).

Cada test verifica exactamente un comportamiento:
  - Que programe pacientes factibles.
  - Que respete la capacidad del quirofano.
  - Que la limpieza descuente capacidad (no sea tiempo ocioso).
  - Que ordene por prioridad clinica.
  - Que filtre por disponibilidad del cirujano.
  - Que filtre por compatibilidad de sala.
  - Que respete el contrato horario del cirujano.
  - Que no programe al mismo paciente dos veces.
"""
import pytest
from models import Block, Patient, Procedure, Room, Surgeon
from decoder import build_agenda


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scheduled_ids(agenda):
    return {s.patient_id for s in agenda.all_surgeries()}


# ---------------------------------------------------------------------------
# Tests básicos
# ---------------------------------------------------------------------------

class TestBasicScheduling:

    def test_programa_paciente_factible(
        self, simple_chromosome, patients, procedures, surgeons, rooms_by_id
    ):
        """Un paciente cuyo cirujano está disponible y entra en la capacidad
        debe quedar programado."""
        agenda = build_agenda(simple_chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert "P1" in scheduled_ids(agenda)

    def test_no_programa_paciente_ya_programado(
        self, simple_chromosome, patients, procedures, surgeons, rooms_by_id
    ):
        """Un paciente no puede aparecer dos veces en la agenda."""
        agenda = build_agenda(simple_chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        ids = [s.patient_id for s in agenda.all_surgeries()]
        assert len(ids) == len(set(ids)), "Un paciente fue programado más de una vez"

    def test_used_time_no_supera_capacidad(
        self, simple_chromosome, patients, procedures, surgeons, rooms_by_id
    ):
        """El tiempo utilizado por bloque nunca debe superar la capacidad
        diaria del quirofano."""
        agenda = build_agenda(simple_chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=15)
        for block, used in agenda.used_time.items():
            room = rooms_by_id[block.room_id]
            assert used <= room.daily_capacity_minutes, (
                f"{block}: usado={used} > capacidad={room.daily_capacity_minutes}"
            )

    def test_prioridad_alta_se_programa_primero(
        self, simple_chromosome, patients, procedures, surgeons, rooms_by_id
    ):
        """Cuando no entran todos los pacientes, los de mayor prioridad
        deben quedar programados."""
        # Reducir la capacidad a solo 90 min (una cirugía TRA)
        rooms_by_id["Q1"] = Room(
            id="Q1", name="Quirofano 1", room_type=2, daily_capacity_minutes=90
        )
        agenda = build_agenda(simple_chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        ids = scheduled_ids(agenda)
        # P1 (prioridad 9) debe entrar; P2 (prioridad 5) no
        assert "P1" in ids
        assert "P2" not in ids


# ---------------------------------------------------------------------------
# Tests de ventana de limpieza
# ---------------------------------------------------------------------------

class TestCleaningWindow:

    def test_limpieza_descuenta_capacidad(
        self, procedures, surgeons, rooms_by_id
    ):
        """Con 2 cirugías de 90 min y limpieza de 15: se usan 90+15+90=195 min."""
        block = Block("lunes", "Q1")
        chromosome = {block: "TRA"}
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
            Patient(id="P2", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=5.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=15)
        assert agenda.used_time[block] == 195
        assert agenda.cleaning_time[block] == 15

    def test_limpieza_cero_no_descuenta_nada(
        self, procedures, surgeons, rooms_by_id
    ):
        """Con cleaning_minutes=0, used_time debe ser exactamente la suma
        de duraciones, sin penalización de limpieza."""
        block = Block("lunes", "Q1")
        chromosome = {block: "TRA"}
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
            Patient(id="P2", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=5.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert agenda.used_time[block] == 180  # 90 + 90
        assert agenda.cleaning_time[block] == 0

    def test_primera_cirugia_no_paga_limpieza(
        self, procedures, surgeons, rooms_by_id
    ):
        """La primera cirugía del bloque nunca paga ventana de limpieza."""
        block = Block("lunes", "Q1")
        chromosome = {block: "TRA"}
        # Paciente con duración 90, capacidad exactamente 90
        rooms_by_id["Q1"] = Room(
            id="Q1", name="Q1", room_type=2, daily_capacity_minutes=90
        )
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=15)
        # Debe programarse aunque capacidad == duración (sin pagar limpieza previa)
        assert "P1" in scheduled_ids(agenda)

    def test_limpieza_impide_segunda_cirugia_si_no_hay_espacio(
        self, procedures, surgeons, rooms_by_id
    ):
        """Con capacidad=170, dos cirugías de 90 min y limpieza=15:
        90+15+90=195 > 170, así que solo entra la primera."""
        block = Block("lunes", "Q1")
        chromosome = {block: "TRA"}
        rooms_by_id["Q1"] = Room(
            id="Q1", name="Q1", room_type=2, daily_capacity_minutes=170
        )
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
            Patient(id="P2", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=5.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=15)
        ids = scheduled_ids(agenda)
        assert "P1" in ids
        assert "P2" not in ids

    def test_used_time_igual_duraciones_mas_limpieza(
        self, procedures, surgeons, rooms_by_id
    ):
        """Para N cirugías programadas: used = sum(duraciones) + (N-1)*cleaning."""
        block = Block("lunes", "Q1")
        chromosome = {block: "TRA"}
        cleaning = 10
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
            Patient(id="P2", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=5.0),
            Patient(id="P3", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=3.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=cleaning)
        n = len(agenda.assignments[block])
        sum_dur = sum(s.duration for s in agenda.assignments[block])
        assert agenda.used_time[block] == sum_dur + (n - 1) * cleaning


# ---------------------------------------------------------------------------
# Tests de filtros de factibilidad
# ---------------------------------------------------------------------------

class TestFeasibilityFilters:

    def test_cirujano_no_disponible_ese_dia(
        self, procedures, rooms_by_id
    ):
        """Un paciente cuyo cirujano no está disponible el día del bloque
        no debe ser programado."""
        surgeons = {
            "S1": Surgeon(id="S1", name="Dr. Lopez", specialty_id="TRA",
                          available_days={"martes"},  # solo martes
                          contract_hours_week=20),
        }
        block = Block("lunes", "Q1")  # bloque en lunes
        chromosome = {block: "TRA"}
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert "P1" not in scheduled_ids(agenda)

    def test_sala_incompatible_excluye_paciente(
        self, surgeons, rooms_by_id
    ):
        """Un procedimiento que requiere room_type=3 no puede programarse
        en un quirofano de room_type=2."""
        procedures = {
            "PR_HIGH": Procedure(id="PR_HIGH", name="Compleja",
                                  specialty_id="TRA", required_room_type=3,
                                  estimated_duration=60),
        }
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR_HIGH",
                    surgeon_id="S1", clinical_priority=9.0),
        ]
        block = Block("lunes", "Q1")  # Q1 tiene room_type=2
        chromosome = {block: "TRA"}
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert "P1" not in scheduled_ids(agenda)

    def test_sala_compatible_igual_room_type(
        self, surgeons, rooms_by_id
    ):
        """required_room_type == room_type del quirofano debe ser compatible."""
        procedures = {
            "PR_EXACT": Procedure(id="PR_EXACT", name="Exacta",
                                   specialty_id="TRA", required_room_type=2,
                                   estimated_duration=60),
        }
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR_EXACT",
                    surgeon_id="S1", clinical_priority=9.0),
        ]
        block = Block("lunes", "Q1")  # Q1 room_type=2
        chromosome = {block: "TRA"}
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert "P1" in scheduled_ids(agenda)

    def test_especialidad_incorrecta_en_bloque(
        self, patients, procedures, surgeons, rooms_by_id
    ):
        """Pacientes TRA no deben programarse en un bloque asignado a CG."""
        block = Block("lunes", "Q1")
        chromosome = {block: "CG"}  # bloque asignado a CG, no TRA
        tra_patients = [p for p in patients if p.specialty_id == "TRA"]
        agenda = build_agenda(chromosome, tra_patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert len(agenda.all_surgeries()) == 0

    def test_bloque_vacio_sin_pacientes_de_esa_especialidad(
        self, procedures, surgeons, rooms_by_id
    ):
        """Si no hay pacientes de la especialidad asignada, el bloque queda vacío."""
        block = Block("lunes", "Q1")
        chromosome = {block: "URO"}  # no hay pacientes de URO en el fixture
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert agenda.used_time[block] == 0
        assert agenda.assignments[block] == []


# ---------------------------------------------------------------------------
# Tests de restricción contractual
# ---------------------------------------------------------------------------

class TestContractHours:

    def test_cirujano_con_contrato_agotado_no_puede_operar(
        self, procedures
    ):
        """Un cirujano con contract_hours_week=5 (=1 jornada) puede operar
        el lunes, pero no puede aparecer el martes porque su contrato ya está agotado.

        Configuración: capacidad del lunes reducida a 90 min (solo entra P1).
        P2 queda fuera del lunes por capacidad, e intenta el martes pero
        el cirujano ya agotó su contrato semanal.
        """
        surgeons = {
            "S1": Surgeon(id="S1", name="Dr. Lopez", specialty_id="TRA",
                          available_days={"lunes", "martes"},
                          contract_hours_week=5),  # exactamente 1 jornada de 5h
        }
        # Lunes: capacidad exacta para 1 cirugía de 90 min
        rooms_by_id = {
            "Q1": Room(id="Q1", name="Q1", room_type=2, daily_capacity_minutes=90)
        }
        chromosome = {
            Block("lunes",  "Q1"): "TRA",
            Block("martes", "Q1"): "TRA",
        }
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
            Patient(id="P2", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=5.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        ids = scheduled_ids(agenda)
        # P1 entra en lunes y agota el contrato del cirujano.
        assert "P1" in ids
        # P2 no entra en lunes (capacidad llena) ni en martes (contrato agotado).
        assert "P2" not in ids

    def test_cirujano_con_contrato_suficiente_opera_dos_dias(
        self, procedures, rooms_by_id
    ):
        """Un cirujano con contrato de 10h puede cubrir dos jornadas de 5h."""
        surgeons = {
            "S1": Surgeon(id="S1", name="Dr. Lopez", specialty_id="TRA",
                          available_days={"lunes", "martes"},
                          contract_hours_week=10),
        }
        chromosome = {
            Block("lunes",  "Q1"): "TRA",
            Block("martes", "Q1"): "TRA",
        }
        patients = [
            Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=9.0),
            Patient(id="P2", specialty_id="TRA", procedure_id="PR1",
                    surgeon_id="S1", clinical_priority=5.0),
        ]
        agenda = build_agenda(chromosome, patients, procedures,
                               surgeons, rooms_by_id, cleaning_minutes=0)
        assert len(agenda.all_surgeries()) == 2
