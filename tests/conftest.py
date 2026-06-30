"""
tests/conftest.py
Fixtures reutilizables para toda la suite de tests.

El escenario base es minimalista a propósito: 2 especialidades, 1 quirofano,
1 dia, pocos cirujanos y pacientes. Eso hace que los tests sean rápidos,
deterministas y fáciles de razonar a mano.

Para tests que necesitan variaciones (sala incompatible, contrato agotado,
capacidad exacta, etc.) cada archivo de test crea sus propias fixtures
locales o sobreescribe las de acá.
"""
import sys
import os

# Asegurar que el directorio raíz del proyecto esté en el path,
# independientemente de desde dónde se ejecute pytest.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import Block, Patient, Procedure, Room, Specialty, Surgeon


# ---------------------------------------------------------------------------
# Datos base
# ---------------------------------------------------------------------------

@pytest.fixture
def days():
    return ["lunes", "martes"]


@pytest.fixture
def specialties():
    return [
        Specialty(id="TRA", name="Traumatologia", min_blocks=1),
        Specialty(id="CG",  name="Cirugia General", min_blocks=1),
    ]


@pytest.fixture
def rooms():
    return [
        Room(id="Q1", name="Quirofano 1", room_type=2, daily_capacity_minutes=300),
    ]


@pytest.fixture
def procedures():
    return {
        "PR1": Procedure(id="PR1", name="Fractura", specialty_id="TRA",
                         required_room_type=1, estimated_duration=90),
        "PR2": Procedure(id="PR2", name="Apendicectomia", specialty_id="CG",
                         required_room_type=1, estimated_duration=60),
    }


@pytest.fixture
def surgeons():
    return {
        "S1": Surgeon(id="S1", name="Dr. Lopez", specialty_id="TRA",
                      available_days={"lunes", "martes"}, contract_hours_week=20),
        "S2": Surgeon(id="S2", name="Dr. Perez", specialty_id="CG",
                      available_days={"lunes", "martes"}, contract_hours_week=20),
    }


@pytest.fixture
def patients():
    """4 pacientes: 2 TRA y 2 CG, todos con cirujano disponible."""
    return [
        Patient(id="P1", specialty_id="TRA", procedure_id="PR1",
                surgeon_id="S1", clinical_priority=9.0),
        Patient(id="P2", specialty_id="TRA", procedure_id="PR1",
                surgeon_id="S1", clinical_priority=5.0),
        Patient(id="P3", specialty_id="CG",  procedure_id="PR2",
                surgeon_id="S2", clinical_priority=8.0),
        Patient(id="P4", specialty_id="CG",  procedure_id="PR2",
                surgeon_id="S2", clinical_priority=3.0),
    ]


@pytest.fixture
def rooms_by_id(rooms):
    return {r.id: r for r in rooms}


@pytest.fixture
def simple_chromosome(days, rooms):
    """Cromosoma sencillo: lunes=TRA, martes=CG."""
    return {
        Block("lunes",  "Q1"): "TRA",
        Block("martes", "Q1"): "CG",
    }
