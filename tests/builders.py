import sys
import os
from pathlib import Path

# Ensure project root is on sys.path when running this file directly
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import GAConfig, OperatingRoom, Patient, Procedure, Specialty, Staff

FREE_SPECIALTY_ID = 0

OR_ALTA_ID = 0
OR_MEDIA_ID = 1

TRAUMA_SPECIALTY_ID = 1
GENERAL_SPECIALTY_ID = 2

DR_ALTA_ID = 1
DR_MEDIA_ID = 2

TRAUMA_PROCEDURE_ID = 1001
GENERAL_PROCEDURE_ID = 2001

TRAUMA_PATIENT_1_ID = 101
TRAUMA_PATIENT_2_ID = 102
TRAUMA_PATIENT_3_ID = 103
TRAUMA_PATIENT_4_ID = 104

GENERAL_PATIENT_1_ID = 201
GENERAL_PATIENT_2_ID = 202
GENERAL_PATIENT_3_ID = 203
GENERAL_PATIENT_4_ID = 204


def make_patient(patient_id, specialty_id, procedure_id, duration, priority, forced_surgeon_id=None):
    return Patient(
        id=patient_id,
        specialty_id=specialty_id,
        procedure_id=procedure_id,
        estimated_duration=duration,
        clinical_priority=priority,
        required_roles=["cirujano"],
        forced_surgeon_id=forced_surgeon_id,
    )


def deterministic_end_to_end_scenario():
    # Esta config está pensada solo para tests:
    # población chica y pocas generaciones para que corra rápido,
    # 2 días y 1 turno para que el caso sea fácil de seguir,
    # y penalizaciones bien altas para obligar al AG a respetar
    # exactamente la cantidad de bloques que le pedimos a cada especialidad.
    config = GAConfig(
        population_size=6,
        max_generations=5,
        convergence_patience=2,
        mutation_rate=0.10,
        crossover_rate=0.85,
        tournament_size=2,
        elite_count=1,
        alpha=0.7,
        beta=0.3,
        n_days=2,
        n_shifts=1,
        block_duration_min=240,
        penalty_below_min_quota=1000.0,
        penalty_above_max_quota=1000.0,
    )

    # Hay dos quirófanos, uno de cada tipo. Cada especialidad puede ir en uno solo.
    operating_rooms = [
        OperatingRoom(
            id=OR_ALTA_ID,
            name="OR Alta",
            or_type="alta_complejidad",
            availability=[[True], [True]],
        ),
        OperatingRoom(
            id=OR_MEDIA_ID,
            name="OR Media",
            or_type="media_complejidad",
            availability=[[True], [True]],
        ),
    ]

    # Acá está la clave del caso de prueba:
    # para cada especialidad, el mínimo y el máximo de bloques valen 2.
    # Eso quiere decir que, en toda la semana de este escenario, Trauma tiene
    # que aparecer exactamente 2 veces y General también exactamente 2 veces.
    #
    # Como solo hay 2 días y 2 quirófanos, terminás con 4 bloques en total.
    # - día 0, OR alta -> tiene que ir Trauma, porque ese quirófano es de alta complejidad
    #   y General no es compatible con ese tipo de quirófano
    # - día 0, OR media -> tiene que ir General, porque ese quirófano es de media complejidad
    #   y Trauma no es compatible con ese tipo de quirófano
    # - día 1, OR alta -> vuelve a ir Trauma, para completar sus 2 bloques
    # - día 1, OR media -> vuelve a ir General, para completar sus 2 bloques
    specialties = [
        Specialty(id=FREE_SPECIALTY_ID, name="Libre", compatible_or_types=[], min_blocks=0, max_blocks=99),
        Specialty(id=TRAUMA_SPECIALTY_ID, name="Trauma", compatible_or_types=["alta_complejidad"], min_blocks=2, max_blocks=2),
        Specialty(id=GENERAL_SPECIALTY_ID, name="General", compatible_or_types=["media_complejidad"], min_blocks=2, max_blocks=2),
    ]
    procedures_by_specialty = {
        TRAUMA_SPECIALTY_ID: [
            Procedure(
                id=TRAUMA_PROCEDURE_ID,
                name="Trauma alta",
                specialty_id=TRAUMA_SPECIALTY_ID,
                required_room_type="alta_complejidad",
            )
        ],
        GENERAL_SPECIALTY_ID: [
            Procedure(
                id=GENERAL_PROCEDURE_ID,
                name="General media",
                specialty_id=GENERAL_SPECIALTY_ID,
                required_room_type="media_complejidad",
            )
        ],
    }

    # Cada paciente dura 120 minutos y cada bloque tiene 240, así que por día entran 2.
    # Como por especialidad cargamos 4 pacientes, los otros 2 quedan para el bloque del día siguiente.
    #
    # Les damos prioridades distintas para que el solver no empate cualquier combinación.
    # Así el test también verifica que elija primero a los pacientes más prioritarios.
    patients_by_specialty = {
        TRAUMA_SPECIALTY_ID: [
            make_patient(patient_id=TRAUMA_PATIENT_1_ID, specialty_id=TRAUMA_SPECIALTY_ID, procedure_id=TRAUMA_PROCEDURE_ID, duration=120, priority=10.0),
            make_patient(patient_id=TRAUMA_PATIENT_2_ID, specialty_id=TRAUMA_SPECIALTY_ID, procedure_id=TRAUMA_PROCEDURE_ID, duration=120, priority=9.0),
            make_patient(patient_id=TRAUMA_PATIENT_3_ID, specialty_id=TRAUMA_SPECIALTY_ID, procedure_id=TRAUMA_PROCEDURE_ID, duration=120, priority=8.0),
            make_patient(patient_id=TRAUMA_PATIENT_4_ID, specialty_id=TRAUMA_SPECIALTY_ID, procedure_id=TRAUMA_PROCEDURE_ID, duration=120, priority=7.0),
        ],
        GENERAL_SPECIALTY_ID: [
            make_patient(patient_id=GENERAL_PATIENT_1_ID, specialty_id=GENERAL_SPECIALTY_ID, procedure_id=GENERAL_PROCEDURE_ID, duration=120, priority=10.0),
            make_patient(patient_id=GENERAL_PATIENT_2_ID, specialty_id=GENERAL_SPECIALTY_ID, procedure_id=GENERAL_PROCEDURE_ID, duration=120, priority=9.0),
            make_patient(patient_id=GENERAL_PATIENT_3_ID, specialty_id=GENERAL_SPECIALTY_ID, procedure_id=GENERAL_PROCEDURE_ID, duration=120, priority=8.0),
            make_patient(patient_id=GENERAL_PATIENT_4_ID, specialty_id=GENERAL_SPECIALTY_ID, procedure_id=GENERAL_PROCEDURE_ID, duration=120, priority=7.0),
        ],
    }

    # Hay un solo cirujano por especialidad y labura los dos días.
    #
    # availability_hours se guarda como {día: (inicio, fin)} en minutos.
    # Por ejemplo, (480, 720) es de 08:00 a 12:00, o sea todo el turno mañana.
    staff_list = [
        Staff(
            id=DR_ALTA_ID,
            name="Dr Alta",
            role="cirujano",
            enabled_procedures_ids=[TRAUMA_PROCEDURE_ID],
            main_specialty_id=TRAUMA_SPECIALTY_ID,
            availability_hours={0: (480, 720), 1: (480, 720)},
        ),
        Staff(
            id=DR_MEDIA_ID,
            name="Dr Media",
            role="cirujano",
            enabled_procedures_ids=[GENERAL_PROCEDURE_ID],
            main_specialty_id=GENERAL_SPECIALTY_ID,
            availability_hours={0: (480, 720), 1: (480, 720)},
        ),
    ]

    # Esto es lo que esperamos que termine armando el AG:
    # día 0 -> [Trauma, General]
    # día 1 -> [Trauma, General]
    expected_chromosome = [
        [[TRAUMA_SPECIALTY_ID, GENERAL_SPECIALTY_ID]],
        [[TRAUMA_SPECIALTY_ID, GENERAL_SPECIALTY_ID]],
    ]

    # Y estos deberían ser los pacientes que asigna en cada bloque, siguiendo prioridad.
    expected_schedule = {
        (0, 0, OR_ALTA_ID): [TRAUMA_PATIENT_1_ID, TRAUMA_PATIENT_2_ID],
        (0, 0, OR_MEDIA_ID): [GENERAL_PATIENT_1_ID, GENERAL_PATIENT_2_ID],
        (1, 0, OR_ALTA_ID): [TRAUMA_PATIENT_3_ID, TRAUMA_PATIENT_4_ID],
        (1, 0, OR_MEDIA_ID): [GENERAL_PATIENT_3_ID, GENERAL_PATIENT_4_ID],
    }

    expected_all_patients = [
        TRAUMA_PATIENT_1_ID,
        TRAUMA_PATIENT_2_ID,
        TRAUMA_PATIENT_3_ID,
        TRAUMA_PATIENT_4_ID,
        GENERAL_PATIENT_1_ID,
        GENERAL_PATIENT_2_ID,
        GENERAL_PATIENT_3_ID,
        GENERAL_PATIENT_4_ID,
    ]

    return {
        "config": config,
        "operating_rooms": operating_rooms,
        "specialties": specialties,
        "procedures_by_specialty": procedures_by_specialty,
        "patients_by_specialty": patients_by_specialty,
        "staff_list": staff_list,
        "expected_chromosome": expected_chromosome,
        "expected_schedule": expected_schedule,
        "expected_all_patients": expected_all_patients,
    }
