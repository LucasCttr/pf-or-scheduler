from genetic_algorithm import GeneticAlgorithm
from models import GAConfig, OperatingRoom, Procedure, Specialty, Staff


def _ga_for_contract(*, room_type="media_complejidad", staff_main_specialty=1):
    specialty = Specialty(
        id=1,
        name="Trauma",
        compatible_or_types=["alta_complejidad", "media_complejidad"],
    )
    procedure = Procedure(
        id=101,
        name="Procedimiento alta",
        specialty_id=1,
        required_room_type="alta_complejidad",
    )
    staff = Staff(
        id=1,
        name="Dr Competente",
        role="cirujano",
        enabled_procedures_ids=[101],
        availability_hours={0: (480, 780)},
        main_specialty_id=staff_main_specialty,
    )
    return GeneticAlgorithm(
        config=GAConfig(n_days=1, n_shifts=1, block_duration_min=300),
        operating_rooms=[
            OperatingRoom(id=1, name="OR", or_type=room_type, availability=[[True]]),
        ],
        specialties=[Specialty(id=0, name="Libre", compatible_or_types=[]), specialty],
        patients_by_specialty={},
        staff_list=[staff],
        procedures_by_specialty={1: [procedure]},
    )


def test_high_complexity_procedure_is_not_compatible_with_lower_room_type():
    ga = _ga_for_contract(room_type="media_complejidad")

    assert ga._compatible_procedures_for_room(1, "media_complejidad") == []
    assert not ga._specialty_valid_for_or(1, 0, 0, 0)


def test_surgeon_must_have_matching_main_specialty():
    ga = _ga_for_contract(room_type="alta_complejidad", staff_main_specialty=2)

    assert ga._compatible_procedures_for_room(1, "alta_complejidad")
    assert not ga._specialty_valid_for_or(1, 0, 0, 0)
