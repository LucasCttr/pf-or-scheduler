from decoder import build_agenda
from genetic_algorithm import GeneticAlgorithm
from models import Block, Patient, Procedure, Room, Specialty, Surgeon


def test_decoder_respects_exact_availability_and_patient_duration():
    block = Block("lunes", "1")
    patient = Patient("1", "10", "100", "20", 9.0, estimated_duration=90)
    agenda = build_agenda(
        {block: "10"},
        [patient],
        {"100": Procedure("100", "Proc", "10", 2, estimated_duration=30)},
        {"20": Surgeon("20", "Dra.", {"10"}, {"lunes": (540, 720)})},
        {"1": Room("1", "OR", 2, 300, {"lunes"})},
    )

    surgery = agenda.all_surgeries()[0]
    assert surgery.start_minute == 540
    assert surgery.end_minute == 630
    assert surgery.duration == 90


def test_decoder_prevents_surgeon_overlap_between_rooms():
    blocks = {Block("lunes", "1"): "10", Block("lunes", "2"): "10"}
    patients = [
        Patient("1", "10", "100", "20", 10.0, 60),
        Patient("2", "10", "100", "20", 9.0, 60),
    ]
    agenda = build_agenda(
        blocks,
        patients,
        {"100": Procedure("100", "Proc", "10", 2)},
        {"20": Surgeon("20", "Dra.", {"10"}, {"lunes": (480, 780)})},
        {
            "1": Room("1", "OR 1", 2, 300, {"lunes"}),
            "2": Room("2", "OR 2", 2, 300, {"lunes"}),
        },
    )

    intervals = sorted((item.start_minute, item.end_minute) for item in agenda.all_surgeries())
    assert intervals == [(480, 540), (540, 600)]


def test_decoder_rejects_incompatible_room_and_unavailable_day():
    patients = [Patient("1", "10", "100", "20", 10.0, 60)]
    procedure = {"100": Procedure("100", "Proc", "10", 3)}
    surgeon = {"20": Surgeon("20", "Dra.", {"10"}, {"martes": (480, 780)})}
    room = {"1": Room("1", "OR", 2, 300, {"lunes"})}

    agenda = build_agenda({Block("lunes", "1"): "10"}, patients, procedure, surgeon, room)

    assert agenda.all_surgeries() == []


def test_ga_repair_respects_maximum_blocks_using_free_blocks():
    room = Room("1", "OR", 2, 300, {"lunes", "martes", "miercoles"})
    specialty = Specialty("10", "General", min_blocks=1, max_blocks=1)
    ga = GeneticAlgorithm(
        ["lunes", "martes", "miercoles"],
        [room],
        [specialty],
        [Surgeon("20", "Dra.", {"10"}, {"lunes": (480, 780)})],
        [Procedure("100", "Proc", "10", 2, 60)],
        [Patient("1", "10", "100", "20", 9.0, 60)],
        population_size=2,
        generations=1,
    )
    chromosome = {block: "10" for block in ga.blocks}

    repaired = ga._repair(chromosome)

    assert list(repaired.values()).count("10") == 1
