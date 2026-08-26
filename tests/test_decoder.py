from decoder import build_agenda
from genetic_algorithm import GeneticAlgorithm
from models import Block, Patient, Procedure, Room, Specialty, Surgeon


def test_decoder_respects_exact_availability_and_patient_duration():
    block = Block("lunes", "1")
    patient = Patient("1", "10", "100", "20", 9.0)
    agenda = build_agenda(
        {block: "10"},
        [patient],
        {"100": Procedure("100", "Proc", "10", 2, 90)},
        {"20": Surgeon("20", "Dra.", "10", {"lunes"}, 40.0)},
        {"1": Room("1", "OR", 2, 300)},
    )

    surgery = agenda.all_surgeries()[0]
    assert surgery.start_time == 0
    assert surgery.end_time == 90
    assert surgery.duration == 90


def test_decoder_prevents_surgeon_overlap_between_rooms():
    blocks = {Block("lunes", "1"): "10", Block("lunes", "2"): "10"}
    patients = [
        Patient("1", "10", "100", "20", 10.0),
        Patient("2", "10", "100", "20", 9.0),
    ]
    agenda = build_agenda(
        blocks,
        patients,
        {"100": Procedure("100", "Proc", "10", 2, 60)},
        {"20": Surgeon("20", "Dra.", "10", {"lunes"}, 40.0)},
        {
            "1": Room("1", "OR 1", 2, 300),
            "2": Room("2", "OR 2", 2, 300),
        },
    )

    intervals = [(item.start_time, item.end_time) for item in agenda.all_surgeries()]
    assert intervals == [(0, 60), (60, 120)]


def test_decoder_rejects_incompatible_room_and_unavailable_day():
    patients = [Patient("1", "10", "100", "20", 10.0)]
    procedure = {"100": Procedure("100", "Proc", "10", 3, 60)}
    surgeon = {"20": Surgeon("20", "Dra.", "10", {"martes"}, 40.0)}
    room = {"1": Room("1", "OR", 2, 300)}

    agenda = build_agenda({Block("lunes", "1"): "10"}, patients, procedure, surgeon, room)

    assert agenda.all_surgeries() == []


def test_ga_repair_respects_maximum_blocks_using_free_blocks():
    room = Room("1", "OR", 2, 300)
    specialties = [
        Specialty("10", "General", min_blocks=1),
        Specialty("20", "Trauma", min_blocks=1),
    ]
    ga = GeneticAlgorithm(
        ["lunes", "martes", "miercoles"],
        [room],
        specialties,
        [Surgeon("20", "Dra.", "10", {"lunes"}, 40.0)],
        [Procedure("100", "Proc", "10", 2, 60)],
        [Patient("1", "10", "100", "20", 9.0)],
        population_size=2,
        generations=1,
    )
    chromosome = {block: "10" for block in ga.blocks}

    repaired = ga._repair(chromosome)

    assert list(repaired.values()).count("10") == 2
    assert list(repaired.values()).count("20") == 1
