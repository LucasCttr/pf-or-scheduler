from fastapi.testclient import TestClient

import api
from api import PlanningJob, PlanningRequest, app, run_planning


def minimal_payload():
    return {
        "week_start": "2026-08-10",
        "pending_surgeries": [
            {
                "id": 1,
                "specialty_id": 10,
                "procedure_id": 100,
                "estimated_duration": 60,
                "clinical_priority": 10,
                "forced_surgeon_id": 20,
            }
        ],
        "operating_rooms": [
            {"id": 30, "name": "OR 1", "or_type": "media_complejidad", "availability": [[True]] * 5}
        ],
        "specialties": [
            {"id": 10, "name": "General", "compatible_or_types": ["media_complejidad"], "min_blocks": 1, "max_blocks": 5}
        ],
        "medical_staff": [
            {
                "id": 20,
                "name": "Dra. Sosa",
                "role": "cirujano",
                "specialties_ids": [10],
                "enabled_procedures_ids": [100],
                "availability_hours": {"0": [480, 840]},
                "main_specialty_id": 10,
            }
        ],
        "procedures_by_specialty": {
            "10": [{"id": 100, "name": "Proc", "specialty_id": 10, "required_room_type": "media_complejidad"}]
        },
        "config": {
            "population_size": 4,
            "max_generations": 2,
            "convergence_patience": 1,
            "tournament_size": 2,
            "block_duration_min": 300,
            "n_days": 5,
        },
    }


def test_create_planning_keeps_async_contract(monkeypatch):
    submitted = []

    class Executor:
        def submit(self, fn, *args):
            submitted.append((fn, args))

    monkeypatch.setattr("api._executor", Executor())
    response = TestClient(app).post("/planning", json=minimal_payload())

    assert response.status_code == 202
    assert response.json()["status"] == "planning"
    assert response.json()["uuid"]
    assert submitted


def test_job_is_completed_only_after_callback_delivery(monkeypatch):
    job_uuid = "11111111-2222-3333-4444-555555555555"
    observed_statuses = []
    api._jobs[job_uuid] = PlanningJob(uuid=job_uuid, status="planning")
    monkeypatch.setattr(api, "run_planning", lambda payload: {"dias": [], "resumen": {}})

    def observe_callback(payload):
        observed_statuses.append(api._jobs[job_uuid].status)

    monkeypatch.setattr(api, "_send_callback", observe_callback)

    api._run_job(job_uuid, PlanningRequest.model_validate(minimal_payload()))

    assert observed_statuses == ["planning"]
    assert api._jobs[job_uuid].status == "completed"


def test_run_planning_returns_back_and_front_contract():
    result = run_planning(PlanningRequest.model_validate(minimal_payload()))

    item = result["dias"][0]["bloques"][0]["cronograma"][0]
    assert item == {
        "paciente_id": 1,
        "medico": "Dra. Sosa",
        "slot_inicio": 4,
        "hora_inicio": "09:00",
        "hora_fin": "10:00",
        "duracion": 60,
    }
    assert result["resumen"]["pacientes_programados"] == 1
    assert result["resumen"]["ids_pendientes"] == []


def test_run_planning_requires_assigned_surgeon():
    payload = minimal_payload()
    payload["pending_surgeries"][0]["forced_surgeon_id"] = None

    try:
        run_planning(PlanningRequest.model_validate(payload))
    except ValueError as exc:
        assert "no tiene cirujano asignado" in str(exc)
    else:
        raise AssertionError("Expected missing surgeon validation")


def test_job_uuid_reuses_existing_dispatch(monkeypatch):
    from uuid import uuid4

    payload = minimal_payload()
    payload["job_uuid"] = str(uuid4())
    submitted = []
    monkeypatch.setattr(api._executor, "submit", lambda *args: submitted.append(args))

    client = TestClient(app)
    first = client.post("/planning", json=payload)
    second = client.post("/planning", json=payload)
    assert first.status_code == second.status_code == 202
    assert first.json()["uuid"] == second.json()["uuid"] == payload["job_uuid"]
    assert len(submitted) == 1


def test_availability_bookings_and_distinct_durations_for_same_procedure():
    payload = minimal_payload()
    payload["operating_rooms"][0]["availability"] = [[False], [True], [False], [False], [False]]
    payload["medical_staff"][0]["availability_hours"] = {"0": [540, 840], "1": [600, 780]}
    payload["pending_surgeries"].append({
        **payload["pending_surgeries"][0], "id": 2, "estimated_duration": 35, "clinical_priority": 1,
    })
    payload["bookings"] = [{"day": 1, "start": 600, "end": 630, "room_id": 30}]
    result = run_planning(PlanningRequest.model_validate(payload))
    assert not result["dias"][0]["bloques"][0]["cronograma"]
    items = result["dias"][1]["bloques"][0]["cronograma"]
    assert [(item["duracion"], item["hora_inicio"], item["hora_fin"]) for item in items] == [
        (60, "10:30", "11:30"), (35, "11:30", "12:05"),
    ]
