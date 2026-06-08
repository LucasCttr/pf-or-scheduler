from fastapi.testclient import TestClient

from api import app


def test_create_planning_returns_uuid_and_planning_status(monkeypatch):
    submitted = []

    class FakeExecutor:
        def submit(self, fn, *args, **kwargs):
            submitted.append((fn, args, kwargs))

    monkeypatch.setattr("api._executor", FakeExecutor())
    client = TestClient(app)

    response = client.post("/planning", json=_minimal_payload())

    assert response.status_code == 202
    payload = response.json()
    assert payload["uuid"]
    assert payload["status"] == "planning"
    assert payload["progress_percentage"] == 0
    assert submitted


def test_get_planning_status(monkeypatch):
    submitted = []

    class FakeExecutor:
        def submit(self, fn, *args, **kwargs):
            submitted.append((fn, args, kwargs))

    monkeypatch.setattr("api._executor", FakeExecutor())
    client = TestClient(app)
    created = client.post("/planning", json=_minimal_payload()).json()

    response = client.get(f"/planning/{created['uuid']}")

    assert response.status_code == 200
    assert response.json() == {
        "uuid": created["uuid"],
        "status": "planning",
        "progress_percentage": 0,
    }


def test_get_unknown_planning_returns_404():
    client = TestClient(app)

    response = client.get("/planning/missing")

    assert response.status_code == 404


def test_run_job_success_sends_completed_callback(monkeypatch):
    callbacks = []

    monkeypatch.setattr(
        "api.run_planning",
        lambda payload, progress_callback=None: {"resumen": {"pacientes_programados": 1}},
    )
    monkeypatch.setattr("api._send_callback", lambda payload: callbacks.append(payload))

    from api import PlanningJob, PlanningRequest, _jobs, _run_job

    job_uuid = "job-success"
    _jobs[job_uuid] = PlanningJob(uuid=job_uuid, status="planning")

    _run_job(job_uuid, PlanningRequest.model_validate(_minimal_payload()))

    assert _jobs[job_uuid].status == "completed"
    assert _jobs[job_uuid].progress_percentage == 100
    assert callbacks[0]["uuid"] == job_uuid
    assert callbacks[0]["status"] == "completed"
    assert callbacks[0]["output_payload"]["resumen"] == {"pacientes_programados": 1}


def test_run_job_failure_sends_failed_callback(monkeypatch):
    callbacks = []

    def fail(_, progress_callback=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("api.run_planning", fail)
    monkeypatch.setattr("api._send_callback", lambda payload: callbacks.append(payload))

    from api import PlanningJob, PlanningRequest, _jobs, _run_job

    job_uuid = "job-failed"
    _jobs[job_uuid] = PlanningJob(uuid=job_uuid, status="planning")

    _run_job(job_uuid, PlanningRequest.model_validate(_minimal_payload()))

    assert _jobs[job_uuid].status == "failed"
    assert callbacks[0]["uuid"] == job_uuid
    assert callbacks[0]["status"] == "failed"
    assert callbacks[0]["error_message"] == "boom"


def test_update_job_progress_caps_at_99():
    from api import PlanningJob, _jobs, _update_job_progress

    job_uuid = "job-progress"
    _jobs[job_uuid] = PlanningJob(uuid=job_uuid, status="planning")

    _update_job_progress(job_uuid, 120)

    assert _jobs[job_uuid].progress_percentage == 99


def _minimal_payload():
    return {
        "week_start": "2026-06-08",
        "pending_surgeries": [
            {
                "id": 1,
                "specialty_id": 1,
                "procedure_id": 101,
                "estimated_duration": 60,
                "clinical_priority": 10.0,
            }
        ],
        "operating_rooms": [
            {
                "id": 1,
                "name": "OR 1",
                "or_type": "alta_complejidad",
                "availability": [[True], [True]],
            }
        ],
        "specialties": [
            {
                "id": 0,
                "name": "Libre",
                "compatible_or_types": [],
                "min_blocks": 0,
                "max_blocks": 99,
            },
            {
                "id": 1,
                "name": "Trauma",
                "compatible_or_types": ["alta_complejidad"],
                "min_blocks": 1,
                "max_blocks": 2,
            },
        ],
        "medical_staff": [
            {
                "id": 1,
                "name": "Dr Alta",
                "role": "cirujano",
                "enabled_procedures_ids": [101],
                "availability_hours": {"0": [480, 720]},
            }
        ],
        "procedures_by_specialty": {"1": [101]},
        "config": {
            "population_size": 2,
            "max_generations": 1,
            "convergence_patience": 1,
            "tournament_size": 1,
            "elite_count": 1,
            "n_days": 2,
            "n_shifts": 1,
            "parallel_workers": 1,
        },
    }



def test_send_callback_retries_until_success(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        if len(calls) < 3:
            from urllib.error import URLError

            raise URLError("not yet")
        return FakeResponse()

    monkeypatch.setenv("BACK_CALLBACK_URL", "http://back.local/api/v1/scheduler/callback")
    monkeypatch.setenv("SCHEDULER_CALLBACK_TOKEN", "token-demo")
    monkeypatch.setattr("api.urllib_request.urlopen", fake_urlopen)
    monkeypatch.setattr("api.time.sleep", lambda _: None)

    from api import _send_callback

    _send_callback({"uuid": "job-1", "status": "completed"})

    assert len(calls) == 3
    assert calls[-1][0].headers["X-scheduler-token"] == "token-demo"
