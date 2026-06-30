"""
tests/test_api.py
Tests de integración de la API FastAPI.

Usa TestClient (httpx síncrono) para evitar levantar un servidor real.
Los tests del flujo async (POST -> GET) mockean el ProcessPoolExecutor
para que el job "complete" de forma síncrona dentro del test, sin tener
que esperar segundos ni lidiar con procesos reales.
"""
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import app


# ---------------------------------------------------------------------------
# Client y payload base
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
    """Payload mínimo pero válido con integridad referencial correcta."""
    return {
        "specialties": [
            {"id": "TRA", "name": "Traumatologia", "min_blocks": 1},
            {"id": "CG",  "name": "Cirugia General", "min_blocks": 1},
        ],
        "rooms": [
            {"id": "Q1", "name": "Quirofano 1", "room_type": 2,
             "daily_capacity_minutes": 300},
        ],
        "procedures": [
            {"id": "PR1", "name": "Fractura", "specialty_id": "TRA",
             "required_room_type": 1, "estimated_duration": 90},
            {"id": "PR2", "name": "Apendice", "specialty_id": "CG",
             "required_room_type": 1, "estimated_duration": 60},
        ],
        "surgeons": [
            {"id": "S1", "name": "Dr. Lopez", "specialty_id": "TRA",
             "available_days": ["lunes", "martes"], "contract_hours_week": 20},
            {"id": "S2", "name": "Dr. Perez", "specialty_id": "CG",
             "available_days": ["lunes", "martes"], "contract_hours_week": 20},
        ],
        "patients": [
            {"id": "P1", "specialty_id": "TRA", "procedure_id": "PR1",
             "surgeon_id": "S1", "clinical_priority": 9.0},
            {"id": "P2", "specialty_id": "CG",  "procedure_id": "PR2",
             "surgeon_id": "S2", "clinical_priority": 7.0},
        ],
        "params": {
            "population_size": 10,
            "generations": 3,
            "stagnation_limit": 3,
            "cleaning_minutes": 15,
            "random_seed": 42,
        },
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /agendas — respuesta inmediata
# ---------------------------------------------------------------------------

class TestCreateAgenda:

    def test_post_valido_devuelve_202(self, client, valid_payload):
        resp = client.post("/agendas", json=valid_payload)
        assert resp.status_code == 202

    def test_post_valido_devuelve_job_id(self, client, valid_payload):
        resp = client.post("/agendas", json=valid_payload)
        data = resp.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    def test_post_valido_estado_inicial_running(self, client, valid_payload):
        resp = client.post("/agendas", json=valid_payload)
        assert resp.json()["status"] == "running"

    def test_post_sin_pacientes_devuelve_422(self, client, valid_payload):
        payload = dict(valid_payload)
        payload["patients"] = []
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422

    def test_post_sin_especialidades_devuelve_422(self, client, valid_payload):
        payload = dict(valid_payload)
        payload["specialties"] = []
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /agendas — validación de integridad referencial
# ---------------------------------------------------------------------------

class TestReferentialIntegrity:

    def test_paciente_con_specialty_id_inexistente(self, client, valid_payload):
        payload = dict(valid_payload)
        payload["patients"] = [
            {"id": "P99", "specialty_id": "NEU_NO_EXISTE",
             "procedure_id": "PR1", "surgeon_id": "S1", "clinical_priority": 5.0}
        ]
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422
        assert "NEU_NO_EXISTE" in resp.text

    def test_paciente_con_procedure_id_inexistente(self, client, valid_payload):
        payload = dict(valid_payload)
        payload["patients"] = [
            {"id": "P99", "specialty_id": "TRA",
             "procedure_id": "PR_NO_EXISTE", "surgeon_id": "S1",
             "clinical_priority": 5.0}
        ]
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422
        assert "PR_NO_EXISTE" in resp.text

    def test_paciente_con_surgeon_id_inexistente(self, client, valid_payload):
        payload = dict(valid_payload)
        payload["patients"] = [
            {"id": "P99", "specialty_id": "TRA",
             "procedure_id": "PR1", "surgeon_id": "S_NO_EXISTE",
             "clinical_priority": 5.0}
        ]
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422
        assert "S_NO_EXISTE" in resp.text

    def test_cirujano_con_specialty_id_inexistente(self, client, valid_payload):
        payload = dict(valid_payload)
        payload["surgeons"] = [
            {"id": "S99", "name": "Dr. X", "specialty_id": "ESP_INEXISTENTE",
             "available_days": ["lunes"], "contract_hours_week": 10}
        ]
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422

    def test_procedimiento_con_specialty_id_inexistente(self, client, valid_payload):
        payload = dict(valid_payload)
        payload["procedures"] = [
            {"id": "PR99", "name": "Test", "specialty_id": "ESP_INEXISTENTE",
             "required_room_type": 1, "estimated_duration": 60}
        ]
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /agendas/{job_id}
# ---------------------------------------------------------------------------

class TestGetAgenda:

    def test_job_id_inexistente_devuelve_404(self, client):
        resp = client.get("/agendas/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_job_recien_creado_esta_running(self, client, valid_payload):
        post_resp = client.post("/agendas", json=valid_payload)
        job_id = post_resp.json()["job_id"]
        get_resp = client.get(f"/agendas/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] in ("running", "done", "error")

    def test_job_terminado_tiene_resultado(self, client, valid_payload):
        """Mockea el Future para que el job ya esté done antes del GET."""
        fake_result = {
            "fitness": 42.0,
            "generaciones_ejecutadas": 3,
            "parametros": {"minutos_limpieza_entre_cirugias": 15},
            "distribucion_semanal_especialidades": {},
            "agenda": {},
            "resumen": {
                "total_pacientes": 2,
                "pacientes_programados": 2,
                "pacientes_pendientes": 0,
                "ids_pacientes_pendientes": [],
            },
            "historial_fitness": [40.0, 41.0, 42.0],
        }

        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = fake_result

        post_resp = client.post("/agendas", json=valid_payload)
        job_id = post_resp.json()["job_id"]

        # Inyectar el future ya terminado en el store
        import jobs
        jobs._JOBS[job_id]["future"] = mock_future

        get_resp = client.get(f"/agendas/{job_id}")
        data = get_resp.json()
        assert data["status"] == "done"
        assert data["result"]["fitness"] == 42.0

    def test_job_con_error_reporta_error(self, client, valid_payload):
        """Mockea un Future que lanzó una excepción."""
        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.side_effect = RuntimeError("algo explotó")

        post_resp = client.post("/agendas", json=valid_payload)
        job_id = post_resp.json()["job_id"]

        import jobs
        jobs._JOBS[job_id]["future"] = mock_future

        get_resp = client.get(f"/agendas/{job_id}")
        data = get_resp.json()
        assert data["status"] == "error"
        assert "algo explotó" in data["error"]


# ---------------------------------------------------------------------------
# Validación de parámetros del GA
# ---------------------------------------------------------------------------

class TestGAParamValidation:

    def test_population_size_menor_a_minimo_devuelve_422(
        self, client, valid_payload
    ):
        payload = dict(valid_payload)
        payload["params"] = {"population_size": 1}  # mínimo es 10
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422

    def test_cleaning_minutes_negativo_devuelve_422(
        self, client, valid_payload
    ):
        payload = dict(valid_payload)
        payload["params"] = {"cleaning_minutes": -5}
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 422

    def test_params_omitidos_usa_defaults(self, client, valid_payload):
        """Si no se envía el bloque params, la API debe aceptarlo con defaults."""
        payload = {k: v for k, v in valid_payload.items() if k != "params"}
        resp = client.post("/agendas", json=payload)
        assert resp.status_code == 202
