# PF OR Scheduler — Decoder

Microservicio de planificación semanal de cirugías. La rama productiva `decoder` combina:

- un algoritmo genético que asigna especialidades a bloques `(día, quirófano)`;
- un decoder determinista que asigna cirugías ya vinculadas a un cirujano;
- una API FastAPI asíncrona con callback autenticado al Back Django.

## Contrato HTTP

- `POST /planning`: recibe el payload autocontenido del Back y devuelve `202` con `uuid` y estado `planning`.
- `GET /planning/{uuid}`: consulta `planning`, `completed` o `failed`.
- Al terminar se envía el resultado a `BACK_CALLBACK_URL` usando `X-Scheduler-Token`.

La salida conserva el contrato `dias[].bloques[].cronograma[]` utilizado por SurgiCare. Una cirugía sólo se agenda si su cirujano asignado, procedimiento y quirófano son compatibles y si cabe dentro de la intersección entre su disponibilidad y la jornada 08:00–13:00.

## Ejecución

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 3020 --reload
```

Variables opcionales:

- `BACK_CALLBACK_URL`
- `SCHEDULER_CALLBACK_TOKEN` (por defecto `dev-scheduler-token`)

## Pruebas

```bash
.venv/bin/pytest
```

El modo CSV sigue disponible con `.venv/bin/python main.py`; usa los archivos de `data/` y genera `agenda_resultado.json`.
