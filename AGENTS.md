# pf-or-scheduler Agent Notes

## Purpose
Optimization service for weekly operating room planning. It receives normalized planning input from `PF-G1-Back`, runs the Genetic Algorithm + MIP scheduler, and posts the final result back to the Back callback.

## Stack
- Python
- FastAPI for the Scheduler API
- Genetic Algorithm for block assignment
- PuLP/CBC MIP for patient and surgeon assignment
- NumPy for chromosome operations
- Pytest

## Key Files
- `api.py`: async planning API, in-memory jobs, callback to Back.
- `genetic_algorithm.py`: main GA loop and fitness evaluation.
- `mip.py`: MIP solver for patient/surgeon assignment.
- `models.py`: domain dataclasses.
- `main.py`: local execution helpers and `reconstruct_agenda`.
- `agenda_resultado.json`: generated schedule output example.
- `tests/`: unit and end-to-end tests.

## Commands
```bash
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 3020 --reload
.venv/bin/python -m pytest tests/test_api.py
```

Example local API run with callback:
```bash
BACK_CALLBACK_URL=http://127.0.0.1:3010/api/v1/scheduler/callback SCHEDULER_CALLBACK_TOKEN=dev-scheduler-token .venv/bin/uvicorn api:app --host 127.0.0.1 --port 3020 --reload
```

## Environment Variables
- `BACK_CALLBACK_URL`: Back endpoint that receives final planning results.
- `SCHEDULER_CALLBACK_TOKEN`: shared secret sent in `X-Scheduler-Token`.

## API Contract
- `POST /planning`: receives the full planning payload, creates an async job, and returns UUID/status.
- `GET /planning/{uuid}`: returns only operational state such as UUID, status, and progress. It must not return the full `output_payload`.
- Final planning output is sent to Back via callback.
- Jobs are in memory only. No Scheduler-side persistence is expected for v1.

## Agent Rules
- Keep the Front out of this service; only Back should consume Scheduler.
- Do not persist jobs in Scheduler unless explicitly requested.
- Do not expose full planning results from `GET /planning/{uuid}`.
- Preserve single-worker behavior unless the concurrency model is explicitly redesigned.
- Avoid broad algorithm changes without focused tests, especially around `GeneticAlgorithm`, `mip.py`, and `reconstruct_agenda`.
- The Back maps string IDs to numeric Scheduler IDs; keep payload contracts compatible with that mapping.
