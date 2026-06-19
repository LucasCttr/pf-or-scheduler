"""
decoder.py — Decodificador Heurístico para asignación de pacientes a quirófanos.

Recibe los bloques del turno (especialidad + pacientes + cirujanos por OR)
y produce un cronograma completo respetando:
  - Ventanas horarias reales de cada cirujano
  - No-solapamiento dentro del mismo quirófano
  - No-solapamiento del mismo cirujano entre quirófanos (reloj global)
  - Competencia del cirujano para el procedimiento del paciente

No usa slots discretos ni MIP — asigna minutos exactos de forma greedy.
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _overlaps(start_new: int, end_new: int, intervals: List[Tuple[int, int]]) -> bool:
    """True si el intervalo [start_new, end_new) se solapa con alguno de intervals."""
    return any(start_new < end and start < end_new for start, end in intervals)


def build_shift_schedule(
    blocks: List[Dict],
    day_idx: int,
    capacity_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parámetros
    ----------
    blocks : salida de _build_shift_blocks — lista de dicts con:
        { or_idx, spec_id, patients: List[Patient], surgeons: List[Staff], t_max }
    day_idx : índice del día (0=Lunes ... 4=Viernes)
    capacity_params : {
        "block_start": int,   minutos desde medianoche en que arranca el bloque (default 480 = 08:00)
        "block_duration": int  duración total del bloque en minutos (default 720)
    }

    Retorna
    -------
    { "fitness": float, "all_pacientes_ids": List[int],
      "per_or": { or_idx: { pacientes_ids, asignaciones, t_max, uso_tiempo, utilizacion_porcentaje } } }
    """
    block_start = capacity_params.get("block_start", 480)
    block_duration = capacity_params.get("block_duration", 720)
    block_end = block_start + block_duration

    # Bloques activos (con pacientes y cirujanos)
    active = [
        b
        for b in blocks
        if b["spec_id"] > 0 and b["patients"] and b["surgeons"] and b["t_max"] > 0
    ]

    per_or: Dict[int, Dict] = {
        b["or_idx"]: {
            "or_idx": b["or_idx"],
            "t_max": b["t_max"],
            "pacientes_ids": [],
            "asignaciones": [],
            "uso_tiempo": 0,
            "utilizacion_porcentaje": 0.0,
        }
        for b in blocks
    }

    if not active:
        return {"fitness": 0.0, "all_pacientes_ids": [], "per_or": per_or}

    # ── Ventanas reales de cada cirujano ──────────────────────────────────────
    # surgeon_window[s_id] = (inicio_real, fin_real) en minutos absolutos
    surgeon_window: Dict[int, Tuple[int, int]] = {}
    for b in active:
        for s in b["surgeons"]:
            if s.id in surgeon_window:
                continue
            s_start, s_end = s.get_range_for_block(day_idx, True, block_duration)
            if s_start == s_end == 0:
                # Sin disponibilidad — no puede operar hoy
                surgeon_window[s.id] = (0, 0)
            else:
                surgeon_window[s.id] = (s_start, s_end)

    # ── Relojes globales ──────────────────────────────────────────────────────
    # or_clock[q]   = próximo minuto disponible en el quirófano q
    # surg_clock[s] = próximo minuto disponible del cirujano s (cross-OR)
    or_clock: Dict[int, int] = {b["or_idx"]: block_start for b in active}
    surg_clock: Dict[int, int] = {
        s.id: surgeon_window[s.id][0]  # empieza en su hora de entrada real
        for b in active
        for s in b["surgeons"]
        if surgeon_window.get(s.id, (0, 0))[1] > 0
    }

    # ── Candidatos globales ordenados por prioridad ───────────────────────────
    # Juntamos todos los pacientes de todos los bloques activos.
    # Cada paciente lleva referencia a su OR asignado y sus cirujanos elegibles.
    candidatos = []
    vistos: set = set()
    for b in active:
        for p in b["patients"]:
            if p.id in vistos:
                continue
            vistos.add(p.id)
            forced = getattr(p, "forced_surgeon_id", None)
            proc_id = getattr(p, "procedure_id", None)
            elegibles = [
                s
                for s in b["surgeons"]
                if (forced is None or s.id == forced)
                and (proc_id is None or proc_id in s.enabled_procedures_ids)
                and surgeon_window.get(s.id, (0, 0))[1]
                > surgeon_window.get(s.id, (0, 0))[0]
            ]
            if elegibles:
                candidatos.append((p, b["or_idx"], elegibles))

    # Ordenar por prioridad clínica descendente
    candidatos.sort(key=lambda x: -x[0].clinical_priority)

    fitness_total = 0.0
    all_ids: List[int] = []
    asignados: set = set()

    for p, q, elegibles in candidatos:
        if p.id in asignados:
            continue

        mejor: Optional[Tuple[int, Any]] = None  # (inicio, cirujano)

        for s in sorted(elegibles, key=lambda s: surg_clock.get(s.id, 0)):
            ws, we = surgeon_window.get(s.id, (0, 0))
            if we == 0:
                continue

            # El inicio más temprano es el máximo entre:
            # - cuando se libera el quirófano
            # - cuando se libera el cirujano
            # - cuando empieza la ventana del cirujano
            inicio = max(or_clock.get(q, block_start), surg_clock.get(s.id, ws), ws)
            fin = inicio + p.estimated_duration

            # Verificar que entra dentro de la ventana del cirujano y del bloque
            if fin > we or fin > block_end:
                continue

            if mejor is None or inicio < mejor[0]:
                mejor = (inicio, s)

        if mejor is None:
            continue  # no hay hueco disponible para este paciente

        inicio, cirujano = mejor
        fin = inicio + p.estimated_duration

        # Registrar
        per_or[q]["asignaciones"].append(
            {
                "p": p.id,
                "doc": cirujano.name,
                "doc_id": cirujano.id,
                "hora_inicio": _fmt(inicio),
                "hora_fin": _fmt(fin),
                "duracion": p.estimated_duration,
            }
        )
        per_or[q]["pacientes_ids"].append(p.id)
        per_or[q]["uso_tiempo"] += p.estimated_duration

        or_clock[q] = fin
        surg_clock[cirujano.id] = fin

        asignados.add(p.id)
        all_ids.append(p.id)
        fitness_total += p.clinical_priority

    # Calcular utilización
    for q, data in per_or.items():
        data["asignaciones"].sort(key=lambda a: a["hora_inicio"])
        if data["t_max"] > 0:
            data["utilizacion_porcentaje"] = round(
                data["uso_tiempo"] / data["t_max"] * 100, 2
            )

    return {"fitness": fitness_total, "all_pacientes_ids": all_ids, "per_or": per_or}
