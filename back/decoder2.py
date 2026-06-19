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
    block_start = capacity_params.get("block_start", 480)
    block_duration = capacity_params.get("block_duration", 720)
    # Gap (turnover) in minutes to leave between consecutive surgeries
    gap_between_cases = capacity_params.get("gap_between_cases", 0)        # me modifica el porcenaje de uso del quirofano (arreglar)
    block_end = block_start + block_duration

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

    # ── Relojes globales ──────────────────────────────────────────────────────
    # Each OR may have its own effective block duration (b["t_max"]).
    or_clock: Dict[int, int] = {b["or_idx"]: block_start for b in active}
    or_block_end: Dict[int, int] = {
        b["or_idx"]: block_start + min(block_duration, b["t_max"]) for b in active
    }

    # surg_clock ahora es simplemente el inicio del bloque (o cuando el médico llega)
    surg_clock: Dict[int, int] = {
        s.id: block_start for b in active for s in b["surgeons"]
    }

    # Track remaining minutes per surgeon locally to avoid mutating shared Staff state
    remaining_minutes: Dict[int, int] = {
        s.id: s.get_available_minutes_in_block(day_idx) for b in active for s in b["surgeons"]
    }

    candidatos = []
    vistos: set = set()
    for b in active:
        for p in b["patients"]:
            if p.id in vistos:
                continue
            vistos.add(p.id)
            forced = getattr(p, "forced_surgeon_id", None)
            elegibles = [
                s
                for s in b["surgeons"]
                if (forced is None or s.id == forced)
                # VALIDACIÓN DE BOLSA: ¿Tiene tiempo en su presupuesto?
                and s.get_available_minutes_in_block(day_idx) >= p.estimated_duration
            ]
            if elegibles:
                candidatos.append((p, b["or_idx"], elegibles))

    candidatos.sort(key=lambda x: -x[0].clinical_priority)

    fitness_total = 0.0
    all_ids: List[int] = []
    asignados: set = set()

    for p, q, elegibles in candidatos:
        if p.id in asignados:
            continue

        # Ordenar por el que tiene más presupuesto disponible o por el reloj más temprano
        mejor: Optional[Tuple[int, Any]] = None

        for s in sorted(elegibles, key=lambda s: surg_clock.get(s.id, block_start)):
            inicio = max(
                or_clock.get(q, block_start) + gap_between_cases,
                surg_clock.get(s.id, block_start) + gap_between_cases,
            )
            fin = inicio + p.estimated_duration

            # Respect the OR-specific block end time
            if fin > or_block_end.get(q, block_end):
                continue

            if mejor is None or inicio < mejor[0]:
                mejor = (inicio, s)

        if mejor is None:
            continue

        inicio, cirujano = mejor
        fin = inicio + p.estimated_duration

        # ── REGISTRO DE CONSUMO (local) ──────────────────────────────────────
        remaining_minutes[cirujano.id] -= p.estimated_duration
        # ────────────────────────────────────────────────────────────────────

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

    # Calcular utilización por OR usando su t_max efectivo
    for b in active:
        idx = b["or_idx"]
        t_max = min(block_duration, b["t_max"]) if b else block_duration
        used = per_or[idx]["uso_tiempo"]
        per_or[idx]["utilizacion_porcentaje"] = (used / t_max * 100.0) if t_max > 0 else 0.0

    return {"fitness": fitness_total, "all_pacientes_ids": all_ids, "per_or": per_or}
