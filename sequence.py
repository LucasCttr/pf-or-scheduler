"""
sequence.py — Nivel 3.5: Secuenciación óptima por quirófano.

Dado el resultado del MIP (qué paciente opera qué cirujano en qué OR),
este módulo determina el ORDEN óptimo de las cirugías dentro de cada
quirófano, garantizando que ningún cirujano exceda su ventana horaria
y maximizando los pacientes efectivamente programados.

Algoritmo:
    1. Agrupa pacientes por cirujano dentro del OR.
    2. Ordena cirujanos por hora de salida (EDF — Earliest Deadline First).
    3. Por cada cirujano, usa knapsack 0/1 para seleccionar el subconjunto
       de mayor prioridad que cabe antes de su deadline.
    4. Retorna una lista ordenada de slots con hora_inicio / hora_fin exactas.

Complejidad por OR: O(n_cirujanos * n_pacientes * T_max) en el DP.
Para tamaños típicos (≤10 cirujanos, ≤8 pacientes por cirujano, T_max≤240 min)
esto es prácticamente instantáneo.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from models import Patient, Staff


# ─────────────────────────────────────────────────────────────────────────────
# Estructuras de salida
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurgerySlot:
    """Un turno quirúrgico ya secuenciado con tiempos exactos."""
    patient_id:  int
    surgeon_name: str
    hora_inicio: str   # "HH:MM"
    hora_fin:    str   # "HH:MM"
    duracion:    int   # minutos
    skipped:     bool = False  # True si el DP lo descartó por tiempo insuficiente


@dataclass
class SequencedOR:
    """Resultado de secuenciar un único quirófano en un turno."""
    or_idx: int
    slots:  List[SurgerySlot]
    utilizacion_porcentaje: float
    skipped_patients: List[int]   # IDs que no entraron


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de formato
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Knapsack 0/1 para maximizar prioridad dentro de una ventana
# ─────────────────────────────────────────────────────────────────────────────

def _knapsack_by_priority(
    patients: List[Patient],
    capacity: int,          # minutos disponibles para este cirujano
) -> List[Patient]:
    """
    Selecciona el subconjunto de `patients` que maximiza la suma de
    clinical_priority sin superar `capacity` minutos.

    Retorna la lista seleccionada en el mismo orden de entrada.
    Cuando `capacity` <= 0 o no hay pacientes, retorna lista vacía.
    """
    if not patients or capacity <= 0:
        return []

    n = len(patients)
    # dp[i][c] = max prioridad acumulada usando los primeros i pacientes con c min
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]

    for i, p in enumerate(patients, start=1):
        d = p.estimated_duration
        for c in range(capacity + 1):
            # No tomar al paciente i
            dp[i][c] = dp[i - 1][c]
            # Tomar al paciente i si cabe
            if d <= c:
                val = dp[i - 1][c - d] + p.clinical_priority
                if val > dp[i][c]:
                    dp[i][c] = val

    # Reconstrucción del conjunto óptimo
    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected.append(patients[i - 1])
            c -= patients[i - 1].estimated_duration
    selected.reverse()
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de secuenciación de un OR
# ─────────────────────────────────────────────────────────────────────────────

def sequence_or(
    or_idx:       int,
    assignments:  List[Dict],
    patients_map: Dict[int, Patient],
    staff_map:    Dict[str, Staff],
    day_idx:      int,
    is_morning:   bool,
    t_max:        int,
    staff_clocks: Dict[str, int],  # <--- NUEVO: Recibe el estado actual de los médicos
) -> SequencedOR:
    if not assignments:
        return SequencedOR(or_idx=or_idx, slots=[], utilizacion_porcentaje=0.0,
                           skipped_patients=[])

    # 1. Agrupar pacientes por cirujano
    surgeon_patients: Dict[str, List[Patient]] = {}
    for asig in assignments:
        p = patients_map[asig["p"]]
        doc = asig["doc"]
        surgeon_patients.setdefault(doc, []).append(p)

    # 2. Ordenar cirujanos por hora de salida (EDF)
    def surgeon_deadline(name: str) -> int:
        s = staff_map.get(name)
        if s is None: return 9999
        _, end = s.get_range_for_block(day_idx, is_morning)
        return end

    sorted_surgeons = sorted(surgeon_patients.keys(), key=surgeon_deadline)

    # 3. Simular el reloj del quirófano
    block_start = 480 if is_morning else 780
    clock = block_start

    slots:            List[SurgerySlot] = []
    skipped_patients: List[int]         = []

    for doc_name in sorted_surgeons:
        surgeon = staff_map.get(doc_name)
        if surgeon is None:
            skipped_patients.extend(p.id for p in surgeon_patients[doc_name])
            continue

        _, s_end = surgeon.get_range_for_block(day_idx, is_morning)
        
        # --- CAMBIO CRÍTICO ---
        # El inicio real depende de: 
        # 1. Cuándo se libera el quirófano (clock)
        # 2. Cuándo se libera el médico en CUALQUIER sala (staff_clocks)
        s_available_at = staff_clocks.get(doc_name, block_start)
        cursor = max(clock, s_available_at)

        if cursor >= s_end:
            skipped_patients.extend(p.id for p in surgeon_patients[doc_name])
            continue

        ventana_disponible = s_end - cursor
        candidatos = surgeon_patients[doc_name]

        # Knapsack para elegir qué entra en este hueco temporal
        seleccionados = _knapsack_by_priority(candidatos, ventana_disponible)
        seleccionados.sort(key=lambda p: (-p.clinical_priority, p.estimated_duration))

        descartados_por_dp = {p.id for p in candidatos} - {p.id for p in seleccionados}
        skipped_patients.extend(descartados_por_dp)

        for p in seleccionados:
            inicio = cursor
            fin    = inicio + p.estimated_duration

            if fin > s_end:
                skipped_patients.append(p.id)
                continue

            slots.append(SurgerySlot(
                patient_id   = p.id,
                surgeon_name = doc_name,
                hora_inicio  = _fmt(inicio),
                hora_fin     = _fmt(fin),
                duracion     = p.estimated_duration,
            ))
            cursor = fin

        # Actualizamos ambos relojes: 
        # El de la sala avanza, y el del médico también (globalmente)
        clock = cursor
        staff_clocks[doc_name] = cursor

    # 4. Calcular utilización
    tiempo_usado = sum(s.duracion for s in slots)
    utilizacion  = round((tiempo_usado / t_max) * 100, 2) if t_max > 0 else 0.0

    return SequencedOR(or_idx, slots, utilizacion, skipped_patients)


def sequence_shift(
    schedule_cache_entry: Dict,
    or_indices:           List[int],
    patients_map:         Dict[int, Patient],
    staff_map:            Dict[str, Staff],
    day_idx:              int,
    is_morning:           bool,
    t_max:                int,
) -> Dict[int, SequencedOR]:
    
    result: Dict[int, SequencedOR] = {}
    
    # --- INICIALIZACIÓN DEL RELOJ GLOBAL DEL STAFF ---
    # Al inicio del turno, todos los médicos están disponibles 
    # según su hora de entrada al hospital.
    block_start = 480 if is_morning else 780
    staff_clocks: Dict[str, int] = {}
    for name, s in staff_map.items():
        s_start, _ = s.get_range_for_block(day_idx, is_morning)
        staff_clocks[name] = s_start if s_start > 0 else block_start

    # IMPORTANTE: Para que la rotación sea justa, podrías ordenar or_indices
    # pero procesarlos secuencialmente con el staff_clocks ya resuelve la colisión.
    for q_idx in or_indices:
        per_or = schedule_cache_entry.get(q_idx) or {}
        assignments = per_or.get("asignaciones", [])
        q_tmax      = per_or.get("t_max", t_max)

        result[q_idx] = sequence_or(
            or_idx       = q_idx,
            assignments  = assignments,
            patients_map = patients_map,
            staff_map    = staff_map,
            day_idx      = day_idx,
            is_morning   = is_morning,
            t_max        = q_tmax,
            staff_clocks = staff_clocks # <--- Pasa el puntero de memoria
        )

    return result