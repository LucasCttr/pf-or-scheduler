"""
tune_ga.py — Búsqueda de hiperparámetros para el Algoritmo Genético.

Estrategia: Random Search con repeticiones por configuración.
  • Es más eficiente que Grid Search para espacios continuos/grandes.
  • Múltiples repeticiones por config permiten medir media y varianza del fitness.
  • Resultados exportados a CSV y mostrados como tabla de ranking.

Uso:
    python tune_ga.py                        # 20 configs × 3 repeticiones
    python tune_ga.py --configs 40 --reps 5  # más exhaustivo
    python tune_ga.py --configs 10 --reps 2  # prueba rápida

Salida:
    tune_results.csv    → tabla completa con todas las métricas
    tune_best.json      → configuración ganadora lista para pegar en main.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import sys
import os
from dataclasses import asdict
from typing import Dict, List, Any

import numpy as np

# ── Asegurarse de que los módulos del proyecto están en el path ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Monkeypatch: forzar timeLimit=2s en el solver MIP durante el tuning ───────
# El MIP tiene timeLimit=10s por defecto. Durante el tuning solo necesitamos
# valores de fitness comparables entre configuraciones, no soluciones óptimas.
# 2s por turno es suficiente para que el CBC encuentre una solución factible buena.
# Este patch no modifica mip.py ni ningún otro archivo del proyecto.
import pulp as _pulp
_OriginalCBC = _pulp.PULP_CBC_CMD

class _TuningCBC(_OriginalCBC):
    """CBC con timeLimit fijo para el tuning. Se ignora cualquier otro timeLimit."""
    TUNING_TIME_LIMIT = 2  # segundos por turno durante el tuning

    def __init__(self, *args, **kwargs):
        kwargs["timeLimit"] = self.TUNING_TIME_LIMIT
        kwargs["msg"]       = 0
        super().__init__(*args, **kwargs)

_pulp.PULP_CBC_CMD = _TuningCBC
# ─────────────────────────────────────────────────────────────────────────────

from models import OperatingRoom, Specialty, Patient, GAConfig, Staff
from genetic_algorithm import GeneticAlgorithm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATOS DE PRUEBA (mismos que main.py, reducidos para velocidad)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_staff() -> List[Staff]:
    return [
        Staff(id=1,  name="Dr. Pérez",     role="cirujano", specialties_ids=[1,2], availability_hours={0:(480,620),  1:(780,1020)}),
        Staff(id=2,  name="Dra. Sosa",     role="cirujano", specialties_ids=[1],   availability_hours={0:(480,1020), 2:(480,720)}),
        Staff(id=3,  name="Dra. Carter",   role="cirujano", specialties_ids=[1],   availability_hours={0:(620,1020), 2:(480,720)}),
        Staff(id=4,  name="Dr. Gomez",     role="cirujano", specialties_ids=[2,4], availability_hours={0:(480,720),  1:(480,720)}),
        Staff(id=5,  name="Dra. Ruiz",     role="cirujano", specialties_ids=[2],   availability_hours={1:(780,1020), 3:(780,1020)}),
        Staff(id=6,  name="Dr. Martinez",  role="cirujano", specialties_ids=[2],   availability_hours={2:(480,600),  4:(480,720)}),
        Staff(id=7,  name="Dra. Blanco",   role="cirujano", specialties_ids=[3],   availability_hours={3:(480,720),  4:(780,1020)}),
        Staff(id=8,  name="Dr. Lopez",     role="cirujano", specialties_ids=[3],   availability_hours={0:(780,1020), 2:(780,1020)}),
        Staff(id=9,  name="Dra. García",   role="cirujano", specialties_ids=[4,5], availability_hours={1:(480,720),  3:(480,720)}),
        Staff(id=10, name="Dr. Rodríguez", role="cirujano", specialties_ids=[4,5], availability_hours={2:(780,1020), 4:(780,1020)}),
        Staff(id=11, name="Dr. Morales",   role="cirujano", specialties_ids=[6],   availability_hours={0:(480,720),  3:(480,1020)}),
        Staff(id=12, name="Dra. Herrera",  role="cirujano", specialties_ids=[6,7], availability_hours={1:(480,720),  4:(480,720)}),
        Staff(id=13, name="Dr. Castro",    role="cirujano", specialties_ids=[7],   availability_hours={2:(780,1020), 3:(780,1020)}),
        Staff(id=14, name="Dra. Mendez",   role="cirujano", specialties_ids=[8],   availability_hours={0:(480,720),  4:(480,1020)}),
        Staff(id=15, name="Dr. Silva",     role="cirujano", specialties_ids=[2,8], availability_hours={1:(780,1020), 2:(480,720)}),
        Staff(id=16, name="Dra. Flores",   role="cirujano", specialties_ids=[1,3], availability_hours={0:(780,1020), 4:(480,720)}),
    ]

def _build_operating_rooms() -> List[OperatingRoom]:
    return [
        OperatingRoom(id=0, name="Quirófano 1 (Alta)",  or_type="alta_complejidad",  availability=[[True,True]]*5),
        OperatingRoom(id=1, name="Quirófano 2 (Media)", or_type="media_complejidad", availability=[[True,True]]*5),
        OperatingRoom(id=2, name="Quirófano 3 (Baja)",  or_type="baja_complejidad",  availability=[[True,False]]*5),
    ]

def _build_specialties() -> List[Specialty]:
    return [
        Specialty(id=0, name="Libre",               compatible_or_types=[],                                                          min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología",        compatible_or_types=["alta_complejidad","media_complejidad"],                    min_blocks=3, max_blocks=6),
        Specialty(id=2, name="Cirugía General",      compatible_or_types=["alta_complejidad","media_complejidad","baja_complejidad"], min_blocks=4, max_blocks=8),
        Specialty(id=3, name="Neurología",           compatible_or_types=["alta_complejidad"],                                        min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología",             compatible_or_types=["media_complejidad","baja_complejidad"],                    min_blocks=2, max_blocks=5),
        Specialty(id=5, name="Ginecología",          compatible_or_types=["media_complejidad","baja_complejidad"],                    min_blocks=2, max_blocks=5),
        Specialty(id=6, name="Cardiología",          compatible_or_types=["alta_complejidad","media_complejidad"],                    min_blocks=3, max_blocks=6),
        Specialty(id=7, name="Otorrinolaringología", compatible_or_types=["media_complejidad","baja_complejidad"],                    min_blocks=2, max_blocks=4),
        Specialty(id=8, name="Oftalmología",         compatible_or_types=["baja_complejidad"],                                        min_blocks=2, max_blocks=6),
    ]

def _make_patients(specialty_id: int, count: int, seed: int, staff_list: List[Staff]) -> List[Patient]:
    rng = random.Random(seed)
    duraciones = [30, 45, 60, 90, 120]
    cirujanos_ids = [s.id for s in staff_list if specialty_id in s.specialties_ids]
    patients: List[Patient] = []
    if specialty_id == 1:
        patients.append(Patient(id=2000, specialty_id=1, estimated_duration=60,
                                clinical_priority=99.0, required_roles=["cirujano"],
                                forced_surgeon_id=1))
    for i in range(count):
        forced = None
        if cirujanos_ids and rng.random() < 0.20:
            forced = rng.choice(cirujanos_ids)
        patients.append(Patient(
            id=specialty_id * 100 + i,
            specialty_id=specialty_id,
            estimated_duration=rng.choice(duraciones),
            clinical_priority=round(rng.uniform(1.0, 10.0), 2),
            required_roles=["cirujano"],
            forced_surgeon_id=forced,
        ))
    return patients


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ESPACIO DE BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════════════

# Cada entrada: (valor_min, valor_max, tipo)  o  lista de opciones discretas
SEARCH_SPACE: Dict[str, Any] = {
    "population_size":         [15, 20, 30, 40],
    "max_generations":         [20, 30, 50],        # reducido: con caché converge rápido
    "convergence_patience":    [5, 7, 10, 15],
    "mutation_rate":           (0.05, 0.25, "float"),
    "crossover_rate":          (0.70, 0.95, "float"),
    "tournament_size":         [3, 5, 7, 10],
    "elite_count":             [1, 2, 3],
    "alpha":                   (0.5, 0.9, "float"),
    "penalty_below_min_quota": [30.0, 50.0, 80.0, 100.0],
    "penalty_above_max_quota": [10.0, 20.0, 30.0],
}

# Parámetros fijos para todos los experimentos
FIXED_PARAMS = {
    "n_days":           5,
    "n_shifts":         2,
    "block_duration_min": 240,
    "slot_size_min":    15,
    "parallel_workers": 1,  # 1 para que los experimentos no se pisen entre sí
}

PATIENTS_PER_SPECIALTY = 10   # reducido para velocidad del tuning (10s → ~2s por rep)


def _sample_config(rng: random.Random) -> Dict[str, Any]:
    """Samplea una configuración aleatoria del espacio de búsqueda."""
    cfg: Dict[str, Any] = {}
    for key, space in SEARCH_SPACE.items():
        if isinstance(space, list):
            cfg[key] = rng.choice(space)
        else:
            lo, hi, kind = space
            val = rng.uniform(lo, hi)
            cfg[key] = round(val, 3) if kind == "float" else int(val)

    # alpha + beta deben sumar ~1
    alpha = cfg.pop("alpha")
    cfg["alpha"] = round(alpha, 3)
    cfg["beta"]  = round(1.0 - alpha, 3)

    cfg.update(FIXED_PARAMS)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FUNCIÓN DE EVALUACIÓN DE UNA CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def _run_single(cfg_dict: Dict[str, Any], seed: int,
                operating_rooms, specialties, patients_by_specialty, staff_list) -> Dict[str, Any]:
    """
    Ejecuta el GA con una configuración dada y un seed específico.
    Suprime la salida por consola del GA durante el tuning.
    Retorna un dict con las métricas de interés.
    """
    import io, contextlib

    config = GAConfig(**cfg_dict)
    random.seed(seed)
    np.random.seed(seed)

    ga = GeneticAlgorithm(config, operating_rooms, specialties, patients_by_specialty, staff_list)

    # Suprimir stdout del GA (progress bars, tabla de generaciones)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        best = ga.run()
    elapsed = time.perf_counter() - t0

    # Contar pacientes programados desde el caché del GA
    schedule = ga.get_schedule_details(best)
    pacientes_asignados = set()
    for per_or in schedule.values():
        for a in per_or.get("asignaciones", []):
            pacientes_asignados.add(a["p"])

    total_patients = sum(len(lst) for lst in patients_by_specialty.values())

    return {
        "fitness":           best.fitness,
        "generations_ran":   len(ga.history),
        "patients_scheduled": len(pacientes_asignados),
        "patients_total":    total_patients,
        "schedule_rate":     round(len(pacientes_asignados) / total_patients * 100, 2),
        "cache_hits":        ga._cache_hits,
        "elapsed_s":         round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TUNING PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def tune(n_configs: int = 20, n_reps: int = 3, master_seed: int = 42) -> None:
    rng = random.Random(master_seed)

    operating_rooms = _build_operating_rooms()
    specialties     = _build_specialties()
    staff_list      = _build_staff()
    patients_by_specialty = {
        sid: _make_patients(sid, PATIENTS_PER_SPECIALTY, seed=sid, staff_list=staff_list)
        for sid in range(1, 9)   # especialidades 1-8
    }
    total_patients = sum(len(v) for v in patients_by_specialty.values())

    print(f"\n{'═'*70}")
    print(f"  TUNING DE HIPERPARÁMETROS — Random Search")
    print(f"  {n_configs} configuraciones × {n_reps} repeticiones = {n_configs * n_reps} ejecuciones")
    print(f"  Pacientes en pool: {total_patients}")
    print(f"{'═'*70}\n")

    configs = [_sample_config(rng) for _ in range(n_configs)]
    results = []

    tune_start = time.perf_counter()
    for cfg_idx, cfg_dict in enumerate(configs, start=1):
        rep_results = []
        rep_seeds   = [rng.randint(0, 99999) for _ in range(n_reps)]

        for rep, seed in enumerate(rep_seeds, start=1):
            done_so_far = (cfg_idx - 1) * n_reps + rep - 1
            total_runs  = n_configs * n_reps
            elapsed_total = time.perf_counter() - tune_start
            eta = (elapsed_total / done_so_far * (total_runs - done_so_far)
                   if done_so_far > 0 else 0)
            eta_str = f"ETA ~{eta/60:.1f}min" if eta > 0 else "ETA --"

            print(f"  Config {cfg_idx:>3}/{n_configs}  Rep {rep}/{n_reps}  "
                  f"pop={cfg_dict['population_size']}  "
                  f"gen={cfg_dict['max_generations']}  "
                  f"mut={cfg_dict['mutation_rate']:.2f}  "
                  f"α={cfg_dict['alpha']:.2f}  [{eta_str}]",
                  end=" ... ", flush=True)
            try:
                r = _run_single(cfg_dict, seed, operating_rooms, specialties,
                                patients_by_specialty, staff_list)
                rep_results.append(r)
                print(f"fitness={r['fitness']:.2f}  sched={r['schedule_rate']:.1f}%  "
                      f"t={r['elapsed_s']:.1f}s")
            except Exception as e:
                print(f"ERROR: {e}")
                rep_results.append(None)

        # Filtrar reps fallidas
        valid = [r for r in rep_results if r is not None]
        if not valid:
            continue

        # Agregar estadísticas
        fitnesses = [r["fitness"] for r in valid]
        rates     = [r["schedule_rate"] for r in valid]
        times     = [r["elapsed_s"] for r in valid]

        row = {
            "config_id":          cfg_idx,
            # Parámetros GA
            "population_size":    cfg_dict["population_size"],
            "max_generations":    cfg_dict["max_generations"],
            "convergence_patience": cfg_dict["convergence_patience"],
            "mutation_rate":      cfg_dict["mutation_rate"],
            "crossover_rate":     cfg_dict["crossover_rate"],
            "tournament_size":    cfg_dict["tournament_size"],
            "elite_count":        cfg_dict["elite_count"],
            "alpha":              cfg_dict["alpha"],
            "beta":               cfg_dict["beta"],
            "penalty_below":      cfg_dict["penalty_below_min_quota"],
            "penalty_above":      cfg_dict["penalty_above_max_quota"],
            # Métricas agregadas
            "n_reps":             len(valid),
            "fitness_mean":       round(np.mean(fitnesses), 4),
            "fitness_std":        round(np.std(fitnesses), 4),
            "fitness_min":        round(np.min(fitnesses), 4),
            "fitness_max":        round(np.max(fitnesses), 4),
            "fitness_cv":         round(np.std(fitnesses) / np.mean(fitnesses) * 100, 2)
                                  if np.mean(fitnesses) != 0 else 0.0,
            "schedule_rate_mean": round(np.mean(rates), 2),
            "schedule_rate_std":  round(np.std(rates), 2),
            "time_mean_s":        round(np.mean(times), 2),
            "time_std_s":         round(np.std(times), 2),
            # Score compuesto: premia fitness alto, bajo CV y alta tasa de programación
            # CV (coeficiente de variación): qué tan estable es la config entre seeds
            "score": 0.0,  # se calcula después de normalizar
        }
        results.append(row)

    if not results:
        print("\n⚠  No se obtuvieron resultados válidos.")
        return

    # ── Score compuesto normalizado ───────────────────────────────────────────
    # Normalizar fitness_mean, schedule_rate_mean (más alto = mejor)
    # y fitness_cv (más bajo = mejor → invertir)
    fitness_vals = np.array([r["fitness_mean"]       for r in results])
    rate_vals    = np.array([r["schedule_rate_mean"] for r in results])
    cv_vals      = np.array([r["fitness_cv"]         for r in results])
    time_vals    = np.array([r["time_mean_s"]        for r in results])

    def _norm(arr: np.ndarray) -> np.ndarray:
        rng_val = arr.max() - arr.min()
        return (arr - arr.min()) / rng_val if rng_val > 0 else np.zeros_like(arr)

    norm_fitness = _norm(fitness_vals)
    norm_rate    = _norm(rate_vals)
    norm_cv      = 1.0 - _norm(cv_vals)     # invertido: menor CV → mayor score
    norm_time    = 1.0 - _norm(time_vals)   # invertido: menor tiempo → mayor score

    # Pesos del score compuesto
    W_FITNESS = 0.50
    W_RATE    = 0.25
    W_STABLE  = 0.15
    W_SPEED   = 0.10

    for i, row in enumerate(results):
        row["score"] = round(
            W_FITNESS * norm_fitness[i]
            + W_RATE    * norm_rate[i]
            + W_STABLE  * norm_cv[i]
            + W_SPEED   * norm_time[i],
            4
        )

    # Ordenar por score descendente
    results.sort(key=lambda r: r["score"], reverse=True)

    # ── Imprimir tabla de ranking ─────────────────────────────────────────────
    print(f"\n\n{'═'*100}")
    print(f"  RANKING DE CONFIGURACIONES")
    print(f"{'═'*100}")
    header = (
        f"{'Rank':>4}  {'Score':>6}  {'Fit μ':>9}  {'Fit σ':>7}  {'CV%':>6}  "
        f"{'Sched%':>7}  {'t(s)':>6}  "
        f"{'pop':>4} {'gen':>4} {'mut':>5} {'cx':>5} {'tour':>4} "
        f"{'α':>5} {'pen↓':>5} {'pen↑':>5}"
    )
    print(header)
    print("─" * 100)

    for rank, row in enumerate(results, start=1):
        marker = "★" if rank == 1 else ("▲" if rank <= 3 else " ")
        print(
            f"{marker}{rank:>3}  "
            f"{row['score']:>6.4f}  "
            f"{row['fitness_mean']:>9.3f}  "
            f"{row['fitness_std']:>7.3f}  "
            f"{row['fitness_cv']:>6.2f}  "
            f"{row['schedule_rate_mean']:>7.2f}  "
            f"{row['time_mean_s']:>6.1f}  "
            f"{row['population_size']:>4} "
            f"{row['max_generations']:>4} "
            f"{row['mutation_rate']:>5.2f} "
            f"{row['crossover_rate']:>5.2f} "
            f"{row['tournament_size']:>4} "
            f"{row['alpha']:>5.2f} "
            f"{row['penalty_below']:>5.0f} "
            f"{row['penalty_above']:>5.0f}"
        )

    # ── Guardar CSV completo ──────────────────────────────────────────────────
    csv_path = "tune_results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✔  Resultados completos guardados en: {csv_path}")

    # ── Guardar mejor configuración como JSON ─────────────────────────────────
    best_cfg = configs[results[0]["config_id"] - 1]
    best_cfg_out = {k: v for k, v in best_cfg.items()}
    json_path = "tune_best.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg_out, f, indent=4, ensure_ascii=False)
    print(f"✔  Mejor configuración guardada en: {json_path}")

    # ── Resumen de la ganadora ────────────────────────────────────────────────
    best_row = results[0]
    print(f"\n{'═'*70}")
    print(f"  CONFIGURACIÓN GANADORA (score={best_row['score']:.4f})")
    print(f"{'═'*70}")
    print(f"  fitness_mean      = {best_row['fitness_mean']:.4f}  ± {best_row['fitness_std']:.4f}")
    print(f"  CV (estabilidad)  = {best_row['fitness_cv']:.2f}%  (menor = más estable)")
    print(f"  schedule_rate     = {best_row['schedule_rate_mean']:.2f}%  ± {best_row['schedule_rate_std']:.2f}%")
    print(f"  tiempo_medio      = {best_row['time_mean_s']:.1f}s")
    print(f"\n  GAConfig(")
    for k, v in best_cfg_out.items():
        print(f"      {k}={repr(v)},")
    print(f"  )")
    print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning de hiperparámetros del AG quirúrgico.")
    parser.add_argument("--configs", type=int, default=20,
                        help="Número de configuraciones a probar (default: 20)")
    parser.add_argument("--reps",    type=int, default=3,
                        help="Repeticiones por configuración (default: 3)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Semilla maestra para reproducibilidad (default: 42)")
    args = parser.parse_args()

    tune(n_configs=args.configs, n_reps=args.reps, master_seed=args.seed)
