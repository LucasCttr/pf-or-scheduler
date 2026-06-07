"""
tune_ga.py — Búsqueda de hiperparámetros para el Algoritmo Genético.
Versión optimizada con Scoring de 3 Componentes (Fitness, CV y Tiempo).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
import time
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# ── Monkeypatch Corrección PuLP: Forzar límite de tiempo de 2s durante el tuning ──
import pulp
import pulp.apis as _pulp_apis

_OriginalCBC = getattr(_pulp_apis, "PULP_CBC_CMD", pulp.PULP_CBC_CMD)

class _TuningCBC(_OriginalCBC):
    TUNING_TIME_LIMIT = 2  # Máximo 2 segundos por turno en el tuning

    def __init__(self, *args, **kwargs):
        kwargs["timeLimit"] = self.TUNING_TIME_LIMIT
        kwargs["msg"]       = 0
        super().__init__(*args, **kwargs)

pulp.PULP_CBC_CMD = _TuningCBC
if hasattr(pulp, "solvers"):
    pulp.solvers.PULP_CBC_CMD = _TuningCBC
if hasattr(_pulp_apis, "PULP_CBC_CMD"):
    _pulp_apis.PULP_CBC_CMD = _TuningCBC
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

models_module = importlib.import_module("models")
genetic_algorithm_module = importlib.import_module("genetic_algorithm")
main_module = importlib.import_module("main")

GAConfig = models_module.GAConfig
GeneticAlgorithm = genetic_algorithm_module.GeneticAlgorithm

# ═══════════════════════════════════════════════════════════════════════
# ESPACIO DE BÚSQUEDA PARAMÉTRICO Y PARÁMETROS FIJOS DE COMPATIBILIDAD
# ═══════════════════════════════════════════════════════════════════════
SEARCH_SPACE: Dict[str, Any] = {
    "population_size":         [15, 25, 40],       
    "max_generations":         [15, 30],           
    "convergence_patience":    [4, 6],
    "mutation_rate":           (0.05, 0.20, "float"),
    "crossover_rate":          (0.75, 0.90, "float"),
    "tournament_size":         [3, 5],
    "elite_count":             [1, 2],
    "alpha":                   (0.5, 0.8, "float"),
}

FIXED_PARAMS = {
    "n_days":             5,
    "n_shifts":           1,    # <-- IMPORTANTE: 1 solo turno por día
    "block_duration_min": 720,  # <-- IMPORTANTE: Ventana unificada de 12 horas (720 min)
    "slot_size_min":      15,   # Genera 48 slots de 15 min por día
    "parallel_workers":   24,
}


def _sample_config(rng: random.Random) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key, space in SEARCH_SPACE.items():
        if isinstance(space, list):
            cfg[key] = rng.choice(space)
        else:
            lo, hi, kind = space
            val = rng.uniform(lo, hi)
            cfg[key] = round(val, 3) if kind == "float" else int(val)

    alpha = cfg.pop("alpha")
    cfg["alpha"] = round(alpha, 3)
    cfg["beta"]  = round(1.0 - alpha, 3)

    cfg.update(FIXED_PARAMS)
    return cfg


def _run_single(cfg_dict: Dict[str, Any], seed: int,
                operating_rooms, specialties, patients_by_specialty, staff_list) -> Dict[str, Any]:
    config = GAConfig(**cfg_dict)
    random.seed(seed)
    np.random.seed(seed)

    ga = GeneticAlgorithm(
        config, operating_rooms, specialties, patients_by_specialty, 
        staff_list, procedures_by_specialty=main_module.PROCEDURES_BY_SPECIALTY
    )

    t0 = time.perf_counter()
    best = ga.run()
    elapsed = time.perf_counter() - t0

    schedule = ga.get_schedule_details(best)
    pacientes_asignados = set()
    for per_or in schedule.values():
        for a in per_or.get("asignaciones", []):
            pacientes_asignados.add(a["p"])

    total_patients = sum(len(lst) for lst in patients_by_specialty.values())

    return {
        "fitness":            best.fitness,
        "generations_ran":    len(ga.history),
        "patients_scheduled": len(pacientes_asignados),
        "patients_total":     total_patients,
        "schedule_rate":      round(len(pacientes_asignados) / total_patients * 100, 2),
        "cache_hits":         ga._cache_hits,
        "elapsed_s":          round(elapsed, 2),
    }


def tune(n_configs: int = 10, n_reps: int = 2, master_seed: int = 42) -> None:
    rng = random.Random(master_seed)

    print("[*] Cargando setup real de la clínica desde main.py...")
    operating_rooms = main_module.build_operating_rooms()
    specialties     = main_module.build_specialties()
    staff_list      = main_module.build_staff()
    
    print("[*] Estructurando K-Folds de demanda real mediante main_module.make_patients...")
    scenarios_patients = []
    for fold_idx in range(n_reps):
        patients_fold = {
            sid: main_module.make_patients(sid, count=35, seed=master_seed + fold_idx + sid, staff_list=staff_list)
            for sid in range(1, 9)
        }
        scenarios_patients.append(patients_fold)

    print(f"\n{'═'*70}")
    print("  TUNING CON EXTRACCIÓN DIRECTA — EVALUACIÓN TRI-COMPONENTE")
    print(f"  {n_configs} configuraciones × {n_reps} Folds = {n_configs * n_reps} corridas globales")
    print(f"{'═'*70}\n")

    configs = [_sample_config(rng) for _ in range(n_configs)]
    results = []

    for cfg_idx, cfg_dict in enumerate(configs, start=1):
        rep_results = []
        print(f"▶ Config {cfg_idx:>2}/{n_configs} | pop={cfg_dict['population_size']} gen={cfg_dict['max_generations']} mut={cfg_dict['mutation_rate']:.2f}")
        
        for fold_idx in range(n_reps):
            print(f"   └─ Ejecutando Fold {fold_idx + 1}/{n_reps} ... ", end="", flush=True)
            try:
                patients_current_fold = scenarios_patients[fold_idx]
                ag_seed = master_seed + fold_idx + 99
                
                r = _run_single(cfg_dict, ag_seed, operating_rooms, specialties,
                                patients_current_fold, staff_list)
                rep_results.append(r)
                
                print(f"OK! (Fit: {r['fitness']:.1f} | Sched: {r['patients_scheduled']} pac | t: {r['elapsed_s']:.1f}s)")
            except Exception as e:
                print(f"FALLÓ: {e}")
                rep_results.append(None)

        valid = [r for r in rep_results if r is not None]
        if not valid: continue

        fitnesses = [r["fitness"] for r in valid]
        rates     = [r["schedule_rate"] for r in valid]
        times     = [r["elapsed_s"] for r in valid]

        row = {
            "config_id":            cfg_idx,
            "population_size":      cfg_dict["population_size"],
            "max_generations":      cfg_dict["max_generations"],
            "convergence_patience": cfg_dict["convergence_patience"],
            "mutation_rate":        cfg_dict["mutation_rate"],
            "crossover_rate":       cfg_dict["crossover_rate"],
            "tournament_size":      cfg_dict["tournament_size"],
            "elite_count":          cfg_dict["elite_count"],
            "alpha":                cfg_dict["alpha"],
            "beta":                 cfg_dict["beta"],
            "n_reps":               len(valid),
            "fitness_mean":         round(np.mean(fitnesses), 4),
            "fitness_std":          round(np.std(fitnesses), 4),
            "fitness_min":          round(np.min(fitnesses), 4),
            "fitness_max":          round(np.max(fitnesses), 4),
            # Coeficiente de variación (CV%)
            "fitness_cv":           round(np.std(fitnesses) / np.abs(np.mean(fitnesses)) * 100, 2) if np.mean(fitnesses) != 0 else 0.0,
            "schedule_rate_mean":   round(np.mean(rates), 2),
            "schedule_rate_std":    round(np.std(rates), 2),
            "time_mean_s":          round(np.mean(times), 2),
            "time_std_s":           round(np.std(times), 2),
            "score":                0.0,
        }
        results.append(row)

    if not results:
        print("[-] Error general: ninguna configuración se pudo procesar.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # CÁLCULO DEL SCORE COMPUESTO (NORMALIZACIÓN + INVERSIÓN DE COSTOS)
    # ═══════════════════════════════════════════════════════════════════════
    fitness_vals = np.array([r["fitness_mean"] for r in results])
    cv_vals      = np.array([r["fitness_cv"]   for r in results])
    time_vals    = np.array([r["time_mean_s"]  for r in results])

    def _norm(arr: np.ndarray) -> np.ndarray:
        rng_val = arr.max() - arr.min()
        return (arr - arr.min()) / rng_val if rng_val > 0 else np.zeros_like(arr)

    # Normalización pura [0, 1]
    norm_fitness = _norm(fitness_vals)
    norm_cv      = _norm(cv_vals)
    norm_time    = _norm(time_vals)

    # Pesos vectoriales definidos en la sección 10.7.3 de la tesis
    W_FITNESS = 0.70
    W_STABLE  = 0.20
    W_SPEED   = 0.10

    for i, row in enumerate(results):
        # Aplicamos la ecuación: 0.70*F + 0.20*(1 - CV) + 0.10*(1 - T)
        row["score"] = round(
            W_FITNESS * norm_fitness[i] + 
            W_STABLE  * (1.0 - norm_cv[i]) + 
            W_SPEED   * (1.0 - norm_time[i]), 4
        )

    # Ordenamos el ranking según el score final de mayor a menor
    results.sort(key=lambda r: r["score"], reverse=True)
    
    print(f"\n\n{'═'*85}")
    print("  RANKING FINAL DE CONFIGURACIONES PARAMÉTRICAS (MÓDELO TRI-COMPONENTE)")
    print(f"{'═'*85}")
    print(f"{'Rank':>4}  {'Score':>6}  {'Fit μ':>9}  {'CV%':>6}  {'t(s)':>6}  {'pop':>4} {'gen':>4} {'mut':>5} {'α':>5}")
    print("─" * 85)
    for rank, row in enumerate(results, start=1):
        marker = "★" if rank == 1 else " "
        print(
            f"{marker}{rank:>3}  "
            f"{row['score']:>6.4f}  "
            f"{row['fitness_mean']:>9.2f}  "
            f"{row['fitness_cv']:>6.2f}  "
            f"{row['time_mean_s']:>6.1f}  "
            f"{row['population_size']:>4} "
            f"{row['max_generations']:>4} "
            f"{row['mutation_rate']:>5.2f} "
            f"{row['alpha']:>5.2f}"
        )

    print(f"\n✔ Calibración terminada. Configuración ganadora exportada con éxito a 'tune_best.json'")
    
    # Exportamos los hiperparámetros de la configuración ganadora (Rank 1)
    best_config_id = results[0]["config_id"]
    with open("tune_best.json", "w", encoding="utf-8") as f:
        json.dump(configs[best_config_id - 1], f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    tune(n_configs=10, n_reps=2)