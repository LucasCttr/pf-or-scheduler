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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

models_module = importlib.import_module("models")
genetic_algorithm_module = importlib.import_module("genetic_algorithm")

OperatingRoom = models_module.OperatingRoom
Specialty = models_module.Specialty
Patient = models_module.Patient
GAConfig = models_module.GAConfig
Staff = models_module.Staff
GeneticAlgorithm = genetic_algorithm_module.GeneticAlgorithm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATOS DE PRUEBA (mismos que main.py, reducidos para velocidad)
# ═══════════════════════════════════════════════════════════════════════════════

PROCEDURES_BY_SPECIALTY = {
    1: [101, 102, 103],
    2: [201, 202, 203],
    3: [301, 302],
    4: [401, 402],
    5: [501, 502],
    6: [601, 602],
    7: [701, 702],
    8: [801, 802],
}

def _build_staff() -> List[Staff]:
    return [
        Staff(id=1,  name="Dr. Pérez",     role="cirujano", enabled_procedures_ids=[101,102,103,201,202], availability_hours={0:(480,620),  1:(780,1020)}),
        Staff(id=2,  name="Dra. Sosa",     role="cirujano", enabled_procedures_ids=[101,102,103],         availability_hours={0:(480,1020), 2:(480,720)}),
        Staff(id=3,  name="Dra. Carter",   role="cirujano", enabled_procedures_ids=[101,102],             availability_hours={0:(620,1020), 2:(480,720)}),
        Staff(id=4,  name="Dr. Gomez",     role="cirujano", enabled_procedures_ids=[201,202,203,401,402], availability_hours={0:(480,720),  1:(480,720)}),
        Staff(id=5,  name="Dra. Ruiz",     role="cirujano", enabled_procedures_ids=[201,202,203],         availability_hours={1:(780,1020), 3:(780,1020)}),
        Staff(id=6,  name="Dr. Martinez",  role="cirujano", enabled_procedures_ids=[201,202],             availability_hours={2:(480,600),  4:(480,720)}),
        Staff(id=7,  name="Dra. Blanco",   role="cirujano", enabled_procedures_ids=[301,302],             availability_hours={3:(480,720),  4:(780,1020)}),
        Staff(id=8,  name="Dr. Lopez",     role="cirujano", enabled_procedures_ids=[301,302],             availability_hours={0:(780,1020), 2:(780,1020)}),
        Staff(id=9,  name="Dra. García",   role="cirujano", enabled_procedures_ids=[401,402,501,502],     availability_hours={1:(480,720),  3:(480,720)}),
        Staff(id=10, name="Dr. Rodríguez", role="cirujano", enabled_procedures_ids=[401,501,502],         availability_hours={2:(780,1020), 4:(780,1020)}),
        Staff(id=11, name="Dr. Morales",   role="cirujano", enabled_procedures_ids=[601,602],             availability_hours={0:(480,720),  3:(480,1020)}),
        Staff(id=12, name="Dra. Herrera",  role="cirujano", enabled_procedures_ids=[601,701,702],         availability_hours={1:(480,720),  4:(480,720)}),
        Staff(id=13, name="Dr. Castro",    role="cirujano", enabled_procedures_ids=[701,702],             availability_hours={2:(780,1020), 3:(780,1020)}),
        Staff(id=14, name="Dra. Mendez",   role="cirujano", enabled_procedures_ids=[801,802],             availability_hours={0:(480,720),  4:(480,1020)}),
        Staff(id=15, name="Dr. Silva",     role="cirujano", enabled_procedures_ids=[201,202,801],         availability_hours={1:(780,1020), 2:(480,720)}),
        Staff(id=16, name="Dra. Flores",   role="cirujano", enabled_procedures_ids=[101,301,302],         availability_hours={0:(780,1020), 4:(480,720)}),
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
    proc_pool = PROCEDURES_BY_SPECIALTY.get(specialty_id, [specialty_id * 100])
    cirujanos_ids = [
        s.id for s in staff_list
        if any(pid in s.enabled_procedures_ids for pid in proc_pool)
    ]
    patients: List[Patient] = []
    if specialty_id == 1:
        patients.append(Patient(id=2000, specialty_id=1, procedure_id=101, estimated_duration=60,
                                clinical_priority=99.0, required_roles=["cirujano"],
                                forced_surgeon_id=1))
    for i in range(count):
        forced = None
        chosen_proc = rng.choice(proc_pool)
        if cirujanos_ids and rng.random() < 0.20:
            forced = rng.choice(cirujanos_ids)
        patients.append(Patient(
            id=specialty_id * 100 + i,
            specialty_id=specialty_id,
            procedure_id=chosen_proc,
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
    print("  TUNING DE HIPERPARÁMETROS — Random Search")
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
    print("  RANKING DE CONFIGURACIONES")
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

    # ── Generar gráficos comparativos ─────────────────────────────────────────
    plot_results(results)

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
    print("\n  GAConfig(")
    for k, v in best_cfg_out.items():
        print(f"      {k}={repr(v)},")
    print("  )")
    print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIÓN DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: List[Dict]) -> None:
    """
    Genera un dashboard de 6 gráficos comparativos guardado como PNG.

    Gráficos:
      1. Ranking de score compuesto (barras horizontales, top-15)
      2. Fitness medio ± std por configuración (barras de error)
      3. Trade-off: fitness medio vs tiempo de ejecución
      4. Trade-off: fitness medio vs tasa de programación
      5. Influencia de parámetros clave sobre el score (scatter × 4 params)
      6. Coeficiente de variación (estabilidad entre seeds)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # sin GUI, solo genera archivos
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
    except ImportError:
        print("⚠  matplotlib/seaborn no encontrados. Omitiendo gráficos.")
        return

    # ── Paleta y estilo ───────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=0.9)
    BLUE    = "#2E4057"
    TEAL    = "#0F6E56"
    AMBER   = "#BA7517"
    CORAL   = "#993C1D"
    GRAY    = "#888780"
    GOLD    = "#E5A020"

    n      = len(results)
    ids    = [f"C{r['config_id']:02d}" for r in results]
    scores = [r["score"]              for r in results]
    f_mean = [r["fitness_mean"]       for r in results]
    f_std  = [r["fitness_std"]        for r in results]
    f_cv   = [r["fitness_cv"]         for r in results]
    sched  = [r["schedule_rate_mean"] for r in results]
    s_std  = [r["schedule_rate_std"]  for r in results]
    t_mean = [r["time_mean_s"]        for r in results]
    pops   = [r["population_size"]    for r in results]
    muts   = [r["mutation_rate"]      for r in results]
    tours  = [r["tournament_size"]    for r in results]
    alphas = [r["alpha"]              for r in results]

    # Colores de barras según ranking (oro / teal / resto)
    bar_colors = [GOLD if i == 0 else (TEAL if i < 3 else BLUE) for i in range(n)]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Tuning de hiperparámetros — Algoritmo Genético Quirúrgico",
                 fontsize=14, fontweight="bold", color=BLUE, y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── 1. Ranking score compuesto ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    top = min(15, n)
    ypos = range(top - 1, -1, -1)
    bars = ax1.barh(list(ypos), scores[:top], color=bar_colors[:top],
                    edgecolor="white", linewidth=0.5, height=0.7)
    ax1.set_yticks(list(ypos))
    ax1.set_yticklabels(ids[:top], fontsize=8)
    ax1.set_xlabel("Score compuesto (fitness 50% · sched 25% · estabilidad 15% · velocidad 10%)")
    ax1.set_title("① Ranking de configuraciones (score compuesto)", fontweight="bold", color=BLUE)
    for bar, score in zip(bars, scores[:top]):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{score:.4f}", va="center", ha="left", fontsize=7.5, color=BLUE)
    ax1.set_xlim(0, max(scores) * 1.15)
    ax1.axvline(scores[0], color=GOLD, linewidth=1.2, linestyle="--", alpha=0.7)

    # ── 2. Fitness medio ± std ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    xpos = range(n)
    ax2.bar(xpos, f_mean, color=bar_colors, edgecolor="white", linewidth=0.4, width=0.7)
    ax2.errorbar(xpos, f_mean, yerr=f_std, fmt="none",
                 color=GRAY, capsize=3, linewidth=1.2)
    ax2.set_xticks(list(xpos))
    ax2.set_xticklabels(ids, rotation=70, fontsize=6.5)
    ax2.set_ylabel("Fitness medio")
    ax2.set_title("② Fitness medio ± σ por config.", fontweight="bold", color=BLUE)
    ax2.yaxis.set_tick_params(labelsize=7.5)

    # ── 3. Tasa de programación ± std ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(xpos, sched, color=bar_colors, edgecolor="white", linewidth=0.4, width=0.7)
    ax3.errorbar(xpos, sched, yerr=s_std, fmt="none",
                 color=GRAY, capsize=2.5, linewidth=1.1)
    ax3.set_xticks(list(xpos))
    ax3.set_xticklabels(ids, rotation=70, fontsize=6.5)
    ax3.set_ylabel("Pacientes programados (%)")
    ax3.set_ylim(0, 100)
    ax3.set_title("③ Tasa de programación ± σ", fontweight="bold", color=BLUE)
    ax3.yaxis.set_tick_params(labelsize=7.5)

    # ── 4. Estabilidad: CV por config ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    cv_colors = [TEAL if cv < 5 else (AMBER if cv < 10 else CORAL) for cv in f_cv]
    ax4.bar(xpos, f_cv, color=cv_colors, edgecolor="white", linewidth=0.4, width=0.7)
    ax4.set_xticks(list(xpos))
    ax4.set_xticklabels(ids, rotation=70, fontsize=6.5)
    ax4.set_ylabel("CV (%) — menor = más estable")
    ax4.set_title("④ Estabilidad entre seeds (CV%)", fontweight="bold", color=BLUE)
    ax4.yaxis.set_tick_params(labelsize=7.5)
    # Leyenda de colores
    from matplotlib.patches import Patch
    ax4.legend(handles=[
        Patch(facecolor=TEAL,  label="CV < 5% (muy estable)"),
        Patch(facecolor=AMBER, label="CV 5-10%"),
        Patch(facecolor=CORAL, label="CV > 10% (inestable)"),
    ], fontsize=7, loc="upper right")

    # ── 5. Trade-off fitness vs tiempo ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    sc5 = ax5.scatter(t_mean, f_mean, c=scores, cmap="RdYlGn",
                      s=60, edgecolors=BLUE, linewidths=0.5, zorder=3)
    # Anotar top-3
    for i in range(min(3, n)):
        ax5.annotate(ids[i], (t_mean[i], f_mean[i]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=7, color=BLUE, fontweight="bold")
    plt.colorbar(sc5, ax=ax5, label="Score", shrink=0.85)
    ax5.set_xlabel("Tiempo medio (s)")
    ax5.set_ylabel("Fitness medio")
    ax5.set_title("⑤ Fitness vs Tiempo\n(color = score)", fontweight="bold", color=BLUE)

    # ── 6. Influencia de parámetros sobre el score (4 scatter) ───────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    sub_params = [
        (pops,   "population_size",  "Tamaño de población"),
        (muts,   "mutation_rate",    "Tasa de mutación"),
        (alphas, "alpha",            "Alpha (peso prioridad)"),
        (tours,  "tournament_size",  "Tamaño de torneo"),
    ]
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2, :], wspace=0.4)

    for col, (param_vals, param_key, param_label) in enumerate(sub_params):
        axi = fig.add_subplot(inner_gs[col])
        axi.scatter(param_vals, scores, c=scores, cmap="RdYlGn",
                    s=50, edgecolors=BLUE, linewidths=0.4, zorder=3)
        # Línea de tendencia (regresión lineal simple)
        pv = np.array(param_vals, dtype=float)
        sv = np.array(scores,     dtype=float)
        if pv.std() > 0:
            m, b = np.polyfit(pv, sv, 1)
            x_line = np.linspace(pv.min(), pv.max(), 50)
            axi.plot(x_line, m * x_line + b, color=CORAL, linewidth=1.5,
                     linestyle="--", alpha=0.8, label="tendencia")
            # Correlación de Pearson
            corr = np.corrcoef(pv, sv)[0, 1]
            axi.set_title(f"⑥ {param_label}\nr = {corr:.2f}",
                          fontweight="bold", color=BLUE, fontsize=8.5)
        else:
            axi.set_title(f"⑥ {param_label}", fontweight="bold", color=BLUE, fontsize=8.5)
        axi.set_xlabel(param_key, fontsize=7.5)
        axi.set_ylabel("Score" if col == 0 else "", fontsize=7.5)
        axi.tick_params(labelsize=7)

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path = "tune_comparativa.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"✔  Gráficos guardados en: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLI
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