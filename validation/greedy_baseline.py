"""
greedy_baseline.py

Compara el Algoritmo Genetico actual contra un baseline greedy deterministico
que resuelve el Paso A (asignacion de especialidades a bloques) con una regla
fija tipo knapsack, en lugar de busqueda evolutiva.

Ambos metodos usan:
  - El mismo decoder (build_agenda) para el Paso B.
  - La misma funcion de fitness (GeneticAlgorithm._evaluate), reutilizada
    directamente desde la clase del GA para garantizar que la comparacion
    sea justa (misma formula, mismo max_achievable_priority).

Uso:
    python greedy_baseline.py --data-dir data --seeds 30
"""
import argparse
import copy
import json
import os
import random
import statistics
import time
from typing import Dict, List

from data_loader import load_all
from models import Block, Procedure, Room, Specialty
from genetic_algorithm import GeneticAlgorithm

DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes"]


# ---------------------------------------------------------------------------
# Paso A resuelto con regla fija (sin GA)
# ---------------------------------------------------------------------------
def greedy_chromosome(
    days: List[str],
    rooms: List[Room],
    specialties: List[Specialty],
    procedures: Dict[str, Procedure],
    patients,
) -> Dict[Block, str]:
    """
    Construye una asignacion especialidad->bloque de una sola pasada,
    sin busqueda evolutiva. Logica tipo knapsack greedy:

      1. Se calcula, por especialidad, la demanda ponderada (suma de
         clinical_priority^2, igual criterio que usa la fitness del GA)
         y el tiempo total requerido por sus pacientes pendientes.
      2. Se ordenan las especialidades por "densidad de valor"
         (prioridad^2 acumulada por minuto de cirugia).
      3. Primero se cubren los min_blocks de cada especialidad
         (restriccion dura, igual que hace _repair en el GA).
      4. Los bloques restantes se reparten a la especialidad con mayor
         densidad de valor que aun tenga demanda pendiente, descontando
         el tiempo estimado a medida que se asignan bloques.
    """
    rooms_by_id = {r.id: r for r in rooms}
    blocks = [Block(d, r.id) for d in days for r in rooms]
    specialty_ids = [s.id for s in specialties]
    min_blocks = {s.id: s.min_blocks for s in specialties}

    demand_priority_sq = {sid: 0.0 for sid in specialty_ids}
    demand_time = {sid: 0 for sid in specialty_ids}

    for p in patients:
        proc = procedures.get(p.procedure_id)
        if proc is None:
            continue
        demand_priority_sq[p.specialty_id] += p.clinical_priority ** 2
        demand_time[p.specialty_id] += proc.estimated_duration

    value_density = {
        sid: (demand_priority_sq[sid] / demand_time[sid]) if demand_time[sid] > 0 else 0.0
        for sid in specialty_ids
    }

    remaining_time_needed = dict(demand_time)

    # Orden fijo y determinista de bloques: salas de mayor capacidad primero,
    # asi las especialidades con mas demanda no quedan atadas a salas chicas.
    available_blocks = sorted(
        blocks, key=lambda b: rooms_by_id[b.room_id].daily_capacity_minutes, reverse=True
    )

    chromosome: Dict[Block, str] = {}

    # --- Paso 1: cubrir min_blocks (restriccion dura) ---
    for sid in sorted(specialty_ids, key=lambda s: min_blocks.get(s, 0), reverse=True):
        needed = min_blocks.get(sid, 0)
        while needed > 0 and available_blocks:
            block = available_blocks.pop(0)
            chromosome[block] = sid
            remaining_time_needed[sid] = max(
                0, remaining_time_needed[sid] - rooms_by_id[block.room_id].daily_capacity_minutes
            )
            needed -= 1

    # --- Paso 2: repartir bloques restantes por densidad de valor ---
    while available_blocks:
        candidates = [sid for sid in specialty_ids if remaining_time_needed[sid] > 0]
        if not candidates:
            # Sin demanda pendiente conocida: no dejar bloques sin asignar,
            # se reparten por densidad de valor original (fallback).
            candidates = specialty_ids

        sid = max(candidates, key=lambda s: value_density[s])
        block = available_blocks.pop(0)
        chromosome[block] = sid
        remaining_time_needed[sid] = max(
            0, remaining_time_needed[sid] - rooms_by_id[block.room_id].daily_capacity_minutes
        )

    return chromosome


# ---------------------------------------------------------------------------
# Baseline aleatorio (piso absoluto, sin ningun criterio)
# ---------------------------------------------------------------------------
def random_chromosome(days: List[str], rooms: List[Room], specialties: List[Specialty]) -> Dict[Block, str]:
    """
    Asigna una especialidad al azar a cada bloque, sin ningun criterio.
    No respeta min_blocks a proposito: es el piso absoluto de comparacion,
    "no aplicar ninguna heuristica".
    """
    blocks = [Block(d, r.id) for d in days for r in rooms]
    specialty_ids = [s.id for s in specialties]
    return {b: random.choice(specialty_ids) for b in blocks}


# ---------------------------------------------------------------------------
# Comparacion
# ---------------------------------------------------------------------------
def run_comparison(data_dir: str, seeds: int, ga_kwargs: dict):
    specialties, rooms, procedures, surgeons, patients = load_all(data_dir)
    procedures_by_id = {p.id: p for p in procedures}

    print(
        f"Datos cargados: {len(specialties)} especialidades, {len(rooms)} quirofanos, "
        f"{len(surgeons)} cirujanos, {len(procedures)} procedimientos, "
        f"{len(patients)} pacientes."
    )

    # GeneticAlgorithm calcula max_achievable_priority en su __init__, y expone
    # _evaluate(). Lo reutilizamos como "arbitro" de fitness para ambos metodos,
    # asi la comparacion queda garantizada como justa.
    ga_ref = GeneticAlgorithm(
        days=DAYS, rooms=rooms, specialties=specialties, surgeons=surgeons,
        procedures=procedures, patients=copy.deepcopy(patients), **ga_kwargs,
    )

    # --- Baseline greedy (determinista, un solo resultado) ---
    greedy_patients = copy.deepcopy(patients)
    t0 = time.perf_counter()
    chromosome_greedy = greedy_chromosome(DAYS, rooms, specialties, procedures_by_id, greedy_patients)

    ga_for_greedy_eval = GeneticAlgorithm(
        days=DAYS, rooms=rooms, specialties=specialties, surgeons=surgeons,
        procedures=procedures, patients=greedy_patients, **ga_kwargs,
    )
    fitness_greedy, agenda_greedy = ga_for_greedy_eval._evaluate(chromosome_greedy)
    tiempo_greedy = time.perf_counter() - t0
    scheduled_greedy = len(agenda_greedy.all_surgeries())

    print(f"\n--- Baseline Greedy (determinista) ---")
    print(f"Fitness: {fitness_greedy:.4f}")
    print(f"Pacientes programados: {scheduled_greedy} / {len(patients)}")
    print(f"Tiempo de ejecucion: {tiempo_greedy:.4f}s")

    # --- GA con multiples semillas ---
    fitness_ga_runs = []
    scheduled_ga_runs = []
    tiempos_ga = []
    print(f"\n--- Corriendo GA con {seeds} semillas ---")
    for seed in range(seeds):
        random.seed(seed)
        ga_patients = copy.deepcopy(patients)
        ga = GeneticAlgorithm(
            days=DAYS, rooms=rooms, specialties=specialties, surgeons=surgeons,
            procedures=procedures, patients=ga_patients, **ga_kwargs,
        )
        t0 = time.perf_counter()
        _, best_fitness, best_agenda = ga.run()
        tiempo_run = time.perf_counter() - t0
        tiempos_ga.append(tiempo_run)
        fitness_ga_runs.append(best_fitness)
        scheduled_ga_runs.append(len(best_agenda.all_surgeries()))
        print(f"  semilla {seed:>2}: fitness={best_fitness:.4f}  "
              f"programados={len(best_agenda.all_surgeries())}/{len(patients)}  "
              f"tiempo={tiempo_run:.2f}s")

    mean_ga = statistics.mean(fitness_ga_runs)
    std_ga = statistics.pstdev(fitness_ga_runs)
    best_ga = max(fitness_ga_runs)
    worst_ga = min(fitness_ga_runs)
    mejora_pct = ((mean_ga - fitness_greedy) / fitness_greedy * 100) if fitness_greedy else float("nan")
    tiempo_ga_medio = statistics.mean(tiempos_ga)
    tiempo_ga_std = statistics.pstdev(tiempos_ga)

    print("\n=== Resumen comparativo ===")
    print(f"Greedy (determinista):        {fitness_greedy:.4f}")
    print(f"GA - media:                   {mean_ga:.4f}")
    print(f"GA - desvio estandar:         {std_ga:.4f}")
    print(f"GA - mejor corrida:           {best_ga:.4f}")
    print(f"GA - peor corrida:            {worst_ga:.4f}")
    print(f"Mejora relativa (media vs greedy): {mejora_pct:.1f}%")
    print(f"\n--- Tiempos de ejecucion ---")
    print(f"Greedy:                       {tiempo_greedy:.4f}s")
    print(f"GA (media por corrida):       {tiempo_ga_medio:.2f}s  (std: {tiempo_ga_std:.2f}s)")
    print(f"GA (corrida mas rapida):      {min(tiempos_ga):.2f}s")
    print(f"GA (corrida mas lenta):       {max(tiempos_ga):.2f}s")
    print(f"Factor de tiempo (GA / Greedy): {tiempo_ga_medio / tiempo_greedy:,.0f}x mas lento")

    # Test estadistico: la media del GA, es significativamente distinta al
    # valor fijo del greedy o podria ser producto del azar?
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(fitness_ga_runs, fitness_greedy)
        print(f"t-test (GA vs greedy):        t={t_stat:.3f}  p-value={p_value:.6f}")
        if p_value < 0.05:
            print("  -> La diferencia es estadisticamente significativa (p < 0.05).")
        else:
            print("  -> La diferencia NO es estadisticamente significativa (p >= 0.05).")
    except ImportError:
        print("(scipy no instalado: se omite el t-test. `pip install scipy` para habilitarlo)")

    # Grafico de barras: greedy (determinista) vs media del GA con barra de
    # error (desvio estandar) y puntos individuales de cada semilla superpuestos.
    # Se evita el boxplot + eje quebrado porque mezclar una linea plana con una
    # caja de detalle en escalas tan distintas termina siendo dificil de leer.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import random as _rnd

        fig, ax = plt.subplots(figsize=(6, 6))

        etiquetas = ["Greedy\n(determinista)", f"GA\n({seeds} semillas)"]
        valores = [fitness_greedy, mean_ga]
        errores = [0, std_ga]
        colores = ["#b0b0b0", "#4c72b0"]

        bars = ax.bar(etiquetas, valores, yerr=errores, capsize=6,
                      color=colores, edgecolor="black", linewidth=0.8, width=0.55,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"})

        # Puntos individuales de cada corrida del GA, con jitter horizontal
        # para mostrar la dispersion real (no solo el resumen media +- std).
        rng = _rnd.Random(0)
        jitter_x = [1 + rng.uniform(-0.12, 0.12) for _ in fitness_ga_runs]
        ax.scatter(jitter_x, fitness_ga_runs, color="black", s=14, zorder=3,
                   alpha=0.5, label="Corridas individuales del GA")

        # Etiquetas de valor sobre cada barra
        max_val = max(valores) + max(errores)
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.03,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Flecha + porcentaje de mejora entre las dos barras
        ax.annotate(
            f"+{mejora_pct:.1f}%",
            xy=(1, mean_ga + max_val * 0.10), xytext=(0, fitness_greedy + max_val * 0.10),
            ha="center", va="center", fontsize=10, color="darkgreen", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=1.5,
                             connectionstyle="arc3,rad=-0.25"),
        )

        ax.set_ylabel("Fitness")
        ax.set_ylim(0, max_val * 1.25)
        ax.set_title(f"GA ({seeds} semillas) vs Greedy determinista")
        ax.legend(loc="lower center")
        fig.tight_layout()

        out_png = os.path.join(os.path.dirname(__file__), "comparacion_ga_vs_greedy.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"\nGrafico guardado en: {out_png}")
    except ImportError:
        print("(matplotlib no instalado: se omite el grafico. `pip install matplotlib` para habilitarlo)")

    # Resultado en JSON para usar en el informe
    result = {
        "greedy": {
            "fitness": fitness_greedy,
            "pacientes_programados": scheduled_greedy,
            "tiempo_segundos": tiempo_greedy,
        },
        "ga": {
            "fitness_runs": fitness_ga_runs,
            "scheduled_runs": scheduled_ga_runs,
            "tiempo_runs_segundos": tiempos_ga,
            "media": mean_ga,
            "std": std_ga,
            "mejor": best_ga,
            "peor": worst_ga,
            "tiempo_medio_segundos": tiempo_ga_medio,
            "tiempo_std_segundos": tiempo_ga_std,
        },
        "mejora_relativa_pct": mejora_pct,
        "factor_tiempo_ga_vs_greedy": tiempo_ga_medio / tiempo_greedy if tiempo_greedy else None,
    }
    out_json = os.path.join(os.path.dirname(__file__), "comparacion_ga_vs_greedy.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Resultado exportado a: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Compara el GA contra un baseline greedy.")
    parser.add_argument("--data-dir", default="data", help="Carpeta con los CSV de entrada")
    parser.add_argument("--seeds", type=int, default=30, help="Cantidad de corridas del GA (semillas distintas)")
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.85)
    parser.add_argument("--mutation-rate", type=float, default=0.04)
    parser.add_argument("--stagnation-limit", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args()

    ga_kwargs = dict(
        population_size=args.population_size,
        generations=args.generations,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        stagnation_limit=args.stagnation_limit,
        alpha=args.alpha,
        beta=args.beta,
    )

    run_comparison(args.data_dir, args.seeds, ga_kwargs)


if __name__ == "__main__":
    main()