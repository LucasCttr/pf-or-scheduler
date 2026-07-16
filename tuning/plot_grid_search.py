"""
plot_grid_search.py

Genera 4 graficos independientes a partir de grid_search_results.csv
para analizar el tuning de hiperparametros del AG.

Uso:
    python plot_grid_search.py [ruta_al_csv]

Si no se pasa ruta, busca "grid_search_results.csv" en el
directorio actual.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "grid_search_results.csv"
OUTPUT_TOP_CONFIGS = "grid_search_top_configs.png"
OUTPUT_QUALITY_VS_STABILITY = "grid_search_quality_vs_stability.png"
OUTPUT_CONVERGENCE = "grid_search_convergence.png"
OUTPUT_CONVERGENCE_PER_CONFIG = "grid_search_convergence_per_config.png"

# ------------------------------------------------------------------
# Estilo general (sobrio, legible, sin depender de seaborn)
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
})

COLOR_MAIN = "#2c6e91"
COLOR_ACCENT = "#d9534f"
COLOR_BEST = "#2e8b57"
COLOR_FAST = "#2e8b57"
COLOR_SLOW = "#d9534f"


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("robust_score", ascending=False).reset_index(drop=True)
    df["config_label"] = df.apply(
        lambda r: f"pop{int(r.population_size)}_t{int(r.tournament_size)}_"
                  f"cx{r.crossover_rate}_mut{r.mutation_rate}",
        axis=1,
    )

    top_n = min(12, len(df))
    top = df.head(top_n)
    best_idx = 0  # df ya esta ordenado por robust_score desc

    # ----------------------------------------------------------------
    # Figura 1: avg_fitness +/- std para el top N configs
    # ----------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    colors = [COLOR_BEST if i == best_idx else COLOR_MAIN for i in range(top_n)]
    ax1.bar(
        range(top_n), top["avg_fitness"], yerr=top["std_fitness"],
        color=colors, capsize=4, edgecolor="white", linewidth=0.5,
        error_kw={"ecolor": "#444", "elinewidth": 1.2},
    )
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(top["config_label"], rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Fitness promedio")
    ax1.set_title(f"Top {top_n} configuraciones por robust_score (avg \u00b1 std)")
    y_margin = top["std_fitness"].max() * 1.6
    ax1.set_ylim(top["avg_fitness"].min() - y_margin, top["avg_fitness"].max() + y_margin)
    ax1.annotate(
        "mejor (robust_score)",
        xy=(best_idx, top["avg_fitness"].iloc[best_idx] + top["std_fitness"].iloc[best_idx]),
        xytext=(best_idx + 1.5, top["avg_fitness"].max() + y_margin * 0.7),
        ha="left", fontsize=9, color=COLOR_BEST, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_BEST),
    )
    fig1.tight_layout()
    fig1.savefig(OUTPUT_TOP_CONFIGS, dpi=160, bbox_inches="tight")
    plt.close(fig1)
    print(f"Grafico guardado en: {OUTPUT_TOP_CONFIGS}")

    # ----------------------------------------------------------------
    # Figura 2: dispersion avg vs std (todas las combinaciones)
    # ----------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(
        df["std_fitness"], df["avg_fitness"],
        c=df["robust_score"], cmap="viridis", s=80,
        edgecolor="white", linewidth=0.6,
    )
    ax2.scatter(
        df["std_fitness"].iloc[best_idx], df["avg_fitness"].iloc[best_idx],
        s=240, facecolors="none", edgecolors=COLOR_ACCENT, linewidth=2,
        label="mejor robust_score",
    )
    ax2.set_xlabel("Desviacion estandar (estabilidad)")
    ax2.set_ylabel("Fitness promedio")
    ax2.set_title("Calidad vs. estabilidad de cada configuracion")
    ax2.legend(fontsize=9, loc="lower right")
    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label("robust_score", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_QUALITY_VS_STABILITY, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"Grafico guardado en: {OUTPUT_QUALITY_VS_STABILITY}")

    # ----------------------------------------------------------------
    # Figura 3: generaciones reales hasta convergencia, por
    # tournament_size
    # ----------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    grouped = df.groupby("tournament_size")["avg_generations_used"]
    means = grouped.mean()
    stds = grouped.std()
    ax3.bar(
        means.index.astype(str), means.values, yerr=stds.values,
        color=COLOR_MAIN, capsize=4, width=0.5,
        error_kw={"ecolor": "#444", "elinewidth": 1.2},
    )
    ax3.set_xlabel("tournament_size")
    ax3.set_ylabel("Generaciones hasta convergencia")
    ax3.set_title("Velocidad de convergencia segun tournament_size")
    fig3.tight_layout()
    fig3.savefig(OUTPUT_CONVERGENCE, dpi=160, bbox_inches="tight")
    plt.close(fig3)
    print(f"Grafico guardado en: {OUTPUT_CONVERGENCE}")

    # ----------------------------------------------------------------
    # Figura 4: generaciones hasta convergencia, por CONFIGURACION
    # individual (todas las combinaciones, no agrupadas), ordenadas
    # de mas rapida a mas lenta. Util cuando el fitness final es
    # practicamente el mismo entre configs y la diferencia real esta
    # en cuanto tardan en llegar ahi.
    # ----------------------------------------------------------------
    df_conv = df.sort_values("avg_generations_used", ascending=True).reset_index(drop=True)
    n_conv = len(df_conv)
    fastest_idx = 0
    slowest_idx = n_conv - 1

    colors_conv = [COLOR_MAIN] * n_conv
    colors_conv[fastest_idx] = COLOR_FAST
    colors_conv[slowest_idx] = COLOR_SLOW

    fig4, ax4 = plt.subplots(figsize=(max(11, n_conv * 0.45), 6))
    ax4.bar(
        range(n_conv), df_conv["avg_generations_used"],
        color=colors_conv, edgecolor="white", linewidth=0.5,
    )
    ax4.set_xticks(range(n_conv))
    ax4.set_xticklabels(df_conv["config_label"], rotation=45, ha="right", fontsize=7)
    ax4.set_ylabel("Generaciones promedio hasta convergencia")
    ax4.set_title("Velocidad de convergencia por configuracion "
                   "(ordenado de mas rapida a mas lenta)")
    ax4.annotate(
        "mas rapida", xy=(fastest_idx, df_conv["avg_generations_used"].iloc[fastest_idx]),
        xytext=(fastest_idx, df_conv["avg_generations_used"].iloc[fastest_idx] + 8),
        ha="center", fontsize=8, color=COLOR_FAST, fontweight="bold",
    )
    ax4.annotate(
        "mas lenta", xy=(slowest_idx, df_conv["avg_generations_used"].iloc[slowest_idx]),
        xytext=(slowest_idx, df_conv["avg_generations_used"].iloc[slowest_idx] + 8),
        ha="center", fontsize=8, color=COLOR_SLOW, fontweight="bold",
    )
    fig4.tight_layout()
    fig4.savefig(OUTPUT_CONVERGENCE_PER_CONFIG, dpi=160, bbox_inches="tight")
    plt.close(fig4)
    print(f"Grafico guardado en: {OUTPUT_CONVERGENCE_PER_CONFIG}")

    # Resumen en texto de la mejor config (por robust_score)
    best = df.iloc[best_idx]
    print("\nMejor configuracion (robust_score):")
    print(
        f"  pop={int(best.population_size)} gen={int(best.generations)} "
        f"tour={int(best.tournament_size)} cross={best.crossover_rate} "
        f"mut={best.mutation_rate}"
    )
    print(f"  avg_fitness={best.avg_fitness:.2f}  std={best.std_fitness:.2f}")

    # Resumen en texto de la configuracion mas rapida en converger
    fastest = df_conv.iloc[fastest_idx]
    print("\nConfiguracion mas rapida en converger (avg_generations_used):")
    print(
        f"  pop={int(fastest.population_size)} gen={int(fastest.generations)} "
        f"tour={int(fastest.tournament_size)} cross={fastest.crossover_rate} "
        f"mut={fastest.mutation_rate}"
    )
    print(
        f"  avg_generations_used={fastest.avg_generations_used:.1f}  "
        f"avg_fitness={fastest.avg_fitness:.4f}"
    )


if __name__ == "__main__":
    main()