"""
plot_grid_search.py (Adaptado)
Genera gráficos a partir de grid_search_results.csv
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "grid_search_results.csv"

# ------------------------------------------------------------------
# Configuración de estilo y salidas
# ------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", 
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.facecolor": "white"})

COLOR_MAIN = "#2c6e91"
COLOR_BEST = "#2e8b57"
COLOR_FAST = "#2e8b57"
COLOR_SLOW = "#d9534f"

def main():
    df = pd.read_csv(CSV_PATH)
    # Ordenamos por fitness promedio (ya que no existe robust_score)
    df = df.sort_values("avg_fitness", ascending=False).reset_index(drop=True)
    df["config_label"] = df.apply(
        lambda r: f"pop{int(r.population_size)}_t{int(r.tournament_size)}_"
                  f"cx{r.crossover_rate}_mut{r.mutation_rate}", axis=1,
    )

    top_n = min(12, len(df))
    top = df.head(top_n)

    # ----------------------------------------------------------------
    # Figura 1: avg_fitness para el top N configs
    # ----------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    colors = [COLOR_BEST if i == 0 else COLOR_MAIN for i in range(top_n)]
    ax1.bar(range(top_n), top["avg_fitness"], color=colors, edgecolor="white")
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(top["config_label"], rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Fitness promedio")
    ax1.set_title(f"Top {top_n} configuraciones por Fitness promedio")
    ax1.set_ylim(top["avg_fitness"].min() * 0.99, top["avg_fitness"].max() * 1.01)
    fig1.tight_layout()
    fig1.savefig("grid_search_top_configs.png")
    plt.close(fig1)

    # ----------------------------------------------------------------
    # Figura 2: Generaciones para converger (por tournament_size)
    # ----------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    grouped = df.groupby("tournament_size")["avg_generations_used"].mean()
    ax2.bar(grouped.index.astype(str), grouped.values, color=COLOR_MAIN, width=0.5)
    ax2.set_xlabel("tournament_size")
    ax2.set_ylabel("Generaciones hasta convergencia")
    ax2.set_title("Velocidad promedio de convergencia")
    fig2.tight_layout()
    fig2.savefig("grid_search_convergence.png")
    plt.close(fig2)

    # ----------------------------------------------------------------
    # Figura 3: Velocidad por configuración individual
    # ----------------------------------------------------------------
    df_conv = df.sort_values("avg_generations_used", ascending=True).reset_index(drop=True)
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    ax3.bar(range(len(df_conv)), df_conv["avg_generations_used"], color=COLOR_MAIN)
    ax3.set_xticks(range(len(df_conv)))
    ax3.set_xticklabels(df_conv["config_label"], rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Generaciones promedio")
    ax3.set_title("Velocidad de convergencia por configuración")
    fig3.tight_layout()
    fig3.savefig("grid_search_convergence_per_config.png")
    plt.close(fig3)

    # ----------------------------------------------------------------
    # Figura 4: Relación Eficiencia (Fitness vs. Generaciones)
    # ----------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sc = ax4.scatter(df["avg_generations_used"], df["avg_fitness"], 
                     c=df["avg_fitness"], cmap="viridis", s=100, edgecolors='black')
    
    ax4.set_xlabel("Generaciones para converger")
    ax4.set_ylabel("Fitness promedio")
    ax4.set_title("Eficiencia: Fitness vs. Velocidad")
    cbar = fig4.colorbar(sc, ax=ax4)
    cbar.set_label("Fitness promedio")
    
    # Marcamos la mejor
    ax4.scatter(df["avg_generations_used"].iloc[0], df["avg_fitness"].iloc[0], 
                s=200, facecolors='none', edgecolors='red', linewidth=2, label="Mejor config")
    
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig("grid_search_efficiency.png")
    plt.close(fig4)
    
    print("Gráficos generados exitosamente.")

if __name__ == "__main__":
    main()