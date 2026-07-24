"""
plot_grid_search.py (Adaptado)
Genera gráficos a partir de grid_search_results.csv
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
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

# Paleta fija para especialidades, para que el color de cada una
# sea consistente entre todos los gráficos
SPECIALTY_PALETTE = [
    "#2c6e91", "#2e8b57", "#d9534f", "#e0a72e", "#7b5ea7", "#3aa6a0"
]

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

    # Columnas de especialidades detectadas automáticamente por prefijo
    specialty_cols = [c for c in df.columns if c.startswith("assigned_")]
    specialty_names = [c.replace("assigned_", "") for c in specialty_cols]

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

    # ----------------------------------------------------------------
    # Figura 5: Vector de especialidades - Top N configs (barras agrupadas)
    # ----------------------------------------------------------------
    if specialty_cols:
        fig5, ax5 = plt.subplots(figsize=(13, 6))
        n_specialties = len(specialty_cols)
        bar_width = 0.8 / n_specialties
        x = np.arange(top_n)

        for i, (col, name) in enumerate(zip(specialty_cols, specialty_names)):
            offset = (i - n_specialties / 2) * bar_width + bar_width / 2
            ax5.bar(x + offset, top[col], width=bar_width,
                    label=name, color=SPECIALTY_PALETTE[i % len(SPECIALTY_PALETTE)],
                    edgecolor="white")

        ax5.set_xticks(x)
        ax5.set_xticklabels(top["config_label"], rotation=40, ha="right", fontsize=8)
        ax5.set_ylabel("Pacientes asignados")
        ax5.set_title(f"Distribución de pacientes por especialidad - Top {top_n} configuraciones")
        ax5.legend(title="Especialidad", ncol=n_specialties, fontsize=8,
                   loc="upper center", bbox_to_anchor=(0.5, -0.25))
        fig5.tight_layout()
        fig5.savefig("grid_search_specialty_distribution.png")
        plt.close(fig5)

        # ------------------------------------------------------------
        # Figura 6: Vector de especialidades - solo la mejor config
        # ------------------------------------------------------------
        best_row = df.iloc[0]
        fig6, ax6 = plt.subplots(figsize=(7, 6))
        values = [best_row[c] for c in specialty_cols]
        bars = ax6.bar(specialty_names, values,
                       color=[SPECIALTY_PALETTE[i % len(SPECIALTY_PALETTE)]
                              for i in range(n_specialties)],
                       edgecolor="white")
        ax6.bar_label(bars, padding=3, fontsize=9)
        ax6.set_ylabel("Pacientes asignados")
        ax6.set_title(f"Pacientes asignados por especialidad\n({best_row['config_label']})")
        fig6.tight_layout()
        fig6.savefig("grid_search_specialty_best_config.png")
        plt.close(fig6)
        
        # ------------------------------------------------------------
        # Figura 7: Vector de especialidades - mejor config, como torta
        # ------------------------------------------------------------
        fig7, ax7 = plt.subplots(figsize=(7, 7))
        ax7.pie(
            values,
            labels=specialty_names,
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(values)))})",
            colors=[SPECIALTY_PALETTE[i % len(SPECIALTY_PALETTE)] for i in range(n_specialties)],
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax7.set_title(f"Proporción de especialidades en la agenda final\n({best_row['config_label']})")
        fig7.tight_layout()
        fig7.savefig("grid_search_specialty_pie.png")
        plt.close(fig7)
        
        # ----------------------------------------------------------------
    # Figura 8: Especialidad asignada por bloque (día x quirófano)
    # ----------------------------------------------------------------
    import os as _os
    chromosome_csv = _os.path.join(_os.path.dirname(CSV_PATH) or ".", "best_chromosome.csv")

    if _os.path.exists(chromosome_csv):
        df_chr = pd.read_csv(chromosome_csv)

        days_order = ["lunes", "martes", "miercoles", "jueves", "viernes"]
        rooms_order = sorted(df_chr["room_id"].unique())

        pivot = df_chr.pivot(index="room_id", columns="day", values="specialty_id")
        pivot = pivot.reindex(index=rooms_order, columns=days_order)

        all_specialties = sorted(df_chr["specialty_id"].unique())
        specialty_to_num = {s: i for i, s in enumerate(all_specialties)}
        pivot_num = pivot.replace(specialty_to_num)

        fig8, ax8 = plt.subplots(figsize=(9, 6))
        cmap = plt.matplotlib.colors.ListedColormap(
            [SPECIALTY_PALETTE[i % len(SPECIALTY_PALETTE)] for i in range(len(all_specialties))]
        )
        im = ax8.imshow(pivot_num.values, cmap=cmap, aspect="auto")

        ax8.set_xticks(range(len(days_order)))
        ax8.set_xticklabels(days_order)
        ax8.set_yticks(range(len(rooms_order)))
        ax8.set_yticklabels(rooms_order)

        # Etiqueta de especialidad dentro de cada celda
        for i in range(len(rooms_order)):
            for j in range(len(days_order)):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax8.text(j, i, val, ha="center", va="center",
                             fontsize=9, color="white", fontweight="bold")

        ax8.set_title("Especialidad asignada por bloque (día x quirófano)")
        fig8.tight_layout()
        fig8.savefig("grid_search_chromosome_grid.png")
        plt.close(fig8)
    else:
        print(f"Aviso: no se encontró {chromosome_csv}, se omite la grilla de cromosoma.")

    print("Gráficos generados exitosamente.")

    print("Gráficos generados exitosamente.")

if __name__ == "__main__":
    main()