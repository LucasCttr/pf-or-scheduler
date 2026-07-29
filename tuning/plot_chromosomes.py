"""
plot_chromosomes.py
Compara visualmente los cromosomas (asignación de especialidad por bloque)
de varias configuraciones, en una grilla de subplots.
"""

import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "chromosomes_by_config.csv"
MAX_CONFIGS = int(sys.argv[2]) if len(sys.argv) > 2 else 12  # cuántas configs mostrar

DAYS_ORDER = ["lunes", "martes", "miercoles", "jueves", "viernes"]

SPECIALTY_PALETTE = [
    "#2c6e91", "#2e8b57", "#d9534f", "#e0a72e", "#7b5ea7", "#3aa6a0"
]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "figure.facecolor": "white"})


def main():
    df = pd.read_csv(CSV_PATH)

    config_labels = list(dict.fromkeys(df["config_label"]))[:MAX_CONFIGS]
    all_specialties = sorted(df["specialty_id"].unique())
    specialty_to_num = {s: i for i, s in enumerate(all_specialties)}

    n_configs = len(config_labels)
    n_cols = 4
    n_rows = math.ceil(n_configs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axes = axes.flatten() if n_configs > 1 else [axes]

    cmap = plt.matplotlib.colors.ListedColormap(
        [SPECIALTY_PALETTE[i % len(SPECIALTY_PALETTE)] for i in range(len(all_specialties))]
    )

    for idx, label in enumerate(config_labels):
        ax = axes[idx]
        df_c = df[df["config_label"] == label]
        rooms_order = sorted(df_c["room_id"].unique())

        pivot = df_c.pivot(index="room_id", columns="day", values="specialty_id")
        pivot = pivot.reindex(index=rooms_order, columns=DAYS_ORDER)
        pivot_num = pivot.replace(specialty_to_num).astype(float)

        ax.imshow(pivot_num.values, cmap=cmap, aspect="auto", vmin=0, vmax=len(all_specialties) - 1)

        for i in range(len(rooms_order)):
            for j in range(len(DAYS_ORDER)):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, val, ha="center", va="center", fontsize=6, color="white")

        ax.set_xticks(range(len(DAYS_ORDER)))
        ax.set_xticklabels([d[:3] for d in DAYS_ORDER], fontsize=6)
        ax.set_yticks(range(len(rooms_order)))
        ax.set_yticklabels(rooms_order, fontsize=6)
        ax.set_title(label, fontsize=7)

    # Ocultar ejes vacíos si sobran celdas en la grilla
    for idx in range(n_configs, len(axes)):
        axes[idx].axis("off")

    # Leyenda de especialidades compartida
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SPECIALTY_PALETTE[i % len(SPECIALTY_PALETTE)])
        for i in range(len(all_specialties))
    ]
    fig.legend(handles, all_specialties, loc="lower center", ncol=len(all_specialties),
               bbox_to_anchor=(0.5, 0.02), fontsize=9, title="Especialidad")

    fig.suptitle("Comparación de cromosomas entre configuraciones", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig("chromosomes_comparison.png", dpi=150)
    plt.close(fig)

    print("Gráfico de comparación de cromosomas generado: chromosomes_comparison.png")


if __name__ == "__main__":
    main()