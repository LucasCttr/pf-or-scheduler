"""
inspeccionar_prioridades.py

Script standalone para diagnosticar la distribucion de clinical_priority
en patients.csv, y detectar si hay demasiados pacientes empatados en la
prioridad maxima por especialidad (lo cual generaria equifinalidad en el AG).

Uso:
    python inspeccionar_prioridades.py [ruta_a_patients.csv]

Por defecto busca data/patients.csv relativo al directorio actual.
"""
import csv
import sys
from collections import Counter, defaultdict


def cargar_patients(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "patients.csv"

    try:
        rows = cargar_patients(path)
    except FileNotFoundError:
        print(f"No se encontro el archivo: {path}")
        print("Uso: python inspeccionar_prioridades.py <ruta_a_patients.csv>")
        sys.exit(1)

    if not rows:
        print("El archivo esta vacio.")
        sys.exit(1)

    priorities = [float(r["clinical_priority"]) for r in rows]
    by_specialty = defaultdict(list)
    for r in rows:
        by_specialty[r["specialty_id"]].append(float(r["clinical_priority"]))

    print(f"Total de pacientes: {len(rows)}")
    print(f"Prioridad minima: {min(priorities)}")
    print(f"Prioridad maxima: {max(priorities)}")
    print(f"Prioridad promedio: {sum(priorities)/len(priorities):.2f}")
    print(f"Valores unicos de prioridad: {sorted(set(priorities))}")

    print("\n--- Distribucion global de prioridades ---")
    counter = Counter(priorities)
    for valor in sorted(counter.keys(), reverse=True):
        cantidad = counter[valor]
        pct = 100 * cantidad / len(rows)
        barra = "#" * int(pct / 2)
        print(f"  {valor:>6}: {cantidad:>4} pacientes ({pct:5.1f}%) {barra}")

    max_priority = max(priorities)
    empatados_en_max = counter[max_priority]
    pct_empatados = 100 * empatados_en_max / len(rows)

    print(f"\n--- Diagnostico de empate en el maximo ---")
    print(f"Pacientes con prioridad maxima ({max_priority}): {empatados_en_max} "
          f"({pct_empatados:.1f}% del total)")

    if pct_empatados > 20:
        print("  ALERTA: mas del 20% de los pacientes comparten la prioridad "
              "maxima. Esto genera equifinalidad: el AG puede elegir distintos "
              "subconjuntos de estos pacientes y obtener el mismo fitness, sin "
              "que la eleccion sea significativa.")
    else:
        print("  La concentracion en el maximo parece razonable.")

    print("\n--- Distribucion por especialidad (top prioridad) ---")
    for spec_id, vals in sorted(by_specialty.items()):
        spec_max = max(vals)
        spec_counter = Counter(vals)
        empatados = spec_counter[spec_max]
        pct = 100 * empatados / len(vals)
        print(f"  Especialidad {spec_id}: {len(vals)} pacientes, "
              f"{empatados} empatados en el maximo ({spec_max}) -> {pct:.1f}%")

    print("\n--- Valores unicos totales de prioridad ---")
    n_unicos = len(set(priorities))
    print(f"Hay solo {n_unicos} valores distintos de prioridad en todo el dataset.")
    if n_unicos <= 5:
        print("  ALERTA: muy pocos valores discretos. Esto favorece empates "
              "masivos y equifinalidad en el AG. Considera generar prioridades "
              "con mayor granularidad (ej. distribucion continua o escala 1-100 "
              "en vez de 1-10 con pocos valores repetidos).")


if __name__ == "__main__":
    main()