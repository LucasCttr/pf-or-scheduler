"""
main.py — Ejemplo de uso del AG con datos similares al Hospital Centenario.

Ejecutar:
    python main.py
"""
import random
from models import OperatingRoom, Specialty, Patient, GAConfig
from genetic_algorithm import GeneticAlgorithm


def make_patients(specialty_id: int, count: int, seed: int = 0) -> list:
    """Genera una lista de pacientes de prueba para una especialidad."""
    rng = random.Random(seed)
    return [
        Patient(
            id=specialty_id * 100 + i,
            specialty_id=specialty_id,
            estimated_duration=rng.randint(45, 240),      # entre 45 min y 4 hs
            clinical_priority=round(rng.uniform(1.0, 10.0), 2),
            required_roles=["cirujano", "anestesista", "instrumentador"],
        )
        for i in range(count)
    ]


def main():
    random.seed(42)

    # ── Quirófanos ────────────────────────────────────────────────────────
    operating_rooms = [
        OperatingRoom(
            id=0, name="Quirófano 1 (Alta)",
            or_type="alta_complejidad",
            availability=[[True, True]] * 5,
        ),
        OperatingRoom(
            id=1, name="Quirófano 2 (Media)",
            or_type="media_complejidad",
            availability=[[True, True]] * 5,
        ),
        OperatingRoom(
            id=2, name="Quirófano 3 (Baja)",
            or_type="baja_complejidad",
            # Solo turno mañana (turno tarde = False)
            availability=[[True, False]] * 5,
        ),
    ]

    # ── Especialidades ────────────────────────────────────────────────────
    # id=0 es el bloque libre (obligatorio, no modificar)
    specialties = [
        Specialty(id=0, name="Libre",            compatible_or_types=[], min_blocks=0, max_blocks=99),
        Specialty(id=1, name="Traumatología",    compatible_or_types=["alta_complejidad", "media_complejidad"],                    min_blocks=3, max_blocks=6),
        Specialty(id=2, name="Cirugía General",  compatible_or_types=["alta_complejidad", "media_complejidad", "baja_complejidad"], min_blocks=4, max_blocks=8),
        Specialty(id=3, name="Neurología",       compatible_or_types=["alta_complejidad"],                                         min_blocks=2, max_blocks=4),
        Specialty(id=4, name="Urología",         compatible_or_types=["media_complejidad", "baja_complejidad"],                    min_blocks=2, max_blocks=5),
        Specialty(id=5, name="Ginecología",      compatible_or_types=["media_complejidad", "baja_complejidad"],                    min_blocks=2, max_blocks=5),
    ]

    # ── Pacientes en lista de espera ──────────────────────────────────────
    patients_by_specialty = {
        1: make_patients(1, count=8,  seed=1),   # Traumatología
        2: make_patients(2, count=14, seed=2),   # Cirugía General
        3: make_patients(3, count=5,  seed=3),   # Neurología
        4: make_patients(4, count=7,  seed=4),   # Urología
        5: make_patients(5, count=6,  seed=5),   # Ginecología
    }

    # ── Configuración del AG ──────────────────────────────────────────────
    config = GAConfig(
        population_size=50,
        max_generations=250,
        convergence_patience=40,
        mutation_rate=0.10,
        crossover_rate=0.85,
        tournament_size=5,
        elite_count=2,
        n_days=5,
        n_shifts=2,
        block_duration_min=480,           # 8 horas por bloque
        penalty_below_min_quota=50.0,
        penalty_above_max_quota=20.0,
    )

    # ── Ejecutar ──────────────────────────────────────────────────────────
    ga = GeneticAlgorithm(
        config=config,
        operating_rooms=operating_rooms,
        specialties=specialties,
        patients_by_specialty=patients_by_specialty,
    )

    print("=" * 70)
    print("  SISTEMA DE PLANIFICACIÓN DE QUIRÓFANOS — Algoritmo Genético")
    print("  Hospital Centenario, Gualeguaychú")
    print("=" * 70 + "\n")

    best = ga.run()
    ga.print_schedule(best)

    # ── Evolución del fitness ─────────────────────────────────────────────
    print("\n  Evolución del fitness (cada 10 generaciones):")
    for i, f in enumerate(ga.history):
        if i % 10 == 0:
            print(f"    Gen {i:4d}: {f:.4f}")

    print(f"\n  Fitness final del mejor individuo: {best.fitness:.4f}")
    print("\n  Agenda lista para ser enviada a revisión de cirujanos.\n")


if __name__ == "__main__":
    main()
