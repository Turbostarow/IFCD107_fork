"""Genera un CSV sintético de 5.000 filas para practicar KNN y validación cruzada."""

import csv
import random
from pathlib import Path


SEED = 1072026
N_ROWS = 5_000
OUTPUT = Path(__file__).resolve().parents[1] / "ejemplos" / "datos_knn_5000.csv"


def main():
    # La semilla fija hace que el mismo script cree siempre el mismo ejemplo.
    rng = random.Random(SEED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # La variable y depende de varias columnas y de un poco de ruido aleatorio.
    # No son datos reales: se han creado solo para ejercitar el algoritmo.
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x1", "x2", "x3", "x4", "y"])
        for _ in range(N_ROWS):
            x1 = rng.uniform(-10, 10)
            x2 = rng.uniform(0, 100)
            x3 = rng.uniform(-5, 5)
            x4 = rng.uniform(10, 40)
            noise = rng.gauss(0, 1.5)
            y = 4 + 1.8 * x1 - 0.12 * x2 + 0.7 * x3**2 + 2.5 * (x4 / 10) ** 0.5 + noise
            writer.writerow([
                f"{x1:.6f}", f"{x2:.6f}", f"{x3:.6f}", f"{x4:.6f}", f"{y:.6f}"
            ])

    print(f"CSV creado: {OUTPUT} ({N_ROWS} filas de datos)")


if __name__ == "__main__":
    main()
