"""Genera datos sintéticos reproducibles, sin información personal."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/random_forest_demo.csv")
    parser.add_argument("--rows", type=int, default=500)
    args = parser.parse_args()
    rng = np.random.default_rng(107)
    size = args.rows
    surface = rng.uniform(40, 180, size)
    rooms = rng.integers(1, 6, size)
    age = rng.integers(0, 80, size)
    zone = rng.choice(["centro", "residencial", "periferia"], size=size)
    zone_bonus = pd.Series(zone).map({"centro": 45_000, "residencial": 22_000,
                                      "periferia": 5_000}).to_numpy()
    target = 900 * surface + 8_000 * rooms - 350 * age + zone_bonus
    target += rng.normal(0, 8_000, size)
    frame = pd.DataFrame({"superficie": surface.round(2), "habitaciones": rooms,
                          "antiguedad": age, "zona": zone, "y": target.round(2)})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"CSV sintético guardado en {destination} ({len(frame)} filas)")


if __name__ == "__main__":
    main()
