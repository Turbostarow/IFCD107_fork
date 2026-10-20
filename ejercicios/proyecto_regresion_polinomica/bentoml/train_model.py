"""Entrena, evalúa y registra el regresor polinómico en BentoML."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import bentoml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


PROJECT_DATA = Path("/app/data/datos_ejemplo_xy.csv")


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lee las columnas x e y del CSV del proyecto y valida los datos."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not {"x", "y"}.issubset(rows[0]):
        raise ValueError(f"El CSV debe contener columnas x e y: {path}")

    x = np.asarray([float(row["x"]) for row in rows], dtype=float).reshape(-1, 1)
    y = np.asarray([float(row["y"]) for row in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("El CSV contiene valores no finitos.")
    return x, y


def main() -> None:
    degree = int(os.getenv("POLYNOMIAL_DEGREE", "2"))
    random_state = int(os.getenv("POLYNOMIAL_RANDOM_STATE", "42"))
    if not 1 <= degree <= 10:
        raise ValueError("POLYNOMIAL_DEGREE debe estar entre 1 y 10.")

    x, y = load_xy(PROJECT_DATA)
    if len(x) <= degree or np.unique(x).size <= degree:
        raise ValueError("Se necesitan más observaciones y valores x distintos que el grado.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    evaluation_model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()
    )
    evaluation_model.fit(x_train, y_train)
    test_predictions = evaluation_model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, test_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "mae": float(mean_absolute_error(y_test, test_predictions)),
    }

    # Para inferencia se vuelve a ajustar el mismo pipeline con todas las filas
    # disponibles, una vez calculadas las métricas sobre el holdout reservado.
    serving_model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()
    )
    serving_model.fit(x, y)
    model = bentoml.sklearn.save_model(
        "polynomial_regression",
        serving_model,
        metadata={
            "algorithm": "polynomial_regression",
            "degree": degree,
            "features": ["x"],
            "training_rows": int(len(x)),
            "test_rows": int(len(x_test)),
            "random_state": random_state,
            "holdout_metrics": metrics,
        },
    )
    print(json.dumps({"model": str(model.tag), "metadata": model.info.metadata}, indent=2))


if __name__ == "__main__":
    main()
