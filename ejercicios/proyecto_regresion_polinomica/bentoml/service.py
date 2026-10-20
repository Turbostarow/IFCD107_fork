"""API de inferencia BentoML para el pipeline polinómico entrenado."""

from __future__ import annotations

import math

import bentoml
import numpy as np


@bentoml.service(name="polynomial_regression")
class PolynomialRegression:
    """Carga el modelo registrado por train_model.py y expone predicciones."""

    model_ref = bentoml.models.BentoModel("polynomial_regression:latest")

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.model_ref)
        metadata = self.model_ref.info.metadata
        self.degree = int(metadata["degree"])
        self.training_rows = int(metadata["training_rows"])
        self.holdout_metrics = metadata["holdout_metrics"]

    @bentoml.api
    def predict(self, x: list[float]) -> dict[str, object]:
        """Devuelve una predicción por cada valor x recibido."""
        if not x:
            raise ValueError("x debe contener al menos un valor.")
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("x debe ser una lista de números finitos.")
        predictions = self.model.predict(values.reshape(-1, 1))
        return {
            "predictions": [float(value) for value in predictions],
            "degree": self.degree,
            "training_rows": self.training_rows,
            "holdout_metrics": self.holdout_metrics,
        }
