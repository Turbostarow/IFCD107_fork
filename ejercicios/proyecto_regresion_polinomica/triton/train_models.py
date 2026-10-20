"""Entrena los modelos del proyecto y crea los artefactos para Triton.

El entrenamiento ocurre antes de arrancar Triton. El servidor publica los
artefactos serializados a través del Python backend.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from regression.linear_regressor import LinearRegressor
from regression.polynomial_regressor import PolynomialRegressor, PolynomialSearchEstimator
from regression.random_forest_regressor import RandomForestRegressorModel


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ejemplos"
XY_CSV = DATA_DIR / "datos_ejemplo_xy.csv"
FOREST_CSV = DATA_DIR / "datos_ejemplo_complejo.csv"
MODEL_REPOSITORY = Path(os.getenv("TRITON_MODEL_REPOSITORY", "/model_repository"))
RESULTS_PATH = Path(os.getenv("TRITON_METRICS_PATH", "/results/training_metrics.json"))
RF_INPUTS = (
    "distrito",
    "tipo_tramite",
    "superficie_solicitada_m2",
    "requerimientos_documentacion",
    "mes_presentacion",
    "x",
)


def save_pickle(model, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)


def fit_linear_and_polynomial(test_size: float, random_state: int, degree: int) -> dict:
    data = pd.read_csv(XY_CSV)
    if not {"x", "y"}.issubset(data.columns):
        raise ValueError(f"{XY_CSV} debe tener columnas x e y.")
    X = data[["x"]].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("El conjunto x/y contiene valores no finitos.")

    linear = LinearRegressor(test_size=test_size, random_state=random_state).fit(X, y)
    save_pickle(linear.model, MODEL_REPOSITORY / "linear_regression" / "1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    polynomial_evaluation = PolynomialRegressor(
        X_train[:, 0], X_test[:, 0], y_train, y_test, degree=degree
    ).fit()
    polynomial_model = PolynomialSearchEstimator(degree=degree).fit(X, y)
    save_pickle(polynomial_model, MODEL_REPOSITORY / "polynomial_regression" / "1")

    return {
        "rows": int(len(data)),
        "test_rows": int(len(X_test)),
        "linear_regression": {"metrics": linear.metrics, "features": ["x"]},
        "polynomial_regression": {
            "metrics": polynomial_evaluation.metrics,
            "degree": degree,
            "features": ["x"],
        },
    }


def fit_random_forest(test_size: float, random_state: int, n_estimators: int) -> dict:
    frame = pd.read_csv(FOREST_CSV)
    missing = sorted(set(RF_INPUTS).difference(frame.columns))
    if missing or "y" not in frame.columns:
        raise ValueError(f"CSV de Random Forest sin las columnas necesarias: {missing}")
    trainer = RandomForestRegressorModel(
        n_estimators=n_estimators,
        test_size=test_size,
        random_state=random_state,
    )
    result = trainer.fit(frame, target="y")
    # La clase existente realiza la partición y genera las métricas. Se conserva
    # su pipeline de preprocesamiento y se reajusta con todas las filas válidas
    # antes de publicar el artefacto definitivo.
    valid = frame.loc[pd.to_numeric(frame["y"], errors="coerce").notna()].copy()
    X_all = valid[result.feature_columns].copy()
    for column in X_all.columns:
        if X_all[column].dtype == object:
            converted = pd.to_numeric(X_all[column], errors="coerce")
            non_empty = X_all[column].notna() & (X_all[column].astype(str).str.strip() != "")
            if non_empty.any() and converted[non_empty].notna().all():
                X_all[column] = converted
    y_all = pd.to_numeric(valid["y"], errors="coerce").astype(float)
    result.model.fit(X_all, y_all)
    save_pickle(result.model, MODEL_REPOSITORY / "random_forest_regression" / "1")
    return {
        "rows": int(len(valid)),
        "test_rows": int(round(len(valid) * test_size)),
        "random_forest_regression": {
            "metrics": result.metrics,
            "n_estimators": n_estimators,
            "features": result.feature_columns,
            "input_schema": [
                {"name": name, "datatype": "BYTES" if name in {"distrito", "tipo_tramite"} else "FP32"}
                for name in RF_INPUTS
            ],
        },
    }


def main() -> None:
    test_size = float(os.getenv("TRITON_TRAIN_TEST_SIZE", "0.2"))
    random_state = int(os.getenv("TRITON_RANDOM_STATE", "42"))
    degree = int(os.getenv("TRITON_POLYNOMIAL_DEGREE", "2"))
    n_estimators = int(os.getenv("TRITON_RF_ESTIMATORS", "200"))
    if not 1 <= degree <= 10:
        raise ValueError("TRITON_POLYNOMIAL_DEGREE debe estar entre 1 y 10.")

    MODEL_REPOSITORY.mkdir(parents=True, exist_ok=True)
    metrics = {
        "training_test_size": test_size,
        "random_state": random_state,
        **fit_linear_and_polynomial(test_size, random_state, degree),
        **fit_random_forest(test_size, random_state, n_estimators),
        "note": "Las métricas corresponden al holdout; los artefactos servidos se reajustan con todas las filas válidas.",
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
