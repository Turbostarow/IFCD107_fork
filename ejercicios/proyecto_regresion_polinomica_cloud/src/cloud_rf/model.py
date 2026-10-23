"""Entrenamiento, comprobación y serialización de un Random Forest tabular."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(frame: pd.DataFrame, n_estimators: int = 100, random_state: int = 42) -> Pipeline:
    """Construye imputación + codificación + Random Forest en un solo pipeline."""
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("Se necesita al menos una fila y una característica.")
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [name for name in frame.columns if name not in numeric]
    steps = []
    if numeric:
        steps.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        steps.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    transformer = ColumnTransformer(steps, remainder="drop")
    return Pipeline([
        ("preprocess", transformer),
        ("random_forest", RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=1
        )),
    ])


def train_from_csv(csv_path: str | Path, output_path: str | Path,
                   target: str = "y", n_estimators: int = 100,
                   random_state: int = 42) -> dict:
    """Evalúa con un holdout, reajusta con todo el CSV y guarda el pipeline."""
    frame = pd.read_csv(csv_path)
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("El CSV debe tener filas y nombres de columna únicos.")
    if target not in frame.columns:
        raise ValueError(f"No existe la columna objetivo '{target}'.")
    features = [name for name in frame.columns if name != target]
    if not features or len(frame) < 10:
        raise ValueError("Se requieren predictores y al menos 10 filas.")
    y = pd.to_numeric(frame[target], errors="coerce")
    usable = y.notna() & np.isfinite(y.to_numpy(dtype=float))
    X, y = frame.loc[usable, features].reset_index(drop=True), y.loc[usable].reset_index(drop=True)
    if len(y) < 10:
        raise ValueError("Se requieren al menos 10 filas con objetivo numérico.")

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    evaluation_model = build_pipeline(train_x, n_estimators, random_state)
    evaluation_model.fit(train_x, train_y)
    predicted = evaluation_model.predict(test_x)
    metrics = {
        "r2": float(r2_score(test_y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(test_y, predicted))),
        "mae": float(mean_absolute_error(test_y, predicted)),
    }

    # El artefacto de servicio se ajusta con todas las filas tras medir el holdout.
    final_model = build_pipeline(X, n_estimators, random_state)
    final_model.fit(X, y)
    artifact = {"pipeline": final_model, "features": features, "target": target,
                "metrics": metrics, "n_rows": int(len(y))}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return artifact


def load_artifact(path: str | Path) -> dict:
    """Carga el modelo empaquetado. Solo cargar artefactos de origen confiable."""
    artifact = joblib.load(path)
    required = {"pipeline", "features", "target", "metrics", "n_rows"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("El artefacto del modelo no tiene el formato esperado.")
    return artifact
