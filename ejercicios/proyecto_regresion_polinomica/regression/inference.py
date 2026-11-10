"""Inferencia con modelos joblib de confianza; no vuelve a entrenarlos."""
from __future__ import annotations

import csv
import io
import joblib

import numpy as np
import pandas as pd

from regression.polynomial_regressor import PolynomialRegressor

MAX_ROWS = 100_000


def predict_csv(raw_model: bytes, raw_csv: bytes, separator=",", decimal=".") -> dict:
    """Carga un .joblib de confianza y devuelve una predicción por fila de CSV.

    Joblib puede ejecutar código durante la carga. La autenticación no convierte
    un archivo de origen desconocido en un archivo seguro.
    """
    if separator not in {",", ";", "\t"} or decimal not in {".", ","}:
        raise ValueError("separator debe ser coma, punto y coma o tabulador; decimal debe ser punto o coma.")
    if separator == decimal:
        raise ValueError("separator y decimal deben ser distintos.")
    try:
        text = raw_csv.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text), delimiter=separator, strict=True)
        header = next(reader)
        if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise ValueError("El CSV necesita cabeceras no vacías y sin duplicados.")
        for row in reader:
            if row and len(row) != len(header):
                raise ValueError("Todas las filas deben tener tantas columnas como la cabecera.")
        frame = pd.read_csv(io.StringIO(text), sep=separator, decimal=decimal, nrows=MAX_ROWS+1)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration, csv.Error) as exc:
        raise ValueError("No se pudo leer el CSV: usa UTF-8, cabecera y un separador válido.") from exc
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError(f"El CSV debe tener entre 1 y {MAX_ROWS} filas de datos.")
    try:
        # Solo modelos de confianza: ni la extensión ni comprobar el tipo después
        # de joblib.load impiden que un archivo malicioso ejecute código.
        model = joblib.load(io.BytesIO(raw_model))
    except Exception as exc:
        raise ValueError("No se pudo cargar el .joblib. Comprueba el archivo y las versiones de sus dependencias.") from exc
    if not callable(getattr(model, "predict", None)):
        raise ValueError("El .joblib no contiene un modelo con método predict.")

    if isinstance(model, PolynomialRegressor):
        expected = ["x"]
    else:
        names = getattr(model, "feature_names_in_", None)
        expected = list(names) if names is not None else list(frame.columns)
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas por el modelo: {', '.join(map(str, missing))}.")
    features = frame.loc[:, expected]
    if not hasattr(model, "feature_names_in_") and not isinstance(model, PolynomialRegressor):
        count = getattr(model, "n_features_in_", None)
        if count is not None and features.shape[1] != count:
            raise ValueError(f"El modelo necesita {count} features en el mismo orden que durante el entrenamiento.")
    try:
        if isinstance(model, PolynomialRegressor):
            predicted = model.predict(pd.to_numeric(features["x"], errors="raise").to_numpy())
        else:
            # Los modelos entrenados con arrays no conservan nombres. En ellos,
            # el contrato es el número y el orden original de las columnas.
            inputs = features if hasattr(model, "feature_names_in_") else features.to_numpy()
            predicted = model.predict(inputs)
        values = np.asarray(predicted, dtype=float)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or len(values) != len(frame) or not np.isfinite(values).all():
            raise ValueError("Se requiere una predicción numérica finita por fila (regresión de un objetivo).")
    except Exception as exc:
        raise ValueError(f"No se pudo predecir con esas features: {exc}") from exc
    return {
        "n_rows": len(frame),
        "feature_columns": [str(name) for name in expected],
        "ignored_columns": [str(name) for name in frame.columns if name not in expected],
        "predictions": [{"row": i+1, "prediction": float(value)} for i, value in enumerate(values)],
    }
