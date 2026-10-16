"""Funciones de validación y ejecución que comparten los endpoints Flask."""
from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from sklearn.model_selection import train_test_split
import json
import numpy as np
from flask import g, jsonify
from scipy.stats import pearsonr

from regression.polynomial_regressor import PolynomialRegressor,build_polynomial_search
from regression.random_forest_regressor import RandomForestRegressorModel

# Límites del procesamiento aplicados tanto a endpoints JSON como a CSV.
MAX_POINTS = 100_000
REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

def validate_reference(value):
    # La validación se aplica antes de formar cualquier ruta. No se normaliza una
    # referencia inválida porque eso podría hacer colisionar expedientes distintos.
    if not isinstance(value, str) or not REFERENCE_RE.fullmatch(value):
        return None, "reference es obligatorio y debe tener 1-64 caracteres: letras, números, punto, guion o guion bajo."
    return value, None

def parse_json_points(points):
    # Limitar número de filas antes de convertirlas a arrays evita asignaciones
    # grandes no necesarias; después se comprueba tipo y finitud en validate_xy.
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_POINTS:
        raise ValueError(f"data debe contener entre 2 y {MAX_POINTS} pares x/y.")
    try:
        x = np.asarray([row["x"] for row in points], dtype=float)
        y = np.asarray([row["y"] for row in points], dtype=float)
    except (TypeError, ValueError, KeyError, IndexError):
        raise ValueError("Cada elemento de data debe tener valores numéricos x e y.") from None
    return validate_xy(x, y)

def parse_integer(name, default, minimum, maximum,source    ):
    # Los números JSON llegan como int; los campos de un formulario llegan
    # como texto. Convertimos ambos formatos y después aplicamos el límite.
    raw = source.get(name, default)
    try:
        if isinstance(raw, bool) or (isinstance(raw, str) and not raw.strip()):
            raise ValueError
        parsed = int(raw)
        if isinstance(raw, float) and raw != parsed:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}.") from None
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}.")
    return parsed

def parse_csv(text):
    # DictReader permite conservar columnas contextuales del caso de terrazas,
    # aunque el algoritmo solo use x e y. Las filas extra no se descartan del
    # CSV archivado; solo se extraen aquí las dos variables del modelo.
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"x", "y"}.issubset(set(reader.fieldnames)):
        raise ValueError("El encabezado debe contener las columnas x,y.")
    x_values, y_values = [], []
    for line_number, row in enumerate(reader, start=2):
        try:
            x_values.append(float(row["x"]))
            y_values.append(float(row["y"]))
        except (TypeError, ValueError):
            raise ValueError(f"Valores no numéricos en la línea {line_number}.") from None
        if len(x_values) > MAX_POINTS:
            raise ValueError(f"El CSV supera {MAX_POINTS} filas de datos.")
    return validate_xy(np.asarray(x_values), np.asarray(y_values))

def validate_xy(x, y):
    # NaN e infinito contaminan coeficientes y métricas sin producir siempre una
    # excepción clara; se rechazan en el borde HTTP antes de invocar NumPy.
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos dos pares x/y del mismo tamaño.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x e y solo pueden contener números finitos.")
    return x, y

def random_forest_options(source):
    """Valida opciones para el formato JSON o multipart de Postman."""
    def parse_int(key, default):
        raw = source.get(key, default)
        try:
            parsed = int(raw)
            if isinstance(raw, float) and raw != parsed:
                raise ValueError
            return parsed
        except (TypeError, ValueError):
            raise ValueError(f"{key} debe ser un entero válido.") from None
    n_estimators = parse_int("n_estimators", 200)
    random_state = parse_int("random_state", 42)
    raw_depth = source.get("max_depth", None)
    max_depth = None if raw_depth in (None, "", "null") else parse_int("max_depth", None)
    try:
        test_size = float(source.get("test_size", 0.2))
    except (TypeError, ValueError):
        raise ValueError("test_size debe ser un número válido.") from None
    target = source.get("target", "y")
    if not 10 <= n_estimators <= 1000:
        raise ValueError("n_estimators debe estar entre 10 y 1000.")
    if max_depth is not None and not 1 <= max_depth <= 100:
        raise ValueError("max_depth debe ser null o estar entre 1 y 100.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size debe estar entre 0.1 y 0.4.")
    if random_state < 0:
        raise ValueError("random_state debe ser un entero no negativo.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target debe indicar el nombre de una columna.")
    return {"n_estimators": n_estimators, "max_depth": max_depth, "test_size": test_size,
            "random_state": random_state, "target": target}

def run_random_forest(app, output_root, reference, raw_csv, options, source_name):
    """Ejecuta el pipeline y guarda artefactos separados por referencia/run_id."""
    from io import BytesIO
    import pandas as pd

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    folders = {name: base / name for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"random_forest.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        # Vincula cada entrenamiento a la identidad auditada por la capa Flask.
        logger.info("Petición algoritmo=random_forest username=%s client_ip=%s token_fingerprint=%s source=%s",
                    getattr(g, "authenticated_user", None) or "anonymous",
                    getattr(g, "client_ip", None) or "unknown",
                    getattr(g, "token_fingerprint", None) or "none", source_name)
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        input_path.write_bytes(raw_csv)
        try:
            frame = pd.read_csv(BytesIO(raw_csv), encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"CSV no válido: {exc}") from None
        if len(frame) > MAX_POINTS:
            raise ValueError(f"El CSV supera {MAX_POINTS} filas de datos.")
        if frame.columns.duplicated().any():
            raise ValueError("El CSV no puede contener nombres de columna duplicados.")
        model = RandomForestRegressorModel(n_estimators=options["n_estimators"],
                                           max_depth=options["max_depth"],
                                           test_size=options["test_size"],
                                           random_state=options["random_state"])
        result = model.fit(frame, target=options["target"])
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        graph_path = folders["graficas"] / f"{run_id}_random_forest.png"
        model_path = folders["modelos"] / f"{run_id}_random_forest.pkt"
        model.save(model_path, result)
        model.write_predictions(csv_path, result)
        model.write_plot(graph_path, result)
        metadata = {
            "reference": reference, "run_id": run_id,
            "algorithm": "RandomForestRegressor", "target": result.target,
            "feature_columns": result.feature_columns,
            "parameters": {key: options[key] for key in ("n_estimators", "max_depth", "test_size", "random_state")},
            "n_rows_input": int(len(frame)), "n_rows_train": int(len(frame) - len(result.predictions)),
            "n_rows_test": int(len(result.predictions)), "metrics": result.metrics,
            "feature_importances_top_15": result.feature_importances[:15],
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        logger.info("Entrenamiento Random Forest completado: n=%d metrics=%s", len(frame), result.metrics)
        return jsonify(metadata), 200
    except (ValueError, UnicodeDecodeError) as exc:
        logger.exception("No se pudo entrenar el Random Forest: %s", exc)
        return jsonify(error=f"No se pudo entrenar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception as exc:
        logger.exception("Error interno entrenando Random Forest: %s", exc)
        return jsonify(error="Error interno durante el entrenamiento.", reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)

def run_model(app, output_root, reference, degree, x, y, input_kind, raw_csv=None):
    # El sufijo aleatorio evita sobrescrituras si llegan dos peticiones para la
    # misma referencia dentro del mismo segundo. UTC facilita correlación de logs.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    # Un directorio raíz por referencia y carpetas por tipo de artefacto hacen
    # posible auditar/reprocesar cada caso sin mezclar entrada, modelo y salida.
    folders = {name: base / name for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"regression.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        # Cada log de ejecución enlaza el trabajo con usuario, IP y huella de
        # token de la petición autenticada, sin almacenar el Bearer en claro.
        logger.info(
            "Petición username=%s client_ip=%s token_fingerprint=%s",
            getattr(g, "authenticated_user", None) or "anonymous",
            getattr(g, "client_ip", None) or "unknown",
            getattr(g, "token_fingerprint", None) or "none",
        )
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        if raw_csv is None:
            # Para JSON también se guarda una entrada tabular mínima; así cada run
            # tiene una fuente de datos recuperable independientemente del canal.
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows(zip(x.tolist(), y.tolist()))
        else:
            # Guardar bytes originales preserva columnas extra, orden, formato y
            # encabezados exactamente como los envió el cliente.
            input_path.write_bytes(raw_csv)
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=42
        )
        model = PolynomialRegressor(X_train, X_test, y_train, y_test,degree=degree)
        result = model.fit()
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        graph_path = folders["graficas"] / f"{run_id}_ajuste.png"
        model_path = folders["modelos"] / f"{run_id}_modelo.pkt"
        # Persistir el modelo inmediatamente tras entrenar permite volver a usarlo
        # aunque la API se reinicie; el endpoint devuelve la ruta para trazabilidad.
        model.save(model_path)
        model.write_predictions(csv_path, result)
        model.write_plot(graph_path, X_test, y_test, result)
        metadata = {
            "reference": reference,
            "run_id": run_id,
            "degree": degree,
            "n_observations": int(len(X_test)),
            "coefficients_descending": result.coefficients.tolist(),
            "metrics": result.metrics,
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        logger.info("Ajuste completado input=%s degree=%d n=%d metrics=%s", input_kind, degree, len(x), result.metrics)
        return jsonify(metadata), 200
    except (ValueError, np.linalg.LinAlgError) as exc:
        # Errores de datos/ajuste se registran con el run_id y reciben 422; el log
        # conserva traceback para diagnóstico sin mezclarlo con la respuesta normal.
        logger.exception("No se pudo ajustar el modelo: %s", exc)
        return jsonify(error=f"No se pudo ajustar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception as exc:  # No devolver rutas, stack traces ni secretos al cliente.
        logger.exception("Error interno en el ajuste: %s", exc)
        return jsonify(error="Error interno durante el ajuste.", reference=reference, run_id=run_id), 500
    finally:
        # Cerrar el handler por solicitud evita descriptores abiertos durante un
        # servicio prolongado y permite leer/archivar el log al terminar la petición.
        handler.close()
        logger.removeHandler(handler)


def run_polynomial_search(output_root, reference, x, y, options, weights=None, raw_csv=None):
    """Separa los datos, busca el modelo y guarda sus resultados para la API."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import pickle
    import matplotlib.pyplot as plt

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    folders = {name: base / name for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"polynomial_search.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        source_kind = "csv" if raw_csv is not None else "json"
        logger.info("RandomizedSearchCV user=%s client_ip=%s source=%s",
                    getattr(g, "authenticated_user", None) or "anonymous",
                    getattr(g, "client_ip", None) or "unknown", source_kind)
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        if raw_csv is not None:
            input_path.write_bytes(raw_csv)
        else:
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows(zip(x.tolist(), y.tolist()))

        # Apartamos un 20% para medir el rendimiento final. RandomizedSearchCV
        # solo ve el 80% restante, así la prueba no influye en la selección.
        indices = np.arange(len(x))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=options["random_state"]
        )
        # El pipeline prueba combinaciones usando validación cruzada dentro del
        # conjunto de entrenamiento y vuelve a ajustar la ganadora (refit=True).
        search = build_polynomial_search(
            n_iter=options["n_iter"], cv=options["cv"], random_state=options["random_state"]
        )
        X_train = x[train_idx].reshape(-1, 1)
        X_test = x[test_idx].reshape(-1, 1)
        fit_kwargs = {}
        if weights is not None:
            # Pipeline necesita el prefijo del paso para enviar los pesos al modelo.
            fit_kwargs["polynomial__sample_weight"] = weights[train_idx]
        search.fit(X_train, y[train_idx], **fit_kwargs)
        # La evaluación se calcula con datos que no participaron en la búsqueda.
        predicted = search.best_estimator_.predict(X_test)
        observed = y[test_idx]
        residuals = observed - predicted
        pearson, pearson_pvalue = pearsonr(observed, predicted)
        metrics = {
            "r2": float(r2_score(observed, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "mae": float(mean_absolute_error(observed, predicted)),
            "pearson": float(pearson),
            "bias": float(np.mean(observed - predicted)),
        }
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        np.savetxt(csv_path, np.column_stack((x[test_idx], observed, predicted, residuals)),
                   delimiter=",", header="x,y_observado,y_predicho,residuo", comments="", fmt="%.10g")
        graph_path = folders["graficas"] / f"{run_id}_randomized_search.png"
        order = np.argsort(x[test_idx])
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(x[test_idx], observed, label="Observaciones", color="#1f77b4", alpha=0.85)
        ax.plot(x[test_idx][order], predicted[order], label="Mejor ajuste", color="#d62728")
        ax.set(title="Regresión polinómica optimizada", xlabel="x", ylabel="y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(graph_path, dpi=160)
        plt.close(fig)
        model_path = folders["modelos"] / f"{run_id}_randomized_search.pkt"
        with model_path.open("wb") as stream:
            pickle.dump(search.best_estimator_, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "reference": reference,
            "run_id": run_id,
            "algorithm": "PolynomialRegressionRandomizedSearchCV",
            "best_parameters": search.best_params_,
            "best_cv_rmse": float(np.sqrt(-search.best_score_)),
            "metrics": metrics,
            "min_observed":np.min(observed),
            "max_observed":np.max(observed),
            "mean_observed": np.mean(observed),
            "n_rows_input": int(len(x)),
            "n_rows_train": int(len(train_idx)),
            "n_rows_test": int(len(test_idx)),
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        best_estimator_params = search.best_estimator_.named_steps["polynomial"].get_params(deep=False)
        logger.info(
            "RandomizedSearchCV completado param_distributions=%s best_estimator_params=%s best=%s metrics=%s",
            json.dumps(search.param_distributions, sort_keys=True, default=str),
            json.dumps(best_estimator_params, sort_keys=True, default=str),
            search.best_params_, metrics,
        )
        return jsonify(metadata), 200
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.exception("No se pudo entrenar RandomizedSearchCV: %s", exc)
        return jsonify(error=f"No se pudo entrenar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception:
        logger.exception("Error interno en RandomizedSearchCV")
        return jsonify(error="Error interno durante la búsqueda del modelo.", reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)
