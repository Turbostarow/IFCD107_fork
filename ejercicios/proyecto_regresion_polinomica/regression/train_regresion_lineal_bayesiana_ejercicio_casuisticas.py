# PRÁCTICA: Regresión Lineal Bayesiana con las seis casuísticas del proyecto
# Entrenador ejecutable de referencia: train_regresion_lineal_bayesiana_casuisticas.py
# Solo pseudocódigo: completa el código Python en los espacios de práctica.
#
# OBJETIVO DE LA PRÁCTICA
# - Completar la plantilla en Python; todas las líneas actuales son comentarios de pseudocódigo.
# - No borrar los límites de validación, el aislamiento del test ni el registro de artefactos.
# - Mantener el entrenador funcional train_<algoritmo>_casuisticas.py como referencia ejecutable.
# 
# FLUJO GENERAL QUE DEBE IMPLEMENTAR EL ALUMNO
"""Entrenamiento común para las casuísticas comparativas de regresión.

Cada algoritmo usa el mismo flujo: separación train/test, preprocesado dentro
de un Pipeline, GridSearchCV y RandomizedSearchCV, comparación por RMSE y
evaluación final sobre el mismo conjunto de prueba.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,  StandardScaler

from regression.eda_preprocesado import EdaPreprocesado
from sklearn.linear_model import BayesianRidge
from regression.utils import r2_ajustado,graficar_cv_results
# FLUJO GENERAL QUE DEBE IMPLEMENTAR EL ALUMNO
# 1. Importar solo dependencias necesarias y definir rutas desde la raíz del proyecto, no desde el directorio actual.

TARGET_COLUMN = "y"
ID_COLUMN = "expediente_id"
TEST_SIZE = 0.2
RANDOM_STATE = 42
ALGORITHM ="bayesian_ridge"
ALGORITHM_LABELS = {
     "bayesian_ridge": "Regresión Lineal Bayesiana",}

def _native_scalar(value):
    """Convierte escalares NumPy a tipos que Flask y JSON pueden serializar."""
    return value.item() if isinstance(value, np.generic) else value


def _randomized_candidate_count(distributions: list[dict]) -> int | None:
    """Cuenta candidatos cuando todas las opciones son listas finitas."""
    total = 0
    for parameter_set in distributions:
        combinations = 1
        for values in parameter_set.values():
            if hasattr(values, "rvs"):
                return None
            combinations *= len(values)
        total += combinations
    return total


def _create_pipeline( frame: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Prepara categorías/números en un Pipeline para evitar fuga entre pliegues."""
    if TARGET_COLUMN not in frame:
        raise ValueError(f"El CSV debe incluir la columna objetivo '{TARGET_COLUMN}'.")
    columns = [column for column in frame.columns if column not in {TARGET_COLUMN, ID_COLUMN}]

    if not columns:
        raise ValueError("El CSV debe contener al menos una característica predictora además de y.")
    X = frame[columns].copy()
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    for column in numeric_columns:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)
    columns = [column for column in columns if X[column].notna().any()]
    X = X[columns]
    numeric_columns = [column for column in numeric_columns if column in columns]
    categorical_columns = [column for column in columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No hay características utilizables después de limpiar valores vacíos.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop",
                                     sparse_threshold=0.0)
    
    pipeline = Pipeline([("preprocess", preprocessor), ("model", BayesianRidge())])
    return pipeline, columns



def train_algorithm_frame_lineal_multiple_regression(
    frame: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    cv: int = 5,
    n_iter: int = 20,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Ejecuta búsquedas, genera informe y persiste modelo/métricas para un CSV."""
    if cv < 2 or n_iter < 1:
        raise ValueError("cv debe ser >= 2 y n_iter debe ser >= 1.")
    frame = frame.copy()
    if len(frame) < 10:
        raise ValueError("Se necesitan al menos 10 filas para una separación train/test fiable.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("El CSV no puede contener nombres de columna duplicados.")
    # 2. Cargar cada CSV de proyecto_eda_preprocesado_latex/datos con la codificación adecuada; comprobar que existen archivo, encabezados y filas.
    processor = EdaPreprocesado()
    # 4. Limpiar nombres de columnas duplicados; convertir y a numérico; tratar NaN e infinito y exigir filas suficientes.
    prepared, mappings = processor.transform(frame)
    # 3. Confirmar que y es el objetivo; separar expediente_id y cualquier columna identificadora o que revele la respuesta.
    if TARGET_COLUMN not in prepared:
        raise ValueError(f"El CSV debe incluir la columna objetivo '{TARGET_COLUMN}'.")
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce")
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(prepared) < 10:
        raise ValueError("Se necesitan al menos 10 filas con y numérico.")
    output_dir = Path(output_dir)
    directories = {name: output_dir / name for name in
                   ("informes", "csv_salida", "modelos", "busquedas")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    # 5. Separar train y test una sola vez, con TEST_SIZE=0.20 y random_state fijo; no consultar test para elegir características ni parámetros.
    indices = np.arange(len(prepared))
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=random_state,
    )
    
    X = prepared.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = prepared[TARGET_COLUMN].astype(float)
    # 6. Construir ColumnTransformer/Pipeline: imputar dentro del pliegue; escalar numéricas cuando 
    #  el algoritmo use magnitudes o distancias; codificar categóricas sin fuga.
    pipeline, feature_columns = _create_pipeline(prepared)
    # 8. Elegir CV con pliegues válidos para el tamaño de train; usar los mismos folds,
        #scorer RMSE negativo y datos de entrenamiento para ambos buscadores.
    # max_iter=300: máximo de iteraciones del algoritmo.
    # El proceso puede detenerse antes si alcanza la convergencia.

    # tol=1e-3: tolerancia usada para decidir si el modelo ha convergido.
    # Un valor menor exige un ajuste más preciso y puede requerir más iteraciones.

    # alpha_1=1e-6 y alpha_2=1e-6: parámetros de la distribución Gamma
    # usada como prior para alpha. alpha_1 es la forma y alpha_2 es la tasa.

    # lambda_1=1e-6 y lambda_2=1e-6: parámetros de la distribución Gamma
    # usada como prior para lambda. lambda_1 es la forma y lambda_2 es la tasa.

    # alpha_init=None: valor inicial de la precisión del ruido.
    # Si se deja en None, se inicializa como 1 / Var(y).

    # lambda_init=None: valor inicial de la precisión de los coeficientes.
    # Si se deja en None, empieza en 1.

    # compute_score=False: si es True, calcula la log-verosimilitud marginal
    # en cada iteración. Permite observar la evolución del ajuste, pero añade coste.

    # fit_intercept=True: indica si el modelo estima un término independiente.
    # El intercepto no se trata como parámetro probabilístico y no tiene varianza
    # asociada. Si es False, se espera que los datos estén centrados.

    # copy_X=True: copia X antes de procesarla.
    # Si es False, el algoritmo podría modificar la matriz original.

    # verbose=False: muestra información del progreso durante el ajuste.
    grid, distributions = ([
        {
            "model__alpha_1": [1e-6, 1e-4],
            "model__alpha_2": [1e-6, 1e-4],
            "model__lambda_1": [1e-6, 1e-4],
            "model__lambda_2": [1e-6, 1e-4],
            "model__max_iter": [300, 1000],
            "model__tol": [1e-3, 1e-4],
            "model__alpha_init": [None, 0.1, 1.0, 10.0],
            "model__lambda_init": [None, 0.1, 1.0, 10.0],
            "model__compute_score": [False, True],
            "model__fit_intercept": [True, False],
            "model__copy_X": [True, False],
        }],

        # La búsqueda aleatoria puede combinar los distintos parámetros.
        [{
        "model__max_iter": [300, 500, 1000],
        "model__tol": loguniform(1e-5, 1e-2),

        # Priors Gamma de alpha (precisión del ruido).
        "model__alpha_1": loguniform(1e-8, 1e-3),
        "model__alpha_2": loguniform(1e-8, 1e-3),

        # Priors Gamma de lambda (precisión de los coeficientes).
        "model__lambda_1": loguniform(1e-8, 1e-3),
        "model__lambda_2": loguniform(1e-8, 1e-3),

        # Valores iniciales; None permite usar la inicialización automática.
        "model__alpha_init": [None, 0.01, 0.1, 1.0, 10.0],
        "model__lambda_init": [None, 0.01, 0.1, 1.0, 10.0],

        "model__compute_score": [False, True],
        "model__fit_intercept": [True, False],
        "model__copy_X": [True, False],
          }])
    cv_folds = min(cv, len(train_indices))
    finite_candidates = _randomized_candidate_count(distributions)
    effective_n_iter = min(n_iter, finite_candidates) if finite_candidates is not None else n_iter
    shared = {"estimator": pipeline, "scoring": "neg_root_mean_squared_error",
              "cv": cv_folds, "n_jobs": -1, "refit": True, "error_score": "raise"}
    # 7. Definir una cuadrícula pequeña y justificable para GridSearchCV y distribuciones para RandomizedSearchCV, 
    # siempre con parámetros del paso correcto del Pipeline.
    grid_search = GridSearchCV(param_grid=grid, **shared)
    random_search = RandomizedSearchCV(
        param_distributions=distributions, n_iter=effective_n_iter,
        random_state=random_state, **shared,
    )



    



    # 9. Fijar semilla, candidatos de RandomizedSearchCV y coste/parallelism; evitar combinaciones inválidas para el estimador(fit).
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    grid_search.fit(X_train, y_train)
    random_search.fit(X_train, y_train)
    searches = {"GridSearchCV": grid_search, "RandomizedSearchCV": random_search}
    # 10. Ajustar los dos buscadores solo sobre train; recopilar best_params_, best_score_, cv_results_ y nombres de candidatos.
    winner = max(searches, key=lambda name: searches[name].best_score_)
    winner_search = searches[winner]
    model_comparisons = {}
    # 11. Convertir el score negativo a RMSE positivo para informar; recordar que el mejor score en sklearn es el valor más alto (menos negativo).
    # "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
    for name, search in searches.items():
        # 12. Elegir el buscador con mejor RMSE medio de CV; no escoger por métricas del conjunto de prueba.
        predicted = search.best_estimator_.predict(X_test)
        model_comparisons[name] = {
            "y_true": y_test.to_numpy(), "y_predicted": predicted,
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "r2": float(r2_score(y_test, predicted)),
            "r2 ajustado": float(r2_ajustado(y_test,winner_prediction,X.shape[1]))
        }
    
    # 13. Evaluar ambos best_estimator_ sobre el mismo test reservado y calcular R², RMSE, MAE y sesgo; 
    # añadir métricas específicas si el método las necesita.
    winner_prediction = model_comparisons[winner]["y_predicted"]
    residuals = y_test.to_numpy() - winner_prediction
    metrics = {
        "r2": float(r2_score(y_test, winner_prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, winner_prediction))),
        "mae": float(mean_absolute_error(y_test, winner_prediction)),
        "bias": float(np.mean(residuals)),
        "r2 ajustado": float(r2_ajustado(y_test,winner_prediction,X.shape[1]))
    }




   



    result_dir = directories["csv_salida"]
    predictions_path = result_dir / f"{run_id}_predicciones.csv"
    predictions_frame = pd.DataFrame({"y_real": y_test.to_numpy(),
                                      "y_predicho": winner_prediction,
                                      "residuo": residuals})
    if ID_COLUMN in prepared:
        predictions_frame.insert(0, ID_COLUMN, prepared[ID_COLUMN].iloc[test_indices].to_numpy())
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    model_path = directories["modelos"] / f"{run_id}_{ALGORITHM}.joblib"
    # 14. Generar para ambos métodos una gráfica y_real frente a y_predicho con línea ideal y mismo conjunto de ejes.
    from joblib import dump
    dump(winner_search.best_estimator_, model_path)
    grid_results_path = directories["busquedas"] / f"{run_id}_gridsearch.csv"
    random_results_path = directories["busquedas"] / f"{run_id}_randomizedsearch.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(grid_results_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(random_search.cv_results_).to_csv(random_results_path, index=False, encoding="utf-8-sig")

    #Graficas  extras
    graficar_cv_results(random_search,directories["csv_salida"])
    # 15. Guardar PDF de EDA/matriz de correlación/gráficas, predicciones CSV, modelo ganador, tablas CV de ambos métodos, JSON de metadatos y log.
    correlation_data = prepared.select_dtypes(include=["number", "bool"])
    correlation_matrix = correlation_data.corr(method="pearson")
    report_path = directories["informes"] / f"{run_id}_informe.pdf"
    EdaPreprocesado().generate_report(
        frame, prepared, mappings, report_path, ALGORITHM_LABELS[ALGORITHM],
        correlation_matrix, model_comparisons=model_comparisons,
    )
    # 16. Escribir en metadatos dataset, algoritmo, variables, partición, CV, semilla, hiperparámetros de ambos ganadores, ganador final
    # , métricas y rutas relativas.

    parameters = {key.removeprefix("model__").removeprefix("polynomial__"): _native_scalar(value)
                  for key, value in winner_search.best_params_.items()}
    search_comparison = {
        name: {"best_parameters": {
                   key.removeprefix("model__").removeprefix("polynomial__"): _native_scalar(value)
                   for key, value in search.best_params_.items()},
               "best_cv_rmse": float(-search.best_score_),
               "candidates_evaluated": int(len(search.cv_results_["params"]))}
        for name, search in searches.items()
    }
    # 18. Añadir un resumen CSV por lotes con una fila por dataset, ganador, parámetros y métricas de test.
    summary = {
        "status": "completed", "algorithm": ALGORITHM,
        "algorithm_label": ALGORITHM_LABELS[ALGORITHM], "winner": winner,
        "best_parameters": parameters,
        "best_cv_rmse": float(-winner_search.best_score_),
        "search_comparison": search_comparison,
        "cv_folds": int(cv_folds), "n_iter_requested": int(n_iter),
        "n_iter_used": int(effective_n_iter), "n_rows": int(len(prepared)),
        "n_features": int(len(feature_columns)), "feature_columns": feature_columns,
        "metrics": metrics,
        "comparison_metrics": {name: {"rmse": values["rmse"], "r2": values["r2"]}
                               for name, values in model_comparisons.items()},
        "artifacts": {
            "report_pdf": str(report_path), "predictions_csv": str(predictions_path),
            "model": str(model_path), "gridsearch_results_csv": str(grid_results_path),
            "randomizedsearch_results_csv": str(random_results_path),
        },
    }

    metadata_path = directories["busquedas"] / f"{run_id}_resumen.json"
    summary["run_id"] = run_id
    summary["artifacts"]["summary_json"] = str(metadata_path)
    # 18. Añadir un resumen CSV por lotes con una fila por dataset, ganador, parámetros y métricas de test.
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                              encoding="utf-8")
    return summary

#
# VALIDACIÓN Y PREVENCIÓN DE FUGAS
# - El imputador, escalador, codificador y expansor de características se ajustan dentro del Pipeline de cada fold.
# - El conjunto test permanece intacto hasta la evaluación final; nunca llamar fit, fit_transform ni search sobre test.
# - Validar rangos y tipos de hiperparámetros antes de iniciar búsquedas costosas.
# - Reducir n_jobs internos cuando el buscador paralelice para no multiplicar hilos de bibliotecas externas.
# - Comprobar que el modelo admite la matriz densa/dispersa producida por ColumnTransformer.
# - Guardar tipos JSON serializables y rutas relativas; no serializar datos de test dentro del modelo.
#
# CONFIGURACIÓN ESPECÍFICA DEL MÉTODO
# ALGORITMO: Regresión Lineal Bayesiana
# - Identificador API esperado: bayesian_ridge.
# - Espacio de hiperparámetros orientativo: alpha_1, alpha_2, lambda_1, lambda_2, fit_intercept y compute_score; mantener hiperparámetros positivos.
# - Ajustar BayesianRidge y distinguir parámetros de hiperparámetros del estimador.
# - Buscar alpha_1, alpha_2, lambda_1 y lambda_2 con valores positivos adecuados; incluir fit_intercept.
# - Solicitar predicción con desviación estándar cuando la implementación lo permita.
# - Guardar predicción media y desviación; explicar que la incertidumbre depende de los supuestos probabilísticos del modelo.
# - No interpretar la desviación estándar como garantía de cobertura sin calibración.
#
# ENTREGABLES MÍNIMOS
# - Entrenador implementado que procese una casuística elegida por parámetro o las seis en un lote.
# - Resultados comparables de GridSearchCV y RandomizedSearchCV con parámetros ganadores.
# - Predicciones de los dos estimadores finalistas sobre el mismo test y dos gráficas real/predicho.
# - Informe y resumen que permitan explicar qué método ganó y por qué, sin afirmar causalidad.
#
# PREGUNTAS PARA EL ALUMNO
# - ¿Qué hiperparámetro tuvo mayor efecto en la búsqueda y cuál fue su evidencia?
# - ¿Coinciden el menor RMSE de CV y el mejor resultado de test? ¿Por qué el test no debe guiar la selección?
# - ¿Qué patrón de error aparece en la gráfica y qué limitaciones tiene la métrica elegida?
# - ¿Qué cambiaría si el dataset fuera temporal, tuviera miles de columnas o muchas categorías?
