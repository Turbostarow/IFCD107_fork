"""Entrenamiento y evaluación de un Random Forest para regresión tabular."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class RandomForestResult:
    model: Pipeline
    metrics: dict
    predictions: pd.DataFrame
    feature_importances: list[dict]
    feature_columns: list[str]
    target: str


class RandomForestRegressorModel:
    """Preprocesa variables mixtas, entrena el bosque y produce métricas."""

    def __init__(self, n_estimators=200, max_depth=None, test_size=0.2, random_state=42):
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or not 10 <= n_estimators <= 1000:
            raise ValueError("n_estimators debe ser un entero entre 10 y 1000.")
        if max_depth is not None and (isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= 100):
            raise ValueError("max_depth debe ser null o un entero entre 1 y 100.")
        if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not 0.1 <= test_size <= 0.4:
            raise ValueError("test_size debe estar entre 0.1 y 0.4.")
        if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state debe ser un entero no negativo.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = float(test_size)
        self.random_state = random_state

    def fit(self, frame: pd.DataFrame, target: str = "y") -> RandomForestResult:
        if target not in frame.columns:
            raise ValueError(f"No existe la columna objetivo '{target}'.")
        # El ID solo enlaza resultados con expedientes; excluirlo evita memorizar claves.
        id_column = "expediente_id" if "expediente_id" in frame.columns else None
        feature_columns = [col for col in frame.columns if col not in {target, id_column}]
        if not feature_columns:
            raise ValueError("El CSV debe aportar al menos una columna predictora además del objetivo.")
        data = frame.copy()
        data[target] = pd.to_numeric(data[target], errors="coerce")
        valid_target = data[target].notna() & np.isfinite(data[target].to_numpy(dtype=float))
        data = data.loc[valid_target].reset_index(drop=True)
        if len(data) < 10:
            raise ValueError("Se requieren al menos 10 filas con objetivo numérico para entrenar y evaluar.")
        X = data[feature_columns].copy()
        y = data[target].astype(float)
        # Las columnas textuales enteramente numéricas pasan a continuas; las demás son categorías.
        for column in feature_columns:
            if X[column].dtype == object:
                converted = pd.to_numeric(X[column], errors="coerce")
                non_empty = X[column].notna() & (X[column].astype(str).str.strip() != "")
                if non_empty.any() and converted[non_empty].notna().all():
                    X[column] = converted
        numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_columns = [col for col in feature_columns if col not in numeric_columns]
        transformers = []
        if numeric_columns:
            transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns))
        if categorical_columns:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_columns))
        pipeline = Pipeline([
            ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("random_forest", RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=self.random_state, n_jobs=-1,
            )),
        ])
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=self.test_size, random_state=self.random_state)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = pipeline.predict(X.iloc[test_idx])
        observed = y.iloc[test_idx].to_numpy(dtype=float)
        residuals = observed - predicted
        metrics = {
            "r2": float(r2_score(observed, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "mae": float(mean_absolute_error(observed, predicted)),
        }
        predictions = pd.DataFrame({"y_observado": observed, "y_predicho": predicted, "residuo": residuals})
        if id_column:
            predictions.insert(0, id_column, data[id_column].iloc[test_idx].to_numpy())
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        scores = pipeline.named_steps["random_forest"].feature_importances_
        importance = [{"feature": str(name), "importance": float(score)} for name, score in
                      sorted(zip(names, scores), key=lambda item: item[1], reverse=True)]
        return RandomForestResult(pipeline, metrics, predictions, importance, feature_columns, target)

    @staticmethod
    def save(path: Path, result: RandomForestResult):
        # Cargar pickle ejecuta código: aceptar únicamente ficheros de origen confiable.
        with Path(path).open("wb") as stream:
            pickle.dump(result.model, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def write_predictions(path: Path, result: RandomForestResult):
        result.predictions.to_csv(path, index=False, encoding="utf-8")

    @staticmethod
    def write_plot(path: Path, result: RandomForestResult):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        observed = result.predictions["y_observado"].to_numpy()
        predicted = result.predictions["y_predicho"].to_numpy()
        limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
        axes[0].scatter(observed, predicted, alpha=0.7, color="#1f77b4")
        axes[0].plot(limits, limits, "--", color="#d62728", label="Predicción ideal")
        axes[0].set(title="Observado frente a predicho", xlabel="Valor observado", ylabel="Valor predicho")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)
        top = result.feature_importances[:15][::-1]
        axes[1].barh([row["feature"] for row in top], [row["importance"] for row in top], color="#2ca02c")
        axes[1].set(title="Importancia de variables (15 principales)", xlabel="Importancia")
        axes[1].grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
