"""Clase de regresión polinómica y generación de artefactos."""
from dataclasses import dataclass
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FitResult:
    coefficients: np.ndarray
    x: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray
    metrics: dict


class PolynomialRegressor:
    """Ajusta un polinomio por mínimos cuadrados usando numpy."""

    def __init__(self, degree: int = 2):
        if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= 10:
            raise ValueError("degree debe estar entre 1 y 10.")
        self.degree = degree
        self.coefficients_ = None

    def fit(self, x, y) -> FitResult:
        # Convertir a arrays numéricos una sola vez simplifica validaciones y
        # garantiza que el ajuste opere sobre datos homogéneos.
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x e y deben ser vectores de igual longitud.")
        # Para grado d se requieren al menos d+1 observaciones y d+1 valores x
        # distintos; de lo contrario el problema está subdeterminado.
        if len(x) <= self.degree:
            raise ValueError("Se necesitan más observaciones que el grado del polinomio.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Los datos deben ser finitos.")
        if np.unique(x).size <= self.degree:
            raise ValueError("Se necesitan más valores x distintos que el grado del polinomio.")
        # Polynomial.fit escala internamente el dominio y mejora el condicionamiento
        # frente a construir directamente potencias grandes de x. convert() pasa
        # después al dominio estándar para guardar coeficientes legibles/reutilizables.
        fitted = np.polynomial.Polynomial.fit(x, y, deg=self.degree)
        polynomial = fitted.convert()
        coefficients = polynomial.coef[::-1]
        self.coefficients_ = coefficients.copy()
        predicted = polynomial(x)
        residuals = y - predicted
        # R² no está definido si y es constante. En el caso perfecto se informa 1;
        # si no lo es, se informa 0 como valor neutro y se conserva RMSE/MAE.
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 if ss_tot == 0 and ss_res == 0 else (0.0 if ss_tot == 0 else 1 - ss_res / ss_tot)
        metrics = {
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
        }
        return FitResult(coefficients, x.copy(), np.asarray(predicted), residuals, metrics)

    def predict(self, x):
        """Predice con un modelo ya entrenado; x puede ser escalar o vector."""
        if self.coefficients_ is None:
            raise ValueError("El modelo aún no está entrenado.")
        values = np.asarray(x, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("x debe contener valores finitos.")
        # Los coeficientes se guardan en orden descendente, que es el convenio
        # esperado por polyval: a_n*x^n + ... + a_1*x + a_0.
        return np.polyval(self.coefficients_, values)

    def save(self, path: Path):
        """Serializa el modelo entrenado como pickle con extensión .pkt.

        Cargar archivos pickle únicamente si proceden de una fuente confiable.
        """
        if self.coefficients_ is None:
            raise ValueError("No se puede guardar un modelo sin entrenar.")
        path = Path(path)
        # Pickle conserva el objeto Python entrenado y sus coeficientes. El archivo
        # solo debe cargarse desde un origen confiable; pickle no es un formato
        # seguro para modelos recibidos de usuarios o servicios externos.
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        """Carga un .pkt creado por esta clase. No cargar archivos no confiables."""
        # Un pickle puede ejecutar código durante unpickle: validar el tipo después
        # no neutraliza un archivo malicioso, por eso el origen debe ser confiable.
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls) or model.coefficients_ is None:
            raise ValueError("El archivo no contiene un modelo PolynomialRegressor entrenado.")
        return model

    @staticmethod
    def write_predictions(path: Path, result: FitResult):
        path = Path(path)
        # La salida conserva observación, predicción y residuo fila a fila para que
        # el usuario pueda revisar errores y no solo el resumen agregado del modelo.
        values = np.column_stack((result.x, result.predicted + result.residuals, result.predicted, result.residuals))
        np.savetxt(path, values, delimiter=",", header="x,y_observado,y_predicho,residuo", comments="", fmt="%.10g")

    def write_plot(self, path: Path, x, y, result: FitResult):
        path = Path(path)
        # Ordenar solo para dibujar la curva evita segmentos zigzagueantes cuando
        # las observaciones llegan desordenadas; el ajuste usa todas las filas.
        order = np.argsort(x)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(x, y, label="Observaciones", color="#1f77b4", alpha=0.85)
        ax.plot(np.asarray(x)[order], result.predicted[order], label=f"Ajuste polinómico (grado {self.degree})", color="#d62728")
        ax.set(title="Regresión polinómica", xlabel="x", ylabel="y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
