from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


def r2_ajustado(y_real, y_predicho, n_features):
    """
    Calcula el R² ajustado.

    n_features: número de variables predictoras del modelo.
                En una regresión polinómica, cuenta las variables
                generadas por PolynomialFeatures, sin contar el intercepto.
    """
    n = len(y_real)

    if n <= n_features + 1:
        raise ValueError("Se necesitan más observaciones que variables predictoras + 1.")

    r2 = r2_score(y_real, y_predicho)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def graficar_cv_results(search, output_path, titulo="Resultados de validación cruzada"):
    """
    Genera gráficos a partir de cv_results_ y los guarda en output_path.

    Parámetros:
        search: GridSearchCV o RandomizedSearchCV ya ajustado con .fit().
        output_path: carpeta donde guardar las imágenes y el CSV.
        titulo: texto para el título de los gráficos.
    """
    carpeta = Path(output_path)
    carpeta.mkdir(parents=True, exist_ok=True)

    resultados = pd.DataFrame(search.cv_results_)
    resultados = resultados.sort_values("rank_test_score").reset_index(drop=True)

    # Guarda la tabla completa para poder consultar todas las combinaciones.
    resultados.to_csv(
        carpeta / "cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nombre_metrica = getattr(search, "scoring", None)
    es_error_negativo = (
        isinstance(nombre_metrica, str)
        and nombre_metrica.startswith("neg_")
    )

    puntuaciones = resultados["mean_test_score"].to_numpy()
    desviaciones = resultados["std_test_score"].to_numpy()

    if es_error_negativo:
        puntuaciones = -puntuaciones
        desviaciones = desviaciones
        etiqueta_y = nombre_metrica.removeprefix("neg_")
    else:
        etiqueta_y = "Puntuación media de validación"

    cantidad = min(20, len(resultados))
    posiciones = np.arange(cantidad)

    # Gráfico de puntuación media y desviación entre pliegues.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        posiciones,
        puntuaciones[:cantidad],
        yerr=desviaciones[:cantidad],
        fmt="o",
        capsize=4,
        alpha=0.8,
    )
    ax.set_title(f"{titulo}: mejores {cantidad} combinaciones")
    ax.set_xlabel("Combinación de hiperparámetros (ordenada por ranking)")
    ax.set_ylabel(etiqueta_y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(carpeta / "puntuaciones_cv.png", dpi=150)
    plt.close(fig)

    # Barras con las puntuaciones de las mejores configuraciones.
    etiquetas = [
        str(resultados.loc[i, "params"])[:70]
        for i in range(cantidad)
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(
        posiciones,
        puntuaciones[:cantidad],
        xerr=desviaciones[:cantidad],
        capsize=3,
    )
    ax.set_yticks(posiciones)
    ax.set_yticklabels(etiquetas, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Mejores {cantidad} configuraciones")
    ax.set_xlabel(etiqueta_y)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(carpeta / "mejores_configuraciones.png", dpi=150)
    plt.close(fig)

    print(f"Gráficos y resultados guardados en: {carpeta.resolve()}")
    print(f"Mejores parámetros: {search.best_params_}")
    print(f"Mejor puntuación CV: {search.best_score_:.4f}")