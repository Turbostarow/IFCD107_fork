# Ejercicio: completar la API y los algoritmos

La carpeta contiene guías para reescribir partes del proyecto desde pseudocódigo. Los archivos `app_ejercicio.py`, `api_functions_ejercicio.py` y `regression/*_def_ejercicio.py` contienen únicamente comentarios; no son módulos funcionales ni se deben importar para iniciar la API.

## Cómo trabajar

1. Leer el objetivo y el flujo paso a paso de un archivo de ejercicio.
2. Crear una copia de trabajo o completar los fuentes originales en una rama de prácticas.
3. Escribir el código Python donde corresponda, manteniendo separadas las capas:
   - `app.py`: rutas, autenticación, entrada/salida HTTP.
   - `api_functions.py`: validación, coordinación y guardado de resultados.
   - `regression/`: pipelines y algoritmos.
4. Comparar la solución con el archivo fuente homónimo, solo después de intentar resolver el ejercicio.
5. Añadir pruebas para entradas válidas, errores, reproducibilidad y prevención de fuga de datos.

## Archivos por componente

| Guía de ejercicio | Fuente de referencia | Contenido |
|---|---|---|
| `app_ejercicio.py` | `app.py` | Autenticación, rutas, límites, respuestas y errores HTTP. |
| `api_functions_ejercicio.py` | `api_functions.py` | Validación, preparación, búsqueda, métricas y artefactos. |
| `regression/polynomial_regressor_def_ejercicio.py` | `regression/polynomial_regressor.py` | Mínimos cuadrados, diagnósticos, covarianza y búsqueda. |
| `regression/random_forest_regressor_def_ejercicio.py` | `regression/random_forest_regressor.py` | Preprocesamiento tabular, bosque, búsqueda e importancias. |
| `regression/knn_regressor_def_ejercicio.py` | `regression/knn_regressor.py` | Escalado, KNN y validación cruzada. |
| `regression/linear_regressor_def_ejercicio.py` | `regression/linear_regressor.py` | Recta, holdout y métricas. |
| `regression/svm_regressor_def_ejercicio.py` | `regression/svm_regressor.py` | SVR y opciones del kernel. |
| `regression/adaboost_regressor_def_ejercicio.py` | `regression/adaboost_regressor.py` | Árbol base y AdaBoost. |
| `regression/gradient_boosting_regressor_def_ejercicio.py` | `regression/gradient_boosting_regressor.py` | Ensamble secuencial y parámetros. |
| `regression/parameter_search_def_ejercicio.py` | `regression/parameter_search.py` | Catálogo de opciones seguras para RandomizedSearchCV. |
| `regression/tabular_preprocessing_def_ejercicio.py` | `regression/tabular_preprocessing.py` | Imputación, escalado y codificación de categorías. |
| `regression/knn_con_ollama_def_ejercicio.py` | `regression/knn_con_ollama.py` | Flujo KNN más explicación auxiliar de Ollama. |

Los pseudocódigos describen el comportamiento esperado, no el único diseño válido. Los alumnos deben justificar decisiones como partición, métricas, tratamiento de faltantes, coste de Leave-One-Out y seguridad de serialización.
