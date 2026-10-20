# NVIDIA Triton para los regresores de IFCD107

Este componente añade NVIDIA Triton en una carpeta independiente y conserva el reparto existente del proyecto:

- `regression/` contiene la implementación de cada algoritmo. El entrenamiento reutiliza `RandomForestRegressorModel` y `PolynomialSearchEstimator`; `linear_regressor.py` añade el entrenador lineal en el mismo paquete.
- `triton/train_models.py` organiza el entrenamiento, el holdout y la creación de artefactos.
- `triton/model_repository/` contiene la configuración y el adaptador de cada modelo para Triton Python backend.
- `triton/Dockerfile.trainer` y `triton/Dockerfile.server` separan las dependencias de entrenamiento y servicio.
- `triton/compose.yaml` orquesta el entrenamiento, comparte el repositorio generado y arranca el servidor después de que termine el entrenamiento.
- `triton/postman/` incluye peticiones de inferencia de ejemplo y comprobaciones para los tres endpoints.

## Entrenar y publicar en Triton

Desde la raíz `proyecto_regresion_polinomica`, inicia el flujo en modo CPU:

```powershell
docker compose -f triton/compose.yaml up --build
```

Con Docker configurado para exponer la GPU NVIDIA y NVIDIA Container Toolkit, usa el override de GPU:

```powershell
docker compose -f triton/compose.yaml -f triton/compose.gpu.yaml up --build
```

El contenedor `train-triton-models` entrena y evalúa los modelos con partición reproducible. Al completarse, registra `model.pkl` en cada carpeta de versión del repositorio y escribe las métricas en `triton/results/training_metrics.json`. A continuación se inicia el servicio NVIDIA Triton en los puertos publicados HTTP 18000, gRPC 18001 y métricas 18002 (puertos internos 8000-8002).

Para regenerar los modelos tras cambiar datos o parámetros:

```powershell
docker compose -f triton/compose.yaml -f triton/compose.gpu.yaml down
docker compose -f triton/compose.yaml -f triton/compose.gpu.yaml up --build --force-recreate
```

El contenedor de entrenamiento siempre vuelve a ejecutarse cuando Compose lo recrea. Para detener el servidor, usa `docker compose -f triton/compose.yaml down` (añade `-f triton/compose.gpu.yaml` si lo arrancaste con el override).

## Modelos y datos

| Nombre Triton | Implementación de entrenamiento | Datos | Entradas para inferencia |
|---|---|---|---|
| `linear_regression` | `LinearRegressor` (scikit-learn) | `ejemplos/datos_ejemplo_xy.csv` | Matriz FP32 `X` con una columna `x` |
| `polynomial_regression` | `PolynomialSearchEstimator` existente, grado 2 por defecto | `ejemplos/datos_ejemplo_xy.csv` | Matriz FP32 `X` con una columna `x` |
| `random_forest_regression` | `RandomForestRegressorModel` existente | `ejemplos/datos_ejemplo_complejo.csv` | Dos entradas STRING y cuatro FP32; ver la colección Postman |

Cada modelo se evalúa en una partición de prueba y después se reajusta con todas las filas válidas para atender predicciones. Las métricas provienen solo de la partición reservada. Los dos CSV del ejemplo son sintéticos; sus métricas son didácticas, no evidencia de rendimiento en datos reales.

## Llamar a la API

La API HTTP V2 de Triton acepta solicitudes JSON como:

```powershell
$body = '{"inputs":[{"name":"X","shape":[3,1],"datatype":"FP32","data":[40,50,60]}],"outputs":[{"name":"Y"}]}'
Invoke-RestMethod -Method Post -Uri http://localhost:18000/v2/models/polynomial_regression/infer -ContentType 'application/json' -Body $body
```

Importa `triton/postman/NVIDIA_Triton_Regressors.postman_collection.json` para los tres ejemplos y sus aserciones.

## Hardware y alcance de Triton

Triton es el servidor de inferencia; no ejecuta este entrenamiento. Compose entrena primero en un contenedor Python independiente y luego sirve los modelos con Triton. Los adaptadores usan el Python backend y scikit-learn, así que estas implementaciones calculan en CPU, incluso si Triton tiene una GPU visible. El override `compose.gpu.yaml` hace visible la GPU al contenedor; para acelerar los cálculos de los modelos habría que exportar a un backend GPU compatible (por ejemplo, ONNX/TensorRT o una implementación basada en RAPIDS cuML) y comprobar equivalencia numérica. El servidor Triton puede ejecutarse sin GPU para este ejemplo.

Las imágenes usan la publicación `26.08-py3` de NVIDIA Triton y un entorno Python 3.12. El trainer y Python backend fijan las mismas versiones de NumPy, pandas y scikit-learn para que los artefactos serializados sean compatibles. Cambia `TRITON_BASE_IMAGE` en el archivo `.env` o en el entorno al actualizar la versión del servidor; conserva el entorno de dependencias idéntico en ambos contenedores.
