# API Flask de regresión polinómica

Ejemplo ejecutable con rutas Flask en `app.py`, lógica compartida en `api_functions.py` y algoritmos separados en `regression/`. La guía completa y las instrucciones de Postman están en `documentacion/guia_proyecto.tex`.

El CSV incluido tiene 1.000 filas sintéticas para una demostración de gestión de expedientes de terrazas. No son registros, plazos ni resultados reales del Ayuntamiento. En el ajuste, `x` es un índice didáctico de complejidad y `y` son días de tramitación simulados.

## Arranque

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

En macOS/Linux, activa con `source .venv/bin/activate`, instala `requirements.txt` y ejecuta `python app.py`.

La API carga automáticamente los valores del archivo local `.env`. En esta copia, `.env` contiene una contraseña aleatoria para `admin` y un secreto aleatorio para firmar tokens. Para Postman, copia `API_ADMIN_PASSWORD` desde ese archivo a la variable de colección `password`.

`.env` está excluido de Git y no debe compartirse ni incluirse en el ZIP distribuible. Para configurar otro entorno, copia `.env.example` a `.env` y asigna credenciales propias. Mantén `API_TOKEN_SECRET` estable entre procesos y reinicios. Si no existe, el ejemplo genera un secreto temporal, por lo que los tokens se invalidan al reiniciar. En producción, utiliza contraseñas con hash y un gestor de secretos.

Importa `postman/Regresion_Polinomica.postman_collection.json` en Postman. Ejecuta `Obtener token` antes de las solicitudes de ajuste; la colección guarda el Bearer token automáticamente. Para la petición CSV, selecciona `ejemplos/datos_ejemplo.csv` en el campo `file` y usa la referencia `terrazas_madrid_demo`.

Para regenerar el conjunto determinista de 1.000 filas: `python scripts/generate_terrace_cases.py`.

La API mantiene endpoints en `app.py`, funciones auxiliares compartidas en `api_functions.py` y clases de algoritmo dentro de `regression/`. Los artefactos se crean bajo `runs/<reference>/` en `graficas/`, `csv_entrada/`, `logs_salida/`, `csv_salida/` y `modelos/`. El registro central `runs/_auditoria/api_audit.jsonl` conserva una entrada por petición con fecha UTC, método, ruta, estado, usuario, IP y huella SHA-256 truncada del token. Los logs nunca guardan el token Bearer completo. Si se usa un proxy inverso, `REMOTE_ADDR` reflejará el proxy salvo que se configure confianza explícita para proxies conocidos. El modelo se guarda como `.pkt`; carga solo archivos de confianza porque utiliza pickle.

Endpoints: `GET /api/v1/health` (público), `POST /api/v1/auth/token` (emite token), regresión polinómica JSON/CSV y búsqueda, KNN con K-Fold/Leave-One-Out, Random Forest desde CSV o desde el ejemplo, y los nuevos regresores SVR, AdaBoost y Gradient Boosting desde CSV. Todas las rutas de entrenamiento requieren Bearer. Los tokens expiran por defecto en una hora; configura `API_TOKEN_TTL_SECONDS` para cambiarlo.


## Random Forest

La API dispone de `POST /api/v1/random-forest/terrazas`, que entrena con `ejemplos/datos_ejemplo_complejo.csv`, y `POST /api/v1/random-forest/csv`, que recibe un fichero mediante `multipart/form-data`. Ambas rutas requieren el token Bearer obtenido con `/api/v1/auth/token`.

Ejemplo para la ruta incluida:

```json
{
  "reference": "terrazas_rf_demo",
  "target": "y",
  "n_estimators": 200,
  "max_depth": null,
  "test_size": 0.2,
  "random_state": 42
}
```

`expediente_id` se conserva para identificar predicciones pero no se usa como predictor. Se procesan variables numéricas y categóricas; los artefactos se guardan por referencia como en la regresión polinómica. El conjunto de terrazas es sintético. La colección Postman incluye solicitudes para las dos rutas.

## SVR y boosting para regresión

Los modelos nuevos siguen el mismo contrato CSV y guardado de artefactos:

- `POST /api/v1/regresion/svm/csv` usa `SVR` dentro de un pipeline que imputa valores ausentes, codifica categorías y escala variables numéricas.
- `POST /api/v1/regresion/boosting/adaboost/csv` usa `AdaBoostRegressor` con árboles base.
- `POST /api/v1/regresion/boosting/gradient/csv` usa `GradientBoostingRegressor`.

Las tres rutas reciben `multipart/form-data` con `reference`, `target`, opciones del modelo y el fichero `file`. Requieren token Bearer y aceptan un CSV de hasta 4 MB. Las salidas mantienen la estructura del proyecto: CSV de entrada y predicciones, gráfico observado frente a predicho, modelo `.pkt`, log y métricas R², RMSE, MAE y sesgo. El preprocesamiento se ajusta dentro del pipeline solo con las filas de entrenamiento.

Cada estimador está en su módulo de `regression/`: `svm_regressor.py`, `adaboost_regressor.py` y `gradient_boosting_regressor.py`. `tabular_preprocessing.py` centraliza la imputación/codificación común y el escalado específico que necesita SVR. La colección Postman incluye una solicitud de ejemplo para cada endpoint nuevo usando el CSV sintético incluido.

## Docker y Ollama

El proyecto incluye un contenedor Flask, un servicio Ollama y un paso de inicialización que descarga el modelo configurado. Ollama no publica su puerto al host; la API web es el único servicio expuesto. Los artefactos de regresión se guardan en un volumen Docker persistente y los modelos Ollama se descargan en la carpeta local `ollama/`.

1. Copia `.env.example` a `.env` y configura `API_ADMIN_PASSWORD` y `API_TOKEN_SECRET` con valores aleatorios propios. No compartas `.env`.
2. Desde la raíz del proyecto ejecuta:

```powershell
docker compose up --build -d
```

La primera ejecución descarga la imagen de Ollama y el modelo (por defecto `qwen2.5:3b`), por lo que puede tardar y ocupar varios GB. Para elegir otro modelo, cambia `OLLAMA_MODEL` en `.env` y recrea los servicios. La API queda disponible en `http://localhost:5000`; cambia `API_PORT` para usar otro puerto del host.

Para pedir ayuda al modelo, primero obtén el token con `POST /api/v1/auth/token` y luego envía `POST /api/v1/ai/chat` con `Authorization: Bearer <token>` y JSON como `{"prompt":"¿Cómo interpreto un R² negativo?"}`. Esta ruta usa Ollama y devuelve `model` y `answer`; las rutas de entrenamiento existentes no dependen del chat.

Comandos útiles:

```powershell
docker compose logs -f api ollama
docker compose down
```

`docker compose down` conserva el volumen de resultados y la carpeta local `ollama/`. Para borrar el volumen de resultados ejecuta `docker compose down -v`; para liberar el espacio de los modelos, detén Compose y elimina manualmente la carpeta `ollama/`.



La variante `POST /api/v1/regresion/polinomica/randomizedsearch/ollama` ejecuta la misma búsqueda y agrega `ollama_documentation` a la respuesta. Si Ollama no está disponible, conserva las métricas y devuelve `ollama_documentation_error`. La colección Postman incluye una solicitud JSON para esta ruta.

## Regresión polinómica servida con BentoML y Docker

El componente autónomo `bentoml/` añade un pipeline de regresión polinómica sin modificar la API Flask ni el servicio de Ollama. Desde esta raíz, inicia el entrenamiento y el servidor con:

```powershell
docker compose -f bentoml/compose.yaml up --build
```

El servicio lee `ejemplos/datos_ejemplo_xy.csv`, evalúa sobre una partición de 200 filas, registra el pipeline de grado 2 y expone `POST http://localhost:3000/predict`. Para enviar una petición desde PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:3000/predict -ContentType 'application/json' -Body '{"x":[40,50,60]}'
```

La ejecución verificada con este CSV obtuvo R² = 0,9287, RMSE = 6,9001 y MAE = 5,5080 en el conjunto reservado. Una llamada con `x=[40,50,60]` devolvió aproximadamente `[45.2401, 56.6759, 70.0502]`. Son resultados sobre datos sintéticos de aprendizaje. Importa `bentoml/BentoML_Polynomial_Regression.postman_collection.json` para repetir la petición. Para parar el servicio, ejecuta `docker compose -f bentoml/compose.yaml down`.

## NVIDIA Triton: regresores lineal, polinómico y Random Forest

El componente independiente `triton/` reutiliza los módulos de `regression/`, entrena y evalúa los modelos, y después los publica mediante NVIDIA Triton Python backend. El entrenamiento ocurre antes de que Triton cargue los modelos; las llamadas HTTP realizan inferencia.

Desde la raíz del proyecto:

```powershell
docker compose -f triton/compose.yaml up --build
```

Para hacer visible la GPU NVIDIA al contenedor Triton (con NVIDIA Container Toolkit disponible):

```powershell
docker compose -f triton/compose.yaml -f triton/compose.gpu.yaml up --build
```

La API V2 se publica en el puerto 8000; importa `triton/postman/NVIDIA_Triton_Regressors.postman_collection.json` para probar los tres modelos. Consulta `triton/README.md` para entradas, datos y explicación del flujo.
