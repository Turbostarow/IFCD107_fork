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

### Ejecución con Docker Compose

El servicio `regression-api` monta `./runs` del proyecto en `/app/runs` dentro del contenedor. Los informes, modelos, CSV y el registro de auditoría quedan así en la carpeta local `runs/` y sobreviven a la recreación del contenedor. En Windows con Docker Desktop, inicia el servicio desde la raíz del proyecto para que la ruta relativa `./runs` corresponda a esta carpeta:

```powershell
docker compose up -d --build regression-api
```

Si se ejecuta en Linux y la carpeta local no permite escribir al usuario `app` del contenedor, asigna permisos de escritura a `runs/` para el UID configurado en la imagen o ajusta el usuario del servicio según la política local. No es necesario borrar volúmenes Docker para aplicar el montaje de carpeta local.

### Interfaz web de Random Forest

El servicio `random-forest-ui` integra el laboratorio interactivo de regresión en este mismo Docker Compose. Se construye desde `random_forest_ui/`, con dependencias independientes de la API Flask. Arranca con:

```powershell
cd D:\onedrive\Documentos\ChatGPT\ifcd107\ejercicios\proyecto_regresion_polinomica
docker compose up -d --build random-forest-ui
```

Abre [http://localhost:8501](http://localhost:8501). Puedes cambiar el puerto mediante `RANDOM_FOREST_UI_PORT` en `.env` (valor predeterminado: `8501`). El puerto se publica únicamente en la interfaz local del equipo. También se inicia al ejecutar `docker compose up -d --build` para el conjunto del proyecto. Este laboratorio no necesita arrancar PostgreSQL, Ollama ni la API.

La interfaz incluye datos sintéticos, carga de CSV, selección de objetivo y predictores numéricos, controles de hiperparámetros con reentrenamiento automático, R², RMSE, MAE, Pearson, bias, gráficos de residuos, importancia de variables y descarga de predicciones e historial. La partición de prueba permanece fija mientras no cambien los datos, su proporción o la semilla. Los valores ausentes de los predictores se imputan con medianas aprendidas exclusivamente en entrenamiento.

Las métricas principales se calculan en prueba. Bias = media(predicho − observado); residuo = observado − predicho. Pearson se muestra como no definido cuando alguna serie es constante. Los predictores categóricos deben codificarse previamente. `max_samples` y OOB se activan solo con bootstrap; Poisson necesita objetivos no negativos con suma positiva. Cada configuración crea un bosque nuevo, por lo que `warm_start` no conserva árboles entre cambios.

El entrenamiento se ejecuta dentro del contenedor de la interfaz y no usa las rutas autenticadas de la API. Los archivos cargados y el historial pertenecen a la sesión: descarga los resultados antes de cerrarla o recrear el contenedor. La exploración reiterada de la prueba puede sobreajustar la elección de parámetros; reserva un conjunto externo para evaluar el modelo final.

Para consultar el estado y los logs o detener solo la interfaz:

```powershell
docker compose ps random-forest-ui
docker compose logs --tail 100 random-forest-ui
docker compose stop random-forest-ui
```

### PostgreSQL: resultados de las casuísticas

Compose levanta un servicio `postgres` con la base `regression_results` y conserva sus datos en el volumen `postgres_data`. Al crear el volumen por primera vez, PostgreSQL ejecuta `database/init/001_regression_results.sql` para crear estas tablas:

- `regression_runs`: una fila al comenzar cada ejecución, con `run_id`, fecha, referencia, algoritmo, nombre del modelo y estado.
- `regression_run_files`: una fila por cada informe, CSV, modelo, resumen, archivo de búsqueda y log generado por la API.
- `regression_run_metrics`: dos filas por ejecución completada, una para `GridSearchCV` y otra para `RandomizedSearchCV`. Incluye RMSE de validación y prueba, R² de prueba, y los hiperparámetros y métricas completas serializados como texto JSON.

La API registra el inicio antes de entrenar. Al terminar, guarda los archivos y las dos filas de métricas; si falla el entrenamiento, marca la ejecución como `failed`. La creación de tablas también se comprueba desde el código, por lo que se añaden si ya existía el volumen de PostgreSQL antes de incorporar el esquema.

Configura en `.env` las variables `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` y `POSTGRES_PORT`; sus valores de desarrollo están en `.env.example`. Cambia la contraseña de ejemplo antes de usar la base fuera del equipo local. Para iniciar la base y la API con sus dependencias:

```powershell
docker compose up -d --build regression-api
```

Para consultar las ejecuciones más recientes (ajusta usuario y base si cambiaste `.env`):

```powershell
docker compose exec postgres psql -U regression -d regression_results -c "SELECT run_id, started_at, model_name, status FROM regression_runs ORDER BY started_at DESC LIMIT 20;"
docker compose exec postgres psql -U regression -d regression_results -c "SELECT run_id, search_name, cv_rmse, test_rmse, test_r2, parameters_text, metrics_text FROM regression_run_metrics ORDER BY created_at DESC LIMIT 20;"
docker compose exec postgres psql -U regression -d regression_results -c "SELECT run_id, file_type, file_path FROM regression_run_files ORDER BY created_at DESC LIMIT 50;"
```

La API carga automáticamente los valores del archivo local `.env`. En esta copia, `.env` contiene una contraseña aleatoria para `admin` y un secreto aleatorio para firmar tokens. Para Postman, copia `API_ADMIN_PASSWORD` desde ese archivo a la variable de colección `password`.

`.env` está excluido de Git y no debe compartirse ni incluirse en el ZIP distribuible. Para configurar otro entorno, copia `.env.example` a `.env` y asigna credenciales propias. Mantén `API_TOKEN_SECRET` estable entre procesos y reinicios. Si no existe, el ejemplo genera un secreto temporal, por lo que los tokens se invalidan al reiniciar. En producción, utiliza contraseñas con hash y un gestor de secretos.

Importa `postman/Regresion_Polinomica.postman_collection.json` en Postman. Ejecuta `Obtener token` antes de las solicitudes de ajuste; la colección guarda el Bearer token automáticamente. Para la petición CSV, selecciona `ejemplos/datos_ejemplo.csv` en el campo `file` y usa la referencia `terrazas_madrid_demo`.

Para regenerar el conjunto determinista de 1.000 filas: `python scripts/generate_terrace_cases.py`.

La API mantiene endpoints en `app.py`, funciones auxiliares compartidas en `api_functions.py` y clases de algoritmo dentro de `regression/`. Los artefactos se crean bajo `runs/<reference>/` en `graficas/`, `csv_entrada/`, `logs_salida/`, `csv_salida/` y `modelos/`. El registro central `runs/_auditoria/api_audit.jsonl` conserva una entrada por petición con fecha UTC, método, ruta, estado, usuario, IP y huella SHA-256 truncada del token. Los logs nunca guardan el token Bearer completo. Si se usa un proxy inverso, `REMOTE_ADDR` reflejará el proxy salvo que se configure confianza explícita para proxies conocidos. El modelo se guarda como `.pkt`; carga solo archivos de confianza porque utiliza pickle.

Endpoints: `GET /api/v1/health` y `GET /api/v1/regresion/parametros` (públicos), `POST /api/v1/auth/token` (emite token), regresión polinómica JSON/CSV y búsqueda, KNN con K-Fold/Leave-One-Out, Random Forest desde CSV o desde el ejemplo, y SVR, AdaBoost y Gradient Boosting desde CSV. Todas las rutas de entrenamiento requieren Bearer. Los tokens expiran por defecto en una hora; configura `API_TOKEN_TTL_SECONDS` para cambiarlo.

`GET /api/v1/regresion/parametros` devuelve un catálogo donde cada parámetro se presenta como un elemento con `parameter` y `items`. Los endpoints de entrenamiento aceptan `search=true`, `n_iter` y `search_cv`; en ese modo `RandomizedSearchCV` toma sus candidatos de las mismas listas. La prueba final queda separada de la búsqueda. La respuesta incluye `parameters` con los valores ganadores, `requested_parameters` y las métricas de validación cruzada. En KNN la búsqueda se hace de forma anidada dentro de cada pliegue externo; al combinarla con Leave-One-Out se limita a 200 filas.

Ejemplo multipart para activar la búsqueda en SVR, AdaBoost, Gradient Boosting o KNN:

```text
search=true
n_iter=10
search_cv=5
```

En Random Forest se pueden enviar esos mismos campos en JSON o multipart. Con `search=false` se conserva el ajuste directo usando los parámetros indicados en la petición.


## Predecir con un modelo .joblib y un CSV

Guarda el modelo o pipeline entrenado con `joblib.dump(modelo, "modelo.joblib")`. El endpoint usa `joblib.load` y admite archivos joblib comprimidos o sin compresión. Los entrenamientos existentes del proyecto conservan su formato `.pkt`; esta ruta de inferencia recibe `.joblib`.

`POST /api/v1/regresion/predecir/csv` requiere el token Bearer habitual y un cuerpo `multipart/form-data`:

| Campo | Tipo | Contenido |
| --- | --- | --- |
| `model` | Archivo | Modelo `.joblib` de confianza, máximo 50 MB. |
| `file` | Archivo | CSV UTF-8 con cabecera y features, máximo 4 MB y 100.000 filas. |
| `separator` | Texto opcional | Separador CSV: `,` (predeterminado), `;` o tabulador. |
| `decimal` | Texto opcional | Separador decimal: `.` (predeterminado) o `,`. Debe ser distinto de `separator`. |

El endpoint carga el modelo y llama a `predict` sin entrenarlo. Conserva el preprocesamiento del pipeline. Si el modelo guarda `feature_names_in_`, exige sus columnas y las ordena como en entrenamiento; las columnas extra se ignoran y se enumeran en `ignored_columns`. Si el modelo se entrenó con arrays, el CSV debe contener únicamente las features y en el mismo orden original. La clase polinómica propia del proyecto requiere la columna `x`; `ejemplos/features_prediccion.csv` sirve para probarla. Para otros modelos, proporciona las features con sus nombres y tipos originales, incluidas las categorías si el pipeline las procesa. No hace falta incluir la variable objetivo.

Ejemplo de respuesta para dos filas:

```json
{
  "n_rows": 2,
  "feature_columns": ["x"],
  "ignored_columns": [],
  "predictions": [
    {"row": 1, "prediction": 12.5},
    {"row": 2, "prediction": 15.2}
  ]
}
```

`row` empieza en 1 y sigue el orden de las filas de datos del CSV. Solo se admite regresión de un objetivo, con una predicción numérica finita por fila. No se calculan métricas sin valores observados. La petición completa admite hasta 56 MB; las demás rutas conservan su límite general de 5 MB. Los errores devuelven JSON: 400 si faltan archivos o sus extensiones no son correctas, 401 sin autenticación, 413 por tamaño y 422 por CSV, modelo o features incompatibles.

Los `.joblib` utilizan serialización basada en pickle: **carga exclusivamente archivos de confianza**, porque deserializarlos puede ejecutar código Python dentro de la API. La extensión `.joblib` no garantiza seguridad. Utiliza las mismas versiones de Python y dependencias con las que se guardó el modelo, y conserva disponibles sus módulos de clases personalizadas. Los archivos recibidos se usan en memoria y no se guardan como nuevos artefactos.

En Postman, importa la colección actualizada, ejecuta **Obtener token** y abre **Predecir desde modelo .joblib y CSV de features**. En Body → form-data selecciona el `.joblib` en `model` y tu CSV en `file`; no añadas manualmente `Content-Type`, pues Postman genera el boundary multipart.

Para aplicar el endpoint a una API Docker ya iniciada:

```powershell
docker compose up -d --build --no-deps regression-api
```

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

### Ejecutar las seis prácticas SVR paso a paso

El script `regression/train_svm_casuisticas.py` es una guía comentada para principiantes y comparte `train_one_dataset` con el endpoint EDA. Para cada CSV preprocesa y guarda los datos transformados; después compara `GridSearchCV` y `RandomizedSearchCV` con validación cruzada sobre el conjunto de entrenamiento. Selecciona el buscador con menor RMSE medio de validación, evalúa los dos mejores estimadores en el mismo conjunto de prueba reservado y persiste los hiperparámetros del ganador en `best_training`. El PDF final incluye una gráfica de valores reales frente a predichos para cada búsqueda.

`GridSearchCV` prueba combinaciones definidas para kernels lineal, RBF y polinómico. `RandomizedSearchCV` explora 40 combinaciones con rangos logarítmicos para `C`, `epsilon` y `gamma`. Ambos incluyen el preprocesador dentro del pipeline, así que imputación, escalado y codificación se ajustan dentro de cada pliegue. Las tablas completas de resultados se guardan en `busquedas/`; el modelo ganador y sus parámetros aparecen en el JSON de métricas, la respuesta del endpoint EDA y el resumen general `resumen_entrenamientos.csv`. Ejecuta el script desde la raíz:

```powershell
.venv\Scripts\python.exe regression\train_svm_casuisticas.py
```

Los PDFs, CSV preprocesados, predicciones, modelos y métricas individuales se guardan en `runs/svm_casuisticas/<casuistica>/`. `runs/svm_casuisticas/resumen_entrenamientos.csv` compara R², RMSE, MAE y sesgo entre los seis casos. Los datos y métricas son didácticos y no describen expedientes reales.

## EDA y preprocesado tabular con informe PDF

`POST /api/v1/eda/preprocesado/csv` acepta `multipart/form-data` con `reference` y el archivo `file`; requiere Bearer token y limita el CSV a 4 MB. El endpoint llama a `train_one_dataset` en `regression/train_svm_casuisticas.py`: primero elabora el informe EDA y el CSV transformado, y después entrena SVR sobre esos datos. La clase `regression/eda_preprocesado.py` convierte `tipo_tramite` y `distrito` a códigos numéricos reproducibles, convierte `requerimientos_documentacion` (o `requiere_documentacion`) a número o códigos si es texto, y sustituye `mes_presentacion` por las variables cíclicas seno/coseno. El informe PDF incluye una matriz de correlación de Pearson para columnas numéricas.

El formulario admite `min_correlation` (predeterminado `0`) y `max_correlation` (predeterminado `1`), ambos entre 0 y 1. Para variables numéricas, el filtro conserva las características cuyo valor absoluto de correlación con el objetivo `y` esté dentro del intervalo, incluidos sus extremos; excluye las que queden fuera. Así, los valores predeterminados conservan todas las características. El objetivo `y`, el identificador `expediente_id` y las categorías codificadas no se eliminan por este filtro. Se puede mantener la exclusión manual opcional con `feaures_eliminar=columna1,columna2` (campo legado). La respuesta detalla los umbrales y las columnas descartadas, además de las métricas y rutas del informe PDF, CSV preprocesado, predicciones y modelo bajo `runs/<reference>/`.

## Casuísticas comparativas para 21 algoritmos de regresión

La subcarpeta **Casuísticas comparativas de regresión** de la colección Postman tiene una solicitud por algoritmo. Todas llaman a `POST /api/v1/regresion/casuisticas/<algorithm>/csv`, con `reference`, `file`, `cv`, `n_iter` y `random_state`. La API ejecuta GridSearchCV y RandomizedSearchCV sobre una partición de entrenamiento, escoge por el menor RMSE medio de validación y evalúa los dos ganadores en la misma partición de prueba. El PDF contiene EDA, matriz de correlación y gráficas de predicción frente a valor real para ambos métodos. Se archivan los hiperparámetros ganadores, predicciones, modelo, resultados de búsqueda y métricas bajo `runs/<reference>/casuisticas/<algorithm>/`.

Cada algoritmo también tiene un lanzador que procesa los seis CSV didácticos de `proyecto_eda_preprocesado_latex/datos/`. Ejecuta desde la raíz del proyecto el módulo que corresponda; por ejemplo:

```powershell
python regression/train_ridge_casuisticas.py
```

Los módulos individuales son:

| Algoritmo | Lanzador funcional | Plantilla de ejercicio con pseudocódigo |
| --- | --- | --- |
| Lineal simple | `train_regresion_lineal_simple_casuisticas.py` | `train_regresion_lineal_simple_ejercicio_casuisticas.py` |
| Lineal múltiple | `train_regresion_lineal_multiple_casuisticas.py` | `train_regresion_lineal_multiple_ejercicio_casuisticas.py` |
| Ridge | `train_ridge_casuisticas.py` | `train_ridge_ejercicio_casuisticas.py` |
| Lasso | `train_lasso_casuisticas.py` | `train_lasso_ejercicio_casuisticas.py` |
| Elastic Net | `train_elastic_net_casuisticas.py` | `train_elastic_net_ejercicio_casuisticas.py` |
| Lineal bayesiana | `train_regresion_lineal_bayesiana_casuisticas.py` | `train_regresion_lineal_bayesiana_ejercicio_casuisticas.py` |
| Polinomial | `train_regresion_polinomial_casuisticas.py` | `train_regresion_polinomial_ejercicio_casuisticas.py` |
| Huber | `train_huber_casuisticas.py` | `train_huber_ejercicio_casuisticas.py` |
| RANSAC | `train_ransac_casuisticas.py` | `train_ransac_ejercicio_casuisticas.py` |
| Theil-Sen | `train_theil_sen_casuisticas.py` | `train_theil_sen_ejercicio_casuisticas.py` |
| Cuantílica | `train_regresion_cuantilica_casuisticas.py` | `train_regresion_cuantilica_ejercicio_casuisticas.py` |
| Árbol de decisión | `train_arbol_decision_casuisticas.py` | `train_arbol_decision_ejercicio_casuisticas.py` |
| Random Forest | `train_random_forest_casuisticas.py` | `train_random_forest_ejercicio_casuisticas.py` |
| Gradient Boosting | `train_gradient_boosting_casuisticas.py` | `train_gradient_boosting_ejercicio_casuisticas.py` |
| XGBoost | `train_xgboost_casuisticas.py` | `train_xgboost_ejercicio_casuisticas.py` |
| LightGBM | `train_lightgbm_casuisticas.py` | `train_lightgbm_ejercicio_casuisticas.py` |
| CatBoost | `train_catboost_casuisticas.py` | `train_catboost_ejercicio_casuisticas.py` |
| SVR | `train_svr_casuisticas.py` | `train_svr_ejercicio_casuisticas.py` |
| KNN | `train_knn_casuisticas.py` | `train_knn_ejercicio_casuisticas.py` |
| Procesos Gaussianos | `train_gaussian_process_casuisticas.py` | `train_gaussian_process_ejercicio_casuisticas.py` |
| MLPRegressor | `train_mlp_casuisticas.py` | `train_mlp_ejercicio_casuisticas.py` |

Las plantillas están compuestas solo por comentarios. Describen paso a paso el flujo común de preprocesado, partición sin fuga, GridSearchCV/RandomizedSearchCV, comparación del ganador, evaluación, gráficas y artefactos, además de los hiperparámetros y consideraciones propias de cada método. El alumnado completa el código en Python usando el lanzador funcional correspondiente como referencia.

## n8n, Dify, Flowise y modelos locales con Ollama

El `compose.yaml` de la raíz levanta la API de regresión, n8n, Flowise y Ollama. También incluye el Compose oficial de Dify para mantener sus servicios y configuración alineados con el proyecto upstream. Se requiere Docker Compose v2.20 o posterior. Dify se incorpora desde `dify/docker/docker-compose.yaml`, por lo que antes del primer arranque hay que preparar su repositorio y su archivo de entorno:

```powershell
git clone https://github.com/langgenius/dify.git dify
if (!(Test-Path .env)) { Copy-Item .env.example .env }
if (!(Test-Path dify/docker/.env)) { Copy-Item dify/docker/.env.example dify/docker/.env }
```

Si ya existe `.env`, conserva sus valores y añade las nuevas variables que falten desde `.env.example`. Asegúrate de que incluya `COMPOSE_PROFILES=weaviate,postgresql,collaboration`; estos perfiles habilitan los servicios de base de datos y vectorización de Dify al lanzar desde el Compose raíz. Sustituye las contraseñas y claves de ejemplo por valores aleatorios seguros. Mantén los mismos nombres de modelos (`llama3:8b` y `mistral:7b`) si quieres que los servicios usen los modelos descargados. La primera ejecución requiere conexión a Internet para descargar las imágenes de contenedor y varios GB de modelos:

El perfil local de Dify requiere el token `DIFY_AGENT_LOCAL_SANDBOX_AUTH_TOKEN`. Genéralo y guárdalo en `dify/docker/.env` (no lo publiques):

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Pega el valor generado en la línea `DIFY_AGENT_LOCAL_SANDBOX_AUTH_TOKEN=`. Dify comparte ese valor entre `agent_backend` y `local_sandbox`; ambos contenedores deben recrearse después de cambiarlo.

En `dify/docker/.env`, limita también la publicación web de Dify al equipo local configurando `EXPOSE_NGINX_PORT=127.0.0.1:8080` y `EXPOSE_NGINX_SSL_PORT=127.0.0.1:8443`.

```powershell
docker compose up -d --build
```

Accesos locales:

- API de regresión: `http://localhost:5000`
- n8n: `http://localhost:5678`
- Flowise: `http://localhost:3001` (crea la cuenta administradora al primer acceso)
- Dify: `http://localhost:8080/install` la primera vez; después, `http://localhost:8080`
- Ollama: API local en `http://localhost:11434`

Ollama descarga por defecto `llama3:8b` y `mistral:7b`, que se guardan en el volumen local `./ollama`. En Flowise y n8n, configura una conexión Ollama con URL `http://ollama:11434`; en Dify, configura el proveedor Ollama con URL `http://host.docker.internal:11434` y el nombre exacto del modelo. Docker Desktop resuelve `host.docker.internal`; en instalaciones Linux puede ser necesario habilitar el acceso al host desde los contenedores.

La inferencia con estos modelos locales no factura llamadas a una API externa. El equipo sigue consumiendo almacenamiento, memoria, CPU/GPU y electricidad; el primer arranque también descarga imágenes y modelos. El Compose deja las interfaces ligadas a `127.0.0.1`, de modo que son accesibles desde el propio equipo. Para detener los servicios sin borrar los datos: `docker compose down`; no añadas `-v` si quieres conservar los volúmenes.

### Solucionar permisos de Flowise

Flowise se ejecuta como usuario `node` (UID 1000) y guarda SQLite, claves, logs y archivos en `/home/node/.flowise`. El servicio `flowise-data-init` ajusta los permisos del volumen persistente antes de iniciar Flowise. Si el contenedor se detuvo con `EACCES` en `/root/.flowise`, actualiza la configuración y recrea ambos servicios:

```powershell
docker compose up -d --force-recreate flowise-data-init flowise
docker compose logs --tail 100 flowise
```

El mensaje `Auth secrets not initialized` que aparece después del error de permisos suele ser consecuencia de que Flowise no pudo inicializar la base de datos y los secretos; revisa que el arranque ya no muestre `EACCES`. La clave de cifrado de credenciales y la base de datos se conservan en el volumen `flowise_data`; no lo borres si quieres conservar los flujos y la cuenta.

### Solucionar `local_sandbox_auth_token is required`

El backend local de Dify no inicia si `DIFY_AGENT_LOCAL_SANDBOX_AUTH_TOKEN` está vacío. Añade un token aleatorio a `dify/docker/.env` como se indica arriba y recrea los dos servicios que lo utilizan:

```powershell
docker compose up -d --force-recreate local_sandbox agent_backend
docker compose logs --tail 100 agent_backend local_sandbox
```

### Solucionar `lookup db_postgres: no such host`

`db_postgres` forma parte del perfil `postgresql`. Al iniciar la composición raíz sin los perfiles de Dify activos, Compose puede omitir la base de datos aunque `plugin_daemon` esté configurado para conectarse a ella. Comprueba que el `.env` raíz tenga `COMPOSE_PROFILES=weaviate,postgresql,collaboration` y recrea los servicios afectados:

```powershell
docker compose up -d --force-recreate db_postgres plugin_daemon api worker worker_beat agent_backend
docker compose logs --tail 100 db_postgres plugin_daemon
```

### Solucionar `Cannot find module '@smithy/eventstream-codec'`

Flowise 3.1.4 presenta fallos de dependencias durante el arranque de nodos, incluido AWS Bedrock. El Compose fija Flowise en `3.1.3`, que evita el problema reportado. Para actualizar la imagen y recrear el servicio:

```powershell
docker compose pull flowise
docker compose up -d --force-recreate flowise
docker compose logs --tail 100 flowise
```

La etiqueta `3.1.3` ha sido reportada con una discrepancia en la versión CLI interna; se fija como solución práctica al fallo de arranque de `3.1.4`, no como actualización definitiva. Consulta las [incidencias de Flowise](https://github.com/FlowiseAI/Flowise/issues/6688) antes de cambiar `FLOWISE_IMAGE`.

## API de análisis histórico de La Primitiva

Los scripts recibidos en `C:\Users\josem\Downloads\primitiva` se han incorporado al proyecto en `primitiva/` para que también estén disponibles dentro de la imagen Docker. La ruta `POST /api/v1/primitiva/analizar/csv` ejecuta el analizador `v02` por defecto; también admite `v01`. Requiere Bearer token y `multipart/form-data` con `reference`, `version` y `file`. Solo se ejecutan los dos scripts incluidos; el archivo subido se trata como datos, no como código. El límite del CSV es 4 MB y el proceso tiene un límite de 90 segundos.

El CSV debe contener las columnas `N1` a `N6`; admite `FECHA`, `R` y `Joker` opcionales. Los scripts esperan separador `;` y buscan automáticamente la cabecera real cuando el fichero tiene líneas introductorias. `primitiva/datos_ejemplo_sinteticos.csv` contiene 60 sorteos ficticios para una primera prueba; no son resultados oficiales. La respuesta devuelve rutas relativas al CSV, informe PDF y log bajo `runs/<reference>/primitiva/`. El análisis de frecuencias históricas no predice sorteos futuros: cada combinación mantiene la misma probabilidad teórica.

En Postman, ejecuta `Obtener token` y después una de las solicitudes de la carpeta **La Primitiva**. Las llamadas ya adjuntan el CSV sintético; puedes sustituirlo por un fichero histórico en el campo `file`.

### Clasificadores didácticos KNN y CNN 1D

Se incorporan `POST /api/v1/primitiva/knn/csv` y `POST /api/v1/primitiva/cnn/csv`. Ambos requieren Bearer token y `multipart/form-data` con `reference` y un `file` CSV con las columnas `N1` a `N6`. El límite de subida es 4 MB. Son rutas distintas para comparar dos métodos; no afirman que sea posible anticipar un sorteo aleatorio.

Parámetros de formulario:

| Campo | KNN | CNN 1D | Predeterminado |
| --- | --- | --- | --- |
| `window` | 3-30 sorteos previos | 3-30 sorteos previos | 10 |
| `neighbors` | 1-25 vecinos | No se usa | 5 |
| `epochs` | No se usa | 10-300 épocas | 80 |
| `random_state` | Semilla | Semilla de inicialización | 42 |

El programa transforma cada sorteo en 49 indicadores (uno por número) y cada ejemplo en una secuencia de sorteos anteriores. KNN usa distancia Hamming y ponderación por distancia. La CNN 1D es una implementación pequeña de propósito didáctico basada en NumPy, con convolución temporal, ReLU, promedio global y salida multietiqueta; no añade TensorFlow o PyTorch al contenedor. Para la validación reserva el 20% final de las secuencias, respetando el orden temporal; luego ajusta el algoritmo con todo el histórico para producir un ranking exploratorio de seis números para el siguiente sorteo.

La respuesta incluye `suggested_numbers`, puntuaciones de ranking y métricas como aciertos medios por boleto y tasa de seis aciertos en el tramo reservado. Una selección aleatoria de seis números tiene un valor esperado de `36/49` aciertos (aprox. 0,735), porque cada boleto elegido contiene seis números y el sorteo extrae seis de 49. El pequeño CSV ficticio solo sirve para probar el flujo, no para evaluar capacidad predictiva. Los resultados y la entrada se guardan bajo `runs/<reference>/primitiva/` en `predicciones/`, `csv_entrada/` y `logs_salida/`.

En Postman se agregan las solicitudes **POST /api/v1/primitiva/knn/csv** y **POST /api/v1/primitiva/cnn/csv** dentro de **La Primitiva**. Ejecuta primero **Obtener token** y después selecciona el modelo; las pruebas de respuesta comprueban que hay seis números y métricas.

## Ejemplo de KNN con Ollama

`ejemplos/knn_con_ollama.py` ejecuta KNN de regresión con scikit-learn sobre `ejemplos/datos_knn_5000_outliers.csv`. Después, envía a Ollama solo un resumen de las métricas para obtener una explicación en lenguaje natural; Ollama no entrena KNN ni recibe las filas del CSV.

Con Ollama instalado y en ejecución, descarga un modelo local y lanza el ejemplo desde la raíz del proyecto:

```powershell
ollama pull llama3.2
python ejemplos/knn_con_ollama.py
```

Para probar la alternativa de escalado robusto ante outliers: `python ejemplos/knn_con_ollama.py --scaler robust`. Para cambiar el modelo, indica `--model nombre:tag` o establece `OLLAMA_MODEL`. La conexión usa la API local de Ollama en `http://localhost:11434`.

## NVIDIA Triton: regresores lineal, polinómico y Random Forest

El directorio independiente `triton/` entrena los modelos con las clases de `regression/`, guarda los artefactos y los publica mediante NVIDIA Triton Python backend. El entrenamiento ocurre antes de que Triton cargue sus modelos; las llamadas HTTP solo realizan inferencia.

Desde la raíz del proyecto:

```powershell
docker compose -f triton/compose.yaml up --build
```

Para hacer visible la GPU NVIDIA al contenedor Triton (con NVIDIA Container Toolkit disponible):

```powershell
docker compose -f triton/compose.yaml -f triton/compose.gpu.yaml up --build
```

La API V2 se publica en el puerto 8000; importa `triton/postman/NVIDIA_Triton_Regressors.postman_collection.json` para probar los tres modelos. Consulta `triton/README.md` para entradas, datos y explicación del flujo.

## Regresión polinómica servida con BentoML y Docker

El componente independiente `bentoml/` incluye el entrenamiento, el servicio de inferencia, la imagen Docker y la composición de contenedores. No modifica la API Flask existente. Consulta [`bentoml/README.md`](bentoml/README.md) para el flujo completo y la petición de Postman.

Desde la raíz del proyecto:

```powershell
docker compose -f bentoml/compose.yaml up --build
```

Compose entrena el pipeline de grado 2 con el CSV sintético `ejemplos/datos_ejemplo_xy.csv`, lo registra en el volumen BentoML y después sirve `POST http://localhost:3000/predict`. La ejecución verificada con 1.000 observaciones reservó 200 para evaluación: R² = 0,9287, RMSE = 6,9001 y MAE = 5,5080. Son métricas del conjunto de ejemplo, no de datos reales. La solicitud acepta JSON con `{"x":[40,50,60]}`; la respuesta añade las métricas, el grado y el número de filas usadas para el ajuste final.
