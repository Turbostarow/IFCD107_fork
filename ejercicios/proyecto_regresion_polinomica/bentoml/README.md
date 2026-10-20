# Regresión polinómica con BentoML y Docker

Este componente es independiente de la API Flask del proyecto. Mantiene la distribución por responsabilidades: `train_model.py` prepara, evalúa y registra el modelo; `service.py` expone la predicción; `Dockerfile` construye la imagen; `compose.yaml` orquesta entrenamiento y servicio; `requirements.txt` contiene las dependencias propias.

## Cómo funciona

1. El contenedor de entrenamiento lee `../ejemplos/datos_ejemplo_xy.csv` (copiado a la imagen en `/app/data`).
2. Divide los datos en entrenamiento y prueba (80/20), ajusta una regresión polinómica de grado 2 y calcula R², RMSE y MAE sobre el conjunto reservado.
3. Ajusta el modelo de servicio con todos los datos y lo guarda en el almacén de modelos de BentoML.
4. Cuando termina correctamente el entrenamiento, Compose inicia el servidor BentoML en el puerto 3000. Ambos contenedores comparten el volumen `bentoml_model_store`.

El conjunto incluido es didáctico y sintético. Las métricas sirven para practicar el flujo; no representan resultados de un proceso real. El grado por defecto es 2 y se puede cambiar con `POLYNOMIAL_DEGREE` (rango permitido 1–10). `POLYNOMIAL_RANDOM_STATE` controla la partición reproducible.

## Arranque

Desde la raíz `proyecto_regresion_polinomica`:

```powershell
docker compose -f bentoml/compose.yaml up --build
```

La primera construcción descarga la imagen de Python y las dependencias. El servicio queda en `http://localhost:3000`; consulta `/docs` para la interfaz OpenAPI.

## Solicitar predicciones

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:3000/predict `
  -ContentType 'application/json' `
  -Body '{"x":[40,50,60]}'
```

La respuesta incluye `predictions`, el grado del polinomio, el número de filas de entrenamiento y las métricas de validación del holdout. La API valida que `x` no esté vacío y que sus elementos sean números finitos. También puedes importar `BentoML_Polynomial_Regression.postman_collection.json`; contiene la misma petición de ejemplo.

## Parada y reconstrucción

```powershell
docker compose -f bentoml/compose.yaml down
```

El volumen persiste el modelo registrado. Para eliminar también ese artefacto y forzar un entrenamiento desde cero:

```powershell
docker compose -f bentoml/compose.yaml down --volumes
docker compose -f bentoml/compose.yaml up --build
```

Al ser un proyecto educativo, la API no añade autenticación. No expongas este servicio a una red no confiable sin incorporar autenticación, límites, controles de acceso y revisión de seguridad.
