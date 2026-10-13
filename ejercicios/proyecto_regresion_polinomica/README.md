# API Flask de regresión polinómica

Ejemplo ejecutable con el algoritmo separado en `regression/polynomial_regressor.py`. La guía completa y las instrucciones de Postman están en `documentacion/guia_proyecto.tex`.

El CSV incluido tiene 1.000 filas sintéticas para una demostración de gestión de expedientes de terrazas. No son registros, plazos ni resultados reales del Ayuntamiento. En el ajuste, `x` es un índice didáctico de complejidad (construido desde superficie, tipo de trámite y requerimientos documentales) e `y` son días de tramitación simulados. Los campos descriptivos adicionales se conservan en el CSV de entrada; el modelo ajusta las columnas `x` e `y`.

## Arranque

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

En macOS/Linux, activar con `source .venv/bin/activate`. La API queda en `http://127.0.0.1:5000`.

Importa `postman/Regresion_Polinomica.postman_collection.json` en Postman. Para la petición de CSV, selecciona `ejemplos/datos_ejemplo.csv` en el campo `file` y usa la referencia `terrazas_madrid_demo`.

Para regenerar el conjunto determinista de 1.000 filas: `python scripts/generate_terrace_cases.py`.

Los artefactos se crean bajo `runs/<reference>/` y se dividen en `graficas/`, `csv_entrada/`, `logs_salida/`, `csv_salida/` y `modelos/`. El ajuste entrenado se guarda como `modelos/<run_id>_modelo.pkt`; para cargarlo y predecir, usa `PolynomialRegressor.load(ruta).predict(x)` desde Python con el proyecto instalado. El `.pkt` se serializa con pickle: carga solo archivos de confianza.

Endpoints: `GET /api/v1/health`, `POST /api/v1/regresion/polinomica` (JSON), `POST /api/v1/regresion/polinomica/csv` (multipart/form-data).
