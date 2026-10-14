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

Endpoints: `GET /api/v1/health` (público), `POST /api/v1/auth/token` (emite token), `POST /api/v1/regresion/polinomica`, `POST /api/v1/regresion/polinomica/csv`, `POST /api/v1/random-forest/terrazas` y `POST /api/v1/random-forest/csv` (rutas de entrenamiento requieren Bearer). Los tokens expiran por defecto en una hora; configura `API_TOKEN_TTL_SECONDS` para cambiarlo.


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
