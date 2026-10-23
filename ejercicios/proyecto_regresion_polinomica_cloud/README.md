# Random Forest como servicio en Vertex AI (IFCD107)

Ejemplo independiente basado en `proyecto_regresion_polinomica`, reducido a un único modelo: `RandomForestRegressor`. El contenedor implementa el protocolo de predicción de Vertex AI: `GET /health` y `POST /predict` con JSON `{ "instances": [...] }`.

## Qué contiene

- `src/cloud_rf/model.py`: preprocesamiento, evaluación holdout y entrenamiento final del pipeline.
- `src/cloud_rf/app.py`: API Flask compatible con el contenedor personalizado de Vertex AI.
- `scripts/generate_data.py`: genera un CSV sintético reproducible con columnas numéricas y categóricas.
- `train_model.py`: entrena, calcula métricas y serializa el modelo.
- `tests/`: tests automatizados del entrenamiento y del contrato HTTP.
- `Dockerfile`: crea una imagen de servicio y entrena el ejemplo dentro de la construcción.
- `.github/workflows/vertex-ai.yml`: ejecuta tests, publica la imagen en Artifact Registry y permite desplegar manualmente el modelo.

El entrenamiento dentro del `Dockerfile` es una simplificación pedagógica para que la imagen sea autocontenida. En un sistema real, entrenamiento, validación, registro de artefactos y despliegue se versionan como etapas independientes.

## Ejecutar localmente

Requiere Python 3.11 o compatible.

```powershell
cd ejercicios/proyecto_regresion_polinomica_cloud
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "src"
python scripts/generate_data.py
python train_model.py
python -m pytest -q
python -m cloud_rf.app
```

El último comando puede ejecutarse en otra terminal después de generar `artifacts/model.joblib`:

```powershell
$env:PYTHONPATH = "src"
$env:MODEL_PATH = "artifacts/model.joblib"
python -m cloud_rf.app
```

Petición local de predicción:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8080/predict `
  -ContentType application/json `
  -Body '{"instances":[{"superficie":85,"habitaciones":3,"antiguedad":12,"zona":"centro"}]}'
```

Respuesta: `{"predictions":[...]}`. El orden y nombres de características deben coincidir con el CSV de entrenamiento. El servicio rechaza columnas que faltan o que sobran.

## Construir y probar el contenedor localmente

```powershell
docker build -t random-forest-ifcd107:local .
docker run --rm -p 8080:8080 random-forest-ifcd107:local
```

En otra consola, solicitar `GET http://localhost:8080/health` y después enviar la petición JSON anterior.

## Crear los recursos de Google Cloud

### 1. Proyecto y APIs

Instalar y configurar Google Cloud CLI, iniciar sesión y seleccionar un proyecto de facturación habilitada. Los siguientes comandos son una plantilla; cambia `PROJECT_ID`, `REGION` y `REPOSITORY` por valores propios:

```bash
gcloud auth login
gcloud config set project PROJECT_ID
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com
gcloud artifacts repositories create REPOSITORY \
  --repository-format=docker --location=REGION \
  --description="Imágenes del ejemplo IFCD107"
```

Usa la misma región para Vertex AI y Artifact Registry si está disponible. La ubicación elegida debe admitir Vertex AI, Artifact Registry y el tipo de máquina deseado.

### 2. Identidad federada para GitHub Actions

La acción oficial `google-github-actions/auth` recomienda Workload Identity Federation (WIF), que evita guardar claves JSON permanentes en GitHub. Un administrador de Google Cloud debe crear un pool y un proveedor OIDC que limite la confianza al repositorio esperado. Ejemplo de comandos (sustituye organización/usuario, repositorio y número de proyecto):

```bash
gcloud iam workload-identity-pools create github-pool \
  --project=PROJECT_ID --location=global \
  --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=PROJECT_ID --location=global --workload-identity-pool=github-pool \
  --display-name="IFCD107 repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='OWNER/REPOSITORY'"

gcloud iam workload-identity-pools providers describe github-provider \
  --project=PROJECT_ID --location=global --workload-identity-pool=github-pool \
  --format='value(name)'
```

Crear una cuenta de servicio exclusiva para CI y limitar los permisos al proyecto de práctica. Como punto de partida para esta demo necesita publicar en Artifact Registry y subir/desplegar modelos en Vertex AI (roles `roles/artifactregistry.writer` y `roles/aiplatform.user`). Evita conceder `Owner`:

```bash
gcloud iam service-accounts create github-ci \
  --project=PROJECT_ID --display-name="GitHub Actions IFCD107"
GITHUB_SA="github-ci@PROJECT_ID.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:${GITHUB_SA}" --role=roles/aiplatform.user
gcloud artifacts repositories add-iam-policy-binding REPOSITORY \
  --project=PROJECT_ID --location=REGION \
  --member="serviceAccount:${GITHUB_SA}" --role=roles/artifactregistry.writer
```

Vincular el principal federado del repositorio a la cuenta de servicio con `roles/iam.workloadIdentityUser`:

```bash
PROJECT_NUMBER="$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')"
gcloud iam service-accounts add-iam-policy-binding GITHUB_SA@PROJECT_ID.iam.gserviceaccount.com \
  --project=PROJECT_ID --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/OWNER/REPOSITORY"
```

El proveedor y la condición deben restringirse a la rama/evento aprobado conforme a la política del centro. La identidad de Vertex AI que despliega el contenedor también debe poder leer la imagen del Artifact Registry; configura `roles/artifactregistry.reader` para la cuenta de servicio de Vertex si el acceso del proyecto no lo concede ya.

Para otorgar lectura explícita a la identidad de servicio de Vertex AI en el repositorio:

```bash
gcloud services identity create --service=aiplatform.googleapis.com --project=PROJECT_ID
VERTEX_AGENT="service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"
gcloud artifacts repositories add-iam-policy-binding REPOSITORY \
  --project=PROJECT_ID --location=REGION \
  --member="serviceAccount:${VERTEX_AGENT}" --role=roles/artifactregistry.reader
```

### 3. Variables del repositorio de GitHub

En **Settings → Secrets and variables → Actions → Variables**, crear:

| Variable | Ejemplo | Significado |
|---|---|---|
| `GCP_PROJECT_ID` | `mi-proyecto-ifcd107` | ID del proyecto Google Cloud. |
| `GCP_REGION` | `europe-west4` | Región de Artifact Registry y Vertex AI. |
| `ARTIFACT_REPOSITORY` | `ifcd107-images` | Repositorio Docker que se creó antes. |
| `GCP_WIF_PROVIDER` | `projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider` | Nombre completo del proveedor OIDC. |
| `GCP_SERVICE_ACCOUNT` | `github-ci@mi-proyecto-ifcd107.iam.gserviceaccount.com` | Cuenta federada de CI. |
| `VERTEX_MACHINE_TYPE` | `n1-standard-2` | Tipo inicial de instancia Vertex AI, sujeto a disponibilidad regional. |

Crear también un entorno de GitHub llamado `vertex-ai-deploy` en **Settings → Environments** y configurar aprobación manual/revisores. El workflow publica imágenes con la etiqueta inmutable del SHA del commit. Un `push` a `main` ejecuta tests y publica imagen. El despliegue de Vertex solo se inicia manualmente mediante **Actions → Test, build and deploy Random Forest to Vertex AI → Run workflow**, activando `deploy_to_vertex`; puede indicarse un endpoint existente o dejarlo vacío para crear uno.

### 4. Ejecutar el pipeline y comprobar el despliegue

1. Abrir una Pull Request: GitHub ejecuta `pytest`; no sube imagen ni crea recursos Vertex.
2. Al integrar en `main`, GitHub construye y publica la imagen en Artifact Registry.
3. Para desplegar, iniciar el workflow manual con `deploy_to_vertex=true`. El job espera la aprobación del entorno protegido, sube un Model a Vertex AI y lo despliega con una réplica mínima y máxima.
4. Consultar endpoints y modelos:

```bash
gcloud ai endpoints list --region=REGION
gcloud ai models list --region=REGION
```

5. Invocar una predicción con una identidad que tenga permiso para llamar al endpoint:

```bash
gcloud ai endpoints predict ENDPOINT_ID --region=REGION \
  --json-request=instances.json
```

Contenido de `instances.json`:

```json
{
  "instances": [
    {"superficie": 85, "habitaciones": 3, "antiguedad": 12, "zona": "centro"}
  ]
}
```

## Retirar recursos y evitar costes

Un endpoint online desplegado puede generar coste mientras mantiene réplicas, incluso sin peticiones. Tras la práctica, primero averigua los modelos desplegados y sus IDs:

```bash
gcloud ai endpoints describe ENDPOINT_ID --region=REGION
```

Despliegue de prueba: `gcloud ai endpoints undeploy-model ENDPOINT_ID --region=REGION --deployed-model-id=DEPLOYED_MODEL_ID`. Después de confirmar que ya no se necesita, borrar el endpoint: `gcloud ai endpoints delete ENDPOINT_ID --region=REGION`. Los modelos registrados e imágenes de Artifact Registry son recursos separados; borrar solo los que ya no se necesiten. Al desplegar una nueva versión en un endpoint ya usado, la versión anterior puede seguir ocupando réplicas; revisa `describe` y retira explícitamente las versiones que ya no necesites. Para las prácticas, usa un endpoint dedicado al ejercicio. Consultar precios oficiales y presupuestos/alertas antes de ejecutar.

## Notas técnicas y límites

- Endpoint Vertex custom container: entrada `POST /predict`, formato `{"instances": [...]}`; salida `{"predictions": [...]}`; comprobación `GET /health`.
- La imagen se ejecuta en el puerto 8080 y serializa el modelo junto con el pipeline de transformación.
- El conjunto sintético es solo demostrativo; no tiene valor estadístico ni representa precios reales.
- La validación holdout informa R², RMSE y MAE; no es un sustituto de validación temporal, evaluación independiente ni revisión de sesgos.
- La demo no incorpora autorización propia para la API local. En Vertex, la llamada se controla mediante IAM; no publicar el contenedor directamente sin una capa de autenticación apropiada.
- La acción de CI no usa llaves JSON duraderas. Las variables de ejemplo deben configurarse antes de que el job de publicación pueda autenticar.
- Las dependencias están acotadas por rangos principales; para producción se recomienda generar un lockfile y fijar también imágenes base por digest.

## Referencias oficiales

- Vertex AI, [usar un contenedor personalizado](https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container).
- Vertex AI, [formato de predicción online](https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions).
- Google Cloud CLI, [subir un modelo](https://cloud.google.com/sdk/gcloud/reference/ai/models/upload) y [desplegar en un endpoint](https://cloud.google.com/sdk/gcloud/reference/ai/endpoints/deploy-model).
- GitHub Actions, [autenticación oficial Google Cloud con WIF](https://github.com/google-github-actions/auth).
- Artifact Registry, [publicar imágenes Docker](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling).
- Vertex AI, [precios](https://cloud.google.com/vertex-ai/pricing).
