# Ejercicio: PostGIS, consultas espaciales y regresores lineales

Duración orientativa: 3–4 horas. Requisitos: Docker Desktop con contenedores Linux, SQL básico y Python básico. Se incluye una solución ejecutable para comparar con tu implementación.

## Caso de aprendizaje

Una oficina quiere explorar viviendas y estimar su precio a partir de superficie, habitaciones, antigüedad y distancia al centro de Madrid. Los 2.500 registros son sintéticos. No representan anuncios ni precios de mercado.

El precio simulado es aproximadamente `80000 + 2800*superficie + 12000*habitaciones - 900*antiguedad - 14000*distancia_centro_km + ruido`, con un suelo de 10.000 €. Las coordenadas cubren un rectángulo cercano a Madrid. El ruido es centrado, con desviación aproximada de 12.728 €. La semilla SQL fija la generación para una misma versión de PostgreSQL y orden de ejecución.

## 1. Arrancar los nuevos contenedores

Desde PowerShell:

```powershell
cd D:\onedrive\Documentos\ChatGPT\ifcd107\ejercicios\proyecto_regresion_polinomica
docker compose up -d --build postgis-lab postgis-lab-ui
docker compose ps postgis-lab postgis-lab-ui
```

Abre http://localhost:8502. La base está en `localhost:5433`, base `georegresion`, usuario `alumno`, contraseña de laboratorio `postgis_lab_local`. Puedes configurar `POSTGIS_LAB_PORT`, `POSTGIS_LAB_UI_PORT` y `POSTGIS_LAB_PASSWORD` en el `.env` existente. La contraseña debe coincidir con la del volumen ya inicializado: cambiar la variable no modifica usuarios existentes.

El Compose principal requiere también sus variables `API_ADMIN_PASSWORD` y `API_TOKEN_SECRET`, aunque se arranquen solo estos dos servicios. Si falta `.env`, copia `.env.example` y completa los valores exigidos. No compartas ese archivo. Los puertos del laboratorio solo escuchan en localhost.

## 2. Crear y estudiar el esquema (30 minutos)

Examina `sql/001_init.sql`. Explica la extensión PostGIS, clave primaria, referencia única, restricciones y columna `geography(Point,4326)`. Justifica el índice espacial GiST.

```powershell
docker compose exec postgis-lab psql -U alumno -d georegresion
```

Dentro de psql:

```sql
SELECT PostGIS_Version();
\d aprendizaje.viviendas
SELECT count(*) FROM aprendizaje.viviendas;
SELECT referencia, ST_AsText(ubicacion::geometry) FROM aprendizaje.viviendas LIMIT 5;
```

Resultado inicial esperado: 2.500 filas, SRID 4326. El orden de `ST_MakePoint` es longitud, latitud. En geometría 4326 las coordenadas están en grados; para radios en metros usamos `geography`.

## 3. Insertar y consultar (45 minutos)

1. Inserta tres viviendas desde la pestaña de inserción, con referencias distintas.
2. Ejecuta las consultas de `sql/consultas.sql` por separado. Su inserción de ejemplo añade otra fila.
3. Obtén las viviendas a menos de 2 km del centro y ordénalas por distancia.
4. Calcula precio medio y número de viviendas por anillos de 1 km.
5. Compara consultas de radio 500 m, 2 km y 10 km. Explica los cambios de población.
6. Ejecuta `EXPLAIN (ANALYZE, BUFFERS)` y explica por qué disponer de índice no obliga al planificador a utilizarlo.

Implementa tus consultas en Python con parámetros `%s`, como en `db.py`. No construyas SQL concatenando entradas del usuario. Las transacciones confirman una inserción correcta o revierten el error.

## 4. Construir la interfaz (45 minutos)

En una copia de trabajo de `app.py`, implementa controles de longitud, latitud, radio y superficie mínima. Presenta tabla, mapa y descarga CSV. Añade un formulario de inserción con validación y tratamiento de referencias duplicadas. Comprueba un filtro sin resultados y que el programa pueda seguir funcionando.

El mapa puede usar Internet para descargar teselas; la tabla, consultas y modelos no dependen de ellas. `db.py` calcula la distancia al centro fijo de Madrid; cambiar el centro del filtro no redefine esa variable del modelo.

## 5. Entrenar y comparar (45 minutos)

Usa `superficie`, `habitaciones`, `antiguedad` y `distancia_centro_km` para predecir `precio`. Excluye precio, referencia e id de las entradas. Sigue este orden:

1. Divide entrenamiento y prueba, semilla 42, 25% de prueba.
2. Ajusta `StandardScaler` solo sobre entrenamiento mediante Pipeline.
3. Entrena Lineal, Ridge, Lasso y Elastic Net.
4. Registra R², RMSE, MAE, Pearson y bias. En la interfaz, bias = media(predicho − real); el residuo mostrado es real − predicho.
5. Explica el gráfico real/predicho, residuos y distribución de residuos.
6. Compara coeficientes en unidades originales e intercepto con la fórmula generadora. No esperes igualdad exacta por el ruido y el suelo del precio.

R² puede ser negativo; Pearson puede quedar sin definir con predicciones constantes. RMSE, MAE y bias se expresan en euros. El escalado hace comparable la penalización; los coeficientes visualizados se transforman de vuelta a sus unidades originales.

Prueba varias intensidades de penalización, pero selecciona hiperparámetros mediante validación cruzada dentro de entrenamiento. Reserva la prueba para la comparación final. Los controles son didácticos: mirar repetidamente la prueba para elegir valores produce una evaluación optimista. Con observaciones reales próximas espacialmente, considera una separación por zonas para medir generalización geográfica.

## Entregables y evaluación

Entrega SQL, código Python, capturas de consultas y mapa, tabla comparativa de métricas y un informe de una página. Incluye las consultas usadas y los filtros, número de registros, partición, parámetros y limitaciones.

| Apartado | Puntos |
|---|---:|
| Esquema, SRID, restricciones e índice | 2 |
| Inserciones y consultas espaciales correctas | 2 |
| Interfaz, parámetros SQL y errores | 2 |
| Modelos, separación de datos y métricas | 3 |
| Interpretación y reproducibilidad | 1 |

Ampliación: añade validación cruzada, separa por cuadrículas espaciales y compara una regresión simple usando solo superficie con la múltiple. Razona si los coeficientes permiten afirmar causalidad.

## Solución y mantenimiento

Comprobación de integración (base, consulta vacía, inserción con rollback, cuatro modelos e interfaz):

```powershell
docker compose exec postgis-lab-ui python verificar.py
```

`db.py` contiene acceso SQL; `modelos.py`, entrenamiento y evaluación; `app.py`, interfaz. El inicializador crea la extensión y carga datos únicamente cuando el volumen está vacío. Para recuperar filas simuladas ausentes sin borrar las inserciones manuales:

```powershell
docker compose exec postgis-lab psql -U alumno -d georegresion -f /docker-entrypoint-initdb.d/001_init.sql
```

La referencia única evita duplicados del lote. Para detener solamente el ejercicio conservando datos:

```powershell
docker compose stop postgis-lab-ui postgis-lab
```

No uses `docker compose down -v`: borraría también otros volúmenes del proyecto.

## Referencias

- [PostGIS: ST_DWithin](https://postgis.net/docs/ST_DWithin.html).
- [PostGIS: ST_Distance](https://postgis.net/docs/ST_Distance.html).
- [scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
- [scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
