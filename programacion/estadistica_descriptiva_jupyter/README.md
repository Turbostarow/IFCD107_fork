# Proyecto Jupyter — Estadística Descriptiva

Proyecto educativo autocontenido para introducir la estadística descriptiva
mediante Python y Jupyter Notebook.

## Contenido

- `estadistica_descriptiva.ipynb`: notebook completo.
- `datos_estadistica_descriptiva.csv`: dataset inicial sintético.
- `README.md`: instrucciones del proyecto.

## Requisitos

Python 3.10 o superior recomendado.

Bibliotecas:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

También puede utilizarse:

```bash
pip install notebook
```

## Ejecución

Desde esta carpeta:

```bash
jupyter notebook
```

o:

```bash
jupyter lab
```

Abrir:

```text
estadistica_descriptiva.ipynb
```

## Dataset

El fichero CSV contiene 120 observaciones sintéticas con las siguientes
variables:

| Variable | Descripción |
|---|---|
| id | Identificador |
| edad | Edad en años |
| altura_m | Altura en metros |
| peso_kg | Peso en kilogramos |
| imc | Índice de masa corporal |
| empleo | Sector profesional |
| actividad_fisica | Nivel de actividad |
| horas_estudio_semana | Horas de estudio semanales |
| satisfaccion | Satisfacción de 1 a 10 |
| ingresos_euros | Ingresos mensuales |

Los datos no representan personas reales.

## Contenidos estadísticos

El notebook incluye:

- exploración de datos;
- tipos de variables;
- frecuencias;
- media, mediana y moda;
- varianza y desviación estándar;
- rango;
- percentiles;
- cuartiles;
- rango intercuartílico;
- coeficiente de variación;
- asimetría;
- curtosis;
- detección de posibles valores atípicos mediante IQR;
- histogramas;
- KDE;
- boxplots;
- gráficos de barras;
- gráficos de dispersión;
- matriz de correlaciones;
- mapa de calor;
- interpretación de resultados;
- ejercicios propuestos.

## Objetivo docente

El proyecto está pensado como material práctico para una sesión de
introducción a la estadística aplicada al análisis de datos y a la
Inteligencia Artificial.

Se recomienda ejecutar las celdas secuencialmente y modificar posteriormente
los parámetros de los gráficos para experimentar con diferentes
representaciones.

## Licencia

MIT License.

Copyright (c) 2026 José Manuel Aroca Fernández.
