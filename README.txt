# Machine Learning — Jupyter Demos y Guía Docente

Repositorio educativo con notebooks Jupyter, material de apoyo en LaTeX e
infografías para introducir y practicar algoritmos fundamentales de
Machine Learning.

## Contenido

El repositorio incluye demostraciones independientes de:

1. Regresión lineal
2. Regresión polinómica
3. Regresión logística
4. K-means
5. K-Nearest Neighbors (K-NN)
6. Máquinas de Vectores de Soporte (SVM)
7. Random Forest

Además, se proporciona una guía docente en LaTeX que explica los conceptos,
la formulación matemática, el funcionamiento de los algoritmos, las
actividades prácticas y una propuesta de evaluación.

## Estructura

```text
.
├── README.txt
├── LICENSE
├── notebooks/
│   ├── 00_Indice_Laboratorio_ML.ipynb
│   ├── 01_Regresion_Lineal.ipynb
│   ├── 02_Regresion_Polinomica.ipynb
│   ├── 03_Regresion_Logistica.ipynb
│   ├── 04_Kmeans.ipynb
│   ├── 05_KNN.ipynb
│   ├── 06_SVM.ipynb
│   └── 07_Random_Forest.ipynb
├── latex/
│   └── Guia_Laboratorio_Jupyter_Machine_Learning.tex
└── imagenes/
    ├── regrliean.png
    ├── regrpolinomica.png
    ├── regrlogistica.png
    ├── K-means.png
    ├── K-nearest neighbors.png
    ├── svm.png
    └── rf.png
```

## Requisitos

Los notebooks están desarrollados para Python 3 y utilizan principalmente:

- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook o JupyterLab

Instalación:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Para ejecutar Jupyter:

```bash
jupyter notebook
```

o:

```bash
jupyter lab
```

## Orden recomendado

Se recomienda seguir el siguiente itinerario:

```text
Regresión lineal
       ↓
Regresión polinómica
       ↓
Regresión logística
       ↓
K-means
       ↓
K-NN
       ↓
SVM
       ↓
Random Forest
```

La secuencia permite avanzar desde modelos sencillos e interpretables hacia
métodos basados en distancia, margen, kernels y ensembles.

## Características de las demos

Cada notebook está pensado como una demostración autónoma y contiene:

- Introducción conceptual.
- Generación o carga de datos de ejemplo.
- Visualización de los datos.
- Entrenamiento del algoritmo.
- Predicciones.
- Métricas de evaluación.
- Visualizaciones de resultados.
- Exploración de parámetros.
- Preguntas de reflexión.

Los ejemplos utilizan datos sintéticos con finalidad docente. Los resultados
obtenidos no deben interpretarse como resultados experimentales de un estudio
científico.

## Guía LaTeX

El repositorio incluye una guía complementaria:

```text
Guia_Laboratorio_Jupyter_Machine_Learning.tex
```

Compilación:

```bash
pdflatex Guia_Laboratorio_Jupyter_Machine_Learning.tex
pdflatex Guia_Laboratorio_Jupyter_Machine_Learning.tex
```

La guía explica:

- fundamentos de cada algoritmo;
- formulación matemática;
- funcionamiento paso a paso;
- actividades;
- preguntas de reflexión;
- comparación de algoritmos;
- actividad integradora;
- criterios de evaluación;
- checklist de entrega.

## Contexto investigador

Este material ha sido desarrollado en el contexto de la experiencia docente
e investigadora de **José Manuel Aroca Fernández**, investigador y docente
universitario vinculado a la **Universidad de Burgos (UBU)**.

Su experiencia académica y técnica incluye trabajo y docencia relacionados
con:

- Inteligencia Artificial y Machine Learning.
- Minería de datos.
- Visualización multidimensional de datos.
- Construcción de ensembles.
- Selección de instancias y características.
- Árboles de decisión y modelos de regresión.
- Bioinformática.
- Programación y desarrollo de software.
- Python y Java.
- CI/CD y herramientas de desarrollo.
- Aplicaciones de Machine Learning sobre datos científicos y ambientales.

Entre las líneas de trabajo desarrolladas se encuentra la aplicación de
Machine Learning y datos de teledetección a problemas de inferencia de
propiedades del suelo, incluyendo el proyecto/plataforma **WALGREEN /
SEN4FARMING**, orientado a la inferencia de Soil Organic Carbon (SOC)
mediante datos de observación de la Tierra y modelos de aprendizaje
automático.

La experiencia investigadora también comprende el estudio de ensembles,
selección de características, explicabilidad de modelos, evaluación de
modelos y aplicación de técnicas de Machine Learning a conjuntos de datos
heterogéneos.

Este repositorio tiene una finalidad principalmente educativa y pretende
servir como material reproducible para cursos, talleres y actividades
prácticas de Inteligencia Artificial y Machine Learning.

## Autor

**José Manuel Aroca Fernández**

Universidad de Burgos (UBU)

Investigador y docente universitario

Áreas de interés:

- Artificial Intelligence
- Machine Learning
- Data Mining
- Data Visualization
- Ensemble Learning
- Feature Selection
- Instance Selection
- Bioinformatics
- Remote Sensing
- Soil Property Prediction

## Cita y atribución

Si utilizas este material en un curso, repositorio, presentación o trabajo
académico, se recomienda citar al autor:

```text
Aroca Fernández, José Manuel.
Machine Learning — Jupyter Demos y Guía Docente.
Repositorio educativo de algoritmos de Machine Learning.
```

Si el repositorio incorpora posteriormente un DOI o una publicación
asociada, se recomienda utilizar dicha referencia bibliográfica en lugar de
esta referencia genérica.

## Licencia

Copyright (c) 2026 José Manuel Aroca Fernández

Este proyecto se distribuye bajo la licencia MIT.

La licencia permite utilizar, copiar, modificar, fusionar, publicar,
distribuir, sublicenciar y vender copias del software y del material
correspondiente, siempre que se conserve el aviso de copyright y la
notificación de licencia.

El texto completo de la licencia se incluye a continuación y puede
almacenarse también como archivo independiente `LICENSE`.

--------------------------------------------------------------------------

MIT License

Copyright (c) 2026 José Manuel Aroca Fernández

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------------------------

## Nota sobre material de terceros

Las bibliotecas de Python, Jupyter, scikit-learn, NumPy, pandas y Matplotlib
mantienen sus propias licencias.

Si se incorporan en futuras versiones datasets, imágenes, iconos, textos o
recursos procedentes de terceros, deberán respetarse las condiciones de
licencia y atribución correspondientes. La licencia MIT de este repositorio
no modifica las licencias de esos materiales de terceros.

## Contribuciones

Las propuestas de mejora son bienvenidas. Las contribuciones deberían
mantener:

- código reproducible;
- documentación clara;
- ejemplos sencillos;
- separación entre datos de ejemplo y resultados experimentales;
- indicación de las dependencias necesarias;
- respeto a las licencias de terceros.

## Finalidad educativa

Este repositorio está diseñado como apoyo a la enseñanza y aprendizaje de
Machine Learning. Las demos simplifican deliberadamente algunos aspectos
de los problemas reales para facilitar la comprensión de los algoritmos.

Antes de utilizar estos métodos en proyectos reales se recomienda estudiar
la calidad de los datos, validación, selección de variables, ajuste de
hiperparámetros, incertidumbre, generalización, reproducibilidad y
limitaciones específicas del problema.
