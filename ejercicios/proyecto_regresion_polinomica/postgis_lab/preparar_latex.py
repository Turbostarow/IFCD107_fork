"""Empaqueta la documentacion del laboratorio como proyecto LaTeX editable."""
from pathlib import Path
import re
import shutil

src = Path(__file__).resolve().parent
dest = src.parent / 'latex_postgis_regresion'
dest.mkdir(exist_ok=True)
(dest / 'referencia' / 'sql').mkdir(parents=True, exist_ok=True)

def esc(s):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
                    '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in s)

def inline(s):
    parts = re.split(r'(`[^`]+`|\[[^\]]+\]\([^)]+\))', s)
    result = []
    for p in parts:
        if p.startswith('`'):
            if len(p) > 65:
                result.append(r'\texttt{'+esc(p[1:-1]).replace(r'\_',r'\_\allowbreak{}')+'}')
            else:
                result.append(r'\path|'+p[1:-1]+'|')
        elif re.fullmatch(r'\[[^\]]+\]\([^)]+\)',p):
            m = re.match(r'\[([^\]]+)\]\(([^)]+)\)',p)
            result.append(r'\href{'+m[2]+'}{'+esc(m[1])+'}')
        else:
            result.append(esc(p))
    return ''.join(result)

lines = (src / 'EJERCICIO.md').read_text(encoding='utf-8').splitlines()
body, listing, mode = [], False, None
def close_list():
    global mode
    if mode:
        body.append(r'\end{'+mode+'}')
        mode = None
for line in lines[1:]:
    if line.startswith('```'):
        if listing:
            body.append(r'\end{Verbatim}')
        else:
            close_list()
            body.append(r'\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\small,frame=single]')
        listing = not listing
        continue
    if listing:
        body.append(line)
        continue
    if line.startswith('|'):
        close_list()
        if 'Apartado' in line:
            body.extend([r'\begin{center}\begin{tabularx}{\linewidth}{Xr}',r'\toprule Apartado & Puntos \\ \midrule'])
        elif '---' in line:
            continue
        else:
            cells = [v.strip() for v in line.strip('|').split('|')]
            body.append(inline(cells[0])+' & '+inline(cells[1])+r' \\')
            if cells[0].startswith('Interpretación'):
                body.append(r'\bottomrule\end{tabularx}\end{center}')
        continue
    if line.startswith('## '):
        close_list()
        body.append(r'\section{'+esc(line[3:])+'}')
    elif re.match(r'^\d+\. ',line) or line.startswith('- '):
        want = 'itemize' if line.startswith('- ') else 'enumerate'
        if mode != want:
            close_list()
            mode = want
            body.append(r'\begin{'+mode+'}')
        text = line[2:] if want=='itemize' else re.sub(r'^\d+\. ','',line)
        body.append(r'\item '+inline(text))
    else:
        close_list()
        body.append(inline(line))
close_list()
header = r'''\documentclass[11pt,a4paper]{article}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2.3cm]{geometry}
\usepackage{xcolor,booktabs,tabularx,fvextra,enumitem,amsmath}
\usepackage{hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={Ejercicio PostGIS y regresores lineales}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlist{itemsep=3pt,topsep=4pt}
\emergencystretch=3em
\begin{document}
\begin{titlepage}
\centering
\vspace*{2cm}
{\Large IFCD107\par}
\vspace{1cm}
{\Huge\bfseries PostGIS y regresores lineales\par}
\vspace{0.7cm}
{\LARGE Enunciado e instrucciones de laboratorio\par}
\vspace{1.5cm}
{\large Base de datos espacial, inserción de viviendas, consultas de proximidad e interfaz Python\par}
\vfill
\begin{tabular}{ll}
Nivel & SQL y Python básicos \\
Duración & 3--4 horas \\
Entorno & Docker Desktop, PostGIS, Streamlit y scikit-learn \\
Datos & 2.500 viviendas sintéticas \\
Fecha & 27 de noviembre de 2026
\end{tabular}
\vspace{1cm}
\end{titlepage}
\tableofcontents
\newpage
\section{Enunciado general}
Desarrolla una aplicación que permita almacenar viviendas con ubicación geográfica, insertar observaciones y consultar por proximidad. Utiliza las filas obtenidas para estimar precios mediante regresores lineales y explica sus errores de predicción.

El trabajo se organiza en tres componentes: un contenedor PostGIS que conserva los datos, un contenedor Python que consulta y entrena los modelos y un navegador que presenta la interfaz. La comunicación entre contenedores utiliza el nombre del servicio y el puerto interno 5432. Desde Windows se utiliza localhost:5433 para la base y localhost:8502 para la interfaz.

\subsection{Resultados de aprendizaje}
\begin{itemize}
\item Crear tablas espaciales, restricciones e índices y distinguir grados de metros.
\item Ejecutar inserciones y consultas parametrizadas desde Python.
\item Entrenar y evaluar regresión lineal, Ridge, Lasso y Elastic Net sin fuga del escalado.
\item Interpretar coeficientes, residuos y limitaciones de una evaluación espacial.
\end{itemize}
\subsection{Modalidad de trabajo}
Para practicar la construcción, crea tu propia implementación siguiendo los requisitos y consulta los archivos de referencia al finalizar cada fase. Para la demostración guiada, arranca la solución ya incluida en el proyecto y completa las preguntas y evidencias solicitadas. El ZIP documental no sustituye al repositorio ejecutable.
'''
tail = r'''
\section{Lista final de comprobación}
\begin{enumerate}
\item Los dos contenedores responden y la carga inicial contiene 2.500 viviendas.
\item Una vivienda manual se inserta y las referencias duplicadas se gestionan.
\item Un radio pequeño devuelve menos filas y una zona sin viviendas muestra un resultado vacío.
\item Los cuatro modelos generan métricas y gráficas con datos de prueba.
\item El informe declara filtros, partición y parámetros y no presenta datos sintéticos como reales.
\end{enumerate}
\appendix
\section{Archivos de referencia incluidos}
La carpeta \path|referencia/| contiene copias de los archivos SQL y Python del laboratorio. Son material de consulta y están separados de los fuentes de este documento. El archivo \path|README.txt| explica cómo compilar y dónde arrancar el proyecto ejecutable.
\subsection{Consultas SQL de ejemplo}
Ejecuta las sentencias por separado. La inserción del ejemplo utiliza una referencia única y falla si se repite sin cambiarla.
\VerbatimInput[breaklines=true,breakanywhere=true,fontsize=\footnotesize,frame=single]{referencia/sql/consultas.sql}
\end{document}
'''
tex = header+'\n'.join(body)+tail
tex = tex.replace('–','--').replace('−','-').replace('€','EUR').replace('²',r'\textsuperscript{2}')
(dest / 'main.tex').write_text(tex,encoding='utf-8')
for name in ['app.py','db.py','modelos.py','verificar.py','requirements.txt','Dockerfile']:
    shutil.copy2(src / name, dest / 'referencia' / name)
for name in ['001_init.sql','consultas.sql']:
    shutil.copy2(src / 'sql' / name, dest / 'referencia' / 'sql' / name)
(dest / 'README.txt').write_text('''PROYECTO LATEX: POSTGIS Y REGRESORES LINEALES

Archivo principal: main.tex. Codificacion UTF-8.
Compilar desde esta carpeta con:
  tectonic main.tex
o dos veces con:
  pdflatex -interaction=nonstopmode -halt-on-error main.tex

Overleaf: subir el ZIP, seleccionar main.tex como documento principal y pdfLaTeX.
La carpeta referencia debe conservarse para incluir las consultas del apendice.

El documento explica el laboratorio ejecutable situado en:
D:\\onedrive\\Documentos\\ChatGPT\\ifcd107\\ejercicios\\proyecto_regresion_polinomica
Desde esa carpeta: docker compose up -d --build postgis-lab postgis-lab-ui
Interfaz: http://localhost:8502

El ZIP es documental e incluye copias de referencia de la solucion.
Los servicios se definen en compose.yaml del proyecto original.
No incluye credenciales del entorno ni datos privados.
''',encoding='utf-8')
print(dest)
