"""Curvas comunes calculadas sin recibir el conjunto de prueba externo."""
import copy
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import KFold, TimeSeriesSplit


def calcular_curva(n_muestras, evaluar, parametros, tipo, valores, parametro=None,
                   folds=2, semilla=42, cronologico=False, progreso=None):
    """evaluar(train, val, params) devuelve RMSE train, RMSE val, n_fit real.

    Todos los índices son relativos al pool de desarrollo: el llamador excluye
    la prueba externa antes de proporcionar datos al evaluador.
    """
    if n_muestras < 12:
        raise ValueError('Necesitas al menos 12 muestras de desarrollo para estas curvas.')
    splitter = TimeSeriesSplit(n_splits=folds) if cronologico else KFold(folds, shuffle=True, random_state=semilla)
    particiones = list(splitter.split(np.arange(n_muestras)))
    filas = []
    for i, valor in enumerate(valores):
        for fold, (train, val) in enumerate(particiones):
            p = copy.deepcopy(parametros)
            if tipo == 'validacion':
                p[parametro] = valor
            elif tipo == 'aprendizaje':
                orden = train if cronologico else np.random.default_rng(semilla+fold).permutation(train)
                cantidad = max(2, int(len(train)*float(valor)))
                train = orden[-cantidad:] if cronologico else orden[:cantidad]
            else:
                raise ValueError('Tipo de curva desconocido.')
            fila = {'Valor': json.dumps(valor), 'Pliegue': fold+1, 'N solicitado': len(train)}
            try:
                with warnings.catch_warnings(record=True) as avisos:
                    warnings.simplefilter('always')
                    rmse_train, rmse_val, n_fit = evaluar(train, val, p)
                if not np.isfinite([rmse_train, rmse_val]).all():
                    raise ValueError('RMSE no finito.')
                fila.update({'Entrenamiento': float(rmse_train), 'Validación': float(rmse_val),
                             'N ajustado': int(n_fit), 'Estado': 'ok',
                             'Aviso': ' | '.join(sorted({str(a.message) for a in avisos}))})
            except Exception as exc:
                fila.update({'Entrenamiento': np.nan, 'Validación': np.nan, 'N ajustado': np.nan,
                             'Estado': 'error', 'Aviso': str(exc)})
            filas.append(fila)
            if progreso: progreso(len(filas)/(len(valores)*len(particiones)))
    return pd.DataFrame(filas)


SUGERIDOS = {
    'random_forest': ('max_depth', [2, 5, 10]), 'bayesian_ridge': ('alpha_1', [1e-6, 1e-4, .01]),
    'ransac': ('residual_threshold', [1., 3., 6.]), 'huber': ('epsilon', [1.1, 1.35, 1.75]),
    'theil_sen': ('max_subpopulation', [100, 250, 500]), 'decision_tree': ('max_depth', [2, 5, 10]),
    'gradient_boosting': ('n_estimators', [25, 50, 100]), 'svr': ('C', [.1, 1., 10.]),
    'knn': ('n_neighbors', [2, 5, 10]), 'catboost': ('depth', [3, 5, 7]),
    'lightgbm': ('num_leaves', [7, 15, 31]), 'xgboost': ('max_depth', [2, 4, 6]),
    'gaussian_process': ('alpha', [1e-8, 1e-4, .1]), 'mlp': ('alpha', [1e-5, .001, .1]),
    'quantile': ('alpha', [0., .1, 1.]), 'cnn1': ('filters', [4, 8, 16]), 'cnn2': ('filters', [4, 8, 16]),
}


def evaluador_tabular(datos_desarrollo, objetivo, variables, algoritmo, semilla=42):
    from modelo import entrenar
    # Recibe únicamente desarrollo: ninguna operación del evaluador puede usar
    # las filas reservadas para la prueba principal.
    datos = datos_desarrollo.reset_index(drop=True).copy()
    def evaluar(fit, val, p):
        modelo, _, validacion, train = entrenar(datos, objetivo, variables, p, semilla=semilla,
                                               algoritmo=algoritmo, particion=(fit, val))
        return train['RMSE'], validacion['RMSE'], len(modelo.training_indices_)
    return evaluar


def evaluador_cnn(x_desarrollo, y_desarrollo, dimension, semilla=42, cronologico=False):
    from cnn_modelo import entrenar_cnn
    x, y = np.asarray(x_desarrollo), np.asarray(y_desarrollo)
    def evaluar(fit, val, p):
        paquete, _, validacion, train, _ = entrenar_cnn(x, y, dimension, p, semilla=semilla,
                                                       cronologico=cronologico, particion=(fit, val))
        return train['RMSE'], validacion['RMSE'], len(paquete['indices']['train'])
    return evaluar


def mostrar_curvas(clave, parametros, evaluar, n_desarrollo, firma_datos, semilla=42, cronologico=False):
    st.caption(f'Las curvas se calculan únicamente con {n_desarrollo} muestras de desarrollo; la prueba externa está excluida. Cada punto vuelve a ajustar el modelo y su preprocesamiento en cada pliegue.')
    st.write('Validación: cambia un hiperparámetro. Aprendizaje: cambia la cantidad de datos. Se muestra RMSE en unidades del objetivo; menor es mejor.')
    st.caption('Se ejecutan con un botón porque requieren varios entrenamientos. No sustituyen al modelo principal ni modifican sus métricas. Con parada anticipada puede existir además una validación interna dentro de cada pliegue de ajuste.')
    if n_desarrollo < 20:
        st.info('Usa al menos 20 muestras de desarrollo para poder explorar ambas curvas.'); return
    folds = st.selectbox('Pliegues de validación interna', [2, 3, 5], key=clave+':cv_folds')
    if cronologico:
        st.caption('Se utiliza validación progresiva por orden (TimeSeriesSplit); cada validación ocurre después de su entrenamiento. No se añade un gap entre ventanas solapadas.')
    else:
        st.caption('Se utiliza KFold con mezcla reproducible. No es validación temporal ni por grupos.')
    sugerido, candidatos = SUGERIDOS[clave]
    opciones = sorted({sugerido} | {k for k, v in parametros.items() if isinstance(v, (str, int, float, bool)) or v is None})
    parametro = st.selectbox('Hiperparámetro para la curva de validación', opciones, index=opciones.index(sugerido), key=clave+':cv_param')
    texto = st.text_input('Valores candidatos · lista JSON', json.dumps(candidatos) if parametro == sugerido else json.dumps([parametros.get(parametro)]), key=clave+':cv_values:'+parametro)
    fracciones_texto = st.text_input('Fracciones para la curva de aprendizaje · lista JSON', '[0.3, 0.65, 1.0]', key=clave+':lc_values')
    try:
        valores = json.loads(texto)
        fracciones = json.loads(fracciones_texto)
        if not isinstance(valores, list) or not 2 <= len(valores) <= 8:
            raise ValueError('Introduce de 2 a 8 candidatos en una lista JSON.')
        if len({json.dumps(v, sort_keys=True, allow_nan=False) for v in valores}) != len(valores):
            raise ValueError('Los valores candidatos deben ser distintos.')
        if not isinstance(fracciones, list) or not 2 <= len(fracciones) <= 8 or any(isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 < f <= 1 for f in fracciones):
            raise ValueError('Introduce de 2 a 8 fracciones numéricas entre cero (excluido) y uno.')
        if len(set(fracciones)) < 2:
            raise ValueError('Introduce al menos dos fracciones distintas.')
    except (ValueError, TypeError) as exc:
        st.info(str(exc)); return
    for tipo, etiquetas, nombre in [('validacion', valores, 'Curva de validación'), ('aprendizaje', sorted(set(fracciones)), 'Curva de aprendizaje')]:
        st.subheader(nombre)
        firma = hashlib.sha256((firma_datos+json.dumps([parametros, tipo, etiquetas, parametro if tipo == 'validacion' else None, folds, semilla, cronologico], sort_keys=True, default=str)).encode()).hexdigest()
        resultado_key = clave+':curva:'+tipo
        if st.button('Calcular '+nombre.lower(), key=clave+':calcular:'+tipo):
            barra = st.progress(0., text='Ajustando modelos en validación interna…')
            tabla = calcular_curva(n_desarrollo, evaluar, parametros, tipo, etiquetas,
                                   parametro, folds, semilla, cronologico, barra.progress)
            barra.empty()
            st.session_state[resultado_key] = (firma, tabla)
        guardado = st.session_state.get(resultado_key)
        if not guardado or guardado[0] != firma:
            st.caption('Pulsa Calcular para obtener resultados con la configuración actual.'); continue
        tabla = guardado[1]
        completos = tabla.groupby('Valor', sort=False).filter(lambda g: len(g) == folds and (g['Estado'] == 'ok').all())
        if len(completos):
            resumen = completos.groupby('Valor', sort=False).agg(train=('Entrenamiento', 'mean'), val=('Validación', 'mean'),
                                                               train_std=('Entrenamiento', 'std'), val_std=('Validación', 'std'), n=('N ajustado', 'mean')).reset_index()
            fig = go.Figure()
            for campo, desviacion, titulo in [('train', 'train_std', 'Entrenamiento'), ('val', 'val_std', 'Validación interna')]:
                eje = resumen['Valor'] if tipo == 'validacion' else resumen['n']
                fig.add_trace(go.Scatter(x=eje, y=resumen[campo], mode='lines+markers', name=titulo,
                                         error_y=dict(type='data', array=resumen[desviacion].fillna(0), visible=True)))
            fig.update_layout(xaxis_title=parametro if tipo == 'validacion' else 'Muestras realmente usadas para ajustar (media)',
                              yaxis_title='RMSE', title=nombre)
            if tipo == 'validacion': fig.update_xaxes(type='category')
            st.plotly_chart(fig, width='stretch')
            st.caption('Barras: una desviación estándar entre pliegues, no un intervalo de confianza. Solo se representan candidatos o tamaños que completaron todos los pliegues; revisa la tabla de errores y avisos.')
        else:
            st.warning('Ningún punto completó todos los pliegues. Revisa tamaño de muestra, parámetros y mensajes.')
        st.dataframe(tabla, width='stretch')
        st.download_button('Descargar '+nombre.lower(), tabla.to_csv(index=False).encode('utf-8-sig'), f'{clave}_{tipo}.csv', key=clave+':descargar:'+tipo)
