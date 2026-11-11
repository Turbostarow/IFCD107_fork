import io
import joblib
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from modelo import datos_ejemplo, entrenar, crear_estimador
from catalogo import CATALOGO
from controles import controles, descripcion_parametros
import sklearn



def mostrar(algoritmo):
    titulo, opciones_parametros = CATALOGO[algoritmo]
    st.set_page_config(page_title=f'Laboratorio {titulo}', page_icon='📈', layout='wide')
    st.title(f'📈 Laboratorio de {titulo} · Regresión')
    st.caption('Modifica un control: el modelo se entrena y los resultados se actualizan automáticamente al soltarlo.')
    
    with st.sidebar:
        st.header('Datos y validación')
        archivo = st.file_uploader('CSV con cabecera', type=['csv'])
        separador = st.selectbox('Separador CSV', [',', ';', '\t'])
        decimal = st.selectbox('Separador decimal', ['.', ','])
        try:
            df = pd.read_csv(archivo, sep=separador, decimal=decimal) if archivo else datos_ejemplo()
        except Exception as exc:
            st.error(f'No se pudo leer el CSV: {exc}')
            st.stop()
        numericas = df.select_dtypes(include=np.number).columns.tolist()
        if len(numericas) < 2:
            st.error('Necesitas al menos dos columnas numéricas: predictores y objetivo.')
            st.stop()
        objetivo = st.selectbox('Variable objetivo', numericas, index=len(numericas)-1)
        candidatas = [c for c in numericas if c != objetivo]
        variables = st.multiselect('Variables predictoras numéricas', candidatas, default=candidatas)
        prueba = st.slider('Proporción de prueba', .1, .5, .25, .05)
        semilla = st.number_input('Semilla de la partición', 0, 2147483647, 42)
        st.header('Hiperparámetros')
        n_validas = len(df.replace([np.inf, -np.inf], np.nan).dropna(subset=[objetivo]))
        p = controles(algoritmo, variables, int(n_validas*(1-prueba)))
        st.caption('Cada cambio entrena un modelo nuevo; warm_start no conserva ajustes entre configuraciones. Los controles de sklearn cubren sus parámetros de constructor; los boosters externos admiten opciones adicionales en JSON.')
        if algoritmo not in {'bayesian_ridge', 'random_forest'}:
            st.caption('Imputación y estandarización ajustadas solo en entrenamiento.')
        if p.get('kernel') == 'precomputed' or p.get('metric') == 'precomputed':
            st.caption('precomputed: el laboratorio transforma las features en un kernel lineal (SVR) o distancias euclídeas (KNN) contra entrenamiento; el pipeline exportado conserva esta transformación.')
    
    if not variables:
        st.info('Selecciona al menos una variable predictora.')
        st.stop()
    st.caption(f'{len(df):,} filas · {len(variables)} predictores · objetivo: {objetivo}. Los predictores ausentes se imputan con la mediana del entrenamiento; objetivos ausentes se excluyen.')
    firma = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes() + json.dumps([objetivo, variables, p, prueba, int(semilla)], sort_keys=True, default=str).encode()).hexdigest()
    cache_key = algoritmo+':entrenamiento_principal'
    guardado = st.session_state.get(cache_key)
    if not guardado or guardado['firma'] != firma:
        inicio = time.perf_counter()
        try:
            with st.spinner(f'Entrenando {titulo}…'), warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter('always')
                modelo, resultados, test, train = entrenar(df, objetivo, variables, p, prueba, int(semilla), algoritmo=algoritmo)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        guardado = dict(firma=firma, modelo=modelo, resultados=resultados, test=test, train=train,
                        avisos=sorted({str(aviso.message) for aviso in avisos}), duracion=time.perf_counter()-inicio)
        st.session_state[cache_key] = guardado
    modelo, resultados, test, train = (guardado[k] for k in ['modelo', 'resultados', 'test', 'train'])
    for aviso in guardado['avisos']:
        st.warning(aviso)
    st.success(f"Modelo principal listo · ajuste en {guardado['duracion']:.2f} s · {len(resultados)} observaciones de prueba")
    for col, (nombre, valor) in zip(st.columns(5), test.items()):
        col.metric(nombre, 'No definido' if not np.isfinite(valor) else f'{valor:.4f}')
    st.caption('Métricas de prueba. Residuo = observado − predicho. Bias = media(predicho − observado): positivo indica sobreestimación. Pearson no está definido si alguna serie es constante.')
    
    diagnostico, comparacion, historial, datos, descripcion, geometria, curvas = st.tabs(['Diagnóstico de residuos', 'Modelo y métricas', 'Historial', 'Datos', 'Descripción', 'Datos y superficie del modelo', 'Curvas de validación y aprendizaje'])
    with diagnostico:
        c1, c2 = st.columns(2)
        fig = px.scatter(resultados, x='predicho', y='residuo', title='Residuos frente a predicción', opacity=.65)
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        c1.plotly_chart(fig, width='stretch')
        c2.plotly_chart(px.histogram(resultados, x='residuo', nbins=35, title='Distribución de residuos'), width='stretch')
        fig = px.scatter(resultados, x='observado', y='predicho', title='Observado frente a predicho', opacity=.65)
        limite = [float(resultados[['observado', 'predicho']].min().min()), float(resultados[['observado', 'predicho']].max().max())]
        fig.add_shape(type='line', x0=limite[0], y0=limite[0], x1=limite[1], y1=limite[1], line=dict(dash='dash', color='red'))
        c1.plotly_chart(fig, width='stretch')
        orden = resultados.reset_index(drop=True).rename_axis('Muestra').reset_index()
        c2.plotly_chart(px.scatter(orden, x='Muestra', y='residuo', title='Residuos por muestra de prueba'), width='stretch')
    with comparacion:
        st.dataframe(pd.DataFrame({'Entrenamiento': train, 'Prueba': test}), width='stretch')
        estimador = modelo.steps[-1][1]
        if algoritmo == 'bayesian_ridge':
            st.write(f'Iteraciones realizadas: {estimador.n_iter_} · alpha estimado: {estimador.alpha_:.6g} · lambda estimado: {estimador.lambda_:.6g}')
            if p['compute_score']:
                scores = pd.DataFrame({'Paso': np.arange(len(estimador.scores_)), 'Log-verosimilitud marginal': estimador.scores_})
                st.plotly_chart(px.line(scores, x='Paso', y='Log-verosimilitud marginal', title='Evolución del ajuste bayesiano'), width='stretch')
        if algoritmo == 'gaussian_process':
            st.write('Kernel ajustado:', str(estimador.kernel_))
            entradas = df.loc[resultados.index, variables].replace([np.inf, -np.inf], np.nan)
            media, desviacion = modelo.predict(entradas, return_std=True)
            bandas = pd.DataFrame({'Predicción': media, 'Inferior 95%': media-1.96*desviacion, 'Superior 95%': media+1.96*desviacion})
            st.line_chart(bandas)
            st.caption('Intervalo gaussiano aproximado basado en la desviación predictiva del kernel; incluye ruido si el kernel lo modela. No garantiza cobertura de futuras observaciones.')
        if algoritmo == 'mlp' and hasattr(estimador, 'loss_curve_'):
            st.line_chart(pd.DataFrame({'Pérdida de entrenamiento': estimador.loss_curve_}))
        if algoritmo == 'quantile':
            st.caption(f'Predicción del cuantil condicional {estimador.quantile:g}; no de la media condicional. R² y RMSE se muestran para comparar, pero no son su objetivo de optimización.')
        if hasattr(estimador, 'feature_importances_'):
            tabla_modelo = pd.DataFrame({'Variable': variables, 'Importancia': estimador.feature_importances_}).sort_values('Importancia')
            st.plotly_chart(px.bar(tabla_modelo, x='Importancia', y='Variable', orientation='h', title='Importancia de variables'), width='stretch')
        lineal = estimador.estimator_ if algoritmo == 'ransac' else estimador
        if hasattr(lineal, 'coef_'):
            coeficientes = pd.DataFrame({'Variable': variables, 'Coeficiente': np.asarray(lineal.coef_).ravel()}).sort_values('Coeficiente')
            st.plotly_chart(px.bar(coeficientes, x='Coeficiente', y='Variable', orientation='h', title='Coeficientes del modelo'), width='stretch')
            st.caption('Los coeficientes dependen de la escala: Bayesian Ridge usa las unidades originales; los otros modelos reciben predictores estandarizados.')
        if algoritmo == 'ransac':
            st.write(f'Muestras de consenso en entrenamiento: {int(estimador.inlier_mask_.sum())} de {len(estimador.inlier_mask_)} · intentos: {estimador.n_trials_}')
        if algoritmo == 'huber':
            st.write(f'Valores atípicos en entrenamiento: {int(estimador.outliers_.sum())} · iteraciones: {estimador.n_iter_}')
        st.json(json.loads(json.dumps({f'model__{nombre}': valor for nombre, valor in p.items()}, default=str)))
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer, compress=3)
        st.download_button('Descargar modelo joblib', buffer.getvalue(), f'{algoritmo}.joblib', 'application/octet-stream')
    
    registros = st.session_state.setdefault(f'{algoritmo}_registros', [])
    if not registros or st.session_state.get(f'{algoritmo}_ultima_firma') != firma:
        registros.append({'Ejecución': len(registros)+1, **test, 'Objetivo': objetivo, 'Parámetros': json.dumps(p, default=str), 'Firma': firma})
        st.session_state[f'{algoritmo}_ultima_firma'] = firma
    with historial:
        st.caption('Configuraciones evaluadas en esta sesión. Compara ejecuciones con los mismos datos y partición; reserva una validación externa para la evaluación final.')
        tabla = pd.DataFrame(registros).drop(columns='Firma')
        st.dataframe(tabla, width='stretch')
        st.download_button('Descargar historial CSV', tabla.to_csv(index=False).encode('utf-8-sig'), f'historial_{algoritmo}.csv', 'text/csv')
    with datos:
        st.dataframe(df.head(200), width='stretch')
        st.dataframe(resultados, width='stretch')
        st.download_button('Descargar predicciones y residuos', resultados.to_csv(index=True).encode('utf-8-sig'), f'residuos_{algoritmo}.csv', 'text/csv')

    with descripcion:
        from descripciones import EXPLICACIONES, ESPECIFICAS
        estimador_base = crear_estimador(algoritmo)
        from boosting_ui import EXTERNOS, REFERENCIAS, PERFILES
        ruta = type(estimador_base).__module__.split('._')[0] + '.' + type(estimador_base).__name__
        version = '.'.join(sklearn.__version__.split('.')[:2])
        url = REFERENCIAS[algoritmo] if algoritmo in EXTERNOS else f'https://scikit-learn.org/{version}/modules/generated/{ruta}.html'
        st.markdown(f'Documentación de **{titulo}**: [referencia oficial]({url}).')
        st.caption('La explicación y los valores actuales corresponden al estimador de esta página. Los parámetros adicionales JSON de boosters externos sobrescriben los controles con el mismo nombre; consulta los tipos y dependencias en la referencia oficial.')
        documentacion = descripcion_parametros(estimador_base)
        for nombre in (list(dict.fromkeys(list(PERFILES[algoritmo])+list(p))) if algoritmo in EXTERNOS else estimador_base.get_params(deep=False)):
            dominio, detalle = documentacion.get(nombre, ('Consultar tipos y restricciones en la referencia oficial', 'Consulta el enlace oficial de esta biblioteca para este parámetro.'))
            with st.expander(f'model__{nombre}', expanded=False):
                st.write(ESPECIFICAS.get((algoritmo, nombre), EXPLICACIONES.get(nombre, 'Configura el comportamiento del estimador según el dominio y la documentación siguientes.')))
                st.markdown(f'**Tipo y valores admitidos:** `{dominio}`')
                st.markdown(f'**Valor actual:** `{p.get(nombre)!s}`')
                st.markdown('**Documentación original de la biblioteca:**')
                st.text(detalle)

    with geometria:
        from geometria import mostrar_geometria
        validos = df[variables + [objetivo]].replace([np.inf, -np.inf], np.nan).dropna(subset=[objetivo])
        train_indices = validos.index.get_indexer(modelo.training_indices_)
        mostrar_geometria(validos[variables].to_numpy(), validos[objetivo].to_numpy(), train_indices,
                         lambda nuevas: modelo.predict(pd.DataFrame(nuevas, columns=variables)), algoritmo,
                         test_indices=validos.index.get_indexer(resultados.index),
                         predicciones_test=resultados['predicho'].to_numpy(), nombres=variables)

    with curvas:
        from curvas import mostrar_curvas, evaluador_tabular
        desarrollo = validos.loc[~validos.index.isin(resultados.index)].reset_index(drop=True)
        firma_desarrollo = hashlib.sha256(pd.util.hash_pandas_object(desarrollo, index=True).values.tobytes()).hexdigest()
        mostrar_curvas(algoritmo, p, evaluador_tabular(desarrollo, objetivo, variables, algoritmo, int(semilla)),
                       len(desarrollo), firma_desarrollo, int(semilla))
