import hashlib
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from cnn_modelo import ejemplo_1d, ejemplo_2d, zip_ejemplo, leer_imagenes, entrenar_cnn


DESCRIPCIONES = {
    'layers': 'Número de bloques convolucionales. Más bloques permiten relaciones más complejas y aumentan el coste y el riesgo de sobreajuste.',
    'filters': 'Canales de salida de cada convolución. Cada filtro aprende un patrón local.',
    'kernel_size': 'Longitud (1D) o lado (2D) del filtro. Se usan tamaños impares y padding simétrico.',
    'stride': 'Paso de la convolución. Valores mayores reducen la resolución de la señal o imagen.',
    'dilation': 'Separación entre elementos del filtro: aumenta el campo receptivo sin añadir pesos al kernel.',
    'activation': 'Función no lineal aplicada tras cada convolución y en la capa densa: ReLU, GELU o tanh.',
    'pooling': 'Reducción espacial tras cada bloque: máximo, promedio o ninguna. Al final siempre se aplica promedio adaptativo a tamaño uno.',
    'dropout': 'Fracción de activaciones anuladas aleatoriamente durante el entrenamiento para regularizar. Se desactiva durante la evaluación.',
    'hidden_units': 'Unidades de la capa densa previa a la salida de regresión. La salida final es un único valor numérico.',
    'optimizer': 'Método de actualización de pesos: Adam, AdamW o SGD. No son estimadores de sklearn.',
    'learning_rate': 'Tamaño del paso del optimizador. Valores excesivos pueden hacer divergir el entrenamiento.',
    'weight_decay': 'Penalización de pesos. AdamW aplica decaimiento desacoplado; Adam y SGD usan su regla propia.',
    'momentum': 'Acumulación de actualizaciones en SGD. Se ignora con Adam y AdamW.',
    'loss': 'Pérdida optimizada sobre el objetivo estandarizado: MSE (cuadrática), MAE (absoluta) o Huber (SmoothL1 de PyTorch). Las métricas finales usan las unidades originales.',
    'epochs': 'Máximo de recorridos completos de entrenamiento. La parada anticipada puede terminar antes.',
    'batch_size': 'Número de muestras por actualización de pesos. El último lote puede ser más pequeño.',
    'patience': 'Épocas sin una mejora suficiente en validación antes de parar. Cero desactiva la parada anticipada. Siempre se restauran los mejores pesos de validación.',
    'min_delta': 'Mejora mínima de la pérdida de validación para reiniciar patience. La prueba nunca participa en esta decisión.',
}


def mostrar_cnn(dimension):
    nombre = f'CNN {dimension}D'
    st.set_page_config(page_title=nombre, page_icon='🧠', layout='wide')
    st.title(f'🧠 {nombre} · Regresión convolucional')
    st.caption('Cada cambio de datos o configuración reentrena la red. La validación selecciona los pesos; la prueba se reserva para las métricas finales.')
    with st.sidebar:
        st.header('Datos')
        fuente = st.selectbox('Origen', ['Ejemplo sintético', 'Cargar datos'])
        cronologico, metadatos = False, {}
        try:
            if dimension == 1:
                st.caption('Cada fila es una señal independiente; las columnas seleccionadas son posiciones ordenadas de un canal. No se crean ventanas temporales automáticamente.')
                archivo = st.file_uploader('CSV de señales', type=['csv']) if fuente == 'Cargar datos' else None
                separador = st.selectbox('Separador CSV', [',', ';', '\t'])
                decimal = st.selectbox('Separador decimal', ['.', ','])
                if fuente == 'Cargar datos' and archivo is None:
                    st.info('Selecciona el CSV de señales.'); st.stop()
                df = pd.read_csv(archivo, sep=separador, decimal=decimal) if archivo else ejemplo_1d()
                numericas = df.select_dtypes(include='number').columns.tolist()
                if len(numericas) < 2:
                    raise ValueError('Necesitas un objetivo y al menos una columna numérica de señal.')
                objetivo = st.selectbox('Objetivo', numericas, index=len(numericas)-1)
                opciones = [c for c in numericas if c != objetivo]
                variables = st.multiselect('Posiciones de la señal, en orden', opciones, default=opciones)
                if not variables:
                    st.info('Selecciona al menos una posición de señal.'); st.stop()
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[objetivo]).reset_index(drop=True)
                if len(df) > 10000:
                    raise ValueError('El laboratorio CNN admite hasta 10.000 señales.')
                x = df[variables].to_numpy(dtype='float32')[:, None, :]
                y = df[objetivo].to_numpy(dtype='float32')
                identificadores = list(range(len(y)))
                cronologico = st.checkbox('Separar por orden de filas (sin barajar)', False)
                metadatos = {'features': variables, 'target': objetivo}
            else:
                tamano = st.select_slider('Tamaño de imagen', [16, 32, 64, 96], value=32)
                color = st.selectbox('Canales', ['L', 'RGB'], help='L: escala de grises; RGB: tres canales.')
                st.download_button('Descargar ZIP de ejemplo', zip_ejemplo(), 'cnn_imagenes_ejemplo.zip', 'application/zip')
                if fuente == 'Cargar datos':
                    archivo = st.file_uploader('ZIP: labels.csv e imágenes', type=['zip'])
                    if archivo is None:
                        st.info('Adjunta el ZIP. labels.csv debe contener file,target.'); st.stop()
                    x, y, identificadores = leer_imagenes(archivo.getvalue(), tamano, color)
                else:
                    # Mismo lector y preprocesamiento que para los ZIP del usuario.
                    x, y, identificadores = leer_imagenes(zip_ejemplo(), tamano, color)
                df = pd.DataFrame({'file': identificadores, 'target': y})
                metadatos = {'image_size': tamano, 'color': color}
            prueba = st.slider('Proporción de prueba', .1, .4, .2, .05)
            semilla = st.number_input('Semilla', 0, 2147483647, 42)
            st.header('Arquitectura y entrenamiento')
            c = {}
            c['layers'] = st.slider('layers', 1, 4, 2)
            c['filters'] = st.select_slider('filters', [4, 8, 16, 32, 64], value=16)
            c['kernel_size'] = st.select_slider('kernel_size', [1, 3, 5, 7], value=3)
            c['stride'] = st.slider('stride', 1, 3, 1)
            c['dilation'] = st.slider('dilation', 1, 3, 1)
            c['activation'] = st.selectbox('activation', ['relu', 'gelu', 'tanh'])
            c['pooling'] = st.selectbox('pooling', ['max', 'avg', 'none'])
            c['dropout'] = st.slider('dropout', 0., .7, .1, .05)
            c['hidden_units'] = st.select_slider('hidden_units', [8, 16, 32, 64, 128], value=32)
            c['optimizer'] = st.selectbox('optimizer', ['Adam', 'AdamW', 'SGD'])
            c['learning_rate'] = st.number_input('learning_rate', min_value=.000001, value=.003, format='%.6f', step=.001)
            c['weight_decay'] = st.number_input('weight_decay', min_value=0., value=0., format='%.6f', step=.0001)
            c['momentum'] = st.slider('momentum', 0., .99, .9, .01, disabled=c['optimizer'] != 'SGD')
            c['loss'] = st.selectbox('loss', ['MSE', 'MAE', 'Huber'])
            c['epochs'] = st.slider('epochs', 1, 200, 20)
            c['batch_size'] = st.select_slider('batch_size', [8, 16, 32, 64, 128], value=32)
            c['patience'] = st.slider('patience', 0, 40, 8)
            c['min_delta'] = st.number_input('min_delta', min_value=0., value=.0001, format='%.6f', step=.0001)
        except Exception as exc:
            st.error(f'Datos inválidos: {exc}'); st.stop()
    firma = hashlib.sha256(x.tobytes()+y.tobytes()+json.dumps([c, metadatos, prueba, int(semilla), cronologico], sort_keys=True).encode()).hexdigest()
    clave = f'cnn{dimension}_resultado'
    try:
        if st.session_state.get(clave, {}).get('firma') != firma:
            barra = st.progress(0., text='Entrenando CNN en CPU…')
            paquete, resultados, test, train, historia = entrenar_cnn(x, y, dimension, c, prueba, int(semilla), cronologico, barra.progress)
            barra.empty()
            paquete['metadata'] = metadatos
            resultados['muestra'] = [identificadores[i] for i in resultados['indice']]
            st.session_state[clave] = dict(firma=firma, paquete=paquete, resultados=resultados, test=test, train=train, historia=historia)
            st.session_state.setdefault(f'cnn{dimension}_historial', []).append({**test, 'Configuración': json.dumps(c), 'Firma': firma})
        resultado = st.session_state[clave]
    except Exception as exc:
        st.error(f'No se pudo entrenar: {exc}'); st.stop()
    paquete, resultados, test, train, historia = (resultado[k] for k in ['paquete', 'resultados', 'test', 'train', 'historia'])
    for col, (label, valor) in zip(st.columns(5), test.items()):
        col.metric(label, f'{valor:.4f}' if np.isfinite(valor) else 'No definido')
    st.caption('Prueba: residuo = observado − predicho; bias = media(predicho − observado). Pearson no se define con series constantes.')
    diagnostico, modelo, historial, datos, descripcion, geometria, curvas = st.tabs(['Diagnóstico de residuos', 'Modelo y métricas', 'Historial', 'Datos', 'Descripción', 'Datos y superficie del modelo', 'Curvas de validación y aprendizaje'])
    with diagnostico:
        a, b = st.columns(2)
        fig = px.scatter(resultados, x='predicho', y='residuo', hover_data=['muestra'], title='Residuos frente a predicción')
        fig.add_hline(y=0, line_dash='dash')
        a.plotly_chart(fig, width='stretch')
        b.plotly_chart(px.histogram(resultados, x='residuo', title='Distribución de residuos'), width='stretch')
        a.plotly_chart(px.scatter(resultados, x='observado', y='predicho', title='Observado frente a predicho'), width='stretch')
        b.plotly_chart(px.scatter(resultados, x='indice', y='residuo', title='Residuos por muestra'), width='stretch')
    with modelo:
        st.dataframe(pd.DataFrame({'Entrenamiento': train, 'Prueba': test}), width='stretch')
        st.plotly_chart(px.line(historia, x='Época', y=['Entrenamiento', 'Validación'], title='Pérdidas sobre objetivo estandarizado'), width='stretch')
        st.write({k: len(v) for k, v in paquete['indices'].items()})
        st.json(c)
        buffer = io.BytesIO(); torch.save(paquete, buffer)
        st.download_button('Descargar pesos, configuración y normalización (.pt)', buffer.getvalue(), f'cnn{dimension}d.pt', 'application/octet-stream')
        st.caption('Formato PyTorch .pt, distinto del endpoint joblib de sklearn. Incluye state_dict, arquitectura, normalización y metadatos para reconstruir la red con cnn_modelo.py.')
    with historial:
        tabla = pd.DataFrame(st.session_state[f'cnn{dimension}_historial']).drop(columns='Firma')
        st.dataframe(tabla, width='stretch')
        st.download_button('Descargar historial', tabla.to_csv(index=False).encode('utf-8-sig'), f'cnn{dimension}_historial.csv')
    with datos:
        st.dataframe(df.head(100), width='stretch')
        if dimension == 2:
            st.image([x[i, 0] if x.shape[1] == 1 else x[i].transpose(1, 2, 0) for i in range(min(6, len(x)))], width=100)
        else:
            st.line_chart(pd.DataFrame(x[:3, 0].T, columns=['Señal 1', 'Señal 2', 'Señal 3']))
        st.dataframe(resultados, width='stretch')
        st.download_button('Descargar predicciones y residuos', resultados.to_csv(index=False).encode('utf-8-sig'), f'cnn{dimension}_residuos.csv')
    with descripcion:
        st.markdown('Redes de **regresión de un objetivo** con [Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) o [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) de PyTorch. Se entrenan todos los pesos por retropropagación en CPU.')
        st.write('CNN 1D: las posiciones de cada fila deben tener un orden significativo (señales, espectros o ventanas ya construidas). No presupone continuidad entre las filas. Para ventanas solapadas o grupos correlacionados, prepara particiones sin solapamiento antes de evaluar; la separación por orden por sí sola no elimina ese solapamiento.')
        st.write('CNN 2D: ZIP con labels.csv en la raíz, columnas file,target e imágenes referenciadas por su ruta dentro del ZIP. Se convierten a gris o RGB y se redimensionan al tamaño elegido. Cada archivo debe aparecer una sola vez. Los ejemplos son sintéticos.')
        st.write('Primero se reserva la prueba. Del resto, el 20% se usa como validación. Imputación, medias, escalas y normalización del objetivo se aprenden solo con entrenamiento. Se restauran los pesos con menor pérdida de validación; las métricas se calculan en prueba con las unidades originales. Elegir repetidamente configuraciones mirando la prueba puede sobreajustar la selección.')
        for parametro, explicacion in DESCRIPCIONES.items():
            with st.expander(parametro):
                st.write(explicacion)
                st.write('Valor actual:', c[parametro])

    with geometria:
        from geometria import mostrar_geometria
        from cnn_modelo import predecir_cnn
        def predecir_geometria(nuevas):
            predicciones = []
            for inicio in range(0, len(nuevas), 16):
                lote = nuevas[inicio:inicio+16].astype('float32').reshape((-1,)+tuple(x.shape[1:]))
                predicciones.extend(predecir_cnn(paquete, lote))
            return np.asarray(predicciones)
        mostrar_geometria(x.reshape(len(x), -1), y, paquete['indices']['train'], predecir_geometria, f'cnn{dimension}',
                         test_indices=resultados['indice'].to_numpy(), predicciones_test=resultados['predicho'].to_numpy(),
                         nombres=metadatos.get('features'), imagenes=dimension == 2)

    with curvas:
        from curvas import mostrar_curvas, evaluador_cnn
        desarrollo = np.sort(np.concatenate([paquete['indices']['train'], paquete['indices']['validation']]))
        x_dev, y_dev = x[desarrollo], y[desarrollo]
        firma_desarrollo = hashlib.sha256(x_dev.tobytes()+y_dev.tobytes()).hexdigest()
        mostrar_curvas(f'cnn{dimension}', c, evaluador_cnn(x_dev, y_dev, dimension, int(semilla), cronologico),
                       len(desarrollo), firma_desarrollo, int(semilla), cronologico)
