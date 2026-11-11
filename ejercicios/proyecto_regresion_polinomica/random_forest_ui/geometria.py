"""Datos de prueba reales y cortes de la función de regresión entrenada."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def datos_prueba(features, objetivo, test_indices, predicciones, nombres):
    x = np.asarray(features, dtype=float)
    indices = np.asarray(test_indices, dtype=int)
    pred = np.asarray(predicciones, dtype=float).reshape(-1)
    if len(indices) != len(pred) or not np.isfinite(pred).all():
        raise ValueError('Las predicciones deben corresponder a las filas de prueba, en el mismo orden.')
    tabla = pd.DataFrame(x[indices], columns=nombres)
    tabla['Muestra'] = indices
    tabla['Observado'] = np.asarray(objetivo)[indices]
    tabla['Predicho'] = pred
    tabla['Residuo'] = tabla['Observado']-pred
    return tabla


def corte_modelo(features, train_indices, eje_x, eje_y, predecir, resolucion=25):
    """Corte condicional en unidades originales; restantes features a mediana train."""
    train = np.asarray(features, dtype=float)[np.asarray(train_indices, dtype=int)]
    train = np.where(np.isfinite(train), train, np.nan)
    base = np.nanmedian(train, axis=0)
    if not np.isfinite(base).all():
        raise ValueError('No se puede construir el corte: hay variables sin valores en entrenamiento.')
    xx = np.linspace(np.nanmin(train[:, eje_x]), np.nanmax(train[:, eje_x]), resolucion)
    if eje_y is None:
        entradas = np.tile(base, (len(xx), 1))
        entradas[:, eje_x] = xx
        return xx, None, np.asarray(predecir(entradas)).reshape(-1), base
    yy = np.linspace(np.nanmin(train[:, eje_y]), np.nanmax(train[:, eje_y]), resolucion)
    gx, gy = np.meshgrid(xx, yy)
    entradas = np.tile(base, (gx.size, 1))
    entradas[:, eje_x], entradas[:, eje_y] = gx.ravel(), gy.ravel()
    zz = np.asarray(predecir(entradas), dtype=float).reshape(gx.shape)
    if not np.isfinite(zz).all():
        raise ValueError('Predicciones no finitas en el corte condicional.')
    return gx, gy, zz, base


def mostrar_geometria(features, objetivo, train_indices, predecir, clave, test_indices,
                      predicciones_test, nombres=None, imagenes=False):
    st.caption('Se muestran exclusivamente observaciones y predicciones guardadas del conjunto de prueba. Los residuos son observado − predicho. No se crean grupos ni fronteras entre clases.')
    x = np.asarray(features, dtype=float)
    indices = np.asarray(test_indices, dtype=int)
    nombres = list(nombres) if nombres is not None else [f'Variable {i+1}' for i in range(x.shape[1])]
    if imagenes:
        # Resúmenes de las imágenes originales, no componentes ni entradas sintéticas.
        resumen = np.column_stack([x.mean(axis=1), x.std(axis=1)])
        nombres = ['Intensidad media', 'Desviación de intensidad']
        tabla = datos_prueba(resumen, objetivo, indices, predicciones_test, nombres)
        st.caption('CNN 2D: ejes descriptivos calculados sobre cada imagen real (píxeles en [0,1]). No representan activaciones de la red ni se usa una superficie construida alterando píxeles.')
    else:
        tabla = datos_prueba(x, objetivo, indices, predicciones_test, nombres)
    if len(tabla) > 1500:
        tabla = tabla.sample(1500, random_state=42)
        st.caption('Se muestra una muestra reproducible de 1.500 observaciones de prueba; las métricas usan toda la prueba.')
    eje_x = st.selectbox('Eje X · variable original', nombres, key=clave+':eje_x')
    color = st.selectbox('Colorear datos de prueba por', ['Residuo', 'Observado', 'Predicho'], key=clave+':color_real')
    opciones_y = [n for n in nombres if n != eje_x]
    if not opciones_y:
        fig = go.Figure()
        for campo in ['Observado', 'Predicho']:
            fig.add_trace(go.Scatter(x=tabla[eje_x], y=tabla[campo], mode='markers', name=campo))
        fig.update_layout(xaxis_title=eje_x, yaxis_title='Objetivo', title='Observaciones y predicciones reales de prueba')
        st.plotly_chart(fig, width='stretch')
        if not imagenes:
            gx, _, z, _ = corte_modelo(x, train_indices, nombres.index(eje_x), None, predecir, 100)
            linea = go.Figure(go.Scatter(x=gx, y=z, name='Función del modelo', mode='lines'))
            linea.update_layout(xaxis_title=eje_x, yaxis_title='Predicción', title='Curva de regresión del modelo entrenado')
            st.plotly_chart(linea, width='stretch')
        return
    eje_y = st.selectbox('Eje Y · variable original', opciones_y, key=clave+':eje_y')
    vista2d, vista3d = st.columns(2)
    fig = px.scatter(tabla, x=eje_x, y=eje_y, color=color, hover_data=['Muestra', 'Observado', 'Predicho', 'Residuo'],
                     title='Distribución de los datos reales de prueba', color_continuous_scale='RdBu')
    vista2d.plotly_chart(fig, width='stretch')
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter3d(x=tabla[eje_x], y=tabla[eje_y], z=tabla['Observado'], mode='markers', name='Observado (prueba)',
                               marker=dict(size=4, color=tabla[color], colorscale='RdBu', colorbar=dict(title=color)),
                               customdata=tabla[['Muestra', 'Predicho', 'Residuo']], hovertemplate='Observado: %{z}<br>Muestra: %{customdata[0]}<br>Predicho: %{customdata[1]}<br>Residuo: %{customdata[2]}<extra></extra>'))
    fig3.add_trace(go.Scatter3d(x=tabla[eje_x], y=tabla[eje_y], z=tabla['Predicho'], mode='markers', name='Predicho (prueba)', marker=dict(size=3, symbol='diamond', color='black')))
    fig3.update_layout(title='Objetivo real y predicción del modelo', scene=dict(xaxis_title=eje_x, yaxis_title=eje_y, zaxis_title='Objetivo'))
    vista3d.plotly_chart(fig3, width='stretch')
    if imagenes:
        st.info('Una CNN 2D depende de la imagen completa. No corresponde dibujar un plano de separación ni una superficie de predicción a partir de estos dos resúmenes. Los puntos negros son las predicciones reales de las imágenes de prueba.')
        return
    st.subheader('Plano o superficie de regresión del modelo entrenado')
    st.caption('Se varían las dos variables elegidas en sus rangos de entrenamiento. Las demás se fijan en la mediana del entrenamiento. Este corte es una entrada sintética identificada, no una nube de observaciones reales ni una frontera de clasificación.')
    resolucion = st.select_slider('Resolución del corte', [15, 25, 35], value=25, key=clave+':resolucion_real')
    try:
        gx, gy, z, base = corte_modelo(x, train_indices, nombres.index(eje_x), nombres.index(eje_y), predecir, resolucion)
        fijas = {n: float(base[i]) for i, n in enumerate(nombres) if n not in {eje_x, eje_y}}
        if fijas:
            with st.expander('Valores fijos de las otras variables'):
                st.json(fijas)
            st.caption('Las predicciones de prueba usan todas las variables reales de cada fila; por eso sus puntos pueden no coincidir con este corte a medianas, incluso en un modelo lineal.')
        a, b = st.columns(2)
        mapa = go.Figure(go.Contour(x=gx[0], y=gy[:, 0], z=z, colorbar=dict(title='Predicción')))
        mapa.update_layout(title='Contornos de la predicción', xaxis_title=eje_x, yaxis_title=eje_y)
        a.plotly_chart(mapa, width='stretch')
        superficie = go.Figure(go.Surface(x=gx, y=gy, z=z, colorbar=dict(title='Predicción')))
        superficie.update_layout(title='Corte de la función de regresión', scene=dict(xaxis_title=eje_x, yaxis_title=eje_y, zaxis_title='Predicción'))
        b.plotly_chart(superficie, width='stretch')
        st.caption('Bayesian Ridge, Huber, Theil-Sen y QuantileRegressor producen planos. RANSAC depende de su estimador base. Los árboles, boosting, KNN, kernels no lineales de SVR y CNN pueden producir superficies no lineales o escalonadas. El gráfico usa predict del modelo, sin ajustar otro algoritmo.')
    except (ValueError, TypeError, RuntimeError) as exc:
        st.warning(f'No se pudo generar el corte del modelo: {exc}')
