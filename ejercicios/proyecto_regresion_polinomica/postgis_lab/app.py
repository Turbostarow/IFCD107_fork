import pandas as pd
import plotly.express as px
import streamlit as st
import psycopg
from db import consultar, insertar
from modelos import FEATURES, entrenar

st.set_page_config(page_title='PostGIS y regresión', layout='wide')
st.title('PostGIS y regresores lineales')
st.caption('Laboratorio con viviendas sintéticas. Importes en euros; no son valoraciones reales.')
with st.sidebar:
    st.header('Consulta espacial')
    lon = st.number_input('Longitud del filtro', -180.0, 180.0, -3.7038, format='%.4f')
    lat = st.number_input('Latitud del filtro', -90.0, 90.0, 40.4168, format='%.4f')
    radio = st.slider('Radio (metros)', 100, 20000, 15000, 100)
    minimo = st.slider('Superficie mínima (m²)', 15, 200, 15)
    st.button('Actualizar consulta')
try:
    df = consultar(lon, lat, radio, minimo)
except psycopg.Error:
    st.error('No se pudo consultar PostGIS. Comprueba que el servicio postgis-lab esté saludable.')
    st.stop()
datos, alta, regresion, guia = st.tabs(['Datos y mapa', 'Insertar vivienda', 'Regresión', 'Ejercicio'])
with datos:
    st.metric('Viviendas encontradas', len(df))
    if not df.empty:
        st.map(df.rename(columns={'latitud':'lat', 'longitud':'lon'}), latitude='lat', longitude='lon')
        st.dataframe(df, hide_index=True)
        st.download_button('Descargar consulta CSV', df.to_csv(index=False), 'consulta_viviendas.csv', 'text/csv')
with alta:
    with st.form('alta'):
        ref = st.text_input('Referencia única', placeholder='MANUAL-001')
        sup = st.number_input('Superficie (m²)', 15.0, 500.0, 90.0)
        hab = st.number_input('Habitaciones', 1, 10, 3)
        edad = st.number_input('Antigüedad (años)', 0, 150, 20)
        precio = st.number_input('Precio observado (€)', 1.0, 10000000.0, 300000.0)
        ilon = st.number_input('Longitud', -180.0, 180.0, -3.70, format='%.5f')
        ilat = st.number_input('Latitud', -90.0, 90.0, 40.42, format='%.5f')
        guardar = st.form_submit_button('Insertar en PostGIS')
    if guardar:
        if not ref.strip():
            st.error('Introduce una referencia.')
        else:
            try:
                insertar(ref.strip(), sup, hab, edad, precio, ilon, ilat)
                st.success('Vivienda insertada. Pulsa Actualizar consulta para verla si cumple los filtros.')
            except psycopg.errors.UniqueViolation:
                st.error('La referencia ya existe.')
            except psycopg.Error:
                st.error('No se pudo insertar la vivienda.')
with regresion:
    algoritmo = st.selectbox('Regresor', ['Lineal', 'Ridge', 'Lasso', 'Elastic Net'])
    alpha = st.number_input('alpha (penalización)', 0.0001, 10000.0, 1.0, disabled=algoritmo=='Lineal')
    ratio = st.slider('l1_ratio', 0.01, 1.0, 0.5, disabled=algoritmo!='Elastic Net')
    fraccion = st.slider('Fracción de prueba', 0.15, 0.4, 0.25)
    st.caption('Se entrena sobre la consulta actual. Escalado ajustado solo con entrenamiento. Partición aleatoria fija: 42.')
    if len(df) < 40:
        st.info('Amplía el radio o reduce la superficie mínima: se necesitan 40 filas.')
    else:
        pipe, metrics, result, coef, intercept = entrenar(df, algoritmo, alpha, ratio, fraccion)
        cols = st.columns(len(metrics))
        for col, (nombre, valor) in zip(cols, metrics.items()):
            col.metric(nombre, 'No definido' if valor is None else f'{valor:.3f}')
        a, b = st.columns(2)
        with a:
            fig = px.scatter(result, x='precio', y='predicho', title='Real frente a predicho (prueba)')
            lo, hi = float(result.precio.min()), float(result.precio.max())
            fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi)
            st.plotly_chart(fig, width='stretch')
        with b:
            st.plotly_chart(px.scatter(result, x='predicho', y='residuo', title='Residuos: real − predicho'), width='stretch')
        st.plotly_chart(px.histogram(result, x='residuo', title='Distribución de residuos'), width='stretch')
        st.dataframe(coef, hide_index=True)
        st.write(f'Intercepto en unidades originales: {intercept:.2f} €')
        st.caption('Habitaciones y superficie pueden estar relacionadas. Los coeficientes no prueban causalidad. Bias = media(predicho − real).')
        st.subheader('Predecir una nueva vivienda')
        with st.form('prediccion'):
            valores = [st.number_input(n, value=float(df[n].median()), min_value=0.0) for n in FEATURES]
            predecir = st.form_submit_button('Predecir precio')
        if predecir:
            x = pd.DataFrame([valores], columns=FEATURES)
            st.success(f'Precio predicho: {pipe.predict(x)[0]:,.2f} €')
        st.download_button('Descargar evaluación', result.to_csv(index=False), 'evaluacion.csv', 'text/csv')
        st.warning('Cambiar hiperparámetros mirando repetidamente la prueba sesga la evaluación. Para seleccionarlos, usa validación cruzada en entrenamiento. En datos reales, considera separar por zonas geográficas.')
with guia:
    st.markdown('''### Objetivos
1. Crear una base espacial, una tabla de puntos y un índice GiST.
2. Insertar viviendas y consultar por radio con parámetros SQL.
3. Obtener distancias en metros usando geography y convertir a km.
4. Comparar regresión lineal, Ridge, Lasso y Elastic Net.
5. Interpretar coeficientes, residuos y métricas sobre prueba.

Consulta el enunciado completo en `postgis_lab/EJERCICIO.md`.
El mapa puede necesitar conexión a Internet para sus teselas.
''')
