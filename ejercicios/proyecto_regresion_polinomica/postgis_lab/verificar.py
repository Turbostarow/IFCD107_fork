"""Comprobacion de integracion: ejecutar dentro del contenedor de interfaz."""
import numpy as np
from streamlit.testing.v1 import AppTest
from db import connect, consultar
from modelos import FEATURES, entrenar

df = consultar()
assert len(df) >= 2500, len(df)
assert consultar(0, 0, 100).empty
near = consultar(radio=500)
assert len(near) < len(df)
with connect() as conn:
    assert conn.execute('SELECT ST_SRID(ubicacion::geometry) AS srid FROM aprendizaje.viviendas LIMIT 1').fetchone()['srid'] == 4326
    # Insercion real y posterior rollback: no deja datos de prueba en el laboratorio.
    conn.execute('''INSERT INTO aprendizaje.viviendas
    (referencia,superficie,habitaciones,antiguedad,precio,ubicacion)
    VALUES ('VERIFICACION-ROLLBACK',80,2,10,200000,
    ST_SetSRID(ST_MakePoint(-3.7038,40.4168),4326)::geography)''')
    conn.rollback()
for name in ['Lineal', 'Ridge', 'Lasso', 'Elastic Net']:
    model, metrics, result, coef, intercept = entrenar(df, name)
    assert len(result) == int(np.ceil(len(df)*0.25))
    assert all(v is None or np.isfinite(v) for v in metrics.values())
    assert metrics['R²'] > 0.7, metrics
    # Comprobar la conversion de coeficientes a unidades originales.
    reconstructed = result[FEATURES].to_numpy() @ coef.iloc[:,1].to_numpy() + intercept
    assert np.allclose(reconstructed, model.predict(result[FEATURES]))
    print(name, {k: round(v, 4) if v is not None else None for k,v in metrics.items()})
app = AppTest.from_file('app.py', default_timeout=30).run()
assert not app.exception, app.exception
app.selectbox[0].select('Elastic Net').run()
assert not app.exception, app.exception
print('OK: PostGIS, filtros, insercion con rollback, cuatro regresores e interfaz.')
