import os
import pandas as pd
import psycopg
from psycopg.rows import dict_row


def connect():
    return psycopg.connect(host=os.getenv('PGHOST', 'postgis-lab'),
                          port=os.getenv('PGPORT', '5433'),
                          dbname=os.getenv('PGDATABASE', 'georegresion'),
                          user=os.getenv('PGUSER', 'alumno'),
                          password=os.getenv('PGPASSWORD', 'postgis_lab_local'),
                          connect_timeout=5, row_factory=dict_row)


def consultar(lon=-3.7038, lat=40.4168, radio=15000, minimo=15):
    # El centro de referencia del modelo permanece fijo aunque cambie el filtro.
    sql = '''SELECT id, referencia, superficie, habitaciones, antiguedad, precio,
      ST_X(ubicacion::geometry) AS longitud,
      ST_Y(ubicacion::geometry) AS latitud,
      ST_Distance(ubicacion, ST_SetSRID(ST_MakePoint(-3.7038,40.4168),4326)::geography)/1000
        AS distancia_centro_km
      FROM aprendizaje.viviendas
      WHERE ST_DWithin(ubicacion,ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,%s)
        AND superficie >= %s ORDER BY id'''
    with connect() as conn:
        return pd.DataFrame(conn.execute(sql, (lon, lat, radio, minimo)).fetchall())


def insertar(referencia, superficie, habitaciones, antiguedad, precio, lon, lat):
    with connect() as conn:
        conn.execute('''INSERT INTO aprendizaje.viviendas
          (referencia,superficie,habitaciones,antiguedad,precio,ubicacion)
          VALUES (%s,%s,%s,%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography)''',
          (referencia, superficie, habitaciones, antiguedad, precio, lon, lat))
