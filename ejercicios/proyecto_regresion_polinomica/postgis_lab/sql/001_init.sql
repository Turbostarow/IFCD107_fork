CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS aprendizaje;
CREATE TABLE IF NOT EXISTS aprendizaje.viviendas (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 referencia text NOT NULL UNIQUE,
 superficie double precision NOT NULL CHECK (superficie BETWEEN 15 AND 500),
 habitaciones integer NOT NULL CHECK (habitaciones BETWEEN 1 AND 10),
 antiguedad integer NOT NULL CHECK (antiguedad BETWEEN 0 AND 150),
 precio double precision NOT NULL CHECK (precio > 0),
 ubicacion geography(Point,4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS viviendas_ubicacion_gist
 ON aprendizaje.viviendas USING gist(ubicacion);
-- Datos reproducibles. Son simulaciones, no anuncios ni valoraciones reales.
SELECT setseed(0.42);
WITH datos AS MATERIALIZED (
 SELECT i, 35 + random()*145 AS superficie,
  1 + floor(random()*5)::int AS habitaciones,
  floor(random()*81)::int AS antiguedad,
  ST_SetSRID(ST_MakePoint(-3.7038 + (random()-0.5)*0.16,
                          40.4168 + (random()-0.5)*0.12),4326)::geography AS ubicacion,
  (random()+random()+random()+random()+random()+random()-3)*18000 AS ruido
 FROM generate_series(1,2500) AS s(i)
), calculados AS (
 SELECT *, ST_Distance(ubicacion,
  ST_SetSRID(ST_MakePoint(-3.7038,40.4168),4326)::geography)/1000 AS distancia
 FROM datos
)
INSERT INTO aprendizaje.viviendas(referencia,superficie,habitaciones,antiguedad,precio,ubicacion)
 SELECT 'SIM-' || lpad(i::text,4,'0'), superficie, habitaciones, antiguedad,
  greatest(10000,80000 + 2800*superficie + 12000*habitaciones
                   - 900*antiguedad - 14000*distancia + ruido), ubicacion
 FROM calculados ON CONFLICT (referencia) DO NOTHING;
ANALYZE aprendizaje.viviendas;
