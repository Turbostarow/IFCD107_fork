-- 1. Comprobar extension, cantidad y SRID.
SELECT PostGIS_Version();
SELECT count(*), ST_SRID(ubicacion::geometry) AS srid
FROM aprendizaje.viviendas GROUP BY srid;

-- 2. Insertar un ejemplo manual. Ejecutar una vez.
INSERT INTO aprendizaje.viviendas
 (referencia,superficie,habitaciones,antiguedad,precio,ubicacion)
VALUES ('EJERCICIO-001',90,3,20,300000,
 ST_SetSRID(ST_MakePoint(-3.70,40.42),4326)::geography);

-- 3. Proximidad: geography devuelve distancias en metros.
SELECT referencia, superficie, precio,
 ST_Distance(ubicacion,ST_SetSRID(ST_MakePoint(-3.7038,40.4168),4326)::geography) AS metros
FROM aprendizaje.viviendas
WHERE ST_DWithin(ubicacion,
 ST_SetSRID(ST_MakePoint(-3.7038,40.4168),4326)::geography,2000)
ORDER BY metros;

-- 4. Agregacion por intervalos de distancia al centro fijo.
SELECT floor(ST_Distance(ubicacion,
 ST_SetSRID(ST_MakePoint(-3.7038,40.4168),4326)::geography)/1000) AS anillo_km,
 count(*) AS viviendas, round(avg(precio)::numeric,2) AS precio_medio
FROM aprendizaje.viviendas GROUP BY anillo_km ORDER BY anillo_km;

-- 5. Examinar el plan. Con pocos registros puede preferirse un escaneo secuencial.
EXPLAIN (ANALYZE, BUFFERS)
SELECT referencia FROM aprendizaje.viviendas
WHERE ST_DWithin(ubicacion,
 ST_SetSRID(ST_MakePoint(-3.7038,40.4168),4326)::geography,500);
