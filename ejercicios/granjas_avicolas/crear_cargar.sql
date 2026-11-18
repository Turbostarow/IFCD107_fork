-- PostgreSQL 16+. Ejecutar en una base vacia o sin esquema avicola.
-- Datos ficticios: explotaciones de gallinas ponedoras, enero de 2026.
-- No borra objetos. Toda la creacion y carga se confirma de forma atomica.
BEGIN;
CREATE SCHEMA avicola;
SET LOCAL search_path = avicola, public;

-- 1. Explotaciones
CREATE TABLE granjas (
 granja_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 codigo varchar(10) NOT NULL UNIQUE,
 nombre text NOT NULL,
 municipio text NOT NULL,
 superficie_m2 numeric(10,2) NOT NULL CHECK(superficie_m2>0)
);
-- 2. Alojamientos
CREATE TABLE naves (
 nave_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 granja_id integer NOT NULL REFERENCES granjas,
 codigo varchar(10) NOT NULL,
 capacidad_aves integer NOT NULL CHECK(capacidad_aves>0),
 sistema text NOT NULL CHECK(sistema IN ('suelo','campero','ecologico')),
 UNIQUE(granja_id,codigo)
);
-- 3. Grupos de aves; una nave puede alojar distintos lotes en el tiempo.
CREATE TABLE lotes (
 lote_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 nave_id integer NOT NULL REFERENCES naves,
 codigo varchar(20) NOT NULL UNIQUE,
 estirpe text NOT NULL,
 fecha_nacimiento date NOT NULL,
 fecha_entrada date NOT NULL,
 fecha_salida date,
 aves_iniciales integer NOT NULL CHECK(aves_iniciales>0),
 CHECK(fecha_entrada>=fecha_nacimiento),
 CHECK(fecha_salida IS NULL OR fecha_salida>=fecha_entrada)
);
-- 4. Proveedores de alimento
CREATE TABLE proveedores (
 proveedor_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 codigo varchar(10) NOT NULL UNIQUE,
 nombre text NOT NULL,
 email text NOT NULL UNIQUE
);
-- 5. Catalogo de alimentos
CREATE TABLE piensos (
 pienso_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 codigo varchar(10) NOT NULL UNIQUE,
 nombre text NOT NULL,
 proteina_pct numeric(5,2) NOT NULL CHECK(proteina_pct>0 AND proteina_pct<=100),
 ecologico boolean NOT NULL DEFAULT false
);
-- 6. Recepciones de alimento: una linea por entrega de un pienso.
CREATE TABLE compras_pienso (
 compra_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 granja_id integer NOT NULL REFERENCES granjas,
 proveedor_id integer NOT NULL REFERENCES proveedores,
 pienso_id integer NOT NULL REFERENCES piensos,
 referencia varchar(30) NOT NULL UNIQUE,
 fecha date NOT NULL,
 kg numeric(12,2) NOT NULL CHECK(kg>0),
 precio_kg numeric(8,4) NOT NULL CHECK(precio_kg>0)
);
-- 7. Raciones registradas por lote, fecha y pienso.
CREATE TABLE consumos_pienso (
 consumo_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 lote_id integer NOT NULL REFERENCES lotes,
 pienso_id integer NOT NULL REFERENCES piensos,
 fecha date NOT NULL,
 kg numeric(10,2) NOT NULL CHECK(kg>0),
 UNIQUE(lote_id,pienso_id,fecha)
);
-- 8. Parte diario: aves al inicio, bajas y huevos del dia.
CREATE TABLE produccion_diaria (
 produccion_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 lote_id integer NOT NULL REFERENCES lotes,
 fecha date NOT NULL,
 aves_inicio integer NOT NULL CHECK(aves_inicio>0),
 bajas integer NOT NULL DEFAULT 0 CHECK(bajas>=0 AND bajas<=aves_inicio),
 huevos_totales integer NOT NULL CHECK(huevos_totales>=0),
 huevos_rotos integer NOT NULL CHECK(huevos_rotos>=0 AND huevos_rotos<=huevos_totales),
 temperatura_c numeric(4,1),
 UNIQUE(lote_id,fecha)
);
-- 9. Compradores; sin datos personales reales.
CREATE TABLE clientes (
 cliente_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 codigo varchar(10) NOT NULL UNIQUE,
 nombre text NOT NULL,
 canal text NOT NULL CHECK(canal IN ('mayorista','comercio','hosteleria')),
 municipio text NOT NULL
);
-- 10. Lineas de venta de huevos trazadas hasta el lote de aves.
CREATE TABLE ventas (
 venta_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 cliente_id integer NOT NULL REFERENCES clientes,
 lote_id integer NOT NULL REFERENCES lotes,
 referencia varchar(30) NOT NULL UNIQUE,
 fecha date NOT NULL,
 huevos integer NOT NULL CHECK(huevos>0),
 precio_docena numeric(8,2) NOT NULL CHECK(precio_docena>0),
 estado text NOT NULL CHECK(estado IN ('pendiente','cobrada','cancelada'))
);

INSERT INTO granjas(codigo,nombre,municipio,superficie_m2) VALUES
 ('G01','Granja Alba','Municipio Norte',12000),
 ('G02','Granja Encina','Municipio Centro',18000),
 ('G03','Granja Ribera','Municipio Sur',24000);
INSERT INTO naves(granja_id,codigo,capacidad_aves,sistema)
SELECT g.granja_id,'N0'||n,1500,
 CASE g.codigo WHEN 'G01' THEN 'suelo' WHEN 'G02' THEN 'campero' ELSE 'ecologico' END
FROM granjas g CROSS JOIN generate_series(1,3) AS t(n)
ORDER BY g.granja_id,n;
INSERT INTO lotes(nave_id,codigo,estirpe,fecha_nacimiento,fecha_entrada,aves_iniciales)
SELECT nave_id,'L2025-'||lpad(nave_id::text,2,'0'),
 CASE WHEN nave_id%2=0 THEN 'Estirpe B' ELSE 'Estirpe A' END,
 DATE '2025-07-01',DATE '2025-11-01',1000+nave_id*20
FROM naves ORDER BY nave_id;
INSERT INTO proveedores(codigo,nombre,email) VALUES
 ('P01','Alimentos Sierra','contacto@sierra.example'),
 ('P02','Nutricion Vega','contacto@vega.example'),
 ('P03','Piensos Horizonte','contacto@horizonte.example');
INSERT INTO piensos(codigo,nombre,proteina_pct,ecologico) VALUES
 ('F01','Puesta convencional',17.5,false),
 ('F02','Puesta campera',18,false),
 ('F03','Puesta ecologica',18.5,true),
 ('F04','Transicion',16,false);
INSERT INTO clientes(codigo,nombre,canal,municipio) VALUES
 ('C01','Distribuciones Alba','mayorista','Municipio Norte'),
 ('C02','Mercado Central','comercio','Municipio Centro'),
 ('C03','Restauracion Ribera','hosteleria','Municipio Sur'),
 ('C04','Tienda Prado','comercio','Municipio Norte'),
 ('C05','Distribuciones Sol','mayorista','Municipio Sur'),
 ('C06','Cocina Escuela','hosteleria','Municipio Centro');
-- Un pienso principal por granja; F04 queda sin compras ni consumo.
INSERT INTO compras_pienso(granja_id,proveedor_id,pienso_id,referencia,fecha,kg,precio_kg)
SELECT granja_id,granja_id,granja_id,'REC-'||codigo||'-'||d,
 DATE '2025-12-31'+d,6000,0.32+granja_id*0.04
FROM granjas CROSS JOIN (VALUES(0),(15)) AS t(d)
ORDER BY granja_id,d;
-- Dos bajas al final del dia 10 y del dia 20: continuidad del censo.
INSERT INTO produccion_diaria(lote_id,fecha,aves_inicio,bajas,huevos_totales,huevos_rotos,temperatura_c)
SELECT l.lote_id,DATE '2026-01-01'+d,
 l.aves_iniciales-CASE WHEN d>=20 THEN 4 WHEN d>=10 THEN 2 ELSE 0 END,
 CASE WHEN d IN (9,19) THEN 2 ELSE 0 END,
 floor((l.aves_iniciales-CASE WHEN d>=20 THEN 4 WHEN d>=10 THEN 2 ELSE 0 END)
 * (0.80+((d+l.lote_id)%8)*0.01))::integer,
 5+(d+l.lote_id)%8,
 CASE WHEN d%11=0 THEN NULL ELSE 19+((d+l.lote_id)%9)*0.5 END
FROM lotes l CROSS JOIN generate_series(0,29) AS t(d)
ORDER BY l.lote_id,d;
INSERT INTO consumos_pienso(lote_id,pienso_id,fecha,kg)
SELECT p.lote_id,n.granja_id,p.fecha,round(p.aves_inicio*0.115,2)
FROM produccion_diaria p JOIN lotes l USING(lote_id) JOIN naves n USING(nave_id)
ORDER BY p.lote_id,p.fecha;
-- Se vende una parte de la puesta de ese mismo dia; no hay saldo inicial.
INSERT INTO ventas(cliente_id,lote_id,referencia,fecha,huevos,precio_docena,estado)
SELECT 1+(p.lote_id%5),p.lote_id,'V-'||p.lote_id||'-'||to_char(p.fecha,'YYYYMMDD'),
 p.fecha,600,2.00+p.lote_id*0.10,
 CASE WHEN extract(day FROM p.fecha)=27 THEN 'pendiente'
 WHEN p.lote_id=9 AND extract(day FROM p.fecha)=3 THEN 'cancelada' ELSE 'cobrada' END
FROM produccion_diaria p WHERE extract(day FROM p.fecha)::integer%3=0
ORDER BY p.lote_id,p.fecha;

CREATE INDEX idx_naves_granja ON naves(granja_id);
CREATE INDEX idx_lotes_nave ON lotes(nave_id);
CREATE INDEX idx_compras_granja_fecha ON compras_pienso(granja_id,fecha);
CREATE INDEX idx_compras_proveedor ON compras_pienso(proveedor_id);
CREATE INDEX idx_compras_pienso ON compras_pienso(pienso_id);
CREATE INDEX idx_consumos_pienso ON consumos_pienso(pienso_id);
CREATE INDEX idx_ventas_lote_fecha ON ventas(lote_id,fecha);
CREATE INDEX idx_ventas_cliente_fecha ON ventas(cliente_id,fecha);
COMMIT;

-- Recuentos esperados: 3,9,9,3,4,6,270,270,6,90 = 670 filas.
SELECT 'granjas' tabla,count(*) filas FROM avicola.granjas
UNION ALL SELECT 'naves',count(*) FROM avicola.naves
UNION ALL SELECT 'lotes',count(*) FROM avicola.lotes
UNION ALL SELECT 'proveedores',count(*) FROM avicola.proveedores
UNION ALL SELECT 'piensos',count(*) FROM avicola.piensos
UNION ALL SELECT 'compras_pienso',count(*) FROM avicola.compras_pienso
UNION ALL SELECT 'consumos_pienso',count(*) FROM avicola.consumos_pienso
UNION ALL SELECT 'produccion_diaria',count(*) FROM avicola.produccion_diaria
UNION ALL SELECT 'clientes',count(*) FROM avicola.clientes
UNION ALL SELECT 'ventas',count(*) FROM avicola.ventas;
