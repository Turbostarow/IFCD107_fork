SET search_path TO avicola,public;
-- 1 y 2
SELECT g.nombre,n.codigo,n.capacidad_aves,l.aves_iniciales,
 round(100.0*l.aves_iniciales/n.capacidad_aves,2) ocupacion_pct
FROM granjas g JOIN naves n USING(granja_id) JOIN lotes l USING(nave_id)
ORDER BY g.granja_id,n.codigo;
-- 3 y 4. Puesta por 100 aves-dia, no promedio simple de porcentajes.
SELECT g.nombre,sum(p.huevos_totales) huevos,sum(p.huevos_rotos) rotos,
 sum(p.huevos_totales-p.huevos_rotos) comercializables,
 round(100.0*sum(p.huevos_totales)/NULLIF(sum(p.aves_inicio),0),2) puesta_pct
FROM granjas g JOIN naves n USING(granja_id) JOIN lotes l USING(nave_id)
JOIN produccion_diaria p USING(lote_id) GROUP BY g.granja_id,g.nombre;
-- 5. Agregar antes de unir: admite varios piensos al dia.
WITH c AS (SELECT lote_id,sum(kg) kg FROM consumos_pienso GROUP BY lote_id),
 p AS (SELECT lote_id,sum(huevos_totales-huevos_rotos) huevos
 FROM produccion_diaria GROUP BY lote_id)
SELECT lote_id,round(c.kg*12/NULLIF(p.huevos,0),3) kg_por_docena
FROM c JOIN p USING(lote_id);
-- 6
SELECT p.nombre,round(sum(c.kg*c.precio_kg),2) gasto
FROM proveedores p JOIN compras_pienso c USING(proveedor_id) GROUP BY p.proveedor_id;
-- 7. Esperados: C06 y F04.
SELECT c.codigo FROM clientes c LEFT JOIN ventas v USING(cliente_id) WHERE v.venta_id IS NULL;
SELECT p.codigo FROM piensos p LEFT JOIN consumos_pienso c USING(pienso_id) WHERE c.consumo_id IS NULL;
-- 8. Ingresos comprometidos y efectivamente cobrados por separado.
SELECT g.nombre,round(sum(v.huevos/12.0*v.precio_docena),2) importe_no_cancelado,
 round(COALESCE(sum(v.huevos/12.0*v.precio_docena) FILTER(WHERE v.estado='cobrada'),0),2) cobrado
FROM ventas v JOIN lotes l USING(lote_id) JOIN naves n USING(nave_id)
JOIN granjas g USING(granja_id) WHERE v.estado<>'cancelada' GROUP BY g.granja_id;
-- 9. Sin saldo inicial, mermas ni movimientos entre granjas en este caso.
WITH entradas AS (
 SELECT granja_id,pienso_id,sum(kg) kg FROM compras_pienso
 WHERE fecha<=DATE '2026-01-30' GROUP BY granja_id,pienso_id
), salidas AS (
 SELECT n.granja_id,c.pienso_id,sum(c.kg) kg FROM consumos_pienso c
 JOIN lotes l USING(lote_id) JOIN naves n USING(nave_id)
 WHERE c.fecha<=DATE '2026-01-30' GROUP BY n.granja_id,c.pienso_id
)
SELECT COALESCE(e.granja_id,s.granja_id) granja_id,
 COALESCE(e.pienso_id,s.pienso_id) pienso_id,
 COALESCE(e.kg,0)-COALESCE(s.kg,0) saldo_kg
FROM entradas e FULL JOIN salidas s USING(granja_id,pienso_id);
-- 10. Hay exactamente un registro diario continuo por lote en la muestra.
SELECT lote_id,fecha,huevos_totales,
 round(avg(huevos_totales) OVER(PARTITION BY lote_id ORDER BY fecha
 ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),2) media_7_registros
FROM produccion_diaria ORDER BY lote_id,fecha;
-- 11
SELECT lote_id,fecha FROM produccion_diaria WHERE temperatura_c IS NULL;
SELECT lote_id,fecha FROM produccion_diaria WHERE 100.0*huevos_totales/aves_inicio<82;
-- 12
BEGIN;
UPDATE ventas SET precio_docena=precio_docena*1.05 WHERE venta_id=1;
ROLLBACK;
-- 13. Corte exclusivo. NULL en temperatura si no hay observaciones.
CREATE OR REPLACE VIEW avicola.variables_ia AS
WITH p AS (
 SELECT lote_id,sum(huevos_totales) huevos,sum(bajas) bajas,
 avg(temperatura_c) temperatura_media,count(temperatura_c) temperaturas_observadas
 FROM avicola.produccion_diaria WHERE fecha<DATE '2026-01-21' GROUP BY lote_id
), c AS (
 SELECT lote_id,sum(kg) consumo_kg FROM avicola.consumos_pienso
 WHERE fecha<DATE '2026-01-21' GROUP BY lote_id
)
SELECT l.lote_id,COALESCE(p.huevos,0) huevos,COALESCE(p.bajas,0) bajas,
 p.temperatura_media,COALESCE(p.temperaturas_observadas,0) temperaturas_observadas,
 COALESCE(c.consumo_kg,0) consumo_kg
FROM avicola.lotes l LEFT JOIN p USING(lote_id) LEFT JOIN c USING(lote_id)
WHERE l.fecha_entrada<DATE '2026-01-21'
 AND (l.fecha_salida IS NULL OR l.fecha_salida>=DATE '2026-01-21');
SELECT * FROM avicola.variables_ia ORDER BY lote_id;
-- 14. En psql, ejecutar en una sola linea:
-- \copy (SELECT * FROM avicola.variables_ia ORDER BY lote_id) TO 'variables_avicolas.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
-- 15
EXPLAIN (ANALYZE,BUFFERS) SELECT * FROM produccion_diaria
WHERE lote_id=1 AND fecha BETWEEN DATE '2026-01-10' AND DATE '2026-01-20';
