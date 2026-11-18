-- Comprobaciones de carga: aborta con excepcion si algo no coincide.
DO $$
DECLARE n integer;
BEGIN
 SELECT count(*) INTO n FROM information_schema.tables
 WHERE table_schema='avicola' AND table_type='BASE TABLE';
 IF n<>10 THEN RAISE EXCEPTION 'Se esperaban 10 tablas: %',n; END IF;
 IF (SELECT count(*) FROM avicola.granjas)<>3
 OR (SELECT count(*) FROM avicola.naves)<>9
 OR (SELECT count(*) FROM avicola.lotes)<>9
 OR (SELECT count(*) FROM avicola.proveedores)<>3
 OR (SELECT count(*) FROM avicola.piensos)<>4
 OR (SELECT count(*) FROM avicola.compras_pienso)<>6
 OR (SELECT count(*) FROM avicola.consumos_pienso)<>270
 OR (SELECT count(*) FROM avicola.produccion_diaria)<>270
 OR (SELECT count(*) FROM avicola.clientes)<>6
 OR (SELECT count(*) FROM avicola.ventas)<>90
 THEN RAISE EXCEPTION 'Recuentos de carga incorrectos'; END IF;
 IF (SELECT sum(bajas) FROM avicola.produccion_diaria)<>36
 THEN RAISE EXCEPTION 'Bajas esperadas: 36'; END IF;
 IF EXISTS(SELECT 1 FROM avicola.lotes l JOIN avicola.naves n USING(nave_id)
 WHERE l.aves_iniciales>n.capacidad_aves)
 THEN RAISE EXCEPTION 'Capacidad excedida'; END IF;
 IF EXISTS(SELECT 1 FROM avicola.produccion_diaria p JOIN avicola.lotes l USING(lote_id)
 WHERE p.fecha<l.fecha_entrada OR (l.fecha_salida IS NOT NULL AND p.fecha>l.fecha_salida))
 THEN RAISE EXCEPTION 'Produccion fuera del periodo del lote'; END IF;
 IF EXISTS(SELECT 1 FROM (
 SELECT lote_id,fecha,aves_inicio,lag(aves_inicio-bajas) OVER(PARTITION BY lote_id ORDER BY fecha) esperado
 FROM avicola.produccion_diaria) q WHERE esperado IS NOT NULL AND aves_inicio<>esperado)
 THEN RAISE EXCEPTION 'Censo discontinuo'; END IF;
 IF EXISTS(SELECT 1 FROM avicola.ventas v LEFT JOIN avicola.produccion_diaria p
 ON p.lote_id=v.lote_id AND p.fecha=v.fecha
 WHERE v.estado<>'cancelada' AND (p.produccion_id IS NULL OR v.huevos>p.huevos_totales-p.huevos_rotos))
 THEN RAISE EXCEPTION 'Venta sin produccion suficiente en la carga'; END IF;
 RAISE NOTICE 'Correcto: 10 tablas, 670 filas y controles de coherencia superados';
END $$;
