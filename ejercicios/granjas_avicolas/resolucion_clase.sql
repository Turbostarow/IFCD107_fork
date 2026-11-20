-- Enunciados. Resolver antes de consultar soluciones.sql.
-- 1. Listar granjas, naves y capacidad, ordenadas por granja/nave.
select codigo,nombre ,n.codigo, n.capacidad_aves
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id;
-- capacidad de cada granja

select g.codigo,nombre , sum(n.capacidad_aves)
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
group by g.codigo,nombre;
-- 2. Calcular ocupacion inicial (%) por nave. Detectar excesos.
select 'Valores granja',g.codigo,nombre ,'valores nave',n.codigo, n.capacidad_aves,
'Valores lotes' , l.aves_iniciales,  round(100.0*l.aves_iniciales/n.capacidad_aves,2)
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id;

select g.codigo,nombre , sum(n.capacidad_aves),sum( l.aves_iniciales),
 round(100.0*sum( l.aves_iniciales)/ sum(n.capacidad_aves),2)
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id
group by g.codigo,nombre;


-- 3. Resumir huevos totales, rotos y comercializables por granja.
select g.codigo,nombre ,n.codigo, sum(p.huevos_rotos), 
sum(p.huevos_totales), sum(p.huevos_totales) -  sum(p.huevos_rotos) as huevos_com 
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id
join avicola.produccion_diaria p ON p.lote_id = l.lote_id
group by g.codigo,nombre ,n.codigo;

select g.codigo,nombre ,sum(p.huevos_rotos), 
sum(p.huevos_totales), sum(p.huevos_totales) -  sum(p.huevos_rotos) as huevos_com 
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id
join avicola.produccion_diaria p ON p.lote_id = l.lote_id
group by g.codigo,nombre
having sum(p.huevos_totales) -  sum(p.huevos_rotos) > 27000;

-- 3.3 actualizar registro de la tabla avicola.produccion_diaria 
-- para clave production_id 72 poniendo a temperatura a 19 grados

-- 3.4 Calcular la temperatura media  por granja y nave
select g.codigo,nombre , n.codigo,
avg(p.temperatura_c)
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id
join avicola.produccion_diaria p ON p.lote_id = l.lote_id
group by g.codigo,nombre,n.codigo
-- temperatura media por día para todas las naves 
select g.codigo,nombre , n.codigo
avg(p.temperatura_c)
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id
join avicola.produccion_diaria p ON p.lote_id = l.lote_id
group by g.codigo,nombre,n.codigo
-- temperatura media por día para todas las semana 
select g.codigo,nombre,p.fecha,
avg(p.temperatura_c)
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id
join avicola.produccion_diaria p ON p.lote_id = l.lote_id
where p.temperatura_c is not null
group by g.codigo,nombre,p.fecha


select g.codigo,nombre,EXTRACT(DOW FROM p.fecha),
round(avg(p.temperatura_c),3)
from avicola.granjas g
join avicola.naves n ON n.granja_id = g.granja_id
join avicola.lotes l on l.nave_id = n.nave_id
join avicola.produccion_diaria p ON p.lote_id = l.lote_id
where p.temperatura_c is not null
group by g.codigo,nombre,EXTRACT(DOW FROM p.fecha)

-- 4. Calcular puesta por cada 100 aves-dia, ponderando por censo.

-- 5. Calcular kg consumidos por docena comercializable por lote.
-- 6. Calcular gasto de compras de pienso por proveedor.
-- 7. Encontrar clientes sin ventas y piensos sin consumo (LEFT JOIN).
-- 8. Calcular ingresos por granja excluyendo ventas canceladas.
-- 9. Calcular saldo teorico de alimento por granja y pienso al 30/1.
-- 10. Mostrar media movil de 7 dias de produccion por lote.
-- 11. Detectar temperaturas ausentes y dias de puesta inferior al 82%.
-- 12. Ensayar una actualizacion de precio y revertir con ROLLBACK.
-- 13. IA: una fila por lote con variables anteriores al 21/1/2026.
--     Variables: huevos, consumo, bajas y temperatura media.
--     Agregar cada tabla por separado para evitar multiplicar filas.
-- 14. Exportar ese conjunto a CSV desde psql.
-- 15. Comparar EXPLAIN y EXPLAIN ANALYZE para consultas por lote/fecha.
