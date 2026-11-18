# Gestión de granjas avícolas · PostgreSQL

Laboratorio para IFCD107 con **10 tablas y 670 filas ficticias**. Caso de gallinas ponedoras: tres granjas, nueve naves y nueve lotes; treinta días de producción y consumo (1–30 de enero de 2026), compras y ventas de huevos.

## Crear la base y cargar todo

Requisitos: PostgreSQL 16 o posterior, cliente psql y usuario con permiso CREATEDB. Desde esta carpeta:

```powershell
psql -X -h localhost -p 5432 -U postgres -d postgres -v ON_ERROR_STOP=1 -f instalar.psql
```

Se solicita la contraseña de forma interactiva. Crea la base `granjas_avicolas`, conecta, crea el esquema `avicola`, carga y comprueba datos. Las rutas internas se resuelven respecto al script. No requiere extensiones ni descargas.

Si la base ya existe, la instalación se detiene sin borrarla. Si se interrumpe después de crear la base y antes de cargar, ejecutar directamente crear_cargar.sql en esa base. La creación de tablas y carga es una transacción: un error revierte ese bloque. No usar la opción --single-transaction con instalar.psql, ya que CREATE DATABASE requiere ejecutarse fuera de una transacción.

## Alternativa pgAdmin o usuario sin CREATEDB

1. Pedir al administrador una base vacía `granjas_avicolas`, o crearla en pgAdmin desde Databases → Create → Database.
2. Abrir Query Tool **en esa base** y ejecutar `crear_cargar.sql` completo.
3. Ejecutar `verificar.sql`.

`instalar.psql` contiene órdenes propias de psql y no funciona en Query Tool. El SQL de creación y carga sí funciona en ambos clientes.

## Modelo y recuentos

| Tabla | Filas | Contenido |
|---|---:|---|
| granjas |3|Explotaciones|
| naves |9|Alojamiento y capacidad|
| lotes |9|Estirpe, fechas y censo inicial|
| proveedores |3|Suministradores ficticios|
| piensos |4|Catálogo de alimentos|
| compras_pienso |6|Recepciones por granja y pienso|
| consumos_pienso |270|Consumo diario por lote y pienso|
| produccion_diaria |270|Censo, bajas, huevos y temperatura|
| clientes |6|Compradores ficticios|
| ventas |90|Ventas de huevos por lote|

Relaciones: granjas → naves → lotes → producción/consumos/ventas. Proveedores y piensos → compras; granjas → compras; piensos → consumos; clientes → ventas. Las claves son identity: las nuevas inserciones continúan automáticamente después de los datos cargados.

## Prácticas

`practicas.sql` contiene 15 ejercicios de filtros, uniones, agregaciones, ventanas, transacciones y preparación de variables para IA. `soluciones.sql` incluye consultas ejecutables y crea una vista adicional, sin añadir tablas. Ejecutarlo después de intentar los ejercicios:

```powershell
psql -X -h localhost -U postgres -d granjas_avicolas -v ON_ERROR_STOP=1 -f soluciones.sql
```

Casos intencionados: cliente C06 sin ventas, pienso F04 sin movimientos, temperaturas NULL, ventas pendientes y una cancelada. La puesta es huevos por aves-día; la conversión alimentaria se expresa en kg por docena comercializable. Los precios de ventas son por docena y los de compra por kg. Son datos didácticos, no parámetros técnicos de manejo animal.

## Alcance del modelo

Las ventas son líneas trazadas al lote de aves, no facturas multilínea ni lotes de envasado. No se modelan IVA, cobros parciales, sanidad, movimientos de aves, saldo inicial ni mermas de almacén. Las granjas tienen un pienso principal durante la muestra. Las bajas suceden al final del día y el censo siguiente descuenta esas bajas.

PRIMARY KEY, FOREIGN KEY, UNIQUE, NOT NULL y CHECK protegen integridad local. Las reglas entre filas (capacidad, cronología, continuidad del censo, ocupación temporal y existencias) no están todas impuestas por triggers. verificar.sql comprueba parte de ellas en la carga inicial; las prácticas pueden ampliar esas validaciones. La comprobación de ventas es válida para esta muestra, con una venta por lote y día; un sistema real necesita saldo acumulado y control concurrente de existencias.

## Referencias

- https://www.postgresql.org/docs/16/sql-createdatabase.html
- https://www.postgresql.org/docs/16/app-psql.html

Consultadas el 18/11/2026. Material original para prácticas.
