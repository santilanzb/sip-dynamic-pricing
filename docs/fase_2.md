# Fase 2: Diseño de Arquitectura y Pipeline ETL

**Fecha:** 2026-02-19  
**Autor:** Santiago Lanz, Diego Blanco

---

## Resumen

Se diseñó la arquitectura completa del sistema SIP Dynamic Pricing y se implementó el pipeline ETL que transforma los datos raw en un modelo estrella optimizado para análisis y entrenamiento de modelos ML.

## Entregables

### 1. Documento de Arquitectura (`docs/arquitectura.md`)

Documento técnico completo que incluye:
- Arquitectura de 4 capas (Datos, Procesamiento, Lógica, Presentación)
- Modelo de datos estrella (fact_ventas + 3 dimensiones)
- Especificación de features para ML (6 categorías, ~30 features)
- Diseño de modelos (XGBoost, Random Forest, LightGBM)
- Función objetivo de optimización de precios
- Interfaces de módulos del sistema

### 2. Pipeline ETL (`src/data/transform.py`)

Script que ejecuta la transformación completa:

```python
from src.data.transform import run_etl
tables = run_etl(data_path='data')
```

## Modelo de Datos

### Esquema Estrella Implementado

```
                    DIM_TIEMPO
                        │
DIM_PRODUCTO ──── FACT_VENTAS ──── DIM_SUCURSAL
```

### Tablas Generadas

| Tabla | Registros | Columnas | Descripción |
|-------|-----------|----------|-------------|
| fact_ventas | 1,326,188 | 19 | Ventas diarias por producto-sucursal |
| dim_tiempo | 1,096 | 12 | Calendario con feriados Venezuela |
| dim_producto | 1,819 | 11 | Catálogo de productos con atributos |
| dim_sucursal | 4 | 4 | Sucursales activas |

### Columnas de fact_ventas

| Columna | Tipo | Descripción |
|---------|------|-------------|
| fecha | datetime | Fecha de venta |
| producto_id | string | Código interno del producto |
| sucursal_id | string | Identificador de sucursal |
| unidades | float | Unidades vendidas |
| precio_unitario_usd | float | Precio por unidad en USD |
| ingreso_usd | float | Ingreso total en USD |
| costo_usd | float | Costo total en USD |
| margen_usd | float | Margen bruto en USD |
| margen_pct | float | Margen porcentual |
| tiene_promocion | int | Flag de promoción activa (0/1) |
| tipo_promocion | int | Tipo de promoción (1-11) |
| pct_descuento | float | Porcentaje de descuento |
| clase | string | Clase de producto |
| es_perecedero | int | Flag perecedero (0/1) |
| dia_semana | int | Día de la semana (0-6) |
| es_fin_semana | int | Flag fin de semana (0/1) |
| mes | int | Mes (1-12) |
| año | int | Año |
| tasa_bcv | float | Tasa BCV del día |

## Integración de Promociones

- **Productos con promociones en ventas:** 422
- **Registros de promoción relevantes:** 9,500
- **Ventas con promoción activa:** 69,884 (5.3%)

### Tipos de Promoción Identificados

| Tipo | Descripción |
|------|-------------|
| 1 | Precio Oferta |
| 2 | % Descuento |
| 3 | Descuento Fijo |
| 4-7 | M×N (varias modalidades) |
| 8-11 | Premios |

## Calendario Venezuela

Se marcaron 27 feriados en el período 2023-2025:

- Año Nuevo (1 Ene)
- Día del Trabajador (1 May)
- Batalla de Carabobo (24 Jun)
- Día de la Independencia (5 Jul)
- Natalicio de Bolívar (24 Jul)
- Día de la Resistencia Indígena (12 Oct)
- Nochebuena (24 Dic)
- Navidad (25 Dic)
- Fin de Año (31 Dic)

## Próximos Pasos (Fase 3)

1. **Feature Engineering:** Crear features de lags, promedios móviles, tendencias
2. **Encoding:** Target encoding para categorías
3. **Preparar datasets:** Train/Validation/Test split temporal
4. **Iniciar entrenamiento** de modelos baseline

## Archivos Modificados/Creados

```
docs/arquitectura.md          # Nuevo - Documento de arquitectura
src/data/transform.py         # Nuevo - Pipeline ETL
data/processed/
├── fact_ventas.parquet       # Nuevo - Tabla de hechos
├── dim_tiempo.parquet        # Nuevo - Dimensión tiempo
├── dim_producto.parquet      # Nuevo - Dimensión producto
└── dim_sucursal.parquet      # Nuevo - Dimensión sucursal
```

---

*Documentación generada como parte del proyecto de tesis SIP Dynamic Pricing*
