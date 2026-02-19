# Fase 0: Setup del Entorno y Extracción de Datos

**Fecha:** 2026-02-19  
**Estado:** ✅ Completada

## Resumen

Configuración inicial del proyecto y extracción de datos desde SQL Server para el sistema SIP Dynamic Pricing.

## Actividades Realizadas

### 0.1 Configuración del Entorno

- ✅ Estructura de directorios creada
- ✅ Entorno virtual Python configurado (`venv/`)
- ✅ Dependencias instaladas (`requirements.txt`)
- ✅ Archivo de configuración creado (`config.yaml`)
- ✅ Script de extracción desarrollado (`src/data/extract.py`)

### 0.2 Conexiones SQL Server

| Servidor | Base de Datos | Propósito |
|----------|---------------|-----------|
| sermgp03 | EMP03 | CompraVenta 2023, 2024 (Ene-Oct) |
| 10.10.0.23 | EMP04 | CompraVenta 2024 (Nov-Dic), 2025, Ajustes |
| serstellar | VAD10 | Promociones |

**Driver:** ODBC Driver 18 for SQL Server  
**Autenticación:** SQL Server (usuario/contraseña)

### 0.3 Datos Extraídos

#### CompraVenta (`data/raw/compraventa_raw.parquet`)

| Métrica | Valor |
|---------|-------|
| Total registros | 1,326,188 |
| Sucursales | 6 |
| Productos únicos | 1,819 |
| Rango de fechas | 2023-01-01 a 2025-12-31 |

**Categorías extraídas:**
- `08FRUV` - Frutas y Verduras
- `03CARN` - Carnes
- `05CHAR` - Charcutería

**Campos principales:**
- Fecha, Sucursal, Codigo_Interno, SKU, Descripcion
- Clase, Departamento, SubDepartamento, Categoria
- Unidades_Venta_Cantidad, Precio_Venta_Total
- Costo_Actual, Costo_Venta_Total
- Unidades_Compra_Cantidad, Costo_Total_Compra

#### Promociones (`data/raw/promociones_raw.parquet`)

| Métrica | Valor |
|---------|-------|
| Total registros | 43,046 |
| Promociones únicas | 3,786 |
| Productos con promoción | 5,762 |

**Distribución por tipo:**
- Tipo 1 (Precio Oferta): 37,471 (87%)
- Tipo 2 (% Descuento): 2,163
- Tipo 5 (M×N Precio Oferta): 1,906
- Otros tipos: 1,506

**Campos principales:**
- Cod_Promocion, Fecha_Inicio, Fecha_Fin
- Tipo_Promocion, Cod_Producto
- Precio_Oferta, Monto_Descuento, Porcentaje_Descuento1

#### Ajustes de Inventario (`data/raw/ajustes_raw.parquet`)

| Métrica | Valor |
|---------|-------|
| Total registros | 16,267,544 |
| Productos con ajustes | 10,790 |

**Fuentes:**
- IV10001 (actual): 126 registros
- IV30300 (histórico): 16,267,418 registros

**Campos principales:**
- Codigo_Producto, Fecha_Ajuste, Tipo_Documento
- Cantidad, Costo_Unitario, Ubicacion
- Codigo_Razon

## Archivos Generados

```
sip-dynamic-pricing/
├── config.yaml                    # Configuración de conexiones
├── requirements.txt               # Dependencias Python
├── README.md                      # Documentación del proyecto
├── .gitignore                     # Archivos a ignorar
├── data/
│   └── raw/
│       ├── compraventa_raw.parquet   # 1.3M registros
│       ├── promociones_raw.parquet   # 43K registros
│       └── ajustes_raw.parquet       # 16.3M registros
└── src/
    └── data/
        └── extract.py             # Script de extracción
```

## Observaciones Técnicas

1. **División de datos 2024:** Los datos de CompraVenta 2024 están divididos entre dos servidores debido a una migración del supermercado (Ene-Oct en EMP03, Nov-Dic en EMP04).

2. **Driver ODBC 18:** Requiere `TrustServerCertificate=yes` para conexiones internas.

3. **Esquemas diferentes en ajustes:**
   - IV30300 (histórico): tiene fecha (`DOCDATE`)
   - IV10001 (actual): no tiene fecha

4. **Promociones tipo 1:** El 87% de las promociones son de tipo "Precio Oferta" (precio directo).

## Próximos Pasos

→ **Fase 1:** Análisis Exploratorio (EDA)
- Distribución temporal de ventas
- Análisis de estacionalidad
- Correlación precio-demanda
- Efecto de promociones
