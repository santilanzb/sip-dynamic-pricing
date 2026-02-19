# Fase 1: Análisis Exploratorio de Datos (EDA)

**Fecha:** 2026-02-19  
**Autor:** Santiago Lanz, Diego Blanco

---

## Resumen Ejecutivo

Se realizó el análisis exploratorio completo de los datos extraídos en la Fase 0, identificando patrones clave de demanda, estacionalidad y elasticidad precio-demanda que servirán como base para el modelo de pricing dinámico.

## Datos Analizados

| Dataset | Registros | Período |
|---------|-----------|---------|
| CompraVenta | 1,326,188 | 2023-01-01 a 2025-12-31 |
| Promociones | 43,046 | Histórico completo |
| Ajustes | 16,267,544 | Histórico completo |

- **Sucursales activas:** 4 (SUC001, SUC002, SUC003, SUC004)
- **Sucursales inactivas:** 2 (DESMIR, SUCCDE - sin ventas en el período)
- **Productos únicos:** 1,819
- **Categorías:** 3 (03CARN, 05CHAR, 08FRUV)

## Métricas Financieras

### Valores Nominales (Bs)
| Métrica | Valor |
|---------|-------|
| Ingresos totales | 5,294,683,078 Bs |
| Margen bruto total | 1,749,533,249 Bs |
| Margen bruto promedio | 33.0% |

### Valores Normalizados (USD) - Con tasas BCV reales
| Métrica | Valor |
|---------|-------|
| Ingresos totales | $61,157,742 USD |
| Margen bruto total | $19,859,075 USD |
| Margen bruto promedio | 32.5% |

**Nota:** Normalización realizada con histórico BCV (2021-08-02 a 2026-02-19).
Tasa inicial (Ene 2023): 17.28 Bs/USD | Tasa final (Dic 2025): 298.14 Bs/USD

### Margen por Categoría

| Categoría | Participación Ingresos | Margen Bruto |
|-----------|------------------------|--------------|
| 03CARN (Carnes) | 45.9% | 35.6% |
| 08FRUV (Frutas/Verduras) | 27.4% | 33.7% |
| 05CHAR (Charcutería) | 26.7% | 27.9% |

## Patrones Identificados

### 1. Estacionalidad Semanal
- **Días pico:** Sábado (19.0%) y Domingo (17.0%)
- **Días valle:** Jueves (11.5%)
- Patrón consistente: incremento de ventas hacia fin de semana

### 2. Distribución por Sucursal
| Sucursal | Participación | Margen |
|----------|---------------|--------|
| SUC002 | 28.1% | 33.5% |
| SUC003 | 27.3% | 32.5% |
| SUC001 | 26.8% | 32.9% |
| SUC004 | 17.9% | 33.5% |

### 3. Top 10 Productos por Ingresos
1. Carne Premium KG - $370.5M
2. Pollo Entero KG - $197.2M
3. Solomo de Res Molido KG - $164.0M
4. Delicia de Pollo KG - $146.3M
5. Queso Artesanal para Rallar KG - $142.0M
6. Muslo Pollo KG - $132.1M
7. Plátano - $116.6M
8. Papa KG - $92.1M
9. Chuleta Ahumada KG - $84.7M
10. Queso Merideño KG - $78.4M

### 4. Análisis de Elasticidad Precio-Demanda
- **Productos analizados:** 1,373 (con >30 días de datos)
- **Correlación media:** 0.057
- **Correlación mediana:** 0.079
- **% con correlación negativa:** 40.6%
- **% altamente elásticos (corr < -0.3):** 10.6% (~145 productos)

**Interpretación:** La mayoría de productos muestra baja correlación precio-demanda, lo que sugiere:
- Demanda relativamente inelástica en general
- ~145 productos son candidatos ideales para optimización de precios
- Factores como disponibilidad y temporada pueden ser más influyentes que el precio

### 5. Promociones
- **Total promociones únicas:** 3,786
- **Productos con promociones:** 5,762

| Tipo de Promoción | Cantidad | % |
|-------------------|----------|---|
| Precio Oferta | 37,471 | 87.0% |
| % Descuento | 2,163 | 5.0% |
| M×N Precio Oferta | 1,906 | 4.4% |
| Premio Precio Oferta | 854 | 2.0% |
| Premio Monto Descuento | 360 | 0.8% |

## ✅ Normalización por Inflación - COMPLETADA

Se normalizaron todos los valores monetarios usando tasas BCV históricas reales:

- **Fuente:** Histórico oficial BCV (1,656 tasas diarias)
- **Rango:** 2021-08-02 a 2026-02-19
- **Devaluación en período:** 17.28 → 298.14 Bs/USD (17x)

### Evolución Real del Negocio (USD)
- Ene 2023: ~$740,000/mes
- Dic 2025: ~$3,100,000/mes
- **Crecimiento real: ~4.2x** (vs 17x nominal)

### Precios Promedio por Unidad (USD)
| Categoría | Precio/Unidad | Margen |
|-----------|---------------|--------|
| 03CARN (Carnes) | $6.64 | 35.0% |
| 05CHAR (Charcutería) | $9.85 | 29.6% |
| 08FRUV (Frutas/Verduras) | $4.67 | 36.5% |

## Archivos Generados

### Scripts
- `run_eda.py` - Script de análisis exploratorio

### Visualizaciones (en `reports/`)
- `ventas_mensuales.png` - Evolución temporal de ventas
- `ventas_dia_semana.png` - Distribución por día de semana
- `participacion_clases.png` - Participación por categoría
- `elasticidad_precio_demanda.png` - Distribución de correlaciones precio-demanda
- `top_productos.png` - Top 15 productos por ingresos

## Próximos Pasos (Fase 2)

1. **Obtener tasas BCV históricas** para normalización de precios
2. **Diseñar arquitectura del pipeline** de datos y modelos
3. **Definir features** para el modelo de pricing dinámico
4. **Seleccionar approach de modelado** (XGBoost principal, Random Forest baseline)

---

*Documentación generada como parte del proyecto de tesis SIP Dynamic Pricing*
