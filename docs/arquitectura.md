# Arquitectura del Sistema SIP Dynamic Pricing

**Versión:** 1.0  
**Fecha:** 2026-02-19  
**Autores:** Santiago Lanz, Diego Blanco

---

## 1. Visión General

Sistema de soporte de decisiones para optimización de precios dinámicos en supermercados, basado en modelos de Machine Learning (XGBoost, Random Forest, LightGBM).

### 1.1 Objetivos del Sistema
- Predecir demanda a nivel producto-sucursal-día
- Estimar elasticidad precio-demanda por producto
- Recomendar precios óptimos que maximicen ingresos
- Simular escenarios de pricing

### 1.2 Alcance
- **Categorías:** Carnes (03CARN), Charcutería (05CHAR), Frutas/Verduras (08FRUV)
- **Sucursales:** 4 activas (SUC001-SUC004)
- **Productos:** ~1,800
- **Horizonte de predicción:** 1-7 días

---

## 2. Arquitectura de Alto Nivel

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CAPA DE DATOS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  SQL Server (EMP03/EMP04/VAD10)  →  ETL  →  Data Lake (Parquet)        │
│  - CompraVenta                        │     - compraventa_normalized    │
│  - Promociones                        │     - fact_ventas               │
│  - Ajustes (IV10001/IV30300)          │     - dim_producto              │
│  - Tasas BCV                          │     - dim_tiempo                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE PROCESAMIENTO                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Feature Engineering  →  Model Training  →  Model Registry              │
│  - Temporales              - XGBoost          - models/xgb_demand.pkl   │
│  - Precio/Demanda          - Random Forest    - models/rf_baseline.pkl │
│  - Promociones             - LightGBM         - models/lgbm_alt.pkl    │
│  - Inventario (proxy)                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAPA DE LÓGICA                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  Demand Predictor  →  Price Optimizer  →  Scenario Simulator            │
│  - predict(product, date, price)                                        │
│  - optimize_price(product, constraints)                                 │
│  - simulate(scenario_params)                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CAPA DE PRESENTACIÓN                               │
├─────────────────────────────────────────────────────────────────────────┤
│                         Streamlit Dashboard                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Overview │  │ Producto │  │ Simulador│  │ Recomend.│                │
│  │   KPIs   │  │ Análisis │  │ Precios  │  │ Óptimas  │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modelo de Datos

### 3.1 Esquema Estrella

```
                        ┌─────────────────┐
                        │   DIM_TIEMPO    │
                        ├─────────────────┤
                        │ fecha (PK)      │
                        │ dia_semana      │
                        │ es_fin_semana   │
                        │ dia_mes         │
                        │ mes             │
                        │ año             │
                        │ es_feriado      │
                        │ es_quincena     │
                        └────────┬────────┘
                                 │
┌─────────────────┐    ┌────────┴────────┐    ┌─────────────────┐
│  DIM_PRODUCTO   │    │   FACT_VENTAS   │    │  DIM_SUCURSAL   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ producto_id(PK) │◄───┤ fecha (FK)      │───►│ sucursal_id(PK) │
│ codigo_interno  │    │ producto_id(FK) │    │ nombre          │
│ sku             │    │ sucursal_id(FK) │    │ tipo            │
│ descripcion     │    │                 │    └─────────────────┘
│ clase           │    │ unidades        │
│ departamento    │    │ precio_unit_usd │    ┌─────────────────┐
│ es_perecedero   │    │ costo_unit_usd  │    │  DIM_PROMOCION  │
│ rotacion        │    │ ingreso_usd     │    ├─────────────────┤
└─────────────────┘    │ costo_usd       │    │ promocion_id(PK)│
                       │ margen_usd      │◄───┤ tipo            │
                       │ margen_pct      │    │ descripcion     │
                       │ tiene_promocion │    │ pct_descuento   │
                       │ promocion_id(FK)│    └─────────────────┘
                       │ tasa_bcv        │
                       └─────────────────┘
```

### 3.2 Granularidad
- **Nivel:** Producto × Sucursal × Día
- **Registros esperados:** ~1,800 productos × 4 sucursales × 1,096 días ≈ 7.9M registros potenciales
- **Registros reales:** ~1.3M (solo días con ventas)

### 3.3 Campos Derivados Clave

| Campo | Fórmula | Descripción |
|-------|---------|-------------|
| precio_unitario_usd | Precio_Venta_Total / Tasa_BCV / Unidades | Precio real en USD |
| margen_pct | (Ingreso - Costo) / Ingreso × 100 | Margen porcentual |
| es_perecedero | Clase IN ('03CARN', '08FRUV') | Flag perecedero |
| rotacion | Clasificación por cuartiles de venta | Alta/Media/Baja |

---

## 4. Feature Engineering

### 4.1 Features Temporales
```python
features_temporales = [
    'dia_semana',           # 0-6 (Lunes-Domingo)
    'es_fin_semana',        # 1 si Sáb/Dom
    'dia_mes',              # 1-31
    'mes',                  # 1-12
    'es_feriado',           # Calendario Venezuela
    'es_quincena',          # 15 o último día
    'dias_para_fin_mes',    # Countdown
    'semana_año',           # 1-52
]
```

### 4.2 Features de Precio
```python
features_precio = [
    'precio_actual_usd',
    'precio_promedio_7d',
    'precio_promedio_30d',
    'variacion_precio_1d',      # vs ayer
    'variacion_precio_7d',      # vs semana anterior
    'precio_vs_categoria',      # índice relativo
    'precio_vs_historico',      # vs promedio histórico
]
```

### 4.3 Features de Demanda (Lags)
```python
features_demanda = [
    'ventas_lag_1',             # ayer
    'ventas_lag_7',             # hace 1 semana
    'ventas_promedio_7d',
    'ventas_promedio_30d',
    'tendencia_14d',            # slope
    'volatilidad_demanda',      # coef. variación
]
```

### 4.4 Features de Promoción
```python
features_promocion = [
    'tiene_promocion',          # 0/1
    'tipo_promocion',           # 1-11 (encoded)
    'pct_descuento',
    'dias_en_promocion',
    'dias_desde_promo',
]
```

### 4.5 Features de Producto
```python
features_producto = [
    'clase_encoded',            # target encoding
    'es_perecedero',
    'rotacion_categoria',       # alta/media/baja
    'precio_medio_historico',
    'variabilidad_demanda',
]
```

### 4.6 Features de Inventario (Proxy)
```python
features_inventario = [
    'ajustes_7d',               # suma ajustes últimos 7 días
    'ajustes_30d',
    'tasa_ajuste',              # ajustes / ventas
    'es_alto_ajuste',           # top 20%
]
```

---

## 5. Modelos de ML

### 5.1 Modelo de Predicción de Demanda

**Target:** `log1p(unidades_vendidas)` (transformación log para estabilizar varianza)

**Modelos a evaluar:**
| Modelo | Rol | Ventajas |
|--------|-----|----------|
| XGBoost | Principal | Balance precisión/velocidad |
| Random Forest | Baseline | Robusto, interpretable |
| LightGBM | Alternativo | Rápido, GPU support |

**Hiperparámetros XGBoost (inicial):**
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'early_stopping_rounds': 50,
}
```

### 5.2 Split Temporal
```
Train:      2023-01-01 → 2024-12-31 (2 años)
Validation: 2025-01-01 → 2025-06-30 (6 meses)
Test:       2025-07-01 → 2025-12-31 (6 meses)
```

### 5.3 Métricas de Evaluación
- **MAPE** (Mean Absolute Percentage Error) - objetivo < 15%
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination) - objetivo > 0.7

---

## 6. Función Objetivo de Optimización

### 6.1 Formulación

```
Maximizar:  Σ [ D(p) × p × (1 - α×Var(p) - β×|Δp|) ]

Sujeto a:
  - p_min ≤ p ≤ p_max           (límites de precio)
  - margen(p) ≥ margen_min      (rentabilidad mínima)
  - |Δp| ≤ Δp_max               (cambio máximo permitido)
```

Donde:
- `D(p)` = Demanda estimada al precio p (del modelo ML)
- `α` = Penalización por dispersión de precios
- `β` = Penalización por cambios abruptos
- `Δp` = Cambio vs precio anterior

### 6.2 Parámetros por Categoría

| Categoría | margen_min | Δp_max | α | β |
|-----------|------------|--------|---|---|
| Carnes | 20% | 15% | 0.1 | 0.2 |
| Charcutería | 15% | 20% | 0.1 | 0.15 |
| FRUV | 25% | 25% | 0.05 | 0.1 |

---

## 7. Módulos del Sistema

### 7.1 Estructura de Código

```
src/
├── data/
│   ├── extract.py          # ✅ Extracción SQL
│   ├── bcv_rates.py        # ✅ Tasas BCV
│   ├── normalize_prices.py # ✅ Normalización Bs→USD
│   ├── transform.py        # 🔨 ETL y fact table
│   └── features.py         # 🔨 Feature engineering
├── models/
│   ├── train.py            # 🔨 Entrenamiento
│   ├── predict.py          # 🔨 Inferencia
│   ├── evaluate.py         # 🔨 Métricas
│   └── optimize.py         # 🔨 Optimización precios
├── simulation/
│   └── simulator.py        # 🔨 Simulador escenarios
└── dashboard/
    └── app.py              # 🔨 Streamlit
```

### 7.2 Interfaces de Módulos

```python
# models/predict.py
class DemandPredictor:
    def load(self, model_path: str) -> None
    def predict(self, product_id: str, date: datetime, 
                price: float, features: dict) -> float
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray

# models/optimize.py
class PriceOptimizer:
    def optimize(self, product_id: str, date: datetime,
                 constraints: dict) -> dict  # {price, demand, revenue}
    def optimize_batch(self, products: list, date: datetime) -> pd.DataFrame

# simulation/simulator.py
class PricingSimulator:
    def simulate_scenario(self, prices: dict, period: tuple) -> dict
    def compare_scenarios(self, scenarios: list) -> pd.DataFrame
```

---

## 8. Pipeline de Datos

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Extract   │ →  │  Transform  │ →  │   Feature   │ →  │    Train    │
│  (SQL→Raw)  │    │ (Raw→Fact)  │    │ Engineering │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  data/raw/         data/processed/    data/processed/     models/
  *.parquet         fact_ventas.pq     features.parquet    *.pkl
```

### 8.1 Frecuencia de Actualización
- **Extracción:** Diaria (automatizable con cron/scheduler)
- **Transformación:** Diaria
- **Re-entrenamiento:** Semanal o mensual

---

## 9. Dashboard

### 9.1 Páginas

1. **Overview**
   - KPIs: Ingresos, Margen, Unidades vendidas
   - Tendencias temporales
   - Top productos

2. **Análisis de Producto**
   - Selector de producto
   - Histórico de precio vs demanda
   - Elasticidad estimada
   - Predicción de demanda

3. **Simulador de Precios**
   - Input: producto, precio propuesto
   - Output: demanda esperada, ingreso proyectado
   - Comparación vs precio actual

4. **Recomendaciones**
   - Tabla de precios óptimos por producto
   - Impacto estimado vs precios actuales
   - Filtros por categoría/sucursal

---

## 10. Consideraciones Técnicas

### 10.1 Rendimiento
- Predicción individual: < 100ms
- Predicción batch (1000 productos): < 5s
- Optimización por producto: < 1s
- Dashboard: Tiempo de carga < 3s

### 10.2 Escalabilidad
- Datos: Parquet soporta hasta TB sin problemas
- Modelos: Serializados en memoria (~100MB)
- Dashboard: Streamlit soporta múltiples usuarios

### 10.3 Monitoreo
- Logging de predicciones
- Tracking de drift en features
- Alertas si MAPE > threshold

---

*Documento generado como parte del proyecto de tesis SIP Dynamic Pricing*
