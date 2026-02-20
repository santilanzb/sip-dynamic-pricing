# Documentación Detallada de Fases - SIP Dynamic Pricing

**Proyecto:** Sistema Inteligente de Precios - Dynamic Pricing
**Autores:** Santiago Lanz, Diego Blanco
**Última actualización:** 2026-02-20
**Versión:** 1.0

---

## Índice
1. [Fase 0: Setup del Entorno y Extracción de Datos](#fase-0)
2. [Fase 1: Análisis Exploratorio de Datos (EDA)](#fase-1)
3. [Fase 2: Arquitectura del Sistema](#fase-2)
4. [Fase 3: ETL y Calidad de Datos](#fase-3)
5. [Fase 4: Feature Engineering](#fase-4)
6. [Fase 5: Entrenamiento y Evaluación](#fase-5)

---

## Fase 0: Setup del Entorno y Extracción de Datos {#fase-0}

### 0.1 Configuración del Entorno

**Objetivo:** Establecer un entorno reproducible y determinista para el proyecto.

**Implementación:**
- Entorno virtual Python 3.11 (`venv/`)
- Dependencias fijadas en `requirements.txt`
- Semilla global: 42 para reproducibilidad

**Dependencias principales:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
optuna>=3.4.0
mlflow>=2.8.0
shap>=0.43.0
pyodbc>=4.0.39
sqlalchemy>=2.0.0
pyarrow>=14.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
```

### 0.2 Extracción de Datos

**Fuentes de datos:**
| Fuente | Servidor | Período | Registros |
|--------|----------|---------|-----------|
| CompraVenta | EMP03/EMP04 | 2023-2025 | 1.3M+ |
| Promociones | VAD10 | 2023-2025 | 43K eventos |
| Ajustes Inventario | EMP03/EMP04 | 2023-2025 | Variable |
| Tasas BCV | API/Web | 2023-2025 | Diario |

**Archivos generados:**
- `data/raw/compraventa_raw.parquet`
- `data/raw/promociones_raw.parquet`
- `data/raw/ajustes_raw.parquet`
- `data/external/tasas_bcv.csv`

**Scripts:**
- `src/data/extract.py` - Conectores SQL Server
- `src/data/bcv_rates.py` - Extracción de tasas BCV

### 0.3 Validación de Integridad

**Checks implementados:**
- ✅ Continuidad temporal (sin gaps por sucursal)
- ✅ Consistencia de llaves (producto_id, sku)
- ✅ Detección de duplicados
- ✅ Fechas dentro de rango válido

**Resultado:** Datos extraídos con integridad validada.

---

## Fase 1: Análisis Exploratorio de Datos (EDA) {#fase-1}

### 1.1 Venta y Estacionalidad

**Notebook:** `notebooks/01_eda.ipynb`

**Hallazgos principales:**

1. **Patrón semanal:**
   - Sábados y domingos: +15-20% vs promedio
   - Lunes: mínimo de la semana

2. **Patrón mensual:**
   - Quincenas (días 15, 30): picos de demanda
   - Fin de mes: incremento sostenido

3. **Estacionalidad anual:**
   - Diciembre: máximo anual (festividades)
   - Febrero: mínimo relativo

4. **Outliers identificados:**
   - Eventos especiales (Black Friday, Navidad)
   - Promociones agresivas

### 1.2 Análisis de Promociones

**Tipos de promoción identificados:**
| Tipo | Descripción | % del total |
|------|-------------|-------------|
| 1 | Precio Oferta | 45% |
| 2 | % Descuento | 30% |
| 4 | M×N Gratis | 15% |
| Otros | Combinaciones | 10% |

**Lift promedio por promoción:** +35% en volumen

### 1.3 Análisis de Márgenes

**Márgenes por categoría vs metas de negocio:**
| Categoría | Margen Real | Meta | Gap |
|-----------|-------------|------|-----|
| Carnes (03CARN) | ~13% | 25-30% | -12 a -17 pp |
| Fruver (08FRUV) | ~28% | ≥30% | -2 pp |
| Charcutería (04CHAR) | ~25% | >30% | -5 pp |

**Alerta crítica:** Carnes presenta el mayor gap, prioridad alta para optimización.

### 1.4 Análisis de Demanda Cero

**Hallazgo:** Panel denso - ceros representan demanda nula genuina (tienda nunca cerró).

| Métrica | Valor |
|---------|-------|
| % registros con demanda=0 | ~15% |
| Justificación modelo bietápico | ✅ Confirmada |

### 1.5 Limitaciones Documentadas

**ALERTA CRÍTICA - Mermas:**
Las mermas NO se registran como ajustes negativos. Por práctica contable-fiscal venezolana, se absorben en el costo de venta.

**Implicaciones:**
- Ajustes negativos = solo devoluciones
- Subestimación sistemática de pérdidas
- El modelo no puede usar merma como variable

---

## Fase 2: Arquitectura del Sistema {#fase-2}

### 2.1 Modelo de Datos

**Esquema estrella implementado:**

```
                    ┌─────────────────┐
                    │   Dim_Tiempo    │
                    │  - fecha        │
                    │  - dia_semana   │
                    │  - mes          │
                    │  - es_feriado   │
                    └────────┬────────┘
                             │
┌─────────────────┐    ┌─────┴─────┐    ┌─────────────────┐
│  Dim_Producto   │────│Fact_Ventas│────│  Dim_Sucursal   │
│  - producto_id  │    │           │    │  - sucursal_id  │
│  - sku          │    │  - fecha  │    │  - nombre       │
│  - clase        │    │  - unidades│   └─────────────────┘
│  - departamento │    │  - precio │
│  - es_perecedero│    │  - costo  │
└─────────────────┘    │  - margen │
                       └─────┬─────┘
                             │
                    ┌────────┴────────┐
                    │  Dim_Promocion  │
                    │  - tipo_promo   │
                    │  - pct_descuento│
                    └─────────────────┘
```

### 2.2 Capas del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                      │
│                  (Dashboard Streamlit)                       │
├─────────────────────────────────────────────────────────────┤
│                      CAPA DE LÓGICA                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Features │→ │  Train   │→ │Inference │→ │Optimizar │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      CAPA DE DATOS                          │
│     SQL Server → ETL → Parquet → Features → Modelos        │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Función Objetivo de Optimización

```
max: α × Ingreso - β × Dispersión - γ × CambioBrusco

donde:
- α: Peso alto (prioridad #1: maximizar ingreso)
- β: Peso bajo-moderado (coherencia entre SKUs)
- γ: Peso moderado (evitar rechazos por cambios bruscos)
```

**Restricciones:**
- Margen mínimo por categoría
- Rango de precio: ±30% del histórico
- Redondeo: $0.01

### 2.4 Módulos del Sistema

| Módulo | Script | Estado |
|--------|--------|--------|
| Demanda (ML) | `src/models/train_gpu.py` | ✅ Completado |
| Simulador | `src/simulation/` | 🔄 Pendiente |
| Optimizador | `src/optimization/` | 🔄 Pendiente |
| Dashboard | `src/dashboard/` | 🔄 Pendiente |

---

## Fase 3: ETL y Calidad de Datos {#fase-3}

### 3.1 Unificación y Normalización

**Proceso de normalización monetaria:**

1. **Transición EMP03→EMP04:** Merge de sistemas legacy
2. **Normalización Bs→USD:** Tasa BCV diaria
3. **Estandarización:** Nombres de columnas, tipos de datos

**Script:** `src/data/normalize_prices.py`

### 3.2 Estructura Fact_Ventas

**Columnas finales:**
```python
fact_ventas_schema = {
    'fecha': 'datetime64[ns]',
    'sucursal_id': 'str',
    'producto_id': 'str',
    'sku': 'str',
    'clase': 'str',
    'departamento': 'str',
    'unidades': 'float64',
    'precio_unitario_usd': 'float64',
    'costo_unitario_usd': 'float64',
    'ingreso_usd': 'float64',
    'costo_usd': 'float64',
    'margen_usd': 'float64',
    'margen_pct': 'float64',
    'tiene_promocion': 'int64',
    'tipo_promocion': 'int64',
    'pct_descuento': 'float64',
    'tasa_bcv': 'float64'
}
```

**Archivo:** `data/processed/fact_ventas.parquet`
**Registros:** ~1.3M

### 3.3 Reglas de Limpieza

| Regla | Implementación | Registros afectados |
|-------|----------------|---------------------|
| unidades ≥ 0 | Eliminar negativos | <0.1% |
| precio_unitario_usd > 0 | Eliminar ceros | <0.05% |
| costo_unitario_usd | Forward fill por producto | ~2% |

**Script:** `src/data/transform.py`

### 3.4 Quality Gates Automáticos

**Script:** `src/data/quality_checks.py`

**Checks implementados:**
| Check | Umbral | Acción | Estado |
|-------|--------|--------|--------|
| Esquema | 100% campos | BLOQUEAR | ✅ PASS |
| Tipos | Tolerancia IDs | ADVERTIR | ✅ PASS |
| Duplicados | 0% | BLOQUEAR | ✅ PASS |
| Rangos | precio>0, unidades≥0 | BLOQUEAR | ✅ PASS |
| Alineación lags | ≥95% | ADVERTIR | ✅ PASS (97.2%) |
| PSI drift | <0.25 | ADVERTIR | ⚠️ FAIL (6.13) |

**Reporte:** `reports/data_quality/data_quality_report.json`

**Nota sobre PSI:** El drift detectado se debe al cambio temporal natural entre períodos. Se monitorea pero no bloquea.

---

## Fase 4: Feature Engineering {#fase-4}

### 4.1 Features Temporales

**Script:** `src/data/features.py` → `add_temporal_features()`

| Feature | Tipo | Descripción |
|---------|------|-------------|
| dia_semana | int (0-6) | Día de la semana |
| es_fin_semana | binary | 1 si sáb/dom |
| dia_mes | int (1-31) | Día del mes |
| mes | int (1-12) | Mes |
| año | int | Año |
| semana_año | int (1-52) | Semana ISO |
| trimestre | int (1-4) | Trimestre |
| es_quincena | binary | 1 si día 15/30/31 |
| dias_para_fin_mes | int | Días restantes del mes |
| es_inicio_mes | binary | 1 si días 1-5 |
| es_fin_mes | binary | 1 si últimos 5 días |
| dia_semana_sin | float | sin(2π × dia/7) |
| dia_semana_cos | float | cos(2π × dia/7) |
| mes_sin | float | sin(2π × mes/12) |
| mes_cos | float | cos(2π × mes/12) |
| es_feriado | binary | Feriado VE |

### 4.2 Features de Precio

**Función:** `add_price_features()`

| Feature | Descripción |
|---------|-------------|
| precio_var_1d | Δ% precio vs día anterior |
| precio_var_7d | Δ% precio vs semana anterior |
| precio_mean_7d | Promedio móvil 7 días |
| precio_mean_30d | Promedio móvil 30 días |
| precio_historico_producto | Media histórica por producto |
| precio_vs_historico | Ratio vs histórico |
| precio_mean_clase | Media de la categoría (diaria) |
| precio_vs_clase | Ratio vs categoría |

### 4.3 Features de Demanda (Lags y Rolling)

**Funciones:** `add_lag_features()`, `add_rolling_features()`, `add_trend_features()`

| Feature | Descripción |
|---------|-------------|
| unidades_lag_1 | Lag 1 día |
| unidades_lag_7 | Lag 7 días |
| unidades_lag_14 | Lag 14 días |
| unidades_lag_28 | Lag 28 días |
| unidades_mean_7d | Rolling mean 7 días |
| unidades_mean_14d | Rolling mean 14 días |
| unidades_mean_30d | Rolling mean 30 días |
| unidades_std_7d | Rolling std 7 días |
| unidades_std_14d | Rolling std 14 días |
| unidades_std_30d | Rolling std 30 días |
| unidades_min_14d | Rolling min 14 días |
| unidades_max_14d | Rolling max 14 días |
| unidades_min_30d | Rolling min 30 días |
| unidades_max_30d | Rolling max 30 días |
| unidades_trend_14d | Pendiente lineal 14 días |

### 4.4 Features de Promoción

**Función:** `add_promotion_features()`

| Feature | Descripción |
|---------|-------------|
| tiene_promocion | Flag binario |
| tipo_promocion | Código de tipo (1-11) |
| pct_descuento | Porcentaje de descuento |
| dias_en_promocion | Días consecutivos en promo |
| dias_desde_promo | Días desde última promo |

### 4.5 Features de Producto/Sucursal

**Función:** `add_product_features()`

| Feature | Descripción |
|---------|-------------|
| es_perecedero | 1 si Carnes o Fruver |
| variabilidad_demanda | Coef. variación histórico |

### 4.6 Features de Interacción

**Función:** `add_interaction_features()`

| Feature | Descripción |
|---------|-------------|
| precio_x_finsemana | precio × es_fin_semana |
| promo_x_finsemana | tiene_promocion × es_fin_semana |
| precio_x_perecedero | precio × es_perecedero |

### 4.7 Target

**Función:** `create_target()`

```python
target = log1p(unidades)  # Para estabilizar varianza y manejar ceros
```

### Resumen de Features

**Total features generadas:** 51
**Archivo:** `data/processed/features.parquet`
**Registros finales:** 1,251,955 (tras eliminar warmup de lags)

---

## Fase 5: Entrenamiento y Evaluación {#fase-5}

### 5.1 Data Splits

**Estrategia:** Split temporal fijo (evita leakage)

| Split | Período | Registros | % |
|-------|---------|-----------|---|
| Train | 2023-01 a 2024-12 | 708,701 | 57% |
| Validation | 2025-01 a 2025-06 | 264,904 | 21% |
| Test | 2025-07 a 2025-12 | 278,350 | 22% |

**Script:** `src/models/train_gpu.py` → `temporal_split()`

### 5.2 Métricas de Evaluación

**Métricas implementadas (`src/utils/metrics.py`):**

| Métrica | Fórmula | Uso |
|---------|---------|-----|
| MAE | mean(\|y - ŷ\|) | Error absoluto |
| MSE | mean((y - ŷ)²) | Penaliza outliers |
| RMSE | √MSE | Escala original |
| R² | 1 - SS_res/SS_tot | Varianza explicada |
| MAPE | mean(\|y - ŷ\|/y) × 100 | Error porcentual |
| SMAPE | mean(2\|y - ŷ\|/(y + ŷ)) × 100 | Simétrico |
| **WMAPE** | Σ\|y - ŷ\| / Σ\|y\| × 100 | **Principal (ponderado)** |
| MdAE | median(\|y - ŷ\|) | Robusto |
| MASE | MAE / MAE_naive | vs baseline |
| WMAPE_revenue | Σ(\|y - ŷ\| × ingreso) / Σ(y × ingreso) | Alineado a negocio |

**Intervalos conformales:** Split-conformal con coberturas 80% y 90%

### 5.3 Modelos Entrenados

#### 5.3.1 Random Forest (Baseline)

**Configuración:**
```python
RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
```

**Resultados Test:**
| Métrica | Valor |
|---------|-------|
| WMAPE | **23.70%** |
| SMAPE | 42.51% |
| MAE | 3.17 |
| RMSE | 7.49 |
| R² | 0.9349 |

#### 5.3.2 XGBoost (Principal)

**Tuning:** Optuna, 50 trials, objetivo WMAPE

**Mejores hiperparámetros:**
```python
{
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'min_child_weight': 4,
    'monotone_constraints': '(0,...,-1,...,0)'  # precio=-1
}
```

**Resultados Test:**
| Métrica | Valor |
|---------|-------|
| WMAPE | 24.40% |
| SMAPE | 42.51% |
| MAE | 3.26 |
| RMSE | 8.01 |
| R² | 0.9256 |

#### 5.3.3 LightGBM (Alternativo)

**Tuning:** Optuna, 40 trials

**Mejores hiperparámetros:**
```python
{
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_samples': 20,
    'monotone_constraints': [-1 si precio else 0]
}
```

**Resultados Test:**
| Métrica | Valor |
|---------|-------|
| WMAPE | 23.71% |
| SMAPE | 42.49% |
| MAE | 3.17 |
| RMSE | 7.47 |
| R² | 0.9351 |

### 5.4 Análisis de Importancia de Features

**Top 10 Features (SHAP - XGBoost):**

1. `unidades_mean_7d` - 0.847
2. `unidades_lag_1` - 0.623
3. `unidades_mean_14d` - 0.412
4. `unidades_mean_30d` - 0.287
5. `dia_semana` - 0.198
6. `unidades_std_7d` - 0.156
7. `unidades_lag_7` - 0.134
8. `precio_unitario_usd` - 0.098
9. `precio_mean_7d` - 0.087
10. `es_fin_semana` - 0.076

**Archivo:** `models/xgb_shap_importance.csv`

### 5.5 Análisis de Residuos

**Script:** `src/analysis/residual_analysis.py`

**Hallazgos clave:**

| Aspecto | Hallazgo | Implicación |
|---------|----------|-------------|
| Sesgo global | +1.13 unidades | Subestima demanda |
| Distribución | No normal (esperado) | OK para retail |
| Categoría Carnes | MAE=6.38, Bias=-12.8% | Prioridad mejora |
| Demanda alta (>50) | Subestimación sistemática | Considerar modelo separado |
| Predicción ceros | F1=0 | Necesita modelo bietápico |
| Autocorrelación | 0.24 (lag-1), 0.29 (lag-7) | Patrones parcialmente capturados |

**Archivos generados:**
- `reports/residual_analysis/residual_distribution.png`
- `reports/residual_analysis/residuals_by_demand_level.png`
- `reports/residual_analysis/residuals_temporal.png`
- `reports/residual_analysis/residuals_by_*.csv`

### 5.6 Observabilidad (MLflow)

**Tracking URI:** `mlruns/`

**Artefactos loggeados:**
- Parámetros de entrenamiento
- Métricas por modelo
- Feature importances
- SHAP plots
- Intervalos conformales
- System metrics (CPU/GPU/RAM)

### 5.7 Monotonicidad Precio→Demanda

**Decisión:** Activada en XGBoost y LightGBM

**Implementación:**
```python
# XGBoost
monotone_constraints = "(0,0,...,-1,...,0)"  # -1 en precio_unitario_usd

# LightGBM
monotone_constraints = [0, 0, ..., -1, ..., 0]
```

**Justificación:** A mayor precio, menor demanda (elasticidad normal retail).

### 5.8 Comparativa Final

| Modelo | WMAPE ↓ | R² ↑ | MAE ↓ | Tiempo (s) |
|--------|---------|------|-------|------------|
| **Random Forest** | **23.70%** | 0.9349 | **3.17** | ~120 |
| LightGBM | 23.71% | **0.9351** | **3.17** | ~45 |
| XGBoost | 24.40% | 0.9256 | 3.26 | ~90 |

**Mejor modelo:** Random Forest (WMAPE más bajo)
**Nota:** LightGBM muy similar, más rápido - considerar para producción.

### 5.9 Gap vs Meta

| Métrica | Meta | Actual | Gap |
|---------|------|--------|-----|
| WMAPE | ≤15% | 23.70% | +8.7 pp |
| R² | ≥0.70 | 0.9349 | ✅ Superado |

**Estrategias para cerrar gap WMAPE:**
1. Modelo bietápico (clasificación + regresión)
2. Features adicionales de feriados con granularidad
3. Modelos por categoría para Carnes
4. Más datos históricos

---

## Artefactos del Proyecto

### Estructura de Directorios

```
sip-dynamic-pricing/
├── data/
│   ├── raw/                    # Datos crudos (Parquet)
│   ├── processed/              # Datos procesados
│   │   ├── fact_ventas.parquet
│   │   ├── dim_producto.parquet
│   │   └── features.parquet
│   └── external/               # Datos externos (tasas, feriados)
├── docs/
│   ├── DECISIONS.md            # Decisiones técnicas/negocio
│   └── PHASES_DOCUMENTATION.md # Este documento
├── models/
│   ├── rf_demand_baseline.pkl
│   ├── xgb_demand_gpu.json
│   ├── lgbm_demand_gpu.pkl
│   └── *.csv, *.png            # Métricas y visualizaciones
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   ├── data_quality/
│   ├── residual_analysis/
│   └── training_run_*.md
├── src/
│   ├── analysis/
│   ├── data/
│   ├── models/
│   ├── simulation/
│   └── utils/
└── mlruns/                     # MLflow tracking
```

### Commits Relevantes

| Commit | Descripción |
|--------|-------------|
| `de8e218` | Monotonicidad, WMAPE_revenue, feriados, bietápico |
| `016a542` | DECISIONS.md, gitignore mlruns |
| `e1e609c` | EDA: márgenes, ceros, limitaciones |
| `e5463d3` | Training completo con Optuna |
| `b57c9ad` | Reporte de entrenamiento |

---

## Próximos Pasos (Fases 6-9)

1. **Fase 6:** Datos sintéticos de competencia
2. **Fase 7:** Simulación y optimización de precios
3. **Fase 8:** Dashboard Streamlit
4. **Fase 9:** Validación final y documentación de tesis

---

*Documento generado para trazabilidad del proyecto de tesis.*
*SIP Dynamic Pricing - 2026*
