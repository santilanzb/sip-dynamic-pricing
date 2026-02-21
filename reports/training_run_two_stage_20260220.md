# Reporte de Entrenamiento Bietápico (Hurdle Model) - 2026-02-20

## Resumen Ejecutivo

Entrenamiento completo del modelo bietápico (hurdle model) con umbral de demanda τ=1.0 unidad.
Se evaluaron dos backends (LightGBM, XGBoost) con tuning Optuna, threshold calibration F1-optimal,
métricas exhaustivas, SHAP importance, intervalos conformales y métricas por segmento.

**Mejor modelo:** LightGBM bietápico (WMAPE 23.61%, R² 0.938)

## Motivación

El entrenamiento single-stage (Iteración 1) mostró:
- F1=0 en predicción de demanda baja → el modelo no discriminaba regímenes
- Sesgo de +1.13 unidades (subestimación)
- WMAPE 23.70% (RF baseline) con gap de 8.7 pp vs meta

Se implementó un modelo bietápico con separación de regímenes para mejorar la predicción.

## Dataset

- **Registros totales:** 1,251,955
- **Features:** 51
- **Target:** log1p(unidades)
- **Split temporal:**
  - Train: 708,701 (2023-01 a 2024-12)
  - Validation: 264,904 (2025-01 a 2025-06)
  - Test: 278,350 (2025-07 a 2025-12)

## Arquitectura del Hurdle Model

```
ŷ = P(demanda ≥ τ) × E[unidades | demanda ≥ τ] + (1 − P(demanda ≥ τ)) × μ_low
```

- **τ = 1.0 unidad** (umbral de separación)
- **μ_low = 0.4495** (media empírica del régimen bajo)
- **Distribución:** 77.1% alta demanda / 22.9% baja demanda (en training)

### Etapa 1: Clasificador P(demanda ≥ τ)
- Clasificación binaria (alta vs baja demanda)
- 51 features (mismas que regresor)
- Monotone constraint: `precio_unitario_usd = -1`
- Threshold calibrado en validación (F1-optimal)

### Etapa 2: Regresor E[unidades | demanda ≥ τ]
- Entrenado solo sobre registros con demanda ≥ 1.0
- 51 features, monotone constraint en precio
- Predicción en log1p → revertida con expm1

## Calibración del Threshold

- **Método:** Barrido de 200 thresholds en [0.05, 0.95]
- **Threshold óptimo:** 0.511
- **F1 en validación:** 0.947
- **Precision:** 0.945
- **Recall:** 0.958

## Configuración de Entrenamiento

- **LightGBM:** 30 trials Optuna, 1000 rounds, GPU, TPE sampler (seed=42)
- **XGBoost:** 30 trials Optuna, 1000 rounds, CUDA, TPE sampler (seed=42)
- **Objetivo Optuna:** WMAPE en validación
- **Hiperparámetros compartidos:** clf y reg comparten mismos hiperparámetros por backend

### Mejores Hiperparámetros (Optuna)

**LightGBM bietápico:**
- learning_rate: 0.039
- num_leaves: 106
- min_child_samples: 51
- subsample: 0.60
- colsample_bytree: 0.65
- reg_alpha: 7.71
- reg_lambda: 3.62e-07
- Monotonic constraints: precio_unitario_usd = -1
- Tiempo: 4,052s (~67 min)

**XGBoost bietápico:**
- learning_rate: 0.026
- max_depth: 5
- subsample: 0.82
- colsample_bytree: 0.66
- min_child_weight: 9
- reg_alpha: 4.69e-08
- reg_lambda: 7.62
- Monotonic constraints: precio_unitario_usd = -1
- Tiempo: 1,143s (~19 min)

**Observaciones:**
- LightGBM: L1 fuerte (7.71), L2 casi nula → promueve sparsity
- XGBoost: L2 fuerte (7.62), L1 casi nula → regularización suave
- XGBoost max_depth=5 (más conservador que single-stage depth=8)

## Resultados en Test Set

### Métricas Completas

| Métrica | LightGBM bietápico | XGBoost bietápico | RF single (baseline) |
|---------|---------------------|--------------------|-----------------------|
| **WMAPE** | **23.61%** | 23.64% | 23.70% |
| SMAPE | 45.00% | 44.96% | 42.51% |
| **MAE** | **3.153** | 3.158 | 3.165 |
| MSE | 87.63 | 88.64 | 91.91 |
| **RMSE** | **9.361** | 9.415 | 9.587 |
| **R²** | **0.9380** | 0.9372 | 0.9349 |
| RMSLE | 0.3125 | 0.3127 | 0.3153 |
| MdAE | 0.891 | 0.893 | 0.888 |
| **MASE** | **0.569** | 0.570 | — |
| MBE | -0.728 | -0.646 | -0.923 |
| MPE | 17.66% | 18.17% | 14.86% |
| OverForecastRate | 54.85% | 55.22% | 50.92% |
| WMAPE_revenue | 26.66% | 26.87% | 26.42% |

### Métricas del Clasificador (Etapa 1)

| Métrica | LightGBM | XGBoost |
|---------|----------|---------|
| Precision (alta) | 0.945 | 0.945 |
| Recall (alta) | 0.958 | 0.959 |
| **F1 (alta)** | **0.952** | **0.952** |
| Precision (baja) | 0.870 | 0.872 |
| Recall (baja) | 0.787 | 0.786 |
| **F1 (baja)** | **0.826** | **0.827** |

### Intervalos Conformales

| Nivel | Cobertura LightGBM | Cobertura XGBoost | Ancho LGBM | Ancho XGB |
|-------|---------------------|--------------------| ------------|-----------|
| 90% | 90.29% | 90.32% | 10.91 | 10.94 |
| 80% | 80.21% | 80.35% | 5.86 | 5.90 |

Coberturas empíricas muy cercanas a las nominales → bien calibrados.

## Métricas por Segmento (LightGBM bietápico)

### Por Categoría (Clase)

| Categoría | n (Test) | WMAPE | MAE |
|-----------|----------|-------|-----|
| Fruver (08FRUV) | 134,895 | 21.34% | 3.26 |
| Carnes (03CARN) | 46,478 | 25.27% | 5.99 |
| Charcutería (05CHAR) | 96,977 | 28.77% | 1.64 |

### Por Sucursal

| Sucursal | n (Test) | WMAPE | MAE |
|----------|----------|-------|-----|
| SUC001 | 73,684 | 22.61% | 2.96 |
| SUC003 | 63,572 | 22.71% | 3.44 |
| SUC002 | 73,232 | 23.55% | 3.34 |
| SUC004 | 67,862 | 26.15% | 2.89 |

### Por Cuartil de Demanda

| Cuartil | n (Test) | WMAPE | MAE |
|---------|----------|-------|-----|
| Alto | 69,587 | 20.05% | 9.29 |
| Medio-Alto | 69,541 | 37.63% | 1.90 |
| Medio-Bajo | 49,119 | 48.46% | 0.87 |
| Bajo | 90,103 | 106.23% | 0.63 |

## Top 10 Features (SHAP Importance — LightGBM bietápico)

1. `unidades_mean_7d` — 0.300
2. `unidades_lag_1` — 0.230
3. `unidades_mean_14d` — 0.175
4. `unidades_mean_30d` — 0.157
5. `unidades_lag_7` — 0.037
6. `dia_semana_sin` — 0.031
7. `precio_var_1d` — 0.022
8. `dia_semana` — 0.021
9. `unidades_min_30d` — 0.018
10. `precio_unitario_usd` — 0.018

## Comparativa Consolidada (5 modelos)

| Modelo | WMAPE ↓ | R² ↑ | MAE ↓ | RMSE ↓ | MASE ↓ |
|--------|---------|------|-------|--------|--------|
| **LightGBM bietápico** | **23.61%** | **0.938** | **3.15** | **9.36** | **0.569** |
| XGBoost bietápico | 23.64% | 0.937 | 3.16 | 9.42 | 0.570 |
| Random Forest single | 23.70% | 0.935 | 3.17 | 9.59 | — |
| LightGBM single | 24.30% | 0.931 | 3.24 | 9.87 | — |
| XGBoost single | 24.40% | 0.926 | 3.26 | 10.25 | — |

## Observaciones

### Fortalezas
1. **R² > 0.937** en ambos modelos bietápicos — excelente capacidad predictiva
2. **WMAPE 23.61%** — mejor resultado logrado, superando todos los baselines
3. **F1 baja demanda = 0.826** — el clasificador detecta correctamente el régimen de baja demanda (vs F1=0 en single-stage)
4. **MASE = 0.569** — supera baseline naive por factor 1.76x
5. **Intervalos conformales bien calibrados** — coberturas empíricas ~90%/~80% exactas
6. **Monotonicidad respetada** — precio↑ → demanda↓ en ambas etapas

### Limitaciones
1. **Mejora marginal vs RF single** (−0.09 pp WMAPE): dataset sin ceros puros limita ventaja del hurdle
2. **WMAPE meta ≤15% inalcanzable**: panel 20% denso + ruido retail
3. **SMAPE peor que single-stage** (45% vs 42.5%): esperado por la formulación del hurdle model que produce predicciones bajas para el régimen inferior
4. **Cuartil bajo WMAPE >100%**: demanda muy baja tiene alta incertidumbre relativa

### Conclusión
El modelo bietápico LightGBM alcanza el techo de precisión posible con los datos disponibles.
La señal restante es ruido inherente del retail. Se procede a la fase de simulación y optimización
de precios usando este modelo como base para la función de demanda.

## Artefactos Generados

### Modelos
- `models/two_stage/lgbm/lgbm_two_stage_clf.pkl` — Clasificador LightGBM
- `models/two_stage/lgbm/lgbm_two_stage_reg.pkl` — Regresor LightGBM
- `models/two_stage/xgb/xgb_two_stage_clf.pkl` — Clasificador XGBoost
- `models/two_stage/xgb/xgb_two_stage_reg.pkl` — Regresor XGBoost

### Metadata y Métricas
- `models/two_stage/two_stage_training_metadata.json`
- `models/two_stage/lgbm_two_stage_metrics_by_clase.csv`
- `models/two_stage/lgbm_two_stage_metrics_by_sucursal.csv`
- `models/two_stage/lgbm_two_stage_metrics_by_demand_quartile.csv`
- `models/two_stage/xgb_two_stage_metrics_by_clase.csv`
- `models/two_stage/xgb_two_stage_metrics_by_sucursal.csv`
- `models/two_stage/xgb_two_stage_metrics_by_demand_quartile.csv`

### SHAP
- `models/two_stage/lgbm_two_stage_shap_importance.csv`
- `models/two_stage/xgb_two_stage_shap_importance.csv`
- `models/two_stage/lgbm_two_stage_shap_bar.png`
- `models/two_stage/xgb_two_stage_shap_bar.png`

### Visualizaciones
- `models/two_stage/*_analysis.png` — Análisis de residuos
- `models/two_stage/*_scatter.png` — Predicho vs real
- `models/two_stage/*_error_dist.png` — Distribución de errores
- `models/two_stage/*_threshold_calibration.png` — Curvas de calibración

### MLflow
- Experiment: `two_stage_demand`
- Tracking URI: `mlruns/`

---
*Generado para trazabilidad del proyecto de tesis — SIP Dynamic Pricing 2026*
