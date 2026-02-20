# Reporte de Entrenamiento - 2026-02-20

## Resumen Ejecutivo
Entrenamiento completo realizado con tuning de hiperparámetros Optuna y observabilidad MLflow.

## Dataset
- **Registros totales:** 1,251,955
- **Features:** 51
- **Split temporal:**
  - Train: 708,701 (2023-2024)
  - Validation: 264,904 (2025-01 a 2025-06)
  - Test: 278,350 (2025-07 a 2025-12)

## Resultados en Test Set

| Modelo | WMAPE | SMAPE | MAE | RMSE | R² |
|--------|-------|-------|-----|------|-----|
| **Random Forest** | **23.70%** | 42.51% | 3.17 | 7.49 | 0.9349 |
| LightGBM | 23.71% | 42.49% | 3.17 | 7.47 | 0.9351 |
| XGBoost | 24.40% | 42.51% | 3.26 | 8.01 | 0.9256 |

**Mejor modelo:** Random Forest (WMAPE 23.70%)

## Configuración de Entrenamiento
- **XGBoost:** 50 trials Optuna, 1500 rounds
- **LightGBM:** 40 trials Optuna, 1500 rounds, early stopping 200
- **Random Forest:** 1000 estimators, max_depth 20

### Mejores Hiperparámetros (Optuna)

**XGBoost:**
- learning_rate: ~0.05
- max_depth: 8
- subsample: 0.9
- colsample_bytree: 0.8
- min_child_weight: 4
- Monotonic constraints: precio_unitario_usd = -1

**LightGBM:**
- learning_rate: ~0.05
- max_depth: 8
- min_child_samples: 20
- Monotonic constraints: precio_unitario_usd = -1

## Top 10 Features (SHAP Importance - XGBoost)

1. `unidades_mean_7d` - Promedio móvil 7 días
2. `unidades_lag_1` - Lag 1 día
3. `unidades_mean_14d` - Promedio móvil 14 días
4. `unidades_mean_30d` - Promedio móvil 30 días
5. `dia_semana` - Día de la semana
6. `unidades_std_7d` - Desviación estándar 7 días
7. `unidades_lag_7` - Lag 7 días
8. `precio_unitario_usd` - Precio unitario
9. `precio_mean_7d` - Promedio precio 7 días
10. `es_fin_semana` - Indicador fin de semana

## Métricas por Segmento

### Por Categoría (Clase)
Las métricas por clase están disponibles en `models/metrics_by_clase.csv`.

### Por Sucursal
Las métricas por sucursal están disponibles en `models/metrics_by_sucursal.csv`.

### Por Cuartil de Demanda
| Cuartil | WMAPE | MAE | n |
|---------|-------|-----|---|
| Bajo | Alto | Bajo | - |
| Medio-Bajo | - | - | - |
| Medio-Alto | - | - | - |
| Alto | Bajo | Alto | - |

(Ver detalle en `models/metrics_by_demand_quartile.csv`)

## Intervalos Conformales
- **Cobertura 90%:** ~90% (target)
- **Cobertura 80%:** ~80% (target)
- Anchos promedio disponibles en MLflow

## Observaciones

### Fortalezas
1. **R² > 0.93** en todos los modelos - excelente capacidad predictiva
2. **WMAPE ~23-24%** - error aceptable para pricing dinámico
3. **Monotonicidad respetada** - precio↑ → demanda↓
4. **Features de lag dominan** - patrón temporal fuerte

### Áreas de Mejora
1. WMAPE meta es ≤15% según el plan - gap de ~8 pp
2. Probar modelo bietápico para mejorar predicción de ceros
3. Evaluar backtesting rolling para robustez temporal
4. Refinar features de feriados con datos reales

## Próximos Pasos
1. [ ] Implementar backtesting rolling (ventanas trimestrales)
2. [ ] Entrenar modelo bietápico (clasificación + regresión)
3. [ ] Analizar residuos por categoría para identificar patrones
4. [ ] Implementar módulo de simulación de precios
5. [ ] Comenzar desarrollo del dashboard

## Artefactos Generados
- `models/xgb_demand_gpu.json` - Modelo XGBoost
- `models/lgbm_demand_gpu.pkl` - Modelo LightGBM
- `models/rf_demand_baseline.pkl` - Modelo Random Forest
- `models/xgb_shap_*.png` - Plots SHAP
- `models/xgb_conformal_intervals.csv` - Intervalos de predicción
- `models/metrics_by_*.csv` - Métricas por segmento
- MLflow tracking en `mlruns/`

---
*Generado automáticamente - SIP Dynamic Pricing*
