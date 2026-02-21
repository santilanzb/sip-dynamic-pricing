# SIP Dynamic Pricing

Sistema Inteligente de Precios — framework generalizable de **dynamic pricing basado en Machine Learning** para supermercados venezolanos.

**Trabajo de Grado** — Universidad Metropolitana (UNIMET), Caracas, Venezuela, 2025-2026.

## Descripción

Este proyecto investiga y desarrolla un sistema end-to-end de optimización de precios dinámicos para cadenas de supermercados, utilizando como caso de estudio datos reales de una cadena venezolana (4 sucursales, 1,800+ productos en 3 categorías de perecederos). La metodología y el sistema son **generalizables a cualquier cadena de supermercados** con datos transaccionales equivalentes.

**Categorías estudiadas:**
- Carnes (03CARN)
- Frutas y Verduras (08FRUV)
- Charcutería (05CHAR)

### Componentes del Sistema

| Módulo | Descripción | Estado |
|--------|-------------|--------|
| Pronóstico de Demanda | LightGBM bietápico (hurdle model), WMAPE 23.61%, R² 0.938 | ✅ Completo |
| Inteligencia Competitiva | Web scraping + generación sintética + ablación | ✅ Completo |
| Simulación Multi-Escenario | 4 escenarios (±5% a ±30%), Phase 1 + Phase 2 backtest | ✅ Completo |
| Optimización de Precios | Grid-search con penalizaciones configurables, 16 KPIs | ✅ Completo |
| Dashboard Gerencial | Streamlit — visualización interactiva | 🔨 En desarrollo |

### Resultados Principales

- **Modelo de demanda:** WMAPE 23.61%, MASE 0.569 (1.76× mejor que naive), intervalos conformales calibrados
- **Optimización (escenario Moderado ±10%):** +8.74% ΔRevenue, +25.54% ΔMargen en test set (out-of-sample)
- **Backtest 23 meses:** +9.20% ΔRevenue sostenido (σ=0.89pp), sin degradación temporal
- **Competencia:** Infraestructura de scraping funcional; datos sintéticos aportan +0.10pp WMAPE (marginal con datos no reales)

## Inicio Rápido

### Requisitos Previos

- Python 3.11+
- GPU NVIDIA (opcional, para LightGBM GPU)
- ODBC Driver 17 for SQL Server (para extracción de datos)

### Instalación

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Estructura del Proyecto

```
sip-dynamic-pricing/
├── data/
│   ├── raw/                    # Datos crudos (Parquet)
│   ├── processed/              # fact_ventas, dim_producto, features.parquet
│   └── external/               # Tasas BCV, feriados, precios competencia
├── docs/
│   ├── PHASES_DOCUMENTATION.md # Documentación exhaustiva de todas las fases
│   ├── DECISIONS.md            # Decisiones técnicas y de negocio
│   └── arquitectura.md         # Diseño del sistema
├── models/
│   ├── two_stage/lgbm/         # Modelo de producción (LightGBM bietápico)
│   ├── two_stage/xgb/          # XGBoost bietápico (comparación)
│   ├── rf_baseline.pkl         # Random Forest single-stage
│   └── *.csv, *.png            # Métricas y visualizaciones de entrenamiento
├── output/
│   ├── simulation/             # Resultados multi-escenario (Phase 1 + Phase 2)
│   └── competition/            # Ablación, coeficientes, plots
├── src/
│   ├── data/                   # ETL, features, quality checks
│   ├── models/                 # TwoStageDemandModel, training, conformal
│   ├── simulation/             # DemandSimulator, PriceOptimizer, KPIs
│   ├── competition/            # Scrapers, synthetic generator, ablation
│   ├── dashboard/              # Streamlit app (en desarrollo)
│   ├── analysis/               # Análisis de residuos
│   └── utils/                  # Métricas compartidas
├── reports/                    # Reportes de training, data quality
├── notebooks/                  # EDA
├── mlruns/                     # MLflow tracking
└── requirements.txt
```

## Datos

| Métrica | Valor |
|---------|-------|
| Registros transaccionales | 1.3M+ |
| Productos | 1,819 |
| Sucursales | 4 (SUC001-SUC004) |
| Período | Ene 2023 – Dic 2025 |
| Features generadas | 60 (53 base + 7 competencia) |
| Normalización | Bs→USD vía tasa BCV diaria |

## Documentación

Ver `docs/PHASES_DOCUMENTATION.md` para documentación exhaustiva de cada fase, incluyendo:
- Metodología y decisiones técnicas
- Resultados detallados con tablas y métricas
- Hallazgos generalizables para la investigación
- Limitaciones y trabajo futuro

## Autores

- **Santiago Lanz** — Universidad Metropolitana
- **Diego Blanco** — Universidad Metropolitana

**Tutores:**
- Nicolás Araque
- Siro Tagliaferro
