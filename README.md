# SIP Dynamic Pricing

Sistema de Soporte de Decisiones para optimización de precios dinámicos en supermercados utilizando Machine Learning.

## 📋 Descripción

Este proyecto implementa un sistema de **dynamic pricing** para cadenas de supermercados venezolanos, enfocado en las categorías de:
- 🥩 Carnes (03CARN)
- 🍎 Frutas y Verduras (08FRUV)  
- 🧀 Charcutería (05CHAR)

### Componentes Principales

1. **Módulo de Pronóstico de Demanda** - XGBoost/LightGBM
2. **Módulo de Simulación de Precios** - Escenarios contrafactuales
3. **Módulo de Optimización** - Recomendación de precio óptimo
4. **Dashboard Gerencial** - Streamlit

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.11+
- Acceso a SQL Server (sermgp03, sermgp04, serestellar)
- ODBC Driver 17 for SQL Server

### Instalación

```bash
# Clonar/navegar al proyecto
cd C:\Users\dblanco\Projects\sip-dynamic-pricing

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Extracción de Datos

```bash
# Extraer todos los datos
python -m src.data.extract --all

# O extraer por separado
python -m src.data.extract --compraventa
python -m src.data.extract --promociones
python -m src.data.extract --ajustes
```

## 📁 Estructura del Proyecto

```
sip-dynamic-pricing/
├── data/
│   ├── raw/                 # Datos extraídos de SQL (Parquet)
│   ├── processed/           # Datos transformados
│   └── synthetic/           # Datos de supermercados sintéticos
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_simulation.ipynb
├── src/
│   ├── data/
│   │   ├── extract.py       # Extracción de SQL
│   │   ├── transform.py     # ETL y feature engineering
│   │   └── synthetic.py     # Generación de datos sintéticos
│   ├── models/
│   │   ├── train.py         # Entrenamiento
│   │   ├── predict.py       # Inferencia
│   │   └── optimize.py      # Optimización de precios
│   ├── simulation/
│   │   └── simulator.py     # Simulador de escenarios
│   └── dashboard/
│       └── app.py           # Streamlit app
├── models/                  # Modelos serializados
├── reports/                 # Visualizaciones y reportes
├── tests/                   # Tests unitarios
├── config.yaml              # Configuración
├── requirements.txt
└── README.md
```

## 📊 Datos

### Fuentes

| Fuente | Servidor | Base de Datos | Descripción |
|--------|----------|---------------|-------------|
| CompraVenta 2023 | sermgp03 | EMP03 | Transacciones históricas |
| CompraVenta 2024 (Ene-Oct) | sermgp03 | EMP03 | Transacciones pre-mudanza |
| CompraVenta 2024 (Nov-Dic) | sermgp04 | EMP04 | Transacciones post-mudanza |
| CompraVenta 2025 | sermgp04 | EMP04 | Transacciones actuales |
| Promociones | serestellar | VAD10 | Histórico de promociones |
| Ajustes | sermgp04 | EMP04 | IV10001/IV30300 |

### Volumen Estimado

- ~1,000,000 registros de CompraVenta
- ~43,000 registros de Promociones
- 4 sucursales
- Período: Sept 2023 - Oct 2025

## 🤖 Modelos

- **Baseline:** Random Forest
- **Principal:** XGBoost
- **Alternativo:** LightGBM (GPU accelerated)

### Métricas Objetivo

- MAPE < 15%
- R² > 0.7
- Mejora de ingresos ≥ 5% vs precios estáticos

## 👥 Autores

- Santiago Lanz
- Diego Blanco

## 📚 Referencias

Trabajo de Grado - Universidad Metropolitana, 2025-2026

Tutores:
- Nicolás Araque
- Siro Tagliaferro
