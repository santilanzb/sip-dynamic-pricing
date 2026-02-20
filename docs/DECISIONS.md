# Decisiones Técnicas y de Negocio - SIP Dynamic Pricing

**Última actualización:** 2026-02-20
**Autores:** Santiago Lanz, Diego Blanco

Este documento registra todas las decisiones críticas tomadas durante el desarrollo del sistema, con su justificación y sus implicaciones para el modelo.

---

## 1. Datos y ETL

### 1.1 Tipo de Panel
**Decisión:** Panel denso real (no sparse).

**Contexto:** La tienda operó todos los días del período analizado (2023-2025) sin excepciones por feriados, contingencias o cierres temporales.

**Implicación:**
- No se requiere imputación de ceros por cierre operativo.
- Un SKU con ventas=0 en un día representa demanda nula genuina, no dato faltante.
- Simplifica el pipeline de features (no hay que distinguir "cerrado" vs "sin ventas").

### 1.2 Feriados y Eventos
**Decisión:** Incorporar feriados venezolanos como features binarias.

**Fuente:** Lista de feriados nacionales y locales de Venezuela (`data/external/feriados_ve.csv`).

**Implicación:**
- Features: `es_feriado`, opcionalmente `tipo_feriado`.
- La tienda opera en feriados, pero el comportamiento de demanda puede variar.
- Se evaluará el efecto estacional (algunos feriados pueden aumentar o disminuir ventas).

### 1.3 Tratamiento de Mermas
**Decisión:** NO usar ajustes negativos como proxy de merma.

**ALERTA CRÍTICA - Limitación de datos:**
Las mermas (pérdidas de inventario por deterioro, robo, etc.) NO se registran como ajustes negativos en el sistema. Por práctica contable-fiscal venezolana, las mermas se absorben en el costo de venta del producto.

**Implicación:**
- Los ajustes negativos solo capturan devoluciones de clientes.
- Existe subestimación sistemática de pérdidas de inventario.
- El modelo no puede predecir ni usar merma real.
- **Recomendación para la empresa:** Implementar registro interno de mermas independiente del tratamiento fiscal.
- **Para la tesis:** Documentar como limitación conocida del dataset.

### 1.4 Reglas de Limpieza
**Decisión:** Filtros estrictos de calidad.

| Campo | Regla | Justificación |
|-------|-------|---------------|
| `unidades` | ≥ 0 | Valores negativos son errores o devoluciones (tratadas aparte) |
| `precio_unitario_usd` | > 0 | Precio cero indica error de registro |
| `costo_unitario_usd` | Imputar forward si faltante | Continuidad por producto |

---

## 2. Reglas de Negocio

### 2.1 Márgenes Mínimos por Categoría
**Decisión:** Usar como restricciones en la capa de optimización.

| Categoría | Margen Meta | Margen Real | Gap |
|-----------|-------------|-------------|-----|
| Carnes | 25-30% | ~13% | -12 a -17 pp |
| Fruver | ≥30% | ~28% | -2 pp |
| Charcutería | >30% | TBD | TBD |

**Implicación:**
- El optimizador debe respetar estos pisos como constraints.
- Carnes tiene el mayor gap: prioridad para ajuste de precios.
- Reportar métricas de simulación por categoría vs metas.

### 2.2 Límites de Variación de Precio
**Decisión:** Sin topes formales, pero con penalización suave.

**Contexto:** Los compradores ajustan precios libremente sin restricción porcentual formal.

**Implicación:**
- El optimizador NO aplica cap de variación por período.
- El parámetro γ (penalización de cambio brusco) actúa como regulador suave.
- Redondeo al centavo ($0.01) para percepción de precio bajo.

### 2.3 SKUs Regulados
**Decisión:** No hay SKUs con precio regulado, congelado o fijo.

**Implicación:**
- Libertad total del optimizador sobre el rango de precios.
- No se requiere lista de exclusión.

---

## 3. Simulación y Optimización

### 3.1 Función Objetivo
**Decisión:** Maximizar ingreso como prioridad #1.

Función objetivo:
```
max: α × Ingreso - β × Dispersión - γ × CambioBrusco
```

**Pesos sugeridos:**
- α (ingreso): **Alto** — componente dominante.
- β (dispersión): **Bajo a moderado** — coherencia entre SKUs similares.
- γ (cambio brusco): **Moderado** — evitar rechazo del consumidor por saltos abruptos.

**Variable adicional requerida:**
- Nivel de stock actual como input para condicionar dirección:
  - Stock alto → favorecer bajada de precio
  - Stock bajo → tolerar subida

### 3.2 Rango de Exploración
**Decisión:** ±30% del precio histórico reciente.

**Contexto:** El stakeholder considera este rango conservador.

**Implicación:**
- Grid de simulación: [precio_base × 0.70, precio_base × 1.30]
- Puede ampliarse en iteraciones futuras si los resultados lo justifican.

---

## 4. Modelado

### 4.1 Monotonicidad Precio-Demanda
**Decisión:** Activar monotonic constraints (precio = -1).

**Justificación económica:** A mayor precio, menor demanda. Relación inversa universal en retail. No aplica efecto Veblen en este contexto.

**Implementación:**
- XGBoost: `monotone_constraints` en parámetros
- LightGBM: `monotone_constraints` en parámetros
- Feature `precio_unitario_usd` con constraint = -1

### 4.2 Modelo Bietápico para Ceros
**Decisión:** Activar modelo en dos etapas.

**Etapa 1 (Clasificación):**
- Predecir P(venta > 0)
- Threshold calibrado para balance FP/FN

**Etapa 2 (Regresión):**
- Predecir cantidad | venta > 0
- Solo se ejecuta si Etapa 1 predice venta

**Predicción final:**
```
ŷ = P(venta > 0) × E[cantidad | venta > 0]
```

### 4.3 Métrica Principal
**Decisión:** WMAPE ponderado por ingreso (revenue-weighted).

**Fórmula:**
```
WMAPE_revenue = Σ(|y - ŷ| × ingreso) / Σ(y × ingreso)
```

**Justificación:**
- Alinea la métrica con el objetivo de negocio (maximizar ingreso).
- SKUs de alto volumen de facturación tienen mayor peso.
- Un error del 10% en SKU que factura $10,000/mes > error del 30% en SKU que factura $500/mes.

**Métricas secundarias:**
- MAE, RMSE, R² (escala original)
- MAPE, SMAPE (interpretabilidad)
- MASE (comparación vs baseline naive)
- Intervalos conformales (cobertura 80%/90%)

### 4.4 Data Splits
**Decisión:** Splits temporales fijos (no aleatorios).

| Split | Período | Uso |
|-------|---------|-----|
| Train | 2023-01 a 2024-12 | Entrenamiento |
| Validation | 2025-01 a 2025-06 | Tuning Optuna |
| Test | 2025-07 a 2025-12 | Evaluación final |

**Justificación:**
- Evita leakage temporal.
- Simula escenario real de predicción futura.
- Permite backtesting rolling para robustez.

---

## 5. Quality Gates

### 5.1 Checks Pre-Entrenamiento
| Check | Umbral | Acción si falla |
|-------|--------|-----------------|
| Esquema | 100% campos requeridos | BLOQUEAR |
| Tipos | Tolerancia string/int en IDs | ADVERTIR |
| Duplicados | 0% duplicados por llave natural | BLOQUEAR |
| Rangos | precio>0, unidades≥0 | BLOQUEAR |
| Alineación lags | ≥95% match | ADVERTIR |
| PSI por feature | <0.25 por feature | ADVERTIR + monitorear |

### 5.2 Checks Post-Entrenamiento
- Gap Train-Val metrics < 10% relativo
- Intervalos conformales con cobertura ≥ target - 5pp
- SHAP consistency (top features estables)

---

## Registro de Cambios

| Fecha | Cambio | Autor |
|-------|--------|-------|
| 2026-02-20 | Documento inicial con todas las decisiones | Diego Blanco |
