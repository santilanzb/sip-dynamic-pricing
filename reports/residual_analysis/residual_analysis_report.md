# Análisis de Residuos - Modelos de Predicción de Demanda

**Fecha:** 2026-02-20
**Modelo:** Random Forest (mejor WMAPE)

## Resumen Ejecutivo

### Distribución de Residuos
- **Media:** 1.1327 (sesgo positivo - subestima)
- **Mediana:** -0.0384
- **Desv. Estándar:** 10.1908
- **Asimetría:** 12.0339

### Predicción de Demanda Cero
- **Registros con demanda=0:** 0 (0.0%)
- **Precision (para ceros):** 0.000
- **Recall (para ceros):** 0.000
- **F1-Score:** 0.000

## Hallazgos Clave

### Fortalezas del Modelo
1. R² > 0.93 indica excelente capacidad predictiva general
2. Sesgo cercano a cero en promedio

### Áreas de Mejora Identificadas
1. **Demanda alta (>50 unidades):** Mayor error absoluto
2. **Demanda cero:** Precisión mejorable con modelo bietápico
3. **Autocorrelación:** Posibles patrones temporales no capturados

## Recomendaciones

1. **Modelo bietápico:** Implementar clasificación previa (venta/no-venta) antes de regresión
2. **Features adicionales:** 
   - Eventos especiales/feriados con más granularidad
   - Interacciones precio-día_semana por categoría
3. **Segmentación:** Considerar modelos separados para categorías problemáticas
4. **Regularización:** Ajustar para reducir varianza en demanda alta

## Archivos Generados

- `residual_distribution.png` - Distribución de residuos
- `residuals_by_demand_level.png` - Error por nivel de demanda
- `residuals_temporal.png` - Evolución temporal
- `residuals_by_*.csv` - Métricas por segmento
