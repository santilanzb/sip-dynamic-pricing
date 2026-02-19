"""
Script para ejecutar EDA y mostrar resultados clave.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Paths
DATA_RAW = Path('data/raw')
REPORTS = Path('reports')
REPORTS.mkdir(exist_ok=True)

print("=" * 70)
print("SIP DYNAMIC PRICING - ANÁLISIS EXPLORATORIO (EDA)")
print("=" * 70)

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("\n📂 CARGANDO DATOS...")

df_cv = pd.read_parquet(DATA_RAW / 'compraventa_raw.parquet')
df_promo = pd.read_parquet(DATA_RAW / 'promociones_raw.parquet')
df_ajustes = pd.read_parquet(DATA_RAW / 'ajustes_raw.parquet')

print(f"   CompraVenta: {len(df_cv):,} registros")
print(f"   Promociones: {len(df_promo):,} registros")
print(f"   Ajustes: {len(df_ajustes):,} registros")

# ============================================================================
# 2. CAMPOS DERIVADOS
# ============================================================================
print("\n🔧 CALCULANDO CAMPOS DERIVADOS...")

df_cv['Precio_Unitario'] = np.where(
    df_cv['Unidades_Venta_Cantidad'] > 0,
    df_cv['Precio_Venta_Total'] / df_cv['Unidades_Venta_Cantidad'],
    0
)

df_cv['Margen_Bruto'] = df_cv['Precio_Venta_Total'] - df_cv['Costo_Venta_Total']

df_cv['Margen_Porcentaje'] = np.where(
    df_cv['Precio_Venta_Total'] > 0,
    (df_cv['Margen_Bruto'] / df_cv['Precio_Venta_Total'] * 100),
    0
)

print("   ✓ Precio_Unitario, Margen_Bruto, Margen_Porcentaje")

# ============================================================================
# 3. RESUMEN GENERAL
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESUMEN GENERAL")
print("=" * 70)

print(f"\n   Período: {df_cv['Fecha'].min().strftime('%Y-%m-%d')} a {df_cv['Fecha'].max().strftime('%Y-%m-%d')}")
print(f"   Días únicos: {df_cv['Fecha'].nunique()}")
print(f"   Sucursales: {df_cv['Sucursal'].nunique()}")
print(f"   Productos únicos: {df_cv['Codigo_Interno'].nunique():,}")

# ============================================================================
# 4. MÉTRICAS FINANCIERAS
# ============================================================================
print("\n" + "=" * 70)
print("💰 MÉTRICAS FINANCIERAS")
print("=" * 70)

ingresos_total = df_cv['Precio_Venta_Total'].sum()
margen_total = df_cv['Margen_Bruto'].sum()
margen_pct = (margen_total / ingresos_total * 100)

print(f"\n   Ingresos totales: ${ingresos_total:,.2f}")
print(f"   Margen bruto total: ${margen_total:,.2f}")
print(f"   Margen bruto promedio: {margen_pct:.1f}%")

# ============================================================================
# 5. ANÁLISIS POR CLASE
# ============================================================================
print("\n" + "=" * 70)
print("📦 ANÁLISIS POR CLASE (CATEGORÍA)")
print("=" * 70)

ventas_clase = df_cv.groupby('Clase').agg({
    'Unidades_Venta_Cantidad': 'sum',
    'Precio_Venta_Total': 'sum',
    'Margen_Bruto': 'sum',
    'Codigo_Interno': 'nunique'
}).reset_index()

ventas_clase.columns = ['Clase', 'Unidades', 'Ingresos', 'Margen', 'Productos']
ventas_clase['Margen_%'] = (ventas_clase['Margen'] / ventas_clase['Ingresos'] * 100).round(2)
ventas_clase['Part_Ingresos_%'] = (ventas_clase['Ingresos'] / ventas_clase['Ingresos'].sum() * 100).round(1)
ventas_clase = ventas_clase.sort_values('Ingresos', ascending=False)

print(f"\n   {'Clase':<10} {'Ingresos':>20} {'Part.%':>8} {'Margen%':>10} {'Productos':>10}")
print("   " + "-" * 60)
for _, row in ventas_clase.iterrows():
    print(f"   {row['Clase']:<10} ${row['Ingresos']:>18,.0f} {row['Part_Ingresos_%']:>7.1f}% {row['Margen_%']:>9.1f}% {row['Productos']:>10}")

# ============================================================================
# 6. ANÁLISIS POR SUCURSAL
# ============================================================================
print("\n" + "=" * 70)
print("🏪 ANÁLISIS POR SUCURSAL")
print("=" * 70)

ventas_sucursal = df_cv.groupby('Sucursal').agg({
    'Unidades_Venta_Cantidad': 'sum',
    'Precio_Venta_Total': 'sum',
    'Margen_Bruto': 'sum'
}).reset_index()

ventas_sucursal.columns = ['Sucursal', 'Unidades', 'Ingresos', 'Margen']
ventas_sucursal['Margen_%'] = (ventas_sucursal['Margen'] / ventas_sucursal['Ingresos'] * 100).round(2)
ventas_sucursal['Part_%'] = (ventas_sucursal['Ingresos'] / ventas_sucursal['Ingresos'].sum() * 100).round(1)
ventas_sucursal = ventas_sucursal.sort_values('Ingresos', ascending=False)

print(f"\n   {'Sucursal':<12} {'Ingresos':>20} {'Part.%':>8} {'Margen%':>10}")
print("   " + "-" * 52)
for _, row in ventas_sucursal.iterrows():
    print(f"   {row['Sucursal']:<12} ${row['Ingresos']:>18,.0f} {row['Part_%']:>7.1f}% {row['Margen_%']:>9.1f}%")

# ============================================================================
# 7. TOP 10 PRODUCTOS
# ============================================================================
print("\n" + "=" * 70)
print("🏆 TOP 10 PRODUCTOS POR INGRESOS")
print("=" * 70)

top_productos = df_cv.groupby(['Codigo_Interno', 'Descripcion', 'Clase']).agg({
    'Unidades_Venta_Cantidad': 'sum',
    'Precio_Venta_Total': 'sum',
    'Margen_Bruto': 'sum'
}).reset_index()

top_productos.columns = ['Codigo', 'Descripcion', 'Clase', 'Unidades', 'Ingresos', 'Margen']
top_ingresos = top_productos.nlargest(10, 'Ingresos')

print(f"\n   {'#':<3} {'Descripción':<35} {'Clase':<8} {'Ingresos':>15}")
print("   " + "-" * 65)
for i, (_, row) in enumerate(top_ingresos.iterrows(), 1):
    desc = row['Descripcion'][:33] + '..' if len(row['Descripcion']) > 35 else row['Descripcion']
    print(f"   {i:<3} {desc:<35} {row['Clase']:<8} ${row['Ingresos']:>13,.0f}")

# ============================================================================
# 8. ANÁLISIS TEMPORAL - DÍA DE SEMANA
# ============================================================================
print("\n" + "=" * 70)
print("📅 DISTRIBUCIÓN POR DÍA DE LA SEMANA")
print("=" * 70)

df_cv['DiaSemana'] = df_cv['Fecha'].dt.dayofweek
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

ventas_dia = df_cv.groupby('DiaSemana')['Precio_Venta_Total'].sum().reset_index()
ventas_dia['Dia'] = [dias[i] for i in ventas_dia['DiaSemana']]
ventas_dia['Part_%'] = (ventas_dia['Precio_Venta_Total'] / ventas_dia['Precio_Venta_Total'].sum() * 100).round(1)

print(f"\n   {'Día':<12} {'Ingresos':>18} {'Part.%':>8}")
print("   " + "-" * 40)
for _, row in ventas_dia.iterrows():
    print(f"   {row['Dia']:<12} ${row['Precio_Venta_Total']:>16,.0f} {row['Part_%']:>7.1f}%")

# ============================================================================
# 9. CORRELACIÓN PRECIO-DEMANDA
# ============================================================================
print("\n" + "=" * 70)
print("📈 ANÁLISIS DE ELASTICIDAD PRECIO-DEMANDA")
print("=" * 70)

producto_dia = df_cv.groupby(['Codigo_Interno', 'Fecha']).agg({
    'Unidades_Venta_Cantidad': 'sum',
    'Precio_Unitario': 'mean'
}).reset_index()

correlaciones = producto_dia.groupby('Codigo_Interno').apply(
    lambda x: x['Precio_Unitario'].corr(x['Unidades_Venta_Cantidad']) if len(x) > 30 else np.nan
).dropna()

print(f"\n   Productos analizados (>30 días de datos): {len(correlaciones)}")
print(f"\n   Correlación Precio-Demanda:")
print(f"      Media: {correlaciones.mean():.3f}")
print(f"      Mediana: {correlaciones.median():.3f}")
print(f"      % con correlación negativa: {(correlaciones < 0).mean()*100:.1f}%")
print(f"      % con correlación < -0.3 (elásticos): {(correlaciones < -0.3).mean()*100:.1f}%")

# ============================================================================
# 10. PROMOCIONES
# ============================================================================
print("\n" + "=" * 70)
print("🎯 ANÁLISIS DE PROMOCIONES")
print("=" * 70)

tipo_promo_desc = {
    1: 'Precio Oferta',
    2: '% Descuento',
    3: 'Descuento Fijo',
    4: 'M×N Gratis',
    5: 'M×N Precio Oferta',
    6: 'M×N % Descuento',
    7: 'M×N Monto Fijo',
    8: 'Premio M×N',
    9: 'Premio Precio Oferta',
    10: 'Premio % Descuento',
    11: 'Premio Monto Descuento'
}

print(f"\n   Total promociones: {df_promo['Cod_Promocion'].nunique():,}")
print(f"   Productos con promoción: {df_promo['Cod_Producto'].nunique():,}")

promo_counts = df_promo['Tipo_Promocion'].value_counts()
print(f"\n   {'Tipo':<25} {'Cantidad':>10} {'%':>8}")
print("   " + "-" * 45)
for tipo, count in promo_counts.head(5).items():
    desc = tipo_promo_desc.get(tipo, f'Tipo {tipo}')
    pct = count / len(df_promo) * 100
    print(f"   {desc:<25} {count:>10,} {pct:>7.1f}%")

# ============================================================================
# 11. AJUSTES DE INVENTARIO
# ============================================================================
print("\n" + "=" * 70)
print("📦 ANÁLISIS DE AJUSTES DE INVENTARIO")
print("=" * 70)

print(f"\n   Total ajustes: {len(df_ajustes):,}")
print(f"   Productos con ajustes: {df_ajustes['Codigo_Producto'].nunique():,}")

if 'Tipo_Documento' in df_ajustes.columns:
    print(f"\n   Distribución por tipo de documento:")
    for tipo, count in df_ajustes['Tipo_Documento'].value_counts().head(5).items():
        print(f"      {tipo}: {count:,}")

# ============================================================================
# 12. GENERAR GRÁFICOS
# ============================================================================
print("\n" + "=" * 70)
print("📊 GENERANDO GRÁFICOS...")
print("=" * 70)

# Gráfico 1: Ventas mensuales
df_cv['AñoMes'] = df_cv['Fecha'].dt.to_period('M')
ventas_mes = df_cv.groupby('AñoMes')['Precio_Venta_Total'].sum().reset_index()
ventas_mes['AñoMes'] = ventas_mes['AñoMes'].astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(ventas_mes['AñoMes'], ventas_mes['Precio_Venta_Total'] / 1e9)
ax.set_ylabel('Ingresos (Miles de Millones $)')
ax.set_title('Evolución Mensual de Ventas - Categorías FRUV, CARN, CHAR')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(REPORTS / 'ventas_mensuales.png', dpi=150, bbox_inches='tight')
print("   ✓ ventas_mensuales.png")

# Gráfico 2: Ventas por día de semana
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(ventas_dia['Dia'], ventas_dia['Precio_Venta_Total'] / 1e9)
ax.set_ylabel('Ingresos (Miles de Millones $)')
ax.set_title('Distribución de Ventas por Día de la Semana')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORTS / 'ventas_dia_semana.png', dpi=150, bbox_inches='tight')
print("   ✓ ventas_dia_semana.png")

# Gráfico 3: Participación por clase
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(ventas_clase['Ingresos'], labels=ventas_clase['Clase'], autopct='%1.1f%%', startangle=90)
ax.set_title('Participación en Ingresos por Clase')
plt.tight_layout()
plt.savefig(REPORTS / 'participacion_clases.png', dpi=150, bbox_inches='tight')
print("   ✓ participacion_clases.png")

# Gráfico 4: Correlación precio-demanda
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(correlaciones, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Sin correlación')
ax.axvline(x=correlaciones.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {correlaciones.median():.2f}')
ax.set_xlabel('Correlación Precio-Demanda')
ax.set_ylabel('Número de Productos')
ax.set_title('Distribución de Elasticidad Precio-Demanda por Producto')
ax.legend()
plt.tight_layout()
plt.savefig(REPORTS / 'elasticidad_precio_demanda.png', dpi=150, bbox_inches='tight')
print("   ✓ elasticidad_precio_demanda.png")

# Gráfico 5: Top productos
fig, ax = plt.subplots(figsize=(12, 8))
top15 = top_productos.nlargest(15, 'Ingresos')
y_pos = range(len(top15))
ax.barh(y_pos, top15['Ingresos'] / 1e6)
ax.set_yticks(y_pos)
ax.set_yticklabels([d[:40] for d in top15['Descripcion']])
ax.invert_yaxis()
ax.set_xlabel('Ingresos (Millones $)')
ax.set_title('Top 15 Productos por Ingresos')
plt.tight_layout()
plt.savefig(REPORTS / 'top_productos.png', dpi=150, bbox_inches='tight')
print("   ✓ top_productos.png")

plt.close('all')

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("✅ RESUMEN EJECUTIVO - HALLAZGOS CLAVE")
print("=" * 70)

print(f"""
📊 DATOS ANALIZADOS
   • {len(df_cv):,} registros de CompraVenta
   • Período: {df_cv['Fecha'].min().strftime('%Y-%m-%d')} a {df_cv['Fecha'].max().strftime('%Y-%m-%d')}
   • {df_cv['Sucursal'].nunique()} sucursales, {df_cv['Codigo_Interno'].nunique():,} productos

💰 MÉTRICAS FINANCIERAS
   • Ingresos totales: ${ingresos_total:,.0f}
   • Margen bruto promedio: {margen_pct:.1f}%

📈 PATRONES IDENTIFICADOS
   • {(correlaciones < 0).mean()*100:.0f}% de productos muestran correlación negativa precio-demanda
   • Productos con alta elasticidad (corr < -0.3): {(correlaciones < -0.3).mean()*100:.0f}%
   • Día con más ventas: {ventas_dia.loc[ventas_dia['Precio_Venta_Total'].idxmax(), 'Dia']}
   • Clase dominante: {ventas_clase.iloc[0]['Clase']} ({ventas_clase.iloc[0]['Part_Ingresos_%']:.0f}% de ingresos)

📁 GRÁFICOS GENERADOS EN: reports/
""")

print("=" * 70)
print("EDA COMPLETADO")
print("=" * 70)
