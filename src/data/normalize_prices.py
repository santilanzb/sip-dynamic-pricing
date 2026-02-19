"""
Script para normalizar precios de Bs a USD usando tasas BCV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.bcv_rates import get_bcv_rates, normalize_prices


def main():
    print("=" * 70)
    print("NORMALIZACIÓN DE PRECIOS Bs -> USD")
    print("=" * 70)
    
    # Paths
    DATA_RAW = Path('data/raw')
    DATA_PROCESSED = Path('data/processed')
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # 1. Obtener tasas BCV
    # Intenta cargar desde CSV si existe, sino genera aproximadas
    csv_path = DATA_RAW / 'tasas_bcv.csv'
    
    rates_df = get_bcv_rates(
        start_date="2023-01-01",
        end_date="2025-12-31",
        csv_path=str(csv_path) if csv_path.exists() else None,
        output_path=str(DATA_PROCESSED / 'bcv_rates.parquet')
    )
    
    # 2. Cargar CompraVenta
    print("\n📂 CARGANDO DATOS DE COMPRAVENTA...")
    df_cv = pd.read_parquet(DATA_RAW / 'compraventa_raw.parquet')
    print(f"   Registros: {len(df_cv):,}")
    
    # 3. Normalizar precios
    print("\n💱 NORMALIZANDO PRECIOS A USD...")
    
    price_columns = [
        'Precio_Venta_Total',
        'Costo_Venta_Total'
    ]
    
    df_normalized = normalize_prices(
        df=df_cv,
        date_col='Fecha',
        price_cols=price_columns,
        rates_df=rates_df,
        suffix='_USD'
    )
    
    # 4. Calcular campos derivados en USD
    print("\n🔧 CALCULANDO CAMPOS DERIVADOS EN USD...")
    
    # Precio unitario USD
    df_normalized['Precio_Unitario_USD'] = np.where(
        df_normalized['Unidades_Venta_Cantidad'] > 0,
        df_normalized['Precio_Venta_Total_USD'] / df_normalized['Unidades_Venta_Cantidad'],
        0
    )
    
    # Margen bruto USD
    df_normalized['Margen_Bruto_USD'] = (
        df_normalized['Precio_Venta_Total_USD'] - df_normalized['Costo_Venta_Total_USD']
    )
    
    # Margen porcentaje (se mantiene igual)
    df_normalized['Margen_Porcentaje'] = np.where(
        df_normalized['Precio_Venta_Total_USD'] > 0,
        (df_normalized['Margen_Bruto_USD'] / df_normalized['Precio_Venta_Total_USD'] * 100),
        0
    )
    
    # 5. Guardar datos normalizados
    output_path = DATA_PROCESSED / 'compraventa_normalized.parquet'
    df_normalized.to_parquet(output_path, index=False)
    print(f"\n✅ Guardado: {output_path}")
    
    # 6. Mostrar resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE NORMALIZACIÓN")
    print("=" * 70)
    
    print(f"\n   Registros procesados: {len(df_normalized):,}")
    print(f"   Período: {df_normalized['Fecha'].min().date()} - {df_normalized['Fecha'].max().date()}")
    
    # Métricas en Bs vs USD
    print("\n   COMPARACIÓN Bs vs USD:")
    print(f"   {'Métrica':<30} {'Bs':>20} {'USD':>20}")
    print("   " + "-" * 72)
    
    ingresos_bs = df_normalized['Precio_Venta_Total'].sum()
    ingresos_usd = df_normalized['Precio_Venta_Total_USD'].sum()
    print(f"   {'Ingresos totales':<30} {ingresos_bs:>20,.0f} {ingresos_usd:>20,.2f}")
    
    margen_bs = df_normalized['Margen_Bruto_USD'].sum() * rates_df['Tasa_USD'].mean()
    margen_usd = df_normalized['Margen_Bruto_USD'].sum()
    print(f"   {'Margen bruto':<30} {'N/A':>20} {margen_usd:>20,.2f}")
    
    # Precio promedio por categoría
    print("\n   PRECIO PROMEDIO POR UNIDAD (USD) POR CLASE:")
    precio_clase = df_normalized.groupby('Clase').agg({
        'Precio_Unitario_USD': 'mean',
        'Margen_Porcentaje': 'mean'
    }).round(2)
    
    for clase, row in precio_clase.iterrows():
        print(f"   {clase}: ${row['Precio_Unitario_USD']:.2f}/unidad, Margen: {row['Margen_Porcentaje']:.1f}%")
    
    # Evolución mensual en USD
    print("\n   EVOLUCIÓN MENSUAL (USD):")
    df_normalized['AñoMes'] = df_normalized['Fecha'].dt.to_period('M')
    evol_mensual = df_normalized.groupby('AñoMes')['Precio_Venta_Total_USD'].sum()
    
    # Mostrar primeros y últimos 3 meses
    print(f"   {'Mes':<12} {'Ingresos USD':>15}")
    print("   " + "-" * 28)
    for periodo in list(evol_mensual.index[:3]) + ['...'] + list(evol_mensual.index[-3:]):
        if periodo == '...':
            print(f"   {'...':<12}")
        else:
            print(f"   {str(periodo):<12} ${evol_mensual[periodo]:>13,.2f}")
    
    print("\n" + "=" * 70)
    print("NORMALIZACIÓN COMPLETADA")
    print("=" * 70)
    
    return df_normalized


if __name__ == "__main__":
    df = main()
