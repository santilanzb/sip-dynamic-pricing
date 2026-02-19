"""
Módulo para obtener tasas BCV históricas y normalizar precios Bs -> USD.

Soporta múltiples fuentes de datos:
1. Archivo CSV local (preferido para datos históricos)
2. pyDolarVenezuela API (para datos recientes)
3. Interpolación para fechas faltantes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


def load_bcv_rates_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Carga tasas BCV desde un archivo CSV.
    
    El CSV debe tener columnas: Fecha, Tasa_USD (o equivalentes)
    Formatos de fecha soportados: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con columnas: Fecha (datetime), Tasa_USD (float)
    """
    df = pd.read_csv(csv_path)
    
    # Detectar columna de fecha
    date_cols = [c for c in df.columns if 'fecha' in c.lower() or 'date' in c.lower()]
    if date_cols:
        date_col = date_cols[0]
    else:
        date_col = df.columns[0]
    
    # Detectar columna de tasa
    rate_cols = [c for c in df.columns if 'usd' in c.lower() or 'dolar' in c.lower() or 'tasa' in c.lower()]
    if rate_cols:
        rate_col = rate_cols[0]
    else:
        rate_col = df.columns[1]
    
    # Parsear fechas
    df['Fecha'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    
    # Parsear tasa (manejar formato venezolano con coma decimal)
    if df[rate_col].dtype == object:
        df['Tasa_USD'] = df[rate_col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    else:
        df['Tasa_USD'] = df[rate_col].astype(float)
    
    df = df[['Fecha', 'Tasa_USD']].dropna()
    df = df.sort_values('Fecha').drop_duplicates(subset='Fecha', keep='last')
    
    print(f"   ✓ Cargadas {len(df)} tasas desde {df['Fecha'].min().date()} hasta {df['Fecha'].max().date()}")
    
    return df


def fetch_bcv_rates_api(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene tasas BCV desde pyDolarVenezuela API.
    
    Nota: La API puede tener limitaciones para rangos de fecha muy largos.
    
    Args:
        start_date: Fecha inicio en formato DD-MM-YYYY
        end_date: Fecha fin en formato DD-MM-YYYY
        
    Returns:
        DataFrame con columnas: Fecha, Tasa_USD
    """
    try:
        from pyDolarVenezuela.pages import BCV
        from pyDolarVenezuela import Monitor
        
        monitor = Monitor(BCV, 'USD')
        
        # Obtener historial
        history = monitor.get_prices_history("bcv", start_date, end_date)
        
        if history:
            records = []
            for item in history:
                records.append({
                    'Fecha': pd.to_datetime(item.date, dayfirst=True),
                    'Tasa_USD': item.price
                })
            
            df = pd.DataFrame(records)
            print(f"   ✓ Obtenidas {len(df)} tasas desde API")
            return df
        else:
            print("   ⚠ No se obtuvieron datos de la API")
            return pd.DataFrame(columns=['Fecha', 'Tasa_USD'])
            
    except Exception as e:
        print(f"   ⚠ Error al obtener tasas de API: {e}")
        return pd.DataFrame(columns=['Fecha', 'Tasa_USD'])


def interpolate_missing_dates(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Interpola fechas faltantes (fines de semana, feriados).
    
    Usa forward-fill para días no hábiles.
    
    Args:
        df: DataFrame con tasas
        start_date: Fecha inicio
        end_date: Fecha fin
        
    Returns:
        DataFrame con todas las fechas del rango
    """
    # Crear rango completo de fechas
    all_dates = pd.DataFrame({
        'Fecha': pd.date_range(start=start_date, end=end_date, freq='D')
    })
    
    # Merge con tasas existentes
    df_complete = all_dates.merge(df, on='Fecha', how='left')
    
    # Forward-fill para días sin tasa (fines de semana, feriados)
    df_complete['Tasa_USD'] = df_complete['Tasa_USD'].ffill()
    
    # Backward-fill para los primeros días si no hay tasa
    df_complete['Tasa_USD'] = df_complete['Tasa_USD'].bfill()
    
    return df_complete


def get_bcv_rates(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    csv_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Obtiene tasas BCV completas para el rango de fechas.
    
    Prioridad:
    1. CSV local si se proporciona
    2. Parquet cacheado si existe
    3. API de pyDolarVenezuela
    4. Generación de datos aproximados con advertencia
    
    Args:
        start_date: Fecha inicio YYYY-MM-DD
        end_date: Fecha fin YYYY-MM-DD
        csv_path: Ruta opcional a CSV con tasas históricas
        output_path: Ruta donde guardar el parquet con tasas
        
    Returns:
        DataFrame con Fecha y Tasa_USD para cada día del rango
    """
    print("\n📊 OBTENIENDO TASAS BCV...")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    df_rates = None
    
    # 1. Intentar cargar desde CSV
    if csv_path and Path(csv_path).exists():
        print(f"   Cargando desde CSV: {csv_path}")
        df_rates = load_bcv_rates_from_csv(csv_path)
    
    # 2. Intentar cargar desde parquet cacheado
    cache_path = Path(output_path) if output_path else Path('data/processed/bcv_rates.parquet')
    if df_rates is None and cache_path.exists():
        print(f"   Cargando desde cache: {cache_path}")
        df_rates = pd.read_parquet(cache_path)
        print(f"   ✓ Cargadas {len(df_rates)} tasas desde cache")
    
    # 3. Intentar API (solo para datos recientes, últimos 30 días)
    if df_rates is None:
        # La API funciona mejor para rangos cortos
        api_start = max(start_dt, datetime.now() - timedelta(days=30))
        if api_start < end_dt:
            print("   Intentando obtener tasas desde API (últimos 30 días)...")
            df_rates = fetch_bcv_rates_api(
                api_start.strftime("%d-%m-%Y"),
                end_dt.strftime("%d-%m-%Y")
            )
    
    # 4. Si no hay datos, crear dataset aproximado con advertencia
    if df_rates is None or len(df_rates) == 0:
        print("\n   ⚠️ ADVERTENCIA: No se encontraron tasas BCV históricas.")
        print("   Se generará un dataset aproximado basado en tendencias conocidas.")
        print("   Para mayor precisión, proporcione un CSV con tasas históricas.")
        print("   Formato esperado: Fecha,Tasa_USD")
        print("   Ejemplo: data/raw/tasas_bcv.csv")
        
        df_rates = _generate_approximate_rates(start_dt, end_dt)
    
    # Interpolar fechas faltantes
    df_complete = interpolate_missing_dates(df_rates, start_dt, end_dt)
    
    # Guardar cache
    if output_path:
        cache_path = Path(output_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_complete.to_parquet(cache_path, index=False)
        print(f"   ✓ Guardado en: {cache_path}")
    
    print(f"\n   📈 Tasas BCV: {len(df_complete)} días")
    print(f"   Rango: {df_complete['Fecha'].min().date()} - {df_complete['Fecha'].max().date()}")
    print(f"   Tasa inicial: {df_complete['Tasa_USD'].iloc[0]:.2f} Bs/USD")
    print(f"   Tasa final: {df_complete['Tasa_USD'].iloc[-1]:.2f} Bs/USD")
    
    return df_complete


def _generate_approximate_rates(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Genera tasas aproximadas basadas en tendencias históricas conocidas.
    
    Basado en datos públicos aproximados:
    - Enero 2023: ~18 Bs/USD
    - Diciembre 2023: ~36 Bs/USD  
    - Diciembre 2024: ~50 Bs/USD
    - Diciembre 2025: ~75 Bs/USD (proyección)
    
    ADVERTENCIA: Estos son valores aproximados para desarrollo.
    Use datos reales del BCV para producción.
    """
    # Puntos de referencia aproximados (Bs/USD)
    reference_points = {
        datetime(2023, 1, 1): 18.0,
        datetime(2023, 6, 1): 27.0,
        datetime(2023, 12, 1): 36.0,
        datetime(2024, 6, 1): 43.0,
        datetime(2024, 12, 1): 50.0,
        datetime(2025, 6, 1): 62.0,
        datetime(2025, 12, 31): 75.0,
    }
    
    dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    # Interpolar entre puntos de referencia
    ref_dates = list(reference_points.keys())
    ref_values = list(reference_points.values())
    
    rates = np.interp(
        [d.timestamp() for d in dates],
        [d.timestamp() for d in ref_dates],
        ref_values
    )
    
    # Agregar ruido pequeño para simular variación diaria
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, len(rates))
    rates = rates + noise
    rates = np.maximum(rates, 1)  # Asegurar valores positivos
    
    return pd.DataFrame({
        'Fecha': dates,
        'Tasa_USD': rates
    })


def normalize_prices(
    df: pd.DataFrame,
    date_col: str,
    price_cols: list,
    rates_df: pd.DataFrame,
    suffix: str = '_USD'
) -> pd.DataFrame:
    """
    Normaliza columnas de precios de Bs a USD.
    
    Args:
        df: DataFrame con precios en Bs
        date_col: Nombre de columna de fecha
        price_cols: Lista de columnas de precio a normalizar
        rates_df: DataFrame con tasas BCV (Fecha, Tasa_USD)
        suffix: Sufijo para nuevas columnas en USD
        
    Returns:
        DataFrame con columnas adicionales en USD
    """
    # Asegurar que la fecha sea datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Merge con tasas
    df = df.merge(
        rates_df[['Fecha', 'Tasa_USD']].rename(columns={'Fecha': date_col}),
        on=date_col,
        how='left'
    )
    
    # Calcular columnas en USD
    for col in price_cols:
        usd_col = f"{col}{suffix}"
        df[usd_col] = df[col] / df['Tasa_USD']
    
    # Eliminar columna de tasa temporal
    df = df.drop(columns=['Tasa_USD'])
    
    return df


if __name__ == "__main__":
    # Test: obtener tasas
    rates = get_bcv_rates(
        start_date="2023-01-01",
        end_date="2025-12-31",
        output_path="data/processed/bcv_rates.parquet"
    )
    
    print("\n📊 Muestra de tasas:")
    print(rates.head(10))
