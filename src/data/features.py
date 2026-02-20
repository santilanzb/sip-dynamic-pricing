"""
Módulo de Feature Engineering.
Genera todas las features necesarias para el modelo de predicción de demanda.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')


def add_temporal_features(df: pd.DataFrame, date_col: str = 'fecha') -> pd.DataFrame:
    """
    Agrega features temporales.
    
    Features generadas:
    - dia_semana (0-6)
    - es_fin_semana (0/1)
    - dia_mes (1-31)
    - mes (1-12)
    - año
    - semana_año (1-52)
    - trimestre (1-4)
    - es_quincena (0/1) - días 15, 30, 31
    - dias_para_fin_mes
    - es_inicio_mes (0/1) - días 1-5
    - es_fin_mes (0/1) - últimos 5 días
    """
    print("   📅 Features temporales...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Básicas
    df['dia_semana'] = df[date_col].dt.dayofweek
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    df['dia_mes'] = df[date_col].dt.day
    df['mes'] = df[date_col].dt.month
    df['año'] = df[date_col].dt.year
    df['semana_año'] = df[date_col].dt.isocalendar().week.astype(int)
    df['trimestre'] = df[date_col].dt.quarter
    
    # Derivadas
    df['es_quincena'] = df['dia_mes'].isin([15, 30, 31]).astype(int)
    df['dias_para_fin_mes'] = df[date_col].dt.daysinmonth - df['dia_mes']
    df['es_inicio_mes'] = (df['dia_mes'] <= 5).astype(int)
    df['es_fin_mes'] = (df['dias_para_fin_mes'] <= 4).astype(int)
    
    # Cíclicas (para capturar periodicidad)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    lags: List[int] = [1, 7, 14, 28],
    date_col: str = 'fecha'
) -> pd.DataFrame:
    """
    Agrega features de lag (valores pasados).
    
    Args:
        df: DataFrame con datos
        group_cols: Columnas para agrupar (ej: ['producto_id', 'sucursal_id'])
        target_col: Columna objetivo para crear lags
        lags: Lista de días de lag
        date_col: Columna de fecha
    """
    print(f"   ⏮️ Lags para {target_col}: {lags}...")
    
    df = df.copy()
    df = df.sort_values(group_cols + [date_col])
    
    for lag in lags:
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    windows: List[int] = [7, 14, 30],
    date_col: str = 'fecha'
) -> pd.DataFrame:
    """
    Agrega features de ventanas móviles (rolling).
    
    Genera:
    - Promedio móvil
    - Desviación estándar móvil
    - Min/Max móvil
    """
    print(f"   📊 Rolling para {target_col}: {windows}...")
    
    df = df.copy()
    df = df.sort_values(group_cols + [date_col])
    
    for window in windows:
        # Promedio móvil (excluyendo el día actual)
        df[f'{target_col}_mean_{window}d'] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        
        # Desviación estándar móvil
        df[f'{target_col}_std_{window}d'] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())
        )
        
        # Para ventanas más grandes, agregar min/max
        if window >= 14:
            df[f'{target_col}_min_{window}d'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
            )
            df[f'{target_col}_max_{window}d'] = (
                df.groupby(group_cols)[target_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
            )
    
    return df


def add_trend_features(
    df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    window: int = 14,
    date_col: str = 'fecha'
) -> pd.DataFrame:
    """
    Agrega features de tendencia.
    
    Calcula la pendiente de regresión lineal sobre una ventana.
    """
    print(f"   📈 Tendencia para {target_col} (ventana={window})...")
    
    df = df.copy()
    df = df.sort_values(group_cols + [date_col])
    
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        slope, _ = np.polyfit(x[mask], y[mask], 1)
        return slope
    
    df[f'{target_col}_trend_{window}d'] = (
        df.groupby(group_cols)[target_col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).apply(calc_slope, raw=False))
    )
    
    return df


def add_price_features(
    df: pd.DataFrame,
    group_cols: List[str],
    price_col: str = 'precio_unitario_usd',
    date_col: str = 'fecha'
) -> pd.DataFrame:
    """
    Agrega features relacionadas con precio.
    
    - Variación de precio vs día anterior
    - Variación de precio vs semana anterior
    - Precio vs promedio histórico del producto
    - Precio vs promedio de la categoría
    """
    print("   💰 Features de precio...")
    
    df = df.copy()
    df = df.sort_values(group_cols + [date_col])
    
    # Variación vs día anterior
    df['precio_lag_1'] = df.groupby(group_cols)[price_col].shift(1)
    df['precio_var_1d'] = (df[price_col] - df['precio_lag_1']) / df['precio_lag_1']
    df['precio_var_1d'] = df['precio_var_1d'].replace([np.inf, -np.inf], np.nan)
    
    # Variación vs semana anterior
    df['precio_lag_7'] = df.groupby(group_cols)[price_col].shift(7)
    df['precio_var_7d'] = (df[price_col] - df['precio_lag_7']) / df['precio_lag_7']
    df['precio_var_7d'] = df['precio_var_7d'].replace([np.inf, -np.inf], np.nan)
    
    # Promedio móvil de precio
    df['precio_mean_7d'] = (
        df.groupby(group_cols)[price_col]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    df['precio_mean_30d'] = (
        df.groupby(group_cols)[price_col]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    )
    
    # Precio vs promedio histórico del producto
    df['precio_historico_producto'] = df.groupby(group_cols[0])[price_col].transform('mean')
    df['precio_vs_historico'] = df[price_col] / df['precio_historico_producto']
    
    # Precio vs promedio de la categoría (si existe columna clase)
    if 'clase' in df.columns:
        df['precio_mean_clase'] = df.groupby(['clase', date_col])[price_col].transform('mean')
        df['precio_vs_clase'] = df[price_col] / df['precio_mean_clase']
        df['precio_vs_clase'] = df['precio_vs_clase'].replace([np.inf, -np.inf], np.nan)
    
    # Limpiar columnas temporales
    df = df.drop(columns=['precio_lag_1', 'precio_lag_7'], errors='ignore')
    
    return df


def add_promotion_features(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str = 'fecha'
) -> pd.DataFrame:
    """
    Agrega features de promoción.
    
    - Días consecutivos en promoción
    - Días desde última promoción
    """
    print("   🎯 Features de promoción...")
    
    df = df.copy()
    df = df.sort_values(group_cols + [date_col])
    
    if 'tiene_promocion' not in df.columns:
        df['tiene_promocion'] = 0
    
    # Días consecutivos en promoción
    def count_consecutive(series):
        result = []
        count = 0
        for val in series:
            if val == 1:
                count += 1
            else:
                count = 0
            result.append(count)
        return result
    
    df['dias_en_promocion'] = (
        df.groupby(group_cols)['tiene_promocion']
        .transform(count_consecutive)
    )
    
    # Días desde última promoción
    def days_since_promo(series):
        result = []
        days = 999  # Valor alto inicial
        for val in series:
            if val == 1:
                days = 0
            else:
                days += 1
            result.append(days)
        return result
    
    df['dias_desde_promo'] = (
        df.groupby(group_cols)['tiene_promocion']
        .transform(days_since_promo)
    )
    
    # Cap días desde promo para evitar valores muy altos
    df['dias_desde_promo'] = df['dias_desde_promo'].clip(upper=365)
    
    return df


def add_product_features(
    df: pd.DataFrame,
    dim_producto: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Agrega features de producto.
    
    - Es perecedero
    - Rotación (alta/media/baja)
    - Variabilidad de demanda histórica
    """
    print("   📦 Features de producto...")
    
    df = df.copy()
    
    # Es perecedero (si no existe, calcularlo)
    if 'es_perecedero' not in df.columns:
        if 'clase' in df.columns:
            df['es_perecedero'] = df['clase'].isin(['03CARN', '08FRUV']).astype(int)
        else:
            df['es_perecedero'] = 0
    
    # Si tenemos dimensión de producto, hacer join
    if dim_producto is not None:
        # Asegurar que la columna de join existe y tipos consistentes
        if 'producto_id' in dim_producto.columns:
            # Alinear tipos para evitar errores de merge
            # Alinear tipo del join a string para evitar conflictos
            dim_producto = dim_producto.copy()
            dim_producto['producto_id'] = dim_producto['producto_id'].astype(str)
            df = df.copy()
            df['producto_id'] = df['producto_id'].astype(str)

            cols_to_join = ['producto_id']
            if 'rotacion' in dim_producto.columns:
                cols_to_join.append('rotacion')
            if 'precio_medio_usd' in dim_producto.columns:
                cols_to_join.append('precio_medio_usd')
            
            df = df.merge(
                dim_producto[cols_to_join],
                on='producto_id',
                how='left'
            )
    
    # Variabilidad de demanda por producto (coeficiente de variación)
    if 'unidades' in df.columns:
        cv_producto = df.groupby('producto_id')['unidades'].agg(['mean', 'std'])
        cv_producto['cv_demanda'] = cv_producto['std'] / cv_producto['mean']
        cv_producto = cv_producto['cv_demanda'].reset_index()
        cv_producto.columns = ['producto_id', 'variabilidad_demanda']
        
        df = df.merge(cv_producto, on='producto_id', how='left')
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de interacción.
    
    Combinaciones de features que pueden ser predictivas.
    """
    print("   🔗 Features de interacción...")
    
    df = df.copy()
    
    # Precio × Fin de semana
    if 'precio_unitario_usd' in df.columns and 'es_fin_semana' in df.columns:
        df['precio_x_finsemana'] = df['precio_unitario_usd'] * df['es_fin_semana']
    
    # Promoción × Fin de semana
    if 'tiene_promocion' in df.columns and 'es_fin_semana' in df.columns:
        df['promo_x_finsemana'] = df['tiene_promocion'] * df['es_fin_semana']
    
    # Precio × Perecedero
    if 'precio_unitario_usd' in df.columns and 'es_perecedero' in df.columns:
        df['precio_x_perecedero'] = df['precio_unitario_usd'] * df['es_perecedero']
    
    return df


def create_target(df: pd.DataFrame, target_col: str = 'unidades') -> pd.DataFrame:
    """
    Crea la variable objetivo (target) transformada.
    
    Usa log1p para estabilizar la varianza.
    """
    print("   🎯 Creando target (log1p)...")
    
    df = df.copy()
    
    # Target: log(1 + unidades) para manejar ceros y estabilizar varianza
    df['target'] = np.log1p(df[target_col].clip(lower=0))
    
    return df


def add_holiday_features(df: pd.DataFrame, date_col: str = 'fecha', holidays_csv: str = 'data/external/feriados_ve.csv') -> pd.DataFrame:
    """
    Agrega features de feriados/eventos a partir de un CSV opcional con columnas:
    - fecha (YYYY-MM-DD)
    - tipo (string/categoría opcional)
    """
    print("   🎉 Features de feriados/eventos...")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    from pathlib import Path
    p = Path(holidays_csv)
    if not p.exists():
        print(f"      ⚠️ Archivo de feriados no encontrado: {p}. Puede añadirse para mejorar el modelo.")
        df['es_feriado'] = 0
        return df
    try:
        h = pd.read_csv(p)
        h['fecha'] = pd.to_datetime(h['fecha'])
        h['es_feriado'] = 1
        h = h[['fecha', 'es_feriado'] + ([c for c in h.columns if c not in ['fecha','es_feriado']])]
        df = df.merge(h[['fecha','es_feriado']], on='fecha', how='left')
        df['es_feriado'] = df['es_feriado'].fillna(0).astype(int)
    except Exception as e:
        print(f"      ⚠️ Error leyendo feriados: {e}")
        df['es_feriado'] = 0
    return df


def engineer_features(
    fact_ventas_path: str,
    dim_producto_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    
    Args:
        fact_ventas_path: Ruta al archivo fact_ventas.parquet
        dim_producto_path: Ruta opcional a dim_producto.parquet
        output_path: Ruta donde guardar el resultado
        
    Returns:
        DataFrame con todas las features
    """
    print("=" * 70)
    print("FEATURE ENGINEERING - SIP DYNAMIC PRICING")
    print("=" * 70)
    
    # 1. Cargar datos
    print("\n📂 Cargando datos...")
    df = pd.read_parquet(fact_ventas_path)
    print(f"   fact_ventas: {len(df):,} registros")

    dim_producto = None
    if dim_producto_path and Path(dim_producto_path).exists():
        dim_producto = pd.read_parquet(dim_producto_path)
        print(f"   dim_producto: {len(dim_producto):,} registros")

    # Limpieza crítica previa (calidad de datos)
    print("\n🧼 Limpieza inicial...")
    # Tipos (preservar IDs tal cual, ya que pueden ser códigos alfanuméricos)
    # No forzamos cast numérico para evitar pérdida de información.
    # Rangos válidos
    before = len(df)
    if 'unidades' in df.columns:
        df = df[df['unidades'] >= 0]
    if 'precio_unitario_usd' in df.columns:
        df = df[df['precio_unitario_usd'] > 0]
    removed_clean = before - len(df)
    if removed_clean > 0:
        print(f"   - Removidos {removed_clean:,} registros por valores inválidos (unidades negativas / precio<=0)")

    # Definir columnas de agrupación
    group_cols = ['producto_id', 'sucursal_id']

    # 2. Features temporales
    print("\n🔧 Generando features...")
    df = add_temporal_features(df)
    
    # 3. Lag features para demanda
    df = add_lag_features(df, group_cols, 'unidades', lags=[1, 7, 14, 28])
    
    # 4. Rolling features para demanda
    df = add_rolling_features(df, group_cols, 'unidades', windows=[7, 14, 30])
    
    # 5. Tendencia de demanda
    df = add_trend_features(df, group_cols, 'unidades', window=14)
    
    # 6. Features de precio
    df = add_price_features(df, group_cols)
    
    # 7. Features de promoción
    df = add_promotion_features(df, group_cols)
    
    # 8. Features de producto
    df = add_product_features(df, dim_producto)
    
    # 9. Features de interacción
    df = add_interaction_features(df)
    
    # 10. Feriados
    df = add_holiday_features(df)

    # 11. Target
    df = create_target(df)
    
    # 11. Filtrar filas con suficientes datos históricos
    # (eliminar primeros 30 días por producto donde no hay lags completos)
    print("\n🧹 Limpiando datos...")
    initial_count = len(df)
    
    # Eliminar filas donde los lags principales son NaN
    df = df.dropna(subset=['unidades_lag_7', 'unidades_mean_7d'])
    
    final_count = len(df)
    removed = initial_count - final_count
    print(f"   Eliminados {removed:,} registros sin historial suficiente")
    print(f"   Registros finales: {final_count:,}")
    
    # 12. Guardar
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\n💾 Guardado en: {output_path}")
    
    # 13. Resumen de features
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE FEATURES")
    print("=" * 70)
    
    feature_cols = [c for c in df.columns if c not in [
        'fecha', 'producto_id', 'sucursal_id', 'target', 'unidades',
        'ingreso_usd', 'costo_usd', 'margen_usd', 'clase', 'tasa_bcv'
    ]]
    
    print(f"\n   Total features: {len(feature_cols)}")
    print(f"   Registros: {len(df):,}")
    print(f"   Período: {df['fecha'].min().date()} - {df['fecha'].max().date()}")
    
    # Categorías de features
    temporal = [c for c in feature_cols if any(x in c for x in ['dia_', 'mes', 'año', 'semana', 'trimestre', 'fin_semana', 'quincena', 'inicio_mes', 'fin_mes'])]
    precio = [c for c in feature_cols if 'precio' in c]
    demanda = [c for c in feature_cols if 'unidades' in c]
    promo = [c for c in feature_cols if 'promo' in c or 'descuento' in c]
    producto = [c for c in feature_cols if any(x in c for x in ['perecedero', 'rotacion', 'variabilidad', 'cv_'])]
    
    print(f"\n   Por categoría:")
    print(f"   - Temporales: {len(temporal)}")
    print(f"   - Precio: {len(precio)}")
    print(f"   - Demanda (lags/rolling): {len(demanda)}")
    print(f"   - Promoción: {len(promo)}")
    print(f"   - Producto: {len(producto)}")
    
    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETADO")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    df = engineer_features(
        fact_ventas_path='data/processed/fact_ventas.parquet',
        dim_producto_path='data/processed/dim_producto.parquet',
        output_path='data/processed/features.parquet'
    )
    
    print("\n📋 Columnas generadas:")
    for col in sorted(df.columns):
        print(f"   - {col}")
