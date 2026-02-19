"""
Módulo de transformación ETL.
Crea la tabla de hechos unificada (fact_ventas) a partir de los datos raw.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_raw_data(data_path: Path) -> dict:
    """Carga todos los datasets raw."""
    print("📂 Cargando datos raw...")
    
    data = {
        'compraventa': pd.read_parquet(data_path / 'processed' / 'compraventa_normalized.parquet'),
        'promociones': pd.read_parquet(data_path / 'raw' / 'promociones_raw.parquet'),
        'ajustes': pd.read_parquet(data_path / 'raw' / 'ajustes_raw.parquet'),
        'bcv_rates': pd.read_parquet(data_path / 'processed' / 'bcv_rates.parquet'),
    }
    
    for name, df in data.items():
        print(f"   {name}: {len(df):,} registros")
    
    return data


def create_dim_tiempo(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Crea dimensión de tiempo con todos los atributos temporales.
    
    Incluye feriados de Venezuela.
    """
    print("\n📅 Creando DIM_TIEMPO...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Feriados Venezuela (principales)
    feriados_fijos = [
        (1, 1),   # Año Nuevo
        (5, 1),   # Día del Trabajador
        (6, 24),  # Batalla de Carabobo
        (7, 5),   # Día de la Independencia
        (7, 24),  # Natalicio de Bolívar
        (10, 12), # Día de la Resistencia Indígena
        (12, 24), # Nochebuena
        (12, 25), # Navidad
        (12, 31), # Fin de año
    ]
    
    # Crear feriados para todos los años
    feriados = set()
    for year in range(dates.min().year, dates.max().year + 1):
        for month, day in feriados_fijos:
            try:
                feriados.add(datetime(year, month, day))
            except ValueError:
                pass
    
    dim_tiempo = pd.DataFrame({'fecha': dates})
    dim_tiempo['dia_semana'] = dim_tiempo['fecha'].dt.dayofweek
    dim_tiempo['dia_semana_nombre'] = dim_tiempo['fecha'].dt.day_name()
    dim_tiempo['es_fin_semana'] = dim_tiempo['dia_semana'].isin([5, 6]).astype(int)
    dim_tiempo['dia_mes'] = dim_tiempo['fecha'].dt.day
    dim_tiempo['mes'] = dim_tiempo['fecha'].dt.month
    dim_tiempo['año'] = dim_tiempo['fecha'].dt.year
    dim_tiempo['semana_año'] = dim_tiempo['fecha'].dt.isocalendar().week.astype(int)
    dim_tiempo['trimestre'] = dim_tiempo['fecha'].dt.quarter
    dim_tiempo['es_feriado'] = dim_tiempo['fecha'].isin(feriados).astype(int)
    dim_tiempo['es_quincena'] = dim_tiempo['dia_mes'].isin([15, 30, 31]).astype(int)
    dim_tiempo['dias_para_fin_mes'] = (
        dim_tiempo['fecha'].dt.daysinmonth - dim_tiempo['dia_mes']
    )
    
    print(f"   ✓ {len(dim_tiempo)} días generados")
    print(f"   ✓ {dim_tiempo['es_feriado'].sum()} feriados marcados")
    
    return dim_tiempo


def create_dim_producto(df_compraventa: pd.DataFrame) -> pd.DataFrame:
    """
    Crea dimensión de producto con atributos derivados.
    """
    print("\n📦 Creando DIM_PRODUCTO...")
    
    # Agrupar para obtener un registro único por producto
    dim_producto = df_compraventa.groupby('Codigo_Interno').agg({
        'Descripcion': 'first',
        'SKU': 'first',
        'Clase': 'first',
        'Departamento': 'first',
        'SubDepartamento': 'first',
        'Categoria': 'first',
        'Unidades_Venta_Cantidad': 'sum',
        'Precio_Unitario_USD': 'mean',
    }).reset_index()
    
    dim_producto.columns = [
        'producto_id', 'descripcion', 'sku', 'clase', 
        'departamento', 'subdepartamento', 'categoria',
        'ventas_totales', 'precio_medio_usd'
    ]
    
    # Atributos derivados
    dim_producto['es_perecedero'] = dim_producto['clase'].isin(['03CARN', '08FRUV']).astype(int)
    
    # Clasificar rotación por cuartiles
    dim_producto['rotacion'] = pd.qcut(
        dim_producto['ventas_totales'],
        q=3,
        labels=['baja', 'media', 'alta']
    )
    
    print(f"   ✓ {len(dim_producto)} productos únicos")
    print(f"   ✓ Perecederos: {dim_producto['es_perecedero'].sum()}")
    
    return dim_producto


def create_dim_sucursal(df_compraventa: pd.DataFrame) -> pd.DataFrame:
    """
    Crea dimensión de sucursal.
    """
    print("\n🏪 Creando DIM_SUCURSAL...")
    
    sucursales = df_compraventa.groupby('Sucursal').agg({
        'Precio_Venta_Total_USD': 'sum',
        'Unidades_Venta_Cantidad': 'sum',
    }).reset_index()
    
    sucursales.columns = ['sucursal_id', 'ingresos_totales', 'unidades_totales']
    
    # Determinar tipo basado en volumen
    sucursales['tipo'] = pd.qcut(
        sucursales['ingresos_totales'],
        q=2,
        labels=['normal', 'alto_volumen']
    )
    
    # Solo sucursales activas
    sucursales = sucursales[sucursales['ingresos_totales'] > 0]
    
    print(f"   ✓ {len(sucursales)} sucursales activas")
    
    return sucursales


def create_promocion_lookup(df_promociones: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara lookup de promociones para join con ventas.
    """
    print("\n🎯 Preparando lookup de promociones...")
    
    # Mapeo de tipos de promoción
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
    
    promo = df_promociones.copy()
    
    # Asegurar tipos de fecha
    promo['Fecha_Inicio'] = pd.to_datetime(promo['Fecha_Inicio'])
    promo['Fecha_Fin'] = pd.to_datetime(promo['Fecha_Fin'])
    
    # Calcular porcentaje de descuento según tipo
    promo['pct_descuento'] = 0.0
    
    # Tipo 2: Porcentaje directo
    mask_pct = promo['Tipo_Promocion'] == 2
    if 'Porcentaje_Descuento1' in promo.columns:
        promo.loc[mask_pct, 'pct_descuento'] = promo.loc[mask_pct, 'Porcentaje_Descuento1']
    
    # Tipo 1: Precio oferta vs precio normal (aproximación)
    # No tenemos precio normal en promociones, se calculará en el join
    
    promo['tipo_promo_desc'] = promo['Tipo_Promocion'].map(tipo_promo_desc)
    
    print(f"   ✓ {len(promo)} registros de promoción")
    
    return promo


def join_promociones(df_ventas: pd.DataFrame, df_promociones: pd.DataFrame, 
                     producto_col: str = 'producto_id', fecha_col: str = 'fecha') -> pd.DataFrame:
    """
    Une ventas con promociones activas por producto y fecha.
    """
    print("\n🔗 Uniendo ventas con promociones...")
    
    df = df_ventas.copy()
    df['tiene_promocion'] = 0
    df['tipo_promocion'] = 0
    df['pct_descuento'] = 0.0
    
    # Para cada venta, verificar si hay promoción activa
    promo = df_promociones[['Cod_Producto', 'Fecha_Inicio', 'Fecha_Fin', 
                            'Tipo_Promocion', 'pct_descuento']].copy()
    
    productos_con_promo = set(promo['Cod_Producto'].unique())
    df_productos = set(df[producto_col].unique())
    productos_match = productos_con_promo.intersection(df_productos)
    
    print(f"   Productos con promociones: {len(productos_match)}")
    
    # Filtrar solo promociones de productos que tenemos
    promo = promo[promo['Cod_Producto'].isin(productos_match)]
    print(f"   Promociones relevantes: {len(promo):,}")
    
    # Optimización: hacer join por rangos de fecha
    for idx, row in promo.iterrows():
        mask = (
            (df[producto_col] == row['Cod_Producto']) &
            (df[fecha_col] >= row['Fecha_Inicio']) &
            (df[fecha_col] <= row['Fecha_Fin'])
        )
        df.loc[mask, 'tiene_promocion'] = 1
        df.loc[mask, 'tipo_promocion'] = row['Tipo_Promocion']
        df.loc[mask, 'pct_descuento'] = row['pct_descuento']
    
    ventas_con_promo = df['tiene_promocion'].sum()
    print(f"   ✓ Ventas con promoción: {ventas_con_promo:,} ({ventas_con_promo/len(df)*100:.1f}%)")
    
    return df


def aggregate_ajustes(df_ajustes: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ajustes de inventario por producto-fecha.
    """
    print("\n📦 Agregando ajustes de inventario...")
    
    # Normalizar columna de fecha
    if 'DOCDATE' in df_ajustes.columns:
        df_ajustes['Fecha'] = pd.to_datetime(df_ajustes['DOCDATE'])
    
    # Normalizar columna de producto
    if 'Codigo_Producto' in df_ajustes.columns:
        prod_col = 'Codigo_Producto'
    elif 'ITEMNMBR' in df_ajustes.columns:
        prod_col = 'ITEMNMBR'
    else:
        prod_col = df_ajustes.columns[0]
    
    # Normalizar cantidad
    if 'TRXQTY' in df_ajustes.columns:
        qty_col = 'TRXQTY'
    elif 'Cantidad' in df_ajustes.columns:
        qty_col = 'Cantidad'
    else:
        qty_col = None
    
    if qty_col:
        ajustes_agg = df_ajustes.groupby([prod_col, 'Fecha']).agg({
            qty_col: ['sum', 'count']
        }).reset_index()
        ajustes_agg.columns = ['producto_id', 'fecha', 'cantidad_ajuste', 'num_ajustes']
    else:
        ajustes_agg = df_ajustes.groupby([prod_col, 'Fecha']).size().reset_index(name='num_ajustes')
        ajustes_agg.columns = ['producto_id', 'fecha', 'num_ajustes']
        ajustes_agg['cantidad_ajuste'] = 0
    
    print(f"   ✓ {len(ajustes_agg):,} registros de ajustes agregados")
    
    return ajustes_agg


def create_fact_ventas(
    df_compraventa: pd.DataFrame,
    df_promociones: pd.DataFrame,
    df_ajustes: pd.DataFrame,
    df_bcv: pd.DataFrame
) -> pd.DataFrame:
    """
    Crea la tabla de hechos unificada fact_ventas.
    """
    print("\n" + "=" * 70)
    print("📊 CREANDO FACT_VENTAS")
    print("=" * 70)
    
    # 1. Base: CompraVenta normalizada
    fact = df_compraventa.copy()
    
    # Renombrar columnas a esquema estándar
    fact = fact.rename(columns={
        'Codigo_Interno': 'producto_id',
        'Sucursal': 'sucursal_id',
        'Fecha': 'fecha',
        'Unidades_Venta_Cantidad': 'unidades',
        'Precio_Unitario_USD': 'precio_unitario_usd',
        'Precio_Venta_Total_USD': 'ingreso_usd',
        'Costo_Venta_Total_USD': 'costo_usd',
        'Margen_Bruto_USD': 'margen_usd',
        'Margen_Porcentaje': 'margen_pct',
        'Clase': 'clase',
    })
    
    # 2. Agregar atributos temporales básicos
    fact['dia_semana'] = fact['fecha'].dt.dayofweek
    fact['es_fin_semana'] = fact['dia_semana'].isin([5, 6]).astype(int)
    fact['mes'] = fact['fecha'].dt.month
    fact['año'] = fact['fecha'].dt.year
    
    # 3. Agregar tasa BCV del día
    fact = fact.merge(
        df_bcv[['Fecha', 'Tasa_USD']].rename(columns={'Fecha': 'fecha', 'Tasa_USD': 'tasa_bcv'}),
        on='fecha',
        how='left'
    )
    
    # 4. Preparar y unir promociones
    promo_lookup = create_promocion_lookup(df_promociones)
    fact = join_promociones(fact, promo_lookup)
    
    # 5. Atributo derivado: es_perecedero
    fact['es_perecedero'] = fact['clase'].isin(['03CARN', '08FRUV']).astype(int)
    
    # 6. Seleccionar y ordenar columnas finales
    cols_fact = [
        'fecha', 'producto_id', 'sucursal_id',
        'unidades', 'precio_unitario_usd', 'ingreso_usd', 'costo_usd',
        'margen_usd', 'margen_pct',
        'tiene_promocion', 'tipo_promocion', 'pct_descuento',
        'clase', 'es_perecedero',
        'dia_semana', 'es_fin_semana', 'mes', 'año',
        'tasa_bcv'
    ]
    
    # Verificar columnas existentes
    cols_disponibles = [c for c in cols_fact if c in fact.columns]
    fact = fact[cols_disponibles]
    
    # Ordenar
    fact = fact.sort_values(['fecha', 'sucursal_id', 'producto_id'])
    
    print(f"\n   ✅ FACT_VENTAS creada:")
    print(f"   Registros: {len(fact):,}")
    print(f"   Columnas: {len(fact.columns)}")
    print(f"   Período: {fact['fecha'].min().date()} - {fact['fecha'].max().date()}")
    print(f"   Productos: {fact['producto_id'].nunique():,}")
    print(f"   Sucursales: {fact['sucursal_id'].nunique()}")
    
    return fact


def run_etl(data_path: str = 'data', output_path: str = None) -> dict:
    """
    Ejecuta el pipeline ETL completo.
    
    Args:
        data_path: Ruta base de datos
        output_path: Ruta de salida (default: data_path/processed)
        
    Returns:
        Dict con todas las tablas creadas
    """
    data_path = Path(data_path)
    output_path = Path(output_path) if output_path else data_path / 'processed'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PIPELINE ETL - SIP DYNAMIC PRICING")
    print("=" * 70)
    
    # 1. Cargar datos raw
    raw_data = load_raw_data(data_path)
    
    # 2. Crear dimensiones
    dim_tiempo = create_dim_tiempo(
        start_date=raw_data['compraventa']['Fecha'].min().strftime('%Y-%m-%d'),
        end_date=raw_data['compraventa']['Fecha'].max().strftime('%Y-%m-%d')
    )
    
    dim_producto = create_dim_producto(raw_data['compraventa'])
    dim_sucursal = create_dim_sucursal(raw_data['compraventa'])
    
    # 3. Crear tabla de hechos
    fact_ventas = create_fact_ventas(
        df_compraventa=raw_data['compraventa'],
        df_promociones=raw_data['promociones'],
        df_ajustes=raw_data['ajustes'],
        df_bcv=raw_data['bcv_rates']
    )
    
    # 4. Guardar todas las tablas
    print("\n" + "=" * 70)
    print("💾 GUARDANDO TABLAS")
    print("=" * 70)
    
    tables = {
        'dim_tiempo': dim_tiempo,
        'dim_producto': dim_producto,
        'dim_sucursal': dim_sucursal,
        'fact_ventas': fact_ventas,
    }
    
    for name, df in tables.items():
        path = output_path / f'{name}.parquet'
        df.to_parquet(path, index=False)
        print(f"   ✓ {name}: {path}")
    
    print("\n" + "=" * 70)
    print("✅ ETL COMPLETADO")
    print("=" * 70)
    
    return tables


if __name__ == "__main__":
    tables = run_etl(data_path='data')
    
    print("\n📊 Resumen de tablas:")
    for name, df in tables.items():
        print(f"   {name}: {len(df):,} registros, {len(df.columns)} columnas")
