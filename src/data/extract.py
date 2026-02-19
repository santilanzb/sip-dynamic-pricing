"""
SIP Dynamic Pricing - Data Extraction Module
=============================================
Extrae datos de SQL Server (CompraVenta, Promociones, Ajustes) y los guarda en Parquet.

Uso:
    python -m src.data.extract --all          # Extrae todo
    python -m src.data.extract --compraventa  # Solo CompraVenta
    python -m src.data.extract --promociones  # Solo Promociones
    python -m src.data.extract --ajustes      # Solo Ajustes
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse
from datetime import datetime

import pandas as pd
import pyodbc
import yaml
from tqdm import tqdm

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


class DataExtractor:
    """Extractor de datos desde SQL Server."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el extractor con la configuración.
        
        Args:
            config_path: Ruta al archivo config.yaml. Si es None, usa el default.
        """
        if config_path is None:
            config_path = ROOT_DIR / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = ROOT_DIR / self.config['paths']['raw_data']
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Clases de interés
        self.clases = self.config['extraction']['clases']
        self.clases_str = "'" + "','".join(self.clases) + "'"
    
    def _get_connection_string(self, db_key: str) -> str:
        """Construye el connection string para una base de datos."""
        db_config = self.config['databases'][db_key]
        
        conn_str = (
            f"DRIVER={{{db_config['driver']}}};"
            f"SERVER={db_config['server']};"
            f"DATABASE={db_config['database']};"
            f"TrustServerCertificate=yes;"
        )
        
        if db_config.get('trusted_connection', True):
            conn_str += "Trusted_Connection=yes;"
        else:
            conn_str += f"UID={db_config['username']};PWD={db_config['password']};"
        
        return conn_str
    
    def _execute_query(self, db_key: str, query: str) -> pd.DataFrame:
        """Ejecuta una query y retorna un DataFrame."""
        conn_str = self._get_connection_string(db_key)
        
        print(f"  Conectando a {db_key}...")
        with pyodbc.connect(conn_str) as conn:
            print(f"  Ejecutando query...")
            df = pd.read_sql(query, conn)
        
        print(f"  Registros obtenidos: {len(df):,}")
        return df
    
    # =========================================================================
    # COMPRAVENTA
    # =========================================================================
    
    def extract_compraventa_2023(self) -> pd.DataFrame:
        """Extrae CompraVentasDiario2023 desde EMP03."""
        print("\n[1/5] Extrayendo CompraVentasDiario2023 (EMP03)...")
        
        query = f"""
        SELECT 
            UltVenta, UltCompra, Sucursal, Año, Mes, Dia,
            [Codigo Interno] as Codigo_Interno, SKU, Descripcion,
            TipoArticulo, Clase, Departamento, SubDepartamento, Categoria,
            Pesado, Regulado, Sunde,
            [Unidad Compra] as Unidad_Compra, [Unidad Ventas] as Unidad_Ventas,
            Iva, [Costo Actual] as Costo_Actual,
            [Unidades Compra Cantidad] as Unidades_Compra_Cantidad,
            [Costo Total Compra] as Costo_Total_Compra,
            [Unidades Venta Cantidad] as Unidades_Venta_Cantidad,
            [Costo Venta Total] as Costo_Venta_Total,
            [Precio Venta Total] as Precio_Venta_Total
        FROM CompraVentasDiario2023
        WHERE Clase IN ({self.clases_str})
        """
        
        df = self._execute_query('emp03', query)
        df['source'] = 'EMP03_2023'
        return df
    
    def extract_compraventa_2024_emp03(self) -> pd.DataFrame:
        """Extrae CompraVentasDiario2024 desde EMP03 (Enero-Octubre)."""
        print("\n[2/5] Extrayendo CompraVentasDiario2024 (EMP03 - Ene-Oct)...")
        
        query = f"""
        SELECT 
            UltVenta, UltCompra, Sucursal, Año, Mes, Dia,
            [Codigo Interno] as Codigo_Interno, SKU, Descripcion,
            TipoArticulo, Clase, Departamento, SubDepartamento, Categoria,
            Pesado, Regulado, Sunde,
            [Unidad Compra] as Unidad_Compra, [Unidad Ventas] as Unidad_Ventas,
            Iva, [Costo Actual] as Costo_Actual,
            [Unidades Compra Cantidad] as Unidades_Compra_Cantidad,
            [Costo Total Compra] as Costo_Total_Compra,
            [Unidades Venta Cantidad] as Unidades_Venta_Cantidad,
            [Costo Venta Total] as Costo_Venta_Total,
            [Precio Venta Total] as Precio_Venta_Total
        FROM CompraVentasDiario2024
        WHERE Clase IN ({self.clases_str})
          AND Mes <= 10  -- Enero a Octubre en EMP03
        """
        
        df = self._execute_query('emp03', query)
        df['source'] = 'EMP03_2024'
        return df
    
    def extract_compraventa_2024_emp04(self) -> pd.DataFrame:
        """Extrae CompraVentasDiario2024 desde EMP04 (Noviembre-Diciembre)."""
        print("\n[3/5] Extrayendo CompraVentasDiario2024 (EMP04 - Nov-Dic)...")
        
        query = f"""
        SELECT 
            UltVenta, UltCompra, Sucursal, Año, Mes, Dia,
            [Codigo Interno] as Codigo_Interno, SKU, Descripcion,
            TipoArticulo, Clase, Departamento, SubDepartamento, Categoria,
            Pesado, Regulado, Sunde,
            [Unidad Compra] as Unidad_Compra, [Unidad Ventas] as Unidad_Ventas,
            Iva, [Costo Actual] as Costo_Actual,
            [Unidades Compra Cantidad] as Unidades_Compra_Cantidad,
            [Costo Total Compra] as Costo_Total_Compra,
            [Unidades Venta Cantidad] as Unidades_Venta_Cantidad,
            [Costo Venta Total] as Costo_Venta_Total,
            [Precio Venta Total] as Precio_Venta_Total
        FROM CompraVentasDiario2024
        WHERE Clase IN ({self.clases_str})
          AND Mes >= 11  -- Noviembre-Diciembre en EMP04
        """
        
        df = self._execute_query('emp04', query)
        df['source'] = 'EMP04_2024'
        return df
    
    def extract_compraventa_2025(self) -> pd.DataFrame:
        """Extrae CompraVentasDiario2025 desde EMP04."""
        print("\n[4/5] Extrayendo CompraVentasDiario2025 (EMP04)...")
        
        query = f"""
        SELECT 
            UltVenta, UltCompra, Sucursal, Año, Mes, Dia,
            [Codigo Interno] as Codigo_Interno, SKU, Descripcion,
            TipoArticulo, Clase, Departamento, SubDepartamento, Categoria,
            Pesado, Regulado, Sunde,
            [Unidad Compra] as Unidad_Compra, [Unidad Ventas] as Unidad_Ventas,
            Iva, [Costo Actual] as Costo_Actual,
            [Unidades Compra Cantidad] as Unidades_Compra_Cantidad,
            [Costo Total Compra] as Costo_Total_Compra,
            [Unidades Venta Cantidad] as Unidades_Venta_Cantidad,
            [Costo Venta Total] as Costo_Venta_Total,
            [Precio Venta Total] as Precio_Venta_Total
        FROM CompraVentasDiario2025
        WHERE Clase IN ({self.clases_str})
        """
        
        df = self._execute_query('emp04', query)
        df['source'] = 'EMP04_2025'
        return df
    
    def extract_all_compraventa(self) -> pd.DataFrame:
        """Extrae y combina todos los datos de CompraVenta."""
        print("\n" + "="*60)
        print("EXTRACCIÓN DE COMPRAVENTA")
        print("="*60)
        
        dfs = []
        
        # 2023 desde EMP03
        dfs.append(self.extract_compraventa_2023())
        
        # 2024 desde ambos servidores
        dfs.append(self.extract_compraventa_2024_emp03())
        dfs.append(self.extract_compraventa_2024_emp04())
        
        # 2025 desde EMP04
        dfs.append(self.extract_compraventa_2025())
        
        # Combinar
        print("\n[5/5] Combinando datasets...")
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Crear fecha
        df_combined['Fecha'] = pd.to_datetime(
            df_combined['Año'].astype(str) + '-' + 
            df_combined['Mes'].astype(str).str.zfill(2) + '-' + 
            df_combined['Dia'].astype(str).str.zfill(2)
        )
        
        # Guardar
        output_path = self.raw_data_path / "compraventa_raw.parquet"
        df_combined.to_parquet(output_path, index=False)
        print(f"\n✓ CompraVenta guardado en: {output_path}")
        print(f"  Total registros: {len(df_combined):,}")
        print(f"  Rango de fechas: {df_combined['Fecha'].min()} a {df_combined['Fecha'].max()}")
        print(f"  Sucursales: {df_combined['Sucursal'].nunique()}")
        print(f"  Productos únicos: {df_combined['Codigo_Interno'].nunique():,}")
        
        return df_combined
    
    # =========================================================================
    # PROMOCIONES
    # =========================================================================
    
    def extract_promociones(self) -> pd.DataFrame:
        """Extrae datos de promociones desde VAD10."""
        print("\n" + "="*60)
        print("EXTRACCIÓN DE PROMOCIONES")
        print("="*60)
        
        print("\n[1/2] Extrayendo promociones con condiciones y valores...")
        
        query = """
        SELECT 
            -- MA_PROMOCION (Master)
            Promo.Cod_Promocion,
            Promo.Descripcion as Promo_Descripcion,
            Promo.Campaña,
            Promo.Fecha_Inicio,
            Promo.Fecha_Fin,
            Promo.Aplica_Dia_Semana,
            Promo.Aplica_Hora_Inicio,
            Promo.Aplica_Hora_Fin,
            Promo.Estatus as Estado,
            Promo.Tipo_Promocion,
            Promo.Prioridad_Promocion,
            
            -- TR_PROMOCION_CONDICION
            CON.Linea_Condicion,
            CON.Cod_Proveedor,
            CON.Cod_Dpto as Cod_Departamento,
            CON.Cod_Grupo,
            CON.Cod_Subgrupo,
            CON.Marca,
            CON.Cod_Producto,
            CON.Tipo_Condicion,
            
            -- TR_PROMOCION_VALORES
            VAL.Linea_Valor,
            VAL.Cantidad_Productos_Requerir as Cantidad_Requerida,
            VAL.Cantidad_Productos_Pagar as Cantidad_Paga,
            VAL.Precio_Oferta,
            VAL.Monto_Descuento,
            VAL.Porcentaje_Descuento1
            
        FROM MA_PROMOCION Promo
        INNER JOIN TR_PROMOCION_CONDICION CON
            ON Promo.Cod_Promocion = CON.Cod_Promocion
        INNER JOIN TR_PROMOCION_VALORES VAL
            ON CON.Linea_Valor = VAL.Linea_Valor
            AND CON.Cod_Promocion = VAL.Cod_Promocion
        """
        
        df = self._execute_query('vad10', query)
        
        # Guardar
        output_path = self.raw_data_path / "promociones_raw.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\n✓ Promociones guardadas en: {output_path}")
        print(f"  Total registros: {len(df):,}")
        print(f"  Promociones únicas: {df['Cod_Promocion'].nunique():,}")
        print(f"  Productos con promoción: {df['Cod_Producto'].nunique():,}")
        
        # Estadísticas por tipo
        print("\n  Distribución por Tipo_Promocion:")
        tipo_counts = df['Tipo_Promocion'].value_counts()
        for tipo, count in tipo_counts.items():
            print(f"    Tipo {tipo}: {count:,} registros")
        
        return df
    
    # =========================================================================
    # AJUSTES DE INVENTARIO
    # =========================================================================
    
    def extract_ajustes(self) -> pd.DataFrame:
        """Extrae ajustes de inventario desde IV10001 e IV30300."""
        print("\n" + "="*60)
        print("EXTRACCIÓN DE AJUSTES DE INVENTARIO")
        print("="*60)
        
        dfs = []
        
        # IV10001 - Ajustes actuales
        print("\n[1/2] Extrayendo IV10001 (ajustes actuales)...")
        query_actual = """
        SELECT 
            ITEMNMBR as Codigo_Producto,
            IVDOCTYP as Tipo_Documento,
            IVDOCNBR as Numero_Documento,
            TRXQTY as Cantidad,
            UNITCOST as Costo_Unitario,
            TRXLOCTN as Ubicacion,
            Reason_Code as Codigo_Razon,
            'IV10001' as Tabla_Origen
        FROM IV10001
        """
        
        try:
            df_actual = self._execute_query('emp04', query_actual)
            dfs.append(df_actual)
        except Exception as e:
            print(f"  ⚠ Error extrayendo IV10001: {e}")
        
        # IV30300 - Ajustes históricos
        print("\n[2/2] Extrayendo IV30300 (ajustes históricos)...")
        query_historico = """
        SELECT 
            ITEMNMBR as Codigo_Producto,
            DOCDATE as Fecha_Ajuste,
            DOCTYPE as Tipo_Documento,
            DOCNUMBR as Numero_Documento,
            TRXQTY as Cantidad,
            UNITCOST as Costo_Unitario,
            TRXLOCTN as Ubicacion,
            Reason_Code as Codigo_Razon,
            'IV30300' as Tabla_Origen
        FROM IV30300
        """
        
        try:
            df_historico = self._execute_query('emp04', query_historico)
            dfs.append(df_historico)
        except Exception as e:
            print(f"  ⚠ Error extrayendo IV30300: {e}")
        
        if not dfs:
            print("  ⚠ No se pudieron extraer ajustes")
            return pd.DataFrame()
        
        # Combinar
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Guardar
        output_path = self.raw_data_path / "ajustes_raw.parquet"
        df_combined.to_parquet(output_path, index=False)
        print(f"\n✓ Ajustes guardados en: {output_path}")
        print(f"  Total registros: {len(df_combined):,}")
        print(f"  Productos con ajustes: {df_combined['Codigo_Producto'].nunique():,}")
        
        return df_combined
    
    # =========================================================================
    # EXTRACCIÓN COMPLETA
    # =========================================================================
    
    def extract_all(self):
        """Ejecuta la extracción completa de todos los datos."""
        print("\n" + "="*60)
        print("SIP DYNAMIC PRICING - EXTRACCIÓN DE DATOS")
        print("="*60)
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Clases: {self.clases}")
        
        # CompraVenta
        df_compraventa = self.extract_all_compraventa()
        
        # Promociones
        df_promociones = self.extract_promociones()
        
        # Ajustes
        df_ajustes = self.extract_ajustes()
        
        print("\n" + "="*60)
        print("EXTRACCIÓN COMPLETADA")
        print("="*60)
        print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nArchivos generados en: {self.raw_data_path}")
        print("  - compraventa_raw.parquet")
        print("  - promociones_raw.parquet")
        print("  - ajustes_raw.parquet")
        
        return {
            'compraventa': df_compraventa,
            'promociones': df_promociones,
            'ajustes': df_ajustes
        }


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description='Extracción de datos SIP')
    parser.add_argument('--all', action='store_true', help='Extraer todos los datos')
    parser.add_argument('--compraventa', action='store_true', help='Solo CompraVenta')
    parser.add_argument('--promociones', action='store_true', help='Solo Promociones')
    parser.add_argument('--ajustes', action='store_true', help='Solo Ajustes')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración')
    
    args = parser.parse_args()
    
    extractor = DataExtractor(config_path=args.config)
    
    if args.all or not any([args.compraventa, args.promociones, args.ajustes]):
        extractor.extract_all()
    else:
        if args.compraventa:
            extractor.extract_all_compraventa()
        if args.promociones:
            extractor.extract_promociones()
        if args.ajustes:
            extractor.extract_ajustes()


if __name__ == "__main__":
    main()
