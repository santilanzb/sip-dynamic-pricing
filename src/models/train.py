"""
Módulo de entrenamiento de modelos.
Entrena XGBoost (principal), Random Forest (baseline) y LightGBM (alternativo).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import json
from typing import Tuple, Dict, List, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula MAPE evitando división por cero."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'target',
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara los datos para entrenamiento.
    
    Args:
        df: DataFrame con features
        target_col: Columna objetivo
        exclude_cols: Columnas a excluir de features
        
    Returns:
        X (features), y (target), feature_names
    """
    if exclude_cols is None:
        exclude_cols = [
            'fecha', 'producto_id', 'sucursal_id', 'target', 'unidades',
            'ingreso_usd', 'costo_usd', 'margen_usd', 'clase', 'tasa_bcv',
            'rotacion'  # Categórica no numérica
        ]
    
    # Separar features y target
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Convertir categóricas a numéricas si existen
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Rellenar NaN con -999 (XGBoost/LightGBM manejan esto bien)
    X = X.fillna(-999)
    
    return X, y, feature_cols


def temporal_train_test_split(
    df: pd.DataFrame,
    train_end: str = '2024-12-31',
    val_end: str = '2025-06-30',
    date_col: str = 'fecha'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal para time series.
    
    Train: hasta train_end
    Validation: train_end hasta val_end  
    Test: después de val_end
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train = df[df[date_col] <= train_end_dt]
    val = df[(df[date_col] > train_end_dt) & (df[date_col] <= val_end_dt)]
    test = df[df[date_col] > val_end_dt]
    
    return train, val, test


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transform_back: bool = True
) -> Dict[str, float]:
    """
    Evalúa el modelo con múltiples métricas.
    
    Args:
        y_true: Valores reales (en log si transform_back=True)
        y_pred: Predicciones (en log si transform_back=True)
        transform_back: Si True, transforma de log a escala original
        
    Returns:
        Dict con métricas
    """
    if transform_back:
        # Transformar de log1p a escala original
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    # Clip negatives (por si acaso)
    y_pred_orig = np.maximum(y_pred_orig, 0)
    
    metrics = {
        'MAE': mean_absolute_error(y_true_orig, y_pred_orig),
        'RMSE': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
        'R2': r2_score(y_true_orig, y_pred_orig),
        'MAPE': mean_absolute_percentage_error(y_true_orig, y_pred_orig),
        'MAE_log': mean_absolute_error(y_true, y_pred),
        'RMSE_log': np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    
    return metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict = None
) -> Tuple[xgb.XGBRegressor, Dict]:
    """Entrena modelo XGBoost."""
    
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'random_state': 42,
            'n_jobs': -1,
        }
    
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Predicciones en validación
    y_pred_val = model.predict(X_val)
    metrics = evaluate_model(y_val.values, y_pred_val)
    
    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict = None
) -> Tuple[RandomForestRegressor, Dict]:
    """Entrena modelo Random Forest (baseline)."""
    
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predicciones en validación
    y_pred_val = model.predict(X_val)
    metrics = evaluate_model(y_val.values, y_pred_val)
    
    return model, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict = None
) -> Tuple[lgb.LGBMRegressor, Dict]:
    """Entrena modelo LightGBM."""
    
    if params is None:
        params = {
            'objective': 'regression',
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    
    # Predicciones en validación
    y_pred_val = model.predict(X_val)
    metrics = evaluate_model(y_val.values, y_pred_val)
    
    return model, metrics


def get_feature_importance(
    model,
    feature_names: List[str],
    model_type: str = 'xgboost'
) -> pd.DataFrame:
    """Obtiene importancia de features."""
    
    if model_type in ['xgboost', 'lightgbm']:
        importance = model.feature_importances_
    elif model_type == 'random_forest':
        importance = model.feature_importances_
    else:
        return pd.DataFrame()
    
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return fi_df


def train_all_models(
    features_path: str,
    output_dir: str = 'models',
    train_end: str = '2024-12-31',
    val_end: str = '2025-06-30'
) -> Dict:
    """
    Pipeline completo de entrenamiento.
    
    Args:
        features_path: Ruta al archivo features.parquet
        output_dir: Directorio donde guardar modelos
        train_end: Fecha fin de entrenamiento
        val_end: Fecha fin de validación
        
    Returns:
        Dict con resultados
    """
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELOS - SIP DYNAMIC PRICING")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos
    print("\n📂 Cargando datos...")
    df = pd.read_parquet(features_path)
    print(f"   Registros totales: {len(df):,}")
    
    # 2. Split temporal
    print("\n📅 Split temporal...")
    train_df, val_df, test_df = temporal_train_test_split(
        df, train_end=train_end, val_end=val_end
    )
    print(f"   Train: {len(train_df):,} ({train_df['fecha'].min().date()} - {train_df['fecha'].max().date()})")
    print(f"   Val:   {len(val_df):,} ({val_df['fecha'].min().date()} - {val_df['fecha'].max().date()})")
    print(f"   Test:  {len(test_df):,} ({test_df['fecha'].min().date()} - {test_df['fecha'].max().date()})")
    
    # 3. Preparar features
    print("\n🔧 Preparando features...")
    X_train, y_train, feature_names = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)
    
    print(f"   Features: {len(feature_names)}")
    
    results = {}
    
    # 4. Entrenar XGBoost
    print("\n" + "=" * 70)
    print("🚀 ENTRENANDO XGBOOST (Principal)")
    print("=" * 70)
    
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluar en test
    y_pred_test = xgb_model.predict(X_test)
    xgb_test_metrics = evaluate_model(y_test.values, y_pred_test)
    
    print(f"\n   📊 Métricas XGBoost (Validación):")
    print(f"      MAPE: {xgb_metrics['MAPE']:.2f}%")
    print(f"      MAE:  {xgb_metrics['MAE']:.2f} unidades")
    print(f"      R²:   {xgb_metrics['R2']:.4f}")
    
    print(f"\n   📊 Métricas XGBoost (Test):")
    print(f"      MAPE: {xgb_test_metrics['MAPE']:.2f}%")
    print(f"      MAE:  {xgb_test_metrics['MAE']:.2f} unidades")
    print(f"      R²:   {xgb_test_metrics['R2']:.4f}")
    
    results['xgboost'] = {
        'val_metrics': xgb_metrics,
        'test_metrics': xgb_test_metrics,
    }
    
    # Guardar modelo
    joblib.dump(xgb_model, output_dir / 'xgb_demand.pkl')
    print(f"\n   💾 Guardado: {output_dir / 'xgb_demand.pkl'}")
    
    # 5. Entrenar Random Forest (baseline)
    print("\n" + "=" * 70)
    print("🌲 ENTRENANDO RANDOM FOREST (Baseline)")
    print("=" * 70)
    
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    
    y_pred_test_rf = rf_model.predict(X_test)
    rf_test_metrics = evaluate_model(y_test.values, y_pred_test_rf)
    
    print(f"\n   📊 Métricas Random Forest (Test):")
    print(f"      MAPE: {rf_test_metrics['MAPE']:.2f}%")
    print(f"      MAE:  {rf_test_metrics['MAE']:.2f} unidades")
    print(f"      R²:   {rf_test_metrics['R2']:.4f}")
    
    results['random_forest'] = {
        'val_metrics': rf_metrics,
        'test_metrics': rf_test_metrics,
    }
    
    joblib.dump(rf_model, output_dir / 'rf_baseline.pkl')
    print(f"\n   💾 Guardado: {output_dir / 'rf_baseline.pkl'}")
    
    # 6. Entrenar LightGBM
    print("\n" + "=" * 70)
    print("⚡ ENTRENANDO LIGHTGBM (Alternativo)")
    print("=" * 70)
    
    lgbm_model, lgbm_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
    
    y_pred_test_lgbm = lgbm_model.predict(X_test)
    lgbm_test_metrics = evaluate_model(y_test.values, y_pred_test_lgbm)
    
    print(f"\n   📊 Métricas LightGBM (Test):")
    print(f"      MAPE: {lgbm_test_metrics['MAPE']:.2f}%")
    print(f"      MAE:  {lgbm_test_metrics['MAE']:.2f} unidades")
    print(f"      R²:   {lgbm_test_metrics['R2']:.4f}")
    
    results['lightgbm'] = {
        'val_metrics': lgbm_metrics,
        'test_metrics': lgbm_test_metrics,
    }
    
    joblib.dump(lgbm_model, output_dir / 'lgbm_alt.pkl')
    print(f"\n   💾 Guardado: {output_dir / 'lgbm_alt.pkl'}")
    
    # 7. Feature importance (XGBoost)
    print("\n" + "=" * 70)
    print("📊 FEATURE IMPORTANCE (Top 15)")
    print("=" * 70)
    
    fi_df = get_feature_importance(xgb_model, feature_names, 'xgboost')
    print("\n   XGBoost:")
    for i, row in fi_df.head(15).iterrows():
        print(f"   {i+1:2}. {row['feature']:<30} {row['importance']:.4f}")
    
    fi_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # 8. Guardar metadatos
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'train_end': train_end,
        'val_end': val_end,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'results': {
            k: {
                'val_metrics': {mk: float(mv) for mk, mv in v['val_metrics'].items()},
                'test_metrics': {mk: float(mv) for mk, mv in v['test_metrics'].items()},
            }
            for k, v in results.items()
        }
    }
    
    with open(output_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 9. Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN COMPARATIVO (Test Set)")
    print("=" * 70)
    
    print(f"\n   {'Modelo':<20} {'MAPE':>10} {'MAE':>12} {'R²':>10}")
    print("   " + "-" * 54)
    
    for name, res in results.items():
        m = res['test_metrics']
        print(f"   {name:<20} {m['MAPE']:>9.2f}% {m['MAE']:>11.2f} {m['R2']:>10.4f}")
    
    # Determinar mejor modelo
    best_model = min(results.items(), key=lambda x: x[1]['test_metrics']['MAPE'])
    print(f"\n   🏆 Mejor modelo: {best_model[0].upper()} (MAPE: {best_model[1]['test_metrics']['MAPE']:.2f}%)")
    
    # Verificar objetivos
    print("\n" + "=" * 70)
    print("✅ VERIFICACIÓN DE OBJETIVOS")
    print("=" * 70)
    
    best_mape = best_model[1]['test_metrics']['MAPE']
    best_r2 = best_model[1]['test_metrics']['R2']
    
    mape_ok = best_mape < 15
    r2_ok = best_r2 > 0.7
    
    print(f"\n   MAPE < 15%: {'✅ SÍ' if mape_ok else '❌ NO'} ({best_mape:.2f}%)")
    print(f"   R² > 0.7:   {'✅ SÍ' if r2_ok else '❌ NO'} ({best_r2:.4f})")
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = train_all_models(
        features_path='data/processed/features.parquet',
        output_dir='models',
        train_end='2024-12-31',
        val_end='2025-06-30'
    )
