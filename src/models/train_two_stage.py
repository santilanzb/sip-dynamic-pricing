"""
Pipeline de entrenamiento para el modelo bietápico de demanda (hurdle model).

Entrena y evalúa modelos bietápicos (LightGBM + XGBoost) con:
- Umbral de demanda τ para separar baja demanda de significativa
- Optuna para tuning de hiperparámetros (clf + reg, objetivo WMAPE en validación)
- Threshold calibration (F1-optimal)
- Métricas exhaustivas (WMAPE, SMAPE, MAE, R², MASE, etc.)
- Intervalos conformales (80%/90%)
- Métricas por segmento (clase, sucursal, cuartiles de demanda)
- SHAP importance
- MLflow tracking completo
- Comparación con baselines single-stage

Uso:
    python -m src.models.train_two_stage

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-20
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lightgbm as lgb
import mlflow
import optuna
import shap
import xgboost as xgb
from optuna.samplers import TPESampler

from src.models.two_stage import (
    TwoStageDemandModel,
    _lgb_monotone_constraints,
    _xgb_monotone_constraints_str,
)
from src.models.conformal import split_conformal_interval, coverage_width
from src.utils.metrics import (
    regression_report,
    group_metrics,
    binned_metrics,
    mase_from_train_series,
    wmape,
    wmape_revenue,
)

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Umbral de demanda para el hurdle model (τ)
DEMAND_THRESHOLD = 1.0


# =============================================================================
# DATA PREPARATION (reutiliza la misma lógica que train_gpu.py)
# =============================================================================

EXCLUDE_COLS = [
    "fecha", "producto_id", "sucursal_id", "target", "unidades",
    "ingreso_usd", "costo_usd", "margen_usd", "clase", "tasa_bcv", "rotacion",
]


def prepare_data(df: pd.DataFrame, target_col: str = "target"):
    """Prepara features y target."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.Categorical(X[col]).codes

    X = X.fillna(-999)
    return X, y, feature_cols


def temporal_split(df: pd.DataFrame, train_end="2024-12-31", val_end="2025-06-30"):
    """Split temporal."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    train = df[df["fecha"] <= train_end]
    val = df[(df["fecha"] > train_end) & (df["fecha"] <= val_end)]
    test = df[df["fecha"] > val_end]
    return train, val, test


# =============================================================================
# OPTUNA TUNING PARA MODELO BIETÁPICO
# =============================================================================

def optimize_two_stage_lgbm(
    X_train, y_train, X_val, y_val,
    demand_threshold: float = DEMAND_THRESHOLD,
    n_trials: int = 30,
):
    """Optimiza hiperparámetros del modelo bietápico LightGBM con Optuna."""

    mono = _lgb_monotone_constraints(X_train.columns.tolist())
    y_val_orig = np.expm1(y_val.values)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        num_leaves = trial.suggest_int("num_leaves", 31, 128)
        min_child = trial.suggest_int("min_child_samples", 5, 80)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

        clf_params = {
            "objective": "binary", "device": "gpu",
            "learning_rate": lr, "num_leaves": num_leaves,
            "min_child_samples": min_child, "subsample": subsample,
            "colsample_bytree": colsample, "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda, "n_estimators": 1000,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
            "monotone_constraints": mono,
        }
        reg_params = {
            "objective": "regression", "device": "gpu",
            "learning_rate": lr, "num_leaves": num_leaves,
            "min_child_samples": min_child, "subsample": subsample,
            "colsample_bytree": colsample, "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda, "n_estimators": 1000,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
            "monotone_constraints": mono,
        }

        model = TwoStageDemandModel(
            backend="lightgbm",
            clf_params=clf_params,
            reg_params=reg_params,
            demand_threshold=demand_threshold,
            calibrate=True,
        )
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_val)

        return float(wmape(y_val_orig, y_pred))

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_two_stage_xgb(
    X_train, y_train, X_val, y_val,
    demand_threshold: float = DEMAND_THRESHOLD,
    n_trials: int = 30,
):
    """Optimiza hiperparámetros del modelo bietápico XGBoost con Optuna."""

    mono_str = _xgb_monotone_constraints_str(X_train.columns.tolist())
    y_val_orig = np.expm1(y_val.values)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

        clf_params = {
            "objective": "binary:logistic", "tree_method": "hist", "device": "cuda",
            "learning_rate": lr, "max_depth": max_depth,
            "subsample": subsample, "colsample_bytree": colsample,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
            "n_estimators": 1000, "random_state": 42, "n_jobs": -1, "verbosity": 0,
            "monotone_constraints": mono_str,
        }
        reg_params = {
            "objective": "reg:squarederror", "tree_method": "hist", "device": "cuda",
            "learning_rate": lr, "max_depth": max_depth,
            "subsample": subsample, "colsample_bytree": colsample,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
            "n_estimators": 1000, "random_state": 42, "n_jobs": -1, "verbosity": 0,
            "monotone_constraints": mono_str,
        }

        model = TwoStageDemandModel(
            backend="xgboost",
            clf_params=clf_params,
            reg_params=reg_params,
            demand_threshold=demand_threshold,
            calibrate=True,
        )
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_val)

        return float(wmape(y_val_orig, y_pred))

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


# =============================================================================
# EVALUACIÓN Y VISUALIZACIÓN
# =============================================================================

def full_evaluation(
    model: TwoStageDemandModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Dict:
    """Evaluación exhaustiva de un modelo bietápico."""

    y_test_orig = np.expm1(y_test.values)
    y_pred = model.predict(X_test)

    # Métricas de regresión + clasificación
    metrics = model.evaluate(X_test, y_test, df_context=test_df)

    # MASE
    mase_val = mase_from_train_series(
        train_df, test_df, y_test_orig, y_pred,
        ["producto_id", "sucursal_id"], "unidades",
    )
    metrics["MASE"] = mase_val

    # Métricas por segmento
    segments = {}
    if "clase" in test_df.columns:
        seg = group_metrics(test_df, y_test_orig, y_pred, "clase")
        seg.to_csv(output_dir / f"{model_name}_metrics_by_clase.csv", index=False)
        segments["by_clase"] = seg.to_dict(orient="records")

    if "sucursal_id" in test_df.columns:
        seg = group_metrics(test_df, y_test_orig, y_pred, "sucursal_id")
        seg.to_csv(output_dir / f"{model_name}_metrics_by_sucursal.csv", index=False)
        segments["by_sucursal"] = seg.to_dict(orient="records")

    # Cuartiles de demanda
    quartiles = pd.qcut(y_test_orig, q=4, labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"], duplicates="drop")
    df_q = pd.DataFrame({"q": quartiles, "y_true": y_test_orig, "y_pred": y_pred})
    df_q["abs_err"] = np.abs(df_q["y_true"] - df_q["y_pred"])
    agg = (
        df_q.groupby("q")
        .agg(WMAPE_num=("abs_err", "sum"), y_true_sum=("y_true", "sum"), MAE=("abs_err", "mean"), n=("y_true", "size"))
        .assign(WMAPE=lambda d: (d["WMAPE_num"] / (d["y_true_sum"].abs() + 1e-12)) * 100)
        .drop(columns=["WMAPE_num", "y_true_sum"])
        .reset_index()
    )
    agg.to_csv(output_dir / f"{model_name}_metrics_by_demand_quartile.csv", index=False)
    segments["by_demand_quartile"] = agg.to_dict(orient="records")

    return {"metrics": metrics, "segments": segments, "y_pred": y_pred}


def plot_two_stage_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    demand_threshold: float = DEMAND_THRESHOLD,
):
    """Genera visualizaciones para el modelo bietápico."""

    # 1. Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sample_idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=10)
    lim = max(np.percentile(y_true, 99), np.percentile(y_pred, 99))
    ax.plot([0, lim], [0, lim], "r--", label="Predicción perfecta")
    ax.set_xlabel("Demanda Real (unidades)")
    ax.set_ylabel("Demanda Predicha (unidades)")
    ax.set_title(f"{model_name}: Real vs Predicho")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_scatter.png", dpi=150)
    plt.close()

    # 2. Distribución de errores
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors.clip(-50, 50), bins=100, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=np.median(errors), color="green", linestyle="--", linewidth=2, label=f"Mediana: {np.median(errors):.2f}")
    ax.set_xlabel("Error (Predicho - Real)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{model_name}: Distribución de Errores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_error_dist.png", dpi=150)
    plt.close()

    # 3. Análisis por régimen de demanda
    tau = demand_threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribución de predicciones por régimen real
    low_mask = y_true < tau
    high_mask = y_true >= tau
    if low_mask.sum() > 0 and high_mask.sum() > 0:
        axes[0].hist(y_pred[low_mask], bins=50, alpha=0.6, label=f"Demanda<{tau} (n={low_mask.sum():,})", density=True)
        axes[0].hist(y_pred[high_mask].clip(0, 30), bins=50, alpha=0.6, label=f"Demanda≥{tau} (n={high_mask.sum():,})", density=True)
        axes[0].set_xlabel("Predicción")
        axes[0].set_ylabel("Densidad")
        axes[0].set_title("Distribución de predicciones por régimen real")
        axes[0].legend()

    # Error por rango de demanda
    bins = pd.cut(y_true, bins=[0, 1, 5, 10, 20, 50, 100, np.inf], labels=["0-1", "1-5", "5-10", "10-20", "20-50", "50-100", "100+"])
    error_by_bin = pd.DataFrame({"bin": bins, "abs_error": np.abs(errors)}).groupby("bin")["abs_error"].mean()
    error_by_bin.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Rango de Demanda")
    axes[1].set_ylabel("MAE Promedio")
    axes[1].set_title(f"{model_name}: MAE por Rango")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_analysis.png", dpi=150)
    plt.close()


def plot_threshold_calibration(calibration_info: Dict, output_dir: Path, model_name: str):
    """Grafica la curva de calibración de threshold."""
    if calibration_info is None or "curve" not in calibration_info:
        return

    curve = pd.DataFrame(calibration_info["curve"])
    best_t = calibration_info["threshold"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve["threshold"], curve["f1"], label="F1", linewidth=2)
    ax.plot(curve["threshold"], curve["precision"], label="Precision", linewidth=1, linestyle="--")
    ax.plot(curve["threshold"], curve["recall"], label="Recall", linewidth=1, linestyle="--")
    ax.axvline(x=best_t, color="red", linestyle=":", linewidth=2, label=f"Óptimo: {best_t:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Métrica")
    ax.set_title(f"{model_name}: Calibración de Threshold (Etapa 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_threshold_calibration.png", dpi=150)
    plt.close()


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def train_two_stage_models(
    features_path: str = "data/processed/features.parquet",
    output_dir: str = "models",
    demand_threshold: float = DEMAND_THRESHOLD,
    optuna_trials_lgbm: int = 30,
    optuna_trials_xgb: int = 30,
):
    """
    Pipeline completo de entrenamiento bietápico (hurdle model).

    Entrena LightGBM y XGBoost two-stage, compara con baselines single-stage,
    y genera artefactos completos para la tesis.
    """
    print("=" * 80)
    print("ENTRENAMIENTO MODELO BIETÁPICO (HURDLE) - SIP DYNAMIC PRICING")
    print("=" * 80)
    print(f"   Umbral de demanda (τ): {demand_threshold}")
    print(f"   Optuna: LGBM {optuna_trials_lgbm} trials | XGB {optuna_trials_xgb} trials")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = output_dir / "two_stage"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri("file:" + str((Path.cwd() / "mlruns").resolve()))
    mlflow.set_experiment("SIP-TwoStage-Training")

    with mlflow.start_run(run_name="two_stage_hurdle_pipeline") as run:
        # 1. Cargar datos
        print("\n Cargando datos...")
        df = pd.read_parquet(features_path)
        print(f"   Registros: {len(df):,}")

        # 2. Split temporal
        print("\n Split temporal...")
        train_df, val_df, test_df = temporal_split(df)
        print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

        # 3. Preparar features
        X_train, y_train, feature_names = prepare_data(train_df)
        X_val, y_val, _ = prepare_data(val_df)
        X_test, y_test, _ = prepare_data(test_df)
        print(f"   Features: {len(feature_names)}")

        # 4. Distribución de clases
        y_train_orig = np.expm1(y_train.values)
        n_high = (y_train_orig >= demand_threshold).sum()
        n_low = len(y_train_orig) - n_high
        print(f"\n   Distribución del clasificador (τ={demand_threshold}):")
        print(f"      Demanda ≥ {demand_threshold}: {n_high:,} ({n_high/len(y_train_orig)*100:.1f}%)")
        print(f"      Demanda < {demand_threshold}: {n_low:,} ({n_low/len(y_train_orig)*100:.1f}%)")

        mlflow.log_params({
            "n_features": len(feature_names),
            "demand_threshold": demand_threshold,
            "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
            "train_pct_high_demand": round(n_high / len(y_train_orig) * 100, 1),
            "optuna_trials_lgbm": optuna_trials_lgbm,
            "optuna_trials_xgb": optuna_trials_xgb,
        })

        y_test_orig = np.expm1(y_test.values)
        y_val_orig = np.expm1(y_val.values)
        results = {}

        # =====================================================================
        # LIGHTGBM TWO-STAGE
        # =====================================================================
        print("\n" + "=" * 80)
        print("LIGHTGBM BIETÁPICO")
        print("=" * 80)

        print("\n   Optimizando hiperparámetros con Optuna...")
        t0 = time.time()
        lgbm_best_params, lgbm_best_wmape = optimize_two_stage_lgbm(
            X_train, y_train, X_val, y_val,
            demand_threshold=demand_threshold,
            n_trials=optuna_trials_lgbm,
        )
        print(f"   Mejores parámetros encontrados (WMAPE val: {lgbm_best_wmape:.2f}%)")
        mlflow.log_dict(lgbm_best_params, "tuning/lgbm_two_stage_best_params.json")

        # Reconstruir parámetros completos con los mejores valores de Optuna
        mono = _lgb_monotone_constraints(feature_names)
        lgbm_clf_params = {
            "objective": "binary", "device": "gpu",
            "learning_rate": lgbm_best_params["learning_rate"],
            "num_leaves": lgbm_best_params["num_leaves"],
            "min_child_samples": lgbm_best_params["min_child_samples"],
            "subsample": lgbm_best_params["subsample"],
            "colsample_bytree": lgbm_best_params["colsample_bytree"],
            "reg_alpha": lgbm_best_params["reg_alpha"],
            "reg_lambda": lgbm_best_params["reg_lambda"],
            "n_estimators": 1000, "random_state": 42, "n_jobs": -1, "verbose": -1,
            "monotone_constraints": mono,
        }
        lgbm_reg_params = {
            "objective": "regression", "device": "gpu",
            "learning_rate": lgbm_best_params["learning_rate"],
            "num_leaves": lgbm_best_params["num_leaves"],
            "min_child_samples": lgbm_best_params["min_child_samples"],
            "subsample": lgbm_best_params["subsample"],
            "colsample_bytree": lgbm_best_params["colsample_bytree"],
            "reg_alpha": lgbm_best_params["reg_alpha"],
            "reg_lambda": lgbm_best_params["reg_lambda"],
            "n_estimators": 1000, "random_state": 42, "n_jobs": -1, "verbose": -1,
            "monotone_constraints": mono,
        }

        print("\n   Entrenando modelo final LightGBM bietápico...")
        lgbm_ts = TwoStageDemandModel(
            backend="lightgbm",
            clf_params=lgbm_clf_params,
            reg_params=lgbm_reg_params,
            demand_threshold=demand_threshold,
            calibrate=True,
        )
        lgbm_ts.fit(X_train, y_train, X_val, y_val)
        lgbm_train_time = time.time() - t0

        # Evaluación completa
        lgbm_eval = full_evaluation(lgbm_ts, X_test, y_test, test_df, train_df, ts_dir, "lgbm_two_stage")

        # Log métricas a MLflow
        for k, v in lgbm_eval["metrics"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"lgbm_ts_{k}", float(v) if v == v else 0.0)
        mlflow.log_metric("lgbm_ts_train_time_sec", lgbm_train_time)

        # Intervalos conformales
        y_pred_val_lgbm = lgbm_ts.predict(X_val)
        y_pred_test_lgbm = lgbm_eval["y_pred"]
        conf = split_conformal_interval(y_val_orig, y_pred_val_lgbm, y_pred_test_lgbm, alphas=(0.1, 0.2))
        for a_key, d in conf.items():
            cov, width = coverage_width(y_test_orig, d["lo"], d["hi"])
            mlflow.log_metric(f"lgbm_ts_{a_key}_coverage_pct", cov)
            mlflow.log_metric(f"lgbm_ts_{a_key}_width", width)
            lgbm_eval["metrics"][f"{a_key}_coverage"] = cov
            lgbm_eval["metrics"][f"{a_key}_width"] = width

        # Visualizaciones
        plot_two_stage_results(y_test_orig, y_pred_test_lgbm, ts_dir, "lgbm_two_stage", demand_threshold)
        plot_threshold_calibration(lgbm_ts.calibration_info_, ts_dir, "lgbm_two_stage")

        # SHAP para el regresor (Etapa 2)
        try:
            shap_sample_idx = np.random.choice(len(X_test), size=min(30000, len(X_test)), replace=False)
            expl = shap.TreeExplainer(lgbm_ts.reg_)
            shap_vals = expl.shap_values(X_test.iloc[shap_sample_idx])
            shap_imp = pd.DataFrame({
                "feature": X_test.columns,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            shap_imp.to_csv(ts_dir / "lgbm_two_stage_shap_importance.csv", index=False)

            plt.figure(figsize=(10, 10))
            shap.summary_plot(shap_vals, X_test.iloc[shap_sample_idx], plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(ts_dir / "lgbm_two_stage_shap_bar.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"   SHAP error (LightGBM): {e}")

        # Guardar modelo
        lgbm_ts.save(str(ts_dir / "lgbm"))
        results["lgbm_two_stage"] = {
            "metrics": lgbm_eval["metrics"],
            "segments": lgbm_eval["segments"],
            "params": lgbm_best_params,
            "train_time_sec": lgbm_train_time,
        }

        print(f"\n   LightGBM bietápico completado:")
        print(f"      WMAPE: {lgbm_eval['metrics']['WMAPE']:.2f}%")
        print(f"      R²:    {lgbm_eval['metrics']['R2']:.4f}")
        print(f"      MAE:   {lgbm_eval['metrics']['MAE']:.2f}")
        print(f"      clf_F1: {lgbm_eval['metrics']['clf_f1']:.3f}")
        print(f"      low_demand_F1: {lgbm_eval['metrics']['low_demand_f1']:.3f}")

        # =====================================================================
        # XGBOOST TWO-STAGE
        # =====================================================================
        print("\n" + "=" * 80)
        print("XGBOOST BIETÁPICO")
        print("=" * 80)

        print("\n   Optimizando hiperparámetros con Optuna...")
        t0 = time.time()
        xgb_best_params, xgb_best_wmape = optimize_two_stage_xgb(
            X_train, y_train, X_val, y_val,
            demand_threshold=demand_threshold,
            n_trials=optuna_trials_xgb,
        )
        print(f"   Mejores parámetros encontrados (WMAPE val: {xgb_best_wmape:.2f}%)")
        mlflow.log_dict(xgb_best_params, "tuning/xgb_two_stage_best_params.json")

        # Reconstruir parámetros completos
        mono_str = _xgb_monotone_constraints_str(feature_names)
        xgb_clf_params = {
            "objective": "binary:logistic", "tree_method": "hist", "device": "cuda",
            "learning_rate": xgb_best_params["learning_rate"],
            "max_depth": xgb_best_params["max_depth"],
            "subsample": xgb_best_params["subsample"],
            "colsample_bytree": xgb_best_params["colsample_bytree"],
            "min_child_weight": xgb_best_params["min_child_weight"],
            "reg_alpha": xgb_best_params["reg_alpha"],
            "reg_lambda": xgb_best_params["reg_lambda"],
            "n_estimators": 1000, "random_state": 42, "n_jobs": -1, "verbosity": 0,
            "monotone_constraints": mono_str,
        }
        xgb_reg_params = {
            "objective": "reg:squarederror", "tree_method": "hist", "device": "cuda",
            "learning_rate": xgb_best_params["learning_rate"],
            "max_depth": xgb_best_params["max_depth"],
            "subsample": xgb_best_params["subsample"],
            "colsample_bytree": xgb_best_params["colsample_bytree"],
            "min_child_weight": xgb_best_params["min_child_weight"],
            "reg_alpha": xgb_best_params["reg_alpha"],
            "reg_lambda": xgb_best_params["reg_lambda"],
            "n_estimators": 1000, "random_state": 42, "n_jobs": -1, "verbosity": 0,
            "monotone_constraints": mono_str,
        }

        print("\n   Entrenando modelo final XGBoost bietápico...")
        xgb_ts = TwoStageDemandModel(
            backend="xgboost",
            clf_params=xgb_clf_params,
            reg_params=xgb_reg_params,
            demand_threshold=demand_threshold,
            calibrate=True,
        )
        xgb_ts.fit(X_train, y_train, X_val, y_val)
        xgb_train_time = time.time() - t0

        # Evaluación completa
        xgb_eval = full_evaluation(xgb_ts, X_test, y_test, test_df, train_df, ts_dir, "xgb_two_stage")

        # Log métricas a MLflow
        for k, v in xgb_eval["metrics"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"xgb_ts_{k}", float(v) if v == v else 0.0)
        mlflow.log_metric("xgb_ts_train_time_sec", xgb_train_time)

        # Intervalos conformales
        y_pred_val_xgb = xgb_ts.predict(X_val)
        y_pred_test_xgb = xgb_eval["y_pred"]
        conf = split_conformal_interval(y_val_orig, y_pred_val_xgb, y_pred_test_xgb, alphas=(0.1, 0.2))
        for a_key, d in conf.items():
            cov, width = coverage_width(y_test_orig, d["lo"], d["hi"])
            mlflow.log_metric(f"xgb_ts_{a_key}_coverage_pct", cov)
            mlflow.log_metric(f"xgb_ts_{a_key}_width", width)
            xgb_eval["metrics"][f"{a_key}_coverage"] = cov
            xgb_eval["metrics"][f"{a_key}_width"] = width

        # Visualizaciones
        plot_two_stage_results(y_test_orig, y_pred_test_xgb, ts_dir, "xgb_two_stage", demand_threshold)
        plot_threshold_calibration(xgb_ts.calibration_info_, ts_dir, "xgb_two_stage")

        # SHAP para el regresor (Etapa 2)
        try:
            shap_sample_idx = np.random.choice(len(X_test), size=min(30000, len(X_test)), replace=False)
            expl = shap.TreeExplainer(xgb_ts.reg_)
            shap_vals = expl.shap_values(X_test.iloc[shap_sample_idx])
            shap_imp = pd.DataFrame({
                "feature": X_test.columns,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            shap_imp.to_csv(ts_dir / "xgb_two_stage_shap_importance.csv", index=False)

            plt.figure(figsize=(10, 10))
            shap.summary_plot(shap_vals, X_test.iloc[shap_sample_idx], plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(ts_dir / "xgb_two_stage_shap_bar.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"   SHAP error (XGBoost): {e}")

        # Guardar modelo
        xgb_ts.save(str(ts_dir / "xgb"))
        results["xgb_two_stage"] = {
            "metrics": xgb_eval["metrics"],
            "segments": xgb_eval["segments"],
            "params": xgb_best_params,
            "train_time_sec": xgb_train_time,
        }

        print(f"\n   XGBoost bietápico completado:")
        print(f"      WMAPE: {xgb_eval['metrics']['WMAPE']:.2f}%")
        print(f"      R²:    {xgb_eval['metrics']['R2']:.4f}")
        print(f"      MAE:   {xgb_eval['metrics']['MAE']:.2f}")
        print(f"      clf_F1: {xgb_eval['metrics']['clf_f1']:.3f}")
        print(f"      low_demand_F1: {xgb_eval['metrics']['low_demand_f1']:.3f}")

        # =====================================================================
        # CARGA DE BASELINES PARA COMPARACIÓN
        # =====================================================================
        print("\n" + "=" * 80)
        print("COMPARACIÓN CON BASELINES SINGLE-STAGE")
        print("=" * 80)

        baseline_results = {}

        # Random Forest baseline
        rf_path = output_dir / "rf_baseline.pkl"
        if rf_path.exists():
            rf_model = joblib.load(rf_path)
            y_pred_rf = np.expm1(rf_model.predict(X_test))
            rf_metrics = regression_report(y_test_orig, y_pred_rf)
            rf_metrics["WMAPE_revenue"] = wmape_revenue(test_df, y_test_orig, y_pred_rf)
            baseline_results["rf_single"] = rf_metrics
            print(f"   RF single-stage: WMAPE={rf_metrics['WMAPE']:.2f}%, R²={rf_metrics['R2']:.4f}")

        # XGBoost baseline
        xgb_path = output_dir / "xgb_demand_gpu.json"
        if xgb_path.exists():
            bst = xgb.Booster()
            bst.load_model(str(xgb_path))
            dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
            y_pred_xgb_single = np.expm1(bst.predict(dtest))
            xgb_single_metrics = regression_report(y_test_orig, y_pred_xgb_single)
            xgb_single_metrics["WMAPE_revenue"] = wmape_revenue(test_df, y_test_orig, y_pred_xgb_single)
            baseline_results["xgb_single"] = xgb_single_metrics
            print(f"   XGB single-stage: WMAPE={xgb_single_metrics['WMAPE']:.2f}%, R²={xgb_single_metrics['R2']:.4f}")

        # LightGBM baseline
        lgbm_path = output_dir / "lgbm_alt.pkl"
        if lgbm_path.exists():
            lgbm_single = joblib.load(lgbm_path)
            y_pred_lgbm_single = np.expm1(lgbm_single.predict(X_test))
            lgbm_single_metrics = regression_report(y_test_orig, y_pred_lgbm_single)
            lgbm_single_metrics["WMAPE_revenue"] = wmape_revenue(test_df, y_test_orig, y_pred_lgbm_single)
            baseline_results["lgbm_single"] = lgbm_single_metrics
            print(f"   LGBM single-stage: WMAPE={lgbm_single_metrics['WMAPE']:.2f}%, R²={lgbm_single_metrics['R2']:.4f}")

        # =====================================================================
        # TABLA COMPARATIVA FINAL
        # =====================================================================
        print("\n" + "=" * 80)
        print("RESUMEN COMPARATIVO (Test Set)")
        print("=" * 80)

        all_models = {}
        for name, res in results.items():
            all_models[name] = res["metrics"]
        for name, metrics in baseline_results.items():
            all_models[name] = metrics

        print(f"\n   {'Modelo':<22} {'WMAPE':>8} {'SMAPE':>8} {'MAE':>8} {'R²':>8} {'clf_F1':>8}")
        print("   " + "-" * 66)
        for name, m in sorted(all_models.items(), key=lambda x: x[1].get("WMAPE", 999)):
            clf_f1 = m.get("clf_f1", "-")
            clf_f1_str = f"{clf_f1:.3f}" if isinstance(clf_f1, float) else clf_f1
            print(f"   {name:<22} {m['WMAPE']:>7.2f}% {m.get('SMAPE', 0):>7.2f}% {m['MAE']:>7.2f} {m['R2']:>8.4f} {clf_f1_str:>8}")

        # Mejor modelo
        best_name = min(all_models.items(), key=lambda x: x[1].get("WMAPE", 999))[0]
        best_wmape = all_models[best_name]["WMAPE"]
        print(f"\n   Mejor modelo: {best_name.upper()} (WMAPE: {best_wmape:.2f}%)")
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric("best_wmape", best_wmape)

        # =====================================================================
        # GUARDAR METADATA
        # =====================================================================
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "demand_threshold": demand_threshold,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "optuna_trials_lgbm": optuna_trials_lgbm,
            "optuna_trials_xgb": optuna_trials_xgb,
            "two_stage_results": {
                k: {
                    "metrics": {mk: float(mv) if isinstance(mv, (int, float, np.floating)) else mv
                                for mk, mv in v["metrics"].items()},
                    "params": v.get("params"),
                    "train_time_sec": v.get("train_time_sec"),
                }
                for k, v in results.items()
            },
            "baseline_results": {
                k: {mk: float(mv) if isinstance(mv, (int, float, np.floating)) else mv
                    for mk, mv in v.items()}
                for k, v in baseline_results.items()
            },
            "best_model": best_name,
        }

        meta_path = ts_dir / "two_stage_training_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        mlflow.log_artifact(str(meta_path))

        # Log all artifacts
        for f in ts_dir.glob("*.csv"):
            mlflow.log_artifact(str(f))
        for f in ts_dir.glob("*.png"):
            mlflow.log_artifact(str(f))

    print("\n" + "=" * 80)
    print("ENTRENAMIENTO BIETÁPICO COMPLETADO")
    print(f"   Artefactos en: {ts_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = train_two_stage_models(
        features_path="data/processed/features.parquet",
        output_dir="models",
        demand_threshold=DEMAND_THRESHOLD,
        optuna_trials_lgbm=30,
        optuna_trials_xgb=30,
    )
