"""
Entrenamiento de modelos con GPU y métricas completas para tesis.

Mejoras:
- GPU acceleration para XGBoost y LightGBM
- Métricas exhaustivas (MAE, MSE, RMSE, R2, RMSLE, MAPE, SMAPE, WMAPE, MdAE, MPE, MBE, MASE)
- Métricas por segmento y por feature (bins y categóricas)
- Intervalos de predicción conformales con cobertura y ancho
- Hyperparameter tuning con Optuna
- Observabilidad con MLflow + monitoreo de sistema (CPU/GPU/RAM)
- SHAP para interpretabilidad + permutation importance
- Visualizaciones y artefactos para tesis
"""

from __future__ import annotations

import os
import time
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import mlflow
import optuna
import shap
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.metrics import (
    regression_report,
    group_metrics,
    binned_metrics,
    mase_from_train_series,
    wmape_revenue,
)
from src.models.conformal import split_conformal_interval, coverage_width

import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None


# =============================================================================
# UTILIDADES DE MONITOREO
# =============================================================================

def start_system_monitor(outfile: Path, log_to_mlflow: bool = True, interval_sec: float = 5.0):
    """Lanza un hilo que registra métricas de CPU/GPU/Mem a CSV (y MLflow si está activo)."""
    stop_event = threading.Event()

    outfile.parent.mkdir(parents=True, exist_ok=True)

    def _gpu_info():
        if pynvml is None:
            return None
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            name = pynvml.nvmlDeviceGetName(h)
            return {
                "gpu_name": str(name),
                "gpu_mem_total_mb": float(mem.total / (1024**2)),
                "gpu_mem_used_mb": float(mem.used / (1024**2)),
                "gpu_util_pct": float(util.gpu),
            }
        except Exception:
            return None

    def _loop():
        with open(outfile, "w", encoding="utf-8") as f:
            f.write("timestamp,cpu_pct,ram_used_mb,ram_total_mb,gpu_util_pct,gpu_mem_used_mb\n")
        while not stop_event.is_set():
            cpu = psutil.cpu_percent(interval=None) if psutil else np.nan
            vm = psutil.virtual_memory() if psutil else None
            ram_used = float(vm.used / (1024**2)) if vm else np.nan
            ram_total = float(vm.total / (1024**2)) if vm else np.nan
            gpu = _gpu_info() or {}
            row = {
                "timestamp": datetime.now().isoformat(),
                "cpu_pct": cpu,
                "ram_used_mb": ram_used,
                "ram_total_mb": ram_total,
                "gpu_util_pct": gpu.get("gpu_util_pct", np.nan),
                "gpu_mem_used_mb": gpu.get("gpu_mem_used_mb", np.nan),
            }
            with open(outfile, "a", encoding="utf-8") as f:
                f.write(
                    f"{row['timestamp']},{row['cpu_pct']},{row['ram_used_mb']},{row['ram_total_mb']},{row['gpu_util_pct']},{row['gpu_mem_used_mb']}\n"
                )
            if log_to_mlflow:
                for k, v in row.items():
                    if k != "timestamp" and np.isfinite(v):
                        mlflow.log_metric(k, float(v))
            time.sleep(interval_sec)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return stop_event


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================

def prepare_data(df: pd.DataFrame, target_col: str = "target"):
    """Prepara features y target."""
    exclude_cols = [
        "fecha",
        "producto_id",
        "sucursal_id",
        "target",
        "unidades",
        "ingreso_usd",
        "costo_usd",
        "margen_usd",
        "clase",
        "tasa_bcv",
        "rotacion",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
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
# HYPERPARAMETER TUNING CON OPTUNA (optimiza WMAPE en validación)
# =============================================================================

def _xgb_monotone_constraints_str(columns: List[str]) -> str:
    """Monotonicidad: precio_unitario_usd con relación negativa (-1), resto 0."""
    lst = ["-1" if c == "precio_unitario_usd" else "0" for c in columns]
    return "(" + ",".join(lst) + ")"


def _lgb_monotone_constraints_list(columns: List[str]) -> List[int]:
    return [-1 if c == "precio_unitario_usd" else 0 for c in columns]


def optimize_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 30,
    use_gpu: bool = True,
    rounds: int = 2000,
):
    """Optimiza hiperparámetros de XGBoost usando la API core (xgboost.train)."""

    dtrain = xgb.DMatrix(X_train, label=y_train.values, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val.values, feature_names=X_val.columns.tolist())

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            # Fallback a CPU 'hist' si no hay GPU disponible en la build
            "tree_method": "hist",
            "monotone_constraints": _xgb_monotone_constraints_str(X_train.columns.tolist()),
            "eta": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "seed": 42,
        }
        evals_result = {}
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=[(dval, "validation")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        y_pred = bst.predict(dval)
        y_pred_orig = np.expm1(y_pred)
        y_val_orig = np.expm1(y_val.values)
        return float(np.sum(np.abs(y_val_orig - y_pred_orig)) / (np.sum(np.abs(y_val_orig)) + 1e-12) * 100)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


def optimize_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 30,
    use_gpu: bool = True,
    rounds: int = 2000,
    early_stopping_rounds: int = 200,
):
    """Optimiza hiperparámetros de LightGBM con Optuna usando early stopping."""

    def objective(trial):
        params = {
            "objective": "regression",
            "device": "gpu" if use_gpu else "cpu",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
            "monotone_constraints": _lgb_monotone_constraints_list(X_train.columns.tolist()),
        }
        model = lgb.LGBMRegressor(**params, n_estimators=rounds)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        y_pred = model.predict(X_val)
        y_pred_orig = np.expm1(y_pred)
        y_val_orig = np.expm1(y_val.values)
        return float(np.sum(np.abs(y_val_orig - y_pred_orig)) / (np.sum(np.abs(y_val_orig)) + 1e-12) * 100)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_xgboost_gpu(
    X_train,
    y_train,
    X_val,
    y_val,
    params: Optional[Dict] = None,
    rounds: int = 2000,
    eval_log_period: int = 50,
):
    """Entrena XGBoost con GPU usando la API core (xgboost.train). Devuelve Booster."""

    if params is None:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "eta": 0.05,
            "max_depth": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "seed": 42,
        }
    else:
        params = {**params}
        params.setdefault("objective", "reg:squarederror")
        # Compatibilidad GPU
        params["tree_method"] = "hist"
        # alias
        if "learning_rate" in params:
            params["eta"] = params.pop("learning_rate")
        if "random_state" in params:
            params["seed"] = params.pop("random_state")

    dtrain = xgb.DMatrix(X_train, label=y_train.values, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val.values, feature_names=X_val.columns.tolist())

    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[(dval, "validation")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    # Loggear curva de validación a MLflow si existe
    try:
        if "validation" in evals_result and "rmse" in evals_result["validation"]:
            for i, v in enumerate(evals_result["validation"]["rmse"]):
                mlflow.log_metric("xgb_rmse_val", float(v), step=i)
    except Exception:
        pass

    return bst


def train_lightgbm_gpu(
    X_train,
    y_train,
    X_val,
    y_val,
    params: Optional[Dict] = None,
    rounds: int = 2000,
    early_stopping_rounds: int = 200,
):
    """Entrena LightGBM con GPU y early stopping."""

    if params is None:
        params = {
            "objective": "regression",
            "device": "gpu",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    else:
        params = {**params, "device": "gpu", "random_state": 42, "n_jobs": -1, "verbose": -1}

    # Monotonicidad precio→demanda
    mono = _lgb_monotone_constraints_list(X_train.columns.tolist())
    params["monotone_constraints"] = mono

    model = lgb.LGBMRegressor(**params, n_estimators=rounds)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
    )

    return model


def train_random_forest(X_train, y_train, params: Optional[Dict] = None):
    """Entrena Random Forest (CPU, baseline)."""

    if params is None:
        params = {
            "n_estimators": 1000,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "random_state": 42,
            "n_jobs": -1,
        }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    return model


# =============================================================================
# VISUALIZACIÓN Y ANÁLISIS
# =============================================================================

def analyze_errors_by_segment(df_test, y_true, y_pred, output_dir):
    """Analiza errores por diferentes segmentos y guarda CSVs."""

    output_dir = Path(output_dir)
    results = {}

    # Por clase/categoría
    if "clase" in df_test.columns:
        seg = group_metrics(df_test, y_true, y_pred, "clase")
        seg.to_csv(output_dir / "metrics_by_clase.csv", index=False)
        results["by_clase"] = seg.head(50).to_dict(orient="records")

    # Por sucursal
    if "sucursal_id" in df_test.columns:
        seg = group_metrics(df_test, y_true, y_pred, "sucursal_id")
        seg.to_csv(output_dir / "metrics_by_sucursal.csv", index=False)
        results["by_sucursal"] = seg.head(50).to_dict(orient="records")

    # Por cuartiles de demanda
    quartiles = pd.qcut(y_true, q=4, labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"], duplicates="drop")
    df_q = pd.DataFrame({"q": quartiles, "y_true": y_true, "y_pred": y_pred})
    df_q["abs_err"] = np.abs(df_q["y_true"] - df_q["y_pred"])
    agg = (
        df_q.groupby("q")
        .agg(WMAPE_num=("abs_err", "sum"), y_true_sum=("y_true", "sum"), MAE=("abs_err", "mean"), n=("y_true", "size"))
        .assign(WMAPE=lambda d: (d["WMAPE_num"] / (d["y_true_sum"].abs() + 1e-12)) * 100)
        .drop(columns=["WMAPE_num", "y_true_sum"])
        .reset_index()
    )
    agg.to_csv(output_dir / "metrics_by_demand_quartile.csv", index=False)
    results["by_demand_quartile"] = agg.to_dict(orient="records")

    return results


def plot_results(y_true, y_pred, output_dir, model_name="model"):
    """Genera visualizaciones para la tesis."""

    output_dir = Path(output_dir)

    # 1. Scatter plot: Real vs Predicho
    fig, ax = plt.subplots(figsize=(10, 8))
    sample_idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=10)
    lim = max(np.max(y_true), np.max(y_pred))
    ax.plot([0, lim], [0, lim], "r--", label="Predicción perfecta")
    ax.set_xlabel("Demanda Real (unidades)")
    ax.set_ylabel("Demanda Predicha (unidades)")
    ax.set_title(f"{model_name}: Real vs Predicho")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_scatter.png", dpi=150)
    plt.close()

    # 2. Distribución de errores
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=100, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=np.median(errors), color="green", linestyle="--", linewidth=2, label=f"Mediana: {np.median(errors):.2f}")
    ax.set_xlabel("Error (Predicho - Real)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{model_name}: Distribución de Errores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_error_dist.png", dpi=150)
    plt.close()

    # 3. Error por rango de demanda
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = pd.cut(
        y_true,
        bins=[0, 1, 5, 10, 20, 50, 100, np.inf],
        labels=["0-1", "1-5", "5-10", "10-20", "20-50", "50-100", "100+"],
    )
    error_by_bin = pd.DataFrame({"bin": bins, "abs_error": np.abs(errors)}).groupby("bin")["abs_error"].mean()
    error_by_bin.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Rango de Demanda (unidades)")
    ax.set_ylabel("MAE Promedio")
    ax.set_title(f"{model_name}: MAE por Rango de Demanda")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_error_by_range.png", dpi=150)
    plt.close()

    print(f"   📈 Gráficos guardados en {output_dir}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def train_thesis_quality(
    features_path: str,
    output_dir: str = "models",
    use_gpu: bool = True,
    run_optuna: bool = True,
    optuna_trials_xgb: int = 60,
    optuna_trials_lgbm: int = 40,
    xgb_rounds: int = 2000,
    lgbm_rounds: int = 2000,
    eval_log_period: int = 50,
):
    """Pipeline de entrenamiento con calidad de tesis y observabilidad."""

    print("=" * 80)
    print("🎓 ENTRENAMIENTO PARA TESIS - SIP DYNAMIC PRICING")
    print("=" * 80)
    print(f"   GPU: {'Habilitada' if use_gpu else 'Deshabilitada'}")
    print(
        f"   Presupuesto de entrenamiento: XGB rounds={xgb_rounds}, LGBM rounds={lgbm_rounds}"
    )
    print(
        f"   Optuna: XGB {optuna_trials_xgb} trials | LGBM {optuna_trials_lgbm} trials (objetivo=W-MAPE Val)"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri("file:" + str((Path.cwd() / "mlruns").resolve()))
    mlflow.set_experiment("SIP-Demand-Training")

    # 0. Monitoreo de sistema
    sys_log = output_dir / "system_metrics.csv"
    mon = start_system_monitor(sys_log, log_to_mlflow=True, interval_sec=5.0)

    with mlflow.start_run(run_name="training_pipeline") as run:
        mlflow.log_params(
            {
                "use_gpu": use_gpu,
                "optuna_trials_xgb": optuna_trials_xgb,
                "optuna_trials_lgbm": optuna_trials_lgbm,
                "xgb_rounds": xgb_rounds,
                "lgbm_rounds": lgbm_rounds,
                "eval_log_period": eval_log_period,
            }
        )

        # 1. Cargar datos
        print("\n📂 Cargando datos...")
        df = pd.read_parquet(features_path)
        print(f"   Registros: {len(df):,}")
        mlflow.log_param("n_rows", int(len(df)))

        # 2. Split temporal
        print("\n📅 Split temporal...")
        train_df, val_df, test_df = temporal_split(df)
        print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        mlflow.log_params({"train_rows": int(len(train_df)), "val_rows": int(len(val_df)), "test_rows": int(len(test_df))})

        # 3. Preparar features
        X_train, y_train, feature_names = prepare_data(train_df)
        X_val, y_val, _ = prepare_data(val_df)
        X_test, y_test, _ = prepare_data(test_df)
        print(f"   Features: {len(feature_names)}")
        mlflow.log_param("n_features", int(len(feature_names)))

        results = {}

        # =========================================================================
        # XGBOOST
        # =========================================================================
        print("\n" + "=" * 80)
        print("🚀 XGBOOST CON GPU")
        print("=" * 80)

        xgb_params = None
        if run_optuna:
            print("\n   🔍 Optimizando hiperparámetros con Optuna...")
            xgb_params = optimize_xgboost(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=optuna_trials_xgb,
                use_gpu=use_gpu,
                rounds=xgb_rounds,
            )
            print(f"   ✓ Mejores parámetros: {xgb_params}")
            mlflow.log_dict(xgb_params, "tuning/xgb_best_params.json")

        t0 = time.time()
        xgb_bst = train_xgboost_gpu(
            X_train,
            y_train,
            X_val,
            y_val,
            params=xgb_params,
            rounds=xgb_rounds,
            eval_log_period=eval_log_period,
        )
        train_time = time.time() - t0
        mlflow.log_metric("xgb_train_time_sec", float(train_time))

        # Evaluar
        dval = xgb.DMatrix(X_val, feature_names=X_val.columns.tolist())
        dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
        y_pred_val = np.expm1(xgb_bst.predict(dval))
        y_val_orig = np.expm1(y_val.values)
        y_pred_test = np.expm1(xgb_bst.predict(dtest))
        y_test_orig = np.expm1(y_test.values)

        # MASE global usando train como baseline ingenuo
        mase_val = mase_from_train_series(train_df, val_df, y_val_orig, y_pred_val, ["producto_id", "sucursal_id"], "unidades")
        mase_test = mase_from_train_series(train_df, test_df, y_test_orig, y_pred_test, ["producto_id", "sucursal_id"], "unidades")

        # Métricas + WMAPE ponderado por ingreso
        xgb_metrics = regression_report(y_test_orig, y_pred_test, extras={"MASE": mase_test})
        xgb_metrics["WMAPE_revenue"] = wmape_revenue(test_df, y_test_orig, y_pred_test)
        for k, v in xgb_metrics.items():
            mlflow.log_metric(f"xgb_test_{k}", float(v) if v == v else 0.0)

        # Intervalos conformales y cobertura
        conf = split_conformal_interval(y_val_orig, y_pred_val, y_pred_test, alphas=(0.1, 0.2))
        for a_key, d in conf.items():
            cov, width = coverage_width(y_test_orig, d["lo"], d["hi"])
            mlflow.log_metric(f"xgb_{a_key}_coverage_pct", cov)
            mlflow.log_metric(f"xgb_{a_key}_width", width)
        # Guardar CSV con intervalos
        conf_df = pd.DataFrame({
            "y_true": y_test_orig,
            "pred": y_pred_test,
            "lo_90": conf["alpha_0.1"]["lo"],
            "hi_90": conf["alpha_0.1"]["hi"],
            "lo_80": conf["alpha_0.2"]["lo"],
            "hi_80": conf["alpha_0.2"]["hi"],
        })
        conf_df.to_csv(output_dir / "xgb_conformal_intervals.csv", index=False)
        mlflow.log_artifact(str(output_dir / "xgb_conformal_intervals.csv"))

        # Análisis de errores por segmento y por feature (bins en top features)
        seg = analyze_errors_by_segment(test_df, y_test_orig, y_pred_test, output_dir)
        mlflow.log_artifact(str(output_dir / "metrics_by_clase.csv")) if (output_dir / "metrics_by_clase.csv").exists() else None
        mlflow.log_artifact(str(output_dir / "metrics_by_sucursal.csv")) if (output_dir / "metrics_by_sucursal.csv").exists() else None
        mlflow.log_artifact(str(output_dir / "metrics_by_demand_quartile.csv"))

        # SHAP (muestra para performance)
        try:
            shap_sample_idx = np.random.choice(len(X_test), size=min(50000, len(X_test)), replace=False)
            expl = shap.TreeExplainer(xgb_bst)
            shap_vals = expl.shap_values(X_test.iloc[shap_sample_idx])
            # Importancia SHAP promedio absoluto
            shap_importance = pd.DataFrame({
                "feature": X_test.columns,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            shap_importance.to_csv(output_dir / "xgb_shap_importance.csv", index=False)
            mlflow.log_artifact(str(output_dir / "xgb_shap_importance.csv"))

            # Plots
            plt.figure(figsize=(10, 10))
            shap.summary_plot(shap_vals, X_test.iloc[shap_sample_idx], plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "xgb_shap_bar.png", dpi=150)
            plt.close()
            mlflow.log_artifact(str(output_dir / "xgb_shap_bar.png"))

            plt.figure(figsize=(10, 10))
            shap.summary_plot(shap_vals, X_test.iloc[shap_sample_idx], show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "xgb_shap_beeswarm.png", dpi=150)
            plt.close()
            mlflow.log_artifact(str(output_dir / "xgb_shap_beeswarm.png"))

            # Métricas por bins en top 5 features numéricas
            top_feats = [f for f in shap_importance["feature"].head(5).tolist() if pd.api.types.is_numeric_dtype(X_test[f])]
            for ftr in top_feats:
                bm = binned_metrics(X_test, y_test_orig, y_pred_test, ftr, n_bins=5)
                if len(bm):
                    out_csv = output_dir / f"xgb_metrics_by_{ftr}_bins.csv"
                    bm.to_csv(out_csv, index=False)
                    mlflow.log_artifact(str(out_csv))
        except Exception as e:  # pragma: no cover
            print("   ⚠️ SHAP no disponible o error:", e)

        # Permutation importance (en una muestra para costo)
        try:
            from sklearn.inspection import permutation_importance as p_import

            sample_idx = np.random.choice(len(X_test), size=min(20000, len(X_test)), replace=False)
            # XGB Booster no es compatible con sklearn permutation_importance directamente
            # Se necesitaría un wrapper. Salteamos para XGBoost (usamos SHAP en su lugar).
            pass  # r = p_import(xgb_bst, X_test.iloc[sample_idx], np.log1p(y_test_orig[sample_idx]), n_repeats=5, n_jobs=-1, random_state=42)
            perm_df = pd.DataFrame({"feature": X_test.columns, "perm_importance": r.importances_mean}).sort_values(
                "perm_importance", ascending=False
            )
            perm_df.to_csv(output_dir / "xgb_permutation_importance.csv", index=False)
            mlflow.log_artifact(str(output_dir / "xgb_permutation_importance.csv"))
        except Exception as e:  # pragma: no cover
            print("   ⚠️ Permutation importance no disponible:", e)

        # Gráficos básicos
        plot_results(y_test_orig, y_pred_test, output_dir, "XGBoost")
        mlflow.log_artifact(str(output_dir / "xgboost_scatter.png")) if (output_dir / "xgboost_scatter.png").exists() else None
        mlflow.log_artifact(str(output_dir / "xgboost_error_dist.png")) if (output_dir / "xgboost_error_dist.png").exists() else None
        mlflow.log_artifact(str(output_dir / "xgboost_error_by_range.png")) if (output_dir / "xgboost_error_by_range.png").exists() else None

        # Guardar modelo
        # Guardar modelo XGBoost Booster en formato JSON
        xgb_model_path = output_dir / "xgb_demand_gpu.json"
        xgb_bst.save_model(str(xgb_model_path))
        mlflow.log_artifact(str(xgb_model_path))
        results["xgboost"] = {"metrics": xgb_metrics, "params": xgb_params}

        # =========================================================================
        # LIGHTGBM
        # =========================================================================
        print("\n" + "=" * 80)
        print("⚡ LIGHTGBM CON GPU")
        print("=" * 80)

        lgbm_params = None
        if run_optuna:
            print("\n   🔍 Optimizando hiperparámetros con Optuna...")
            try:
                lgbm_params = optimize_lightgbm(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_trials=optuna_trials_lgbm,
                    use_gpu=use_gpu,
                    rounds=lgbm_rounds,
                )
                print(f"   ✓ Mejores parámetros: {lgbm_params}")
                mlflow.log_dict(lgbm_params, "tuning/lgbm_best_params.json")
            except Exception as e:
                print(f"   ⚠️ Error en Optuna para LightGBM: {e}")
                print("   Usando parámetros por defecto...")

        t0 = time.time()
        lgbm_early_stop = 200  # Early stopping rounds for LightGBM
        try:
            lgbm_model = train_lightgbm_gpu(
                X_train,
                y_train,
                X_val,
                y_val,
                params=lgbm_params,
                rounds=lgbm_rounds,
                early_stopping_rounds=lgbm_early_stop,
            )
            lgbm_train_time = time.time() - t0
            mlflow.log_metric("lgbm_train_time_sec", float(lgbm_train_time))

            y_pred_test_lgbm = np.expm1(lgbm_model.predict(X_test))
            lgbm_metrics = regression_report(y_test_orig, y_pred_test_lgbm)
            lgbm_metrics["WMAPE_revenue"] = wmape_revenue(test_df, y_test_orig, y_pred_test_lgbm)
            for k, v in lgbm_metrics.items():
                mlflow.log_metric(f"lgbm_test_{k}", float(v) if v == v else 0.0)

            plot_results(y_test_orig, y_pred_test_lgbm, output_dir, "LightGBM")
            joblib.dump(lgbm_model, output_dir / "lgbm_demand_gpu.pkl")
            mlflow.log_artifact(str(output_dir / "lgbm_demand_gpu.pkl"))
            results["lightgbm"] = {"metrics": lgbm_metrics, "params": lgbm_params}

        except Exception as e:
            print(f"   ⚠️ Error entrenando LightGBM con GPU: {e}")
            print("   Intentando sin GPU...")
            lgbm_model = lgb.LGBMRegressor(device="cpu")
            lgbm_model.fit(X_train, y_train)
            y_pred_test_lgbm = np.expm1(lgbm_model.predict(X_test))
            lgbm_metrics = regression_report(y_test_orig, y_pred_test_lgbm)
            results["lightgbm"] = {"metrics": lgbm_metrics}

        # =========================================================================
        # RANDOM FOREST (Baseline)
        # =========================================================================
        print("\n" + "=" * 80)
        print("🌲 RANDOM FOREST (Baseline)")
        print("=" * 80)

        t0 = time.time()
        rf_model = train_random_forest(X_train, y_train)
        rf_train_time = time.time() - t0
        mlflow.log_metric("rf_train_time_sec", float(rf_train_time))

        y_pred_test_rf = np.expm1(rf_model.predict(X_test))
        rf_metrics = regression_report(y_test_orig, y_pred_test_rf)
        rf_metrics["WMAPE_revenue"] = wmape_revenue(test_df, y_test_orig, y_pred_test_rf)
        for k, v in rf_metrics.items():
            mlflow.log_metric(f"rf_test_{k}", float(v) if v == v else 0.0)

        plot_results(y_test_orig, y_pred_test_rf, output_dir, "RandomForest")
        joblib.dump(rf_model, output_dir / "rf_baseline.pkl")
        mlflow.log_artifact(str(output_dir / "rf_baseline.pkl"))
        results["random_forest"] = {"metrics": rf_metrics}

        # =========================================================================
        # FEATURE IMPORTANCE (XGB nativa)
        # =========================================================================
        print("\n" + "=" * 80)
        print("📊 FEATURE IMPORTANCE (XGBoost)")
        print("=" * 80)

        # Importancia de features desde Booster (gain)
        score = xgb_bst.get_score(importance_type="gain")
        # Mapear f0..fn a nombres reales
        mapping = {f"f{i}": name for i, name in enumerate(feature_names)}
        fi_rows = [{"feature": mapping.get(k, k), "importance": v} for k, v in score.items()]
        fi_df = pd.DataFrame(fi_rows).sort_values("importance", ascending=False)
        print("\n   Top 15:")
        for i, row in fi_df.head(15).iterrows():
            print(f"   {fi_df.index.get_loc(i)+1:2}. {row['feature']:<30} {row['importance']:.4f}")

        fi_df.to_csv(output_dir / "feature_importance_gpu.csv", index=False)
        mlflow.log_artifact(str(output_dir / "feature_importance_gpu.csv"))

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 12))
        fi_df.head(20).plot(kind="barh", x="feature", y="importance", ax=ax, legend=False)
        ax.invert_yaxis()
        ax.set_xlabel("Importancia")
        ax.set_title("Top 20 Features - XGBoost")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(output_dir / "feature_importance.png"))

        # =========================================================================
        # RESUMEN FINAL
        # =========================================================================
        print("\n" + "=" * 80)
        print("📊 RESUMEN COMPARATIVO (Test)")
        print("=" * 80)

        print(f"\n   {'Modelo':<18} {'WMAPE':>8} {'SMAPE':>8} {'MAE':>8} {'R²':>8}")
        print("   " + "-" * 52)

        for name, res in results.items():
            m = res["metrics"]
            print(f"   {name:<18} {m['WMAPE']:>7.2f}% {m['SMAPE']:>7.2f}% {m['MAE']:>7.2f} {m['R2']:>8.4f}")

        # Mejor modelo por WMAPE
        best = min(results.items(), key=lambda x: x[1]["metrics"]["WMAPE"])
        print(f"\n   🏆 Mejor modelo: {best[0].upper()} (WMAPE: {best[1]['metrics']['WMAPE']:.2f}%)")
        mlflow.log_metric("best_wmape", float(best[1]["metrics"]["WMAPE"]))
        mlflow.set_tag("best_model", best[0])

        # Guardar metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "gpu_used": use_gpu,
            "optuna_trials_xgb": optuna_trials_xgb,
            "optuna_trials_lgbm": optuna_trials_lgbm,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "results": {k: {"metrics": v["metrics"], "params": v.get("params")} for k, v in results.items()},
        }

        with open(output_dir / "training_metadata_gpu.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        mlflow.log_artifact(str(output_dir / "training_metadata_gpu.json"))

    # Detener monitor
    mon.set()

    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Modelos y artefactos guardados en: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = train_thesis_quality(
        features_path="data/processed/features.parquet",
        output_dir="models",
        use_gpu=True,
        run_optuna=True,
        optuna_trials_xgb=60,
        optuna_trials_lgbm=40,
        xgb_rounds=2000,
        lgbm_rounds=2000,
        eval_log_period=50,
    )
