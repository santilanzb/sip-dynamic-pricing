"""
Utilidades de métricas para evaluación rigurosa del modelo.
Incluye métricas globales, por segmento y por feature, además de utilidades
para cálculo de MASE y PSI (Population Stability Index).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
)

# ===============================
# Métricas escalares
# ===============================

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0 else float("nan")


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8, min_denom: float = 0.5) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > min_denom
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true))
    return _safe_div(num * 100.0, den)


def mpe(y_true: np.ndarray, y_pred: np.ndarray, min_denom: float = 0.5) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > min_denom
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100)


def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_log_error(np.maximum(y_true, 0) + 1, np.maximum(y_pred, 0) + 1)))


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def overforecast_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) > 0) * 100)


# ===============================
# MASE (requiere baseline ingenuo de entrenamiento)
# ===============================

def mase_from_train_series(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_cols: List[str],
    target_col: str = "unidades",
) -> float:
    """
    Calcula MASE global utilizando, por grupo, el MAE de un pronóstico ingenuo (lag-1)
    computado SOBRE EL TRAIN y lo usa como denominador para el test.
    """
    # Denominador por grupo (MAE naive lag-1 en train ordenado por fecha)
    train_sorted = train_df.sort_values(group_cols + ["fecha"]).copy()
    train_sorted["naive_lag1"] = train_sorted.groupby(group_cols)[target_col].shift(1)
    g = (
        train_sorted.dropna(subset=["naive_lag1"]).groupby(group_cols)
        .apply(lambda d: np.mean(np.abs(d[target_col].values - d["naive_lag1"].values)))
        .rename("mae_naive")
        .reset_index()
    )
    # Mapear denominadores al conjunto de test
    test_key = test_df[group_cols].copy()
    test_key = test_key.reset_index(drop=True)
    test_key["__row_id__"] = np.arange(len(test_key))
    denom_map = test_key.merge(g, on=group_cols, how="left")["mae_naive"].values
    # Evitar nulos/ceros
    denom_map = np.where((np.isnan(denom_map)) | (denom_map <= 0), np.nan, denom_map)

    abs_err = np.abs(y_true - y_pred)
    valid = ~np.isnan(denom_map)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sum(abs_err[valid]) / np.sum(denom_map[valid]))


# ===============================
# PSI (Population Stability Index) para drift
# ===============================

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calcula PSI unidimensional con cortes en quantiles de expected.
    Retorna 0 si la variable es (casi) constante en expected.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    m_expected = np.isfinite(expected)
    m_actual = np.isfinite(actual)
    expected = expected[m_expected]
    actual = actual[m_actual]
    if expected.size < 100 or actual.size < 100:
        return float("nan")
    # Cortes por quantiles del expected (sin duplicados)
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    # Número de bins reales
    k = max(len(cuts) - 1, 1)
    if k < 2:
        return float(0.0)
    # Bin a 1..k
    expected_binned = np.clip(np.digitize(expected, cuts[1:-1], right=False), 1, k)
    actual_binned = np.clip(np.digitize(actual, cuts[1:-1], right=False), 1, k)

    def dist(x, k):
        vals, cnts = np.unique(x, return_counts=True)
        d = {int(v): c for v, c in zip(vals, cnts)}
        return np.array([d.get(i, 0) for i in range(1, k + 1)], dtype=float)

    e = dist(expected_binned, k)
    a = dist(actual_binned, k)
    e = e / (e.sum() + 1e-12)
    a = a / (a.sum() + 1e-12)
    # Evitar ceros exactos
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_by_feature(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        # Omitir variables de baja cardinalidad (e.g., binarias) donde PSI no es informativo
        if train_df[col].nunique(dropna=True) <= 5:
            continue
        v = psi(train_df[col].values, test_df[col].values)
        rows.append({"feature": col, "psi": v})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("psi", ascending=False)


# ===============================
# Empaquetadores
# ===============================

def wmape_revenue(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_col: str = "precio_unitario_usd",
) -> float:
    """WMAPE ponderado por ingreso (precio * unidades)."""
    if price_col not in df.columns:
        return float("nan")
    price = np.asarray(df[price_col].values, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    num = np.sum(np.abs(y_true - y_pred) * price)
    den = np.sum(np.abs(y_true) * price)
    return _safe_div(num * 100.0, den)


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extras: Optional[Dict] = None,
) -> Dict[str, float]:
    """Devuelve un conjunto amplio de métricas de regresión en escala original."""
    y_true = np.asarray(y_true)
    y_pred = np.maximum(np.asarray(y_pred), 0)

    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "WMAPE": wmape(y_true, y_pred),
        "MdAE": median_ae(y_true, y_pred),
        "MPE": mpe(y_true, y_pred),
        "MBE": mbe(y_true, y_pred),
        "OverForecastRate": overforecast_rate(y_true, y_pred),
    }
    if extras:
        out.update(extras)
    return {k: (float(v) if v is not None and not isinstance(v, float) else v) for k, v in out.items()}


def group_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    tmp = df[[group_col]].copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = np.maximum(y_pred, 0)
    tmp["abs_err"] = np.abs(tmp["y_true"] - tmp["y_pred"])
    agg = (
        tmp.groupby(group_col)
        .agg(
            n=("y_true", "size"),
            WMAPE=("abs_err", "sum"),
            y_true_sum=("y_true", "sum"),
            MAE=("abs_err", "mean"),
        )
        .assign(WMAPE=lambda d: (d["WMAPE"] / (d["y_true_sum"].abs() + 1e-12)) * 100)
        .drop(columns=["y_true_sum"])
        .reset_index()
        .sort_values("WMAPE", ascending=False)
    )
    return agg


def binned_metrics(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    if not pd.api.types.is_numeric_dtype(X[feature]):
        return pd.DataFrame()
    q = pd.qcut(X[feature], q=n_bins, duplicates="drop")
    df = pd.DataFrame({"bin": q, "y_true": y_true, "y_pred": np.maximum(y_pred, 0)})
    df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
    out = (
        df.groupby("bin")
        .agg(WMAPE_num=("abs_err", "sum"), y_true_sum=("y_true", "sum"), MAE=("abs_err", "mean"), n=("y_true", "size"))
        .assign(WMAPE=lambda d: (d["WMAPE_num"] / (d["y_true_sum"].abs() + 1e-12)) * 100)
        .drop(columns=["WMAPE_num", "y_true_sum"])
        .reset_index()
    )
    out["feature"] = feature
    return out
