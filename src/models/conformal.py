"""
Utilidades para intervalos de predicción mediante Conformal Prediction (split-conformal).
Basado en cuantiles de residuos de validación para obtener cobertura (1 - alpha).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def split_conformal_interval(
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_pred: np.ndarray,
    alphas=(0.1, 0.2),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calcula intervalos de predicción simétricos usando el cuantil de |residuo| en validación.

    Returns un dict con claves f"alpha_{alpha}", cada una con 'lo', 'hi', 'q' (cuantil usado).
    """
    y_val_true = np.asarray(y_val_true)
    y_val_pred = np.asarray(y_val_pred)
    y_test_pred = np.asarray(y_test_pred)

    abs_resid = np.abs(y_val_true - y_val_pred)
    out = {}
    for a in alphas:
        q = np.quantile(abs_resid, 1 - a)
        lo = np.maximum(y_test_pred - q, 0)
        hi = y_test_pred + q
        out[f"alpha_{a}"] = {"lo": lo, "hi": hi, "q": float(q)}
    return out


def coverage_width(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float]:
    """Cálculo de cobertura y ancho promedio de intervalos."""
    y_true = np.asarray(y_true)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    covered = (y_true >= lo) & (y_true <= hi)
    cov = float(np.mean(covered) * 100)
    width = float(np.mean(hi - lo))
    return cov, width
