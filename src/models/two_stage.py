"""
Modelo bietápico generalizado para predicción de demanda (hurdle model).

Etapa 1 (Clasificación): P(demanda ≥ τ) — predice si habrá demanda significativa.
Etapa 2 (Regresión):     E[unidades | demanda ≥ τ] — predice cantidad condicionada a demanda alta.

Predicción final:
    P(demanda ≥ τ) × E[unidades | demanda ≥ τ] + (1 − P(demanda ≥ τ)) × μ_low

Donde:
    τ (demand_threshold): umbral que separa demanda baja de significativa (default: 1.0 unidad)
    μ_low: media de demanda del régimen bajo (< τ), calculada en training

Soporta backends LightGBM y XGBoost con:
- Monotone constraints (precio_unitario_usd = -1)
- Threshold calibration (F1-optimal en validación)
- Evaluación completa con métricas de regresión + clasificación

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-20
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.metrics import regression_report, wmape_revenue


# =============================================================================
# Monotone constraints helpers
# =============================================================================

def _lgb_monotone_constraints(columns: List[str]) -> List[int]:
    """Retorna lista de constraints para LightGBM: -1 en precio_unitario_usd, 0 en el resto."""
    return [-1 if c == "precio_unitario_usd" else 0 for c in columns]


def _xgb_monotone_constraints_str(columns: List[str]) -> str:
    """Retorna string de constraints para XGBoost sklearn API."""
    lst = ["-1" if c == "precio_unitario_usd" else "0" for c in columns]
    return "(" + ",".join(lst) + ")"


# =============================================================================
# Threshold calibration
# =============================================================================

def calibrate_threshold(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    method: Literal["f1", "f1_weighted"] = "f1",
    n_thresholds: int = 200,
) -> Dict:
    """
    Calibra el threshold óptimo para el clasificador de Etapa 1.

    Barre thresholds en [0.05, 0.95] y selecciona el que maximiza F1.

    Args:
        y_true_binary: Labels binarias (0/1) — 1 si demanda ≥ τ.
        y_prob: Probabilidades predichas P(demanda ≥ τ).
        method: 'f1' para F1-score estándar.
        n_thresholds: Número de thresholds a evaluar.

    Returns:
        Dict con 'threshold', 'f1', 'precision', 'recall', y la curva completa 'curve'.
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    curve = []

    for t in thresholds:
        y_pred_bin = (y_prob >= t).astype(int)
        f1 = f1_score(y_true_binary, y_pred_bin, zero_division=0)
        prec = precision_score(y_true_binary, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_bin, zero_division=0)
        curve.append({"threshold": float(t), "f1": float(f1), "precision": float(prec), "recall": float(rec)})

        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(prec), "recall": float(rec)}

    return {**best, "curve": curve}


# =============================================================================
# TwoStageDemandModel
# =============================================================================

@dataclass
class TwoStageDemandModel:
    """
    Modelo bietápico generalizado para predicción de demanda (hurdle model).

    Attributes:
        backend: 'lightgbm' o 'xgboost'.
        clf_params: Hiperparámetros para el clasificador (Etapa 1).
        reg_params: Hiperparámetros para el regresor (Etapa 2).
        demand_threshold: Umbral τ que separa baja demanda de significativa (escala original).
        threshold: Threshold de decisión para P(demanda ≥ τ). Se calibra en fit() si calibrate=True.
        calibrate: Si True, calibra el threshold en el conjunto de validación durante fit().
    """
    backend: Literal["lightgbm", "xgboost"] = "lightgbm"
    clf_params: Optional[Dict] = None
    reg_params: Optional[Dict] = None
    demand_threshold: float = 1.0
    threshold: float = 0.5
    calibrate: bool = True

    # Modelos internos (se asignan en fit)
    clf_: object = field(default=None, init=False, repr=False)
    reg_: object = field(default=None, init=False, repr=False)
    calibration_info_: Optional[Dict] = field(default=None, init=False, repr=False)
    feature_names_: Optional[List[str]] = field(default=None, init=False, repr=False)
    mean_low_demand_: float = field(default=0.0, init=False, repr=False)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "TwoStageDemandModel":
        """
        Entrena el modelo bietápico.

        Args:
            X_train: Features de entrenamiento.
            y_train: Target en log1p(unidades).
            X_val: Features de validación (requerido si calibrate=True).
            y_val: Target de validación en log1p(unidades).

        Returns:
            self
        """
        self.feature_names_ = X_train.columns.tolist()
        tau = self.demand_threshold

        # Preparar labels basadas en umbral τ
        y_train_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        y_train_orig = np.expm1(y_train_arr)
        high_flag_train = (y_train_orig >= tau).astype(int)

        n_high = int(high_flag_train.sum())
        n_low = int(len(high_flag_train) - n_high)
        print(f"\n   Distribución de entrenamiento (τ = {tau}):")
        print(f"      Demanda ≥ {tau}: {n_high:,} ({n_high/len(high_flag_train)*100:.1f}%)")
        print(f"      Demanda < {tau}: {n_low:,} ({n_low/len(high_flag_train)*100:.1f}%)")

        # Calcular μ_low: media de demanda en régimen bajo
        mask_low = high_flag_train == 0
        if mask_low.sum() > 0:
            self.mean_low_demand_ = float(y_train_orig[mask_low].mean())
        else:
            self.mean_low_demand_ = 0.0
        print(f"      μ_low (media demanda baja): {self.mean_low_demand_:.4f}")

        # Validar que existan ambas clases
        n_classes = len(np.unique(high_flag_train))
        if n_classes < 2:
            raise ValueError(
                f"Solo se encontró {n_classes} clase(s) con τ={tau}. "
                f"El clasificador requiere 2 clases. "
                f"Revisa el umbral demand_threshold."
            )

        # =====================================================================
        # Etapa 1: Clasificador P(demanda ≥ τ)
        # =====================================================================
        print(f"\n   Etapa 1: Entrenando clasificador ({self.backend})...")
        self.clf_ = self._build_classifier(X_train)
        self.clf_.fit(X_train, high_flag_train)
        print(f"      Clasificador entrenado")

        # =====================================================================
        # Etapa 2: Regresor E[unidades | demanda ≥ τ]
        # =====================================================================
        print(f"\n   Etapa 2: Entrenando regresor ({self.backend})...")
        mask_high = high_flag_train == 1
        self.reg_ = self._build_regressor(X_train)
        self.reg_.fit(X_train[mask_high], y_train_arr[mask_high])
        print(f"      Regresor entrenado (sobre {mask_high.sum():,} registros con demanda ≥ {tau})")

        # =====================================================================
        # Calibración de threshold
        # =====================================================================
        if self.calibrate and X_val is not None and y_val is not None:
            print(f"\n   Calibrando threshold en validación...")
            y_val_arr = y_val.values if hasattr(y_val, "values") else np.asarray(y_val)
            y_val_orig = np.expm1(y_val_arr)
            high_flag_val = (y_val_orig >= tau).astype(int)

            y_prob_val = self._predict_proba(X_val)
            cal = calibrate_threshold(high_flag_val, y_prob_val, method="f1")
            self.threshold = cal["threshold"]
            self.calibration_info_ = cal
            print(f"      Threshold óptimo: {self.threshold:.3f}")
            print(f"        F1={cal['f1']:.3f}, Precision={cal['precision']:.3f}, Recall={cal['recall']:.3f}")
        else:
            print(f"\n   Sin calibración — usando threshold fijo: {self.threshold:.3f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicción final del hurdle model.

        P(demanda ≥ τ) × E[unidades | demanda ≥ τ] + (1 − P(demanda ≥ τ)) × μ_low

        Returns:
            Array con unidades predichas en escala original (no log).
        """
        p_high = self._predict_proba(X)
        y_log = self._predict_regression(X)
        y_cond = np.expm1(y_log)
        y_cond = np.maximum(y_cond, 0)

        y_pred = p_high * y_cond + (1 - p_high) * self.mean_low_demand_
        return np.maximum(y_pred, 0)

    def predict_classification(self, X: pd.DataFrame) -> np.ndarray:
        """Predicción binaria usando el threshold calibrado."""
        p_high = self._predict_proba(X)
        return (p_high >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades de demanda ≥ τ."""
        return self._predict_proba(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df_context: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Evaluación completa: métricas de regresión + clasificación.

        Args:
            X: Features.
            y: Target en log1p(unidades).
            df_context: DataFrame con contexto (precio, etc.) para WMAPE_revenue.

        Returns:
            Dict con métricas de regresión y clasificación.
        """
        tau = self.demand_threshold
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        y_true = np.expm1(y_arr)
        y_pred = self.predict(X)

        # Métricas de regresión
        extras = {}
        if df_context is not None:
            extras["WMAPE_revenue"] = wmape_revenue(df_context, y_true, y_pred)

        reg_metrics = regression_report(y_true, y_pred, extras=extras)

        # Métricas de clasificación (demanda ≥ τ vs < τ)
        high_flag_true = (y_true >= tau).astype(int)
        high_flag_pred = self.predict_classification(X)
        p_high = self._predict_proba(X)

        clf_metrics = {
            "clf_threshold": self.threshold,
            "demand_threshold": tau,
            "clf_precision": float(precision_score(high_flag_true, high_flag_pred, zero_division=0)),
            "clf_recall": float(recall_score(high_flag_true, high_flag_pred, zero_division=0)),
            "clf_f1": float(f1_score(high_flag_true, high_flag_pred, zero_division=0)),
            "clf_prob_mean_high": float(p_high[high_flag_true == 1].mean()) if high_flag_true.sum() > 0 else 0.0,
            "clf_prob_mean_low": float(p_high[high_flag_true == 0].mean()) if (high_flag_true == 0).sum() > 0 else 0.0,
        }

        # Métricas de predicción de baja demanda
        pred_low = (y_pred < tau)
        true_low = (y_true < tau)
        tp = int(((pred_low) & (true_low)).sum())
        fp = int(((pred_low) & (~true_low)).sum())
        fn = int(((~pred_low) & (true_low)).sum())
        tn = int(((~pred_low) & (~true_low)).sum())
        clf_metrics["low_demand_tp"] = tp
        clf_metrics["low_demand_fp"] = fp
        clf_metrics["low_demand_fn"] = fn
        clf_metrics["low_demand_tn"] = tn
        clf_metrics["low_demand_precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        clf_metrics["low_demand_recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        clf_metrics["low_demand_f1"] = (
            float(2 * clf_metrics["low_demand_precision"] * clf_metrics["low_demand_recall"]
                  / (clf_metrics["low_demand_precision"] + clf_metrics["low_demand_recall"]))
            if (clf_metrics["low_demand_precision"] + clf_metrics["low_demand_recall"]) > 0 else 0.0
        )

        # Legacy alias for comparison table compatibility
        clf_metrics["zero_f1"] = clf_metrics["low_demand_f1"]

        return {**reg_metrics, **clf_metrics}

    def save(self, path_dir: str) -> None:
        """Guarda modelo completo (clasificador, regresor, metadata) en un directorio."""
        os.makedirs(path_dir, exist_ok=True)
        joblib.dump(self.clf_, os.path.join(path_dir, "stage1_clf.pkl"))
        joblib.dump(self.reg_, os.path.join(path_dir, "stage2_reg.pkl"))

        metadata = {
            "backend": self.backend,
            "demand_threshold": self.demand_threshold,
            "threshold": self.threshold,
            "calibrate": self.calibrate,
            "mean_low_demand": self.mean_low_demand_,
            "feature_names": self.feature_names_,
            "calibration_info": (
                {k: v for k, v in self.calibration_info_.items() if k != "curve"}
                if self.calibration_info_ else None
            ),
        }
        with open(os.path.join(path_dir, "two_stage_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load(cls, path_dir: str) -> "TwoStageDemandModel":
        """Carga un modelo bietápico guardado."""
        with open(os.path.join(path_dir, "two_stage_metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        m = cls(
            backend=metadata["backend"],
            demand_threshold=metadata.get("demand_threshold", 1.0),
            threshold=metadata["threshold"],
            calibrate=metadata.get("calibrate", True),
        )
        m.clf_ = joblib.load(os.path.join(path_dir, "stage1_clf.pkl"))
        m.reg_ = joblib.load(os.path.join(path_dir, "stage2_reg.pkl"))
        m.feature_names_ = metadata.get("feature_names")
        m.calibration_info_ = metadata.get("calibration_info")
        m.mean_low_demand_ = metadata.get("mean_low_demand", 0.0)
        return m

    # =========================================================================
    # Internals
    # =========================================================================

    def _build_classifier(self, X: pd.DataFrame):
        """Construye el modelo clasificador según el backend."""
        mono = _lgb_monotone_constraints(X.columns.tolist())

        if self.backend == "lightgbm":
            import lightgbm as lgb
            params = self.clf_params or {
                "objective": "binary",
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 64,
                "n_estimators": 1000,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "device": "gpu",
            }
            params = {**params}
            params["monotone_constraints"] = mono
            return lgb.LGBMClassifier(**params)

        elif self.backend == "xgboost":
            import xgboost as xgb
            params = self.clf_params or {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "learning_rate": 0.05,
                "max_depth": 8,
                "n_estimators": 1000,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }
            params = {**params}
            params["monotone_constraints"] = _xgb_monotone_constraints_str(X.columns.tolist())
            return xgb.XGBClassifier(**params)

        raise ValueError(f"Backend no soportado: {self.backend}")

    def _build_regressor(self, X: pd.DataFrame):
        """Construye el modelo regresor según el backend."""
        mono = _lgb_monotone_constraints(X.columns.tolist())

        if self.backend == "lightgbm":
            import lightgbm as lgb
            params = self.reg_params or {
                "objective": "regression",
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 64,
                "n_estimators": 1000,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "device": "gpu",
            }
            params = {**params}
            params["monotone_constraints"] = mono
            return lgb.LGBMRegressor(**params)

        elif self.backend == "xgboost":
            import xgboost as xgb
            params = self.reg_params or {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cuda",
                "learning_rate": 0.05,
                "max_depth": 8,
                "n_estimators": 1000,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            }
            params = {**params}
            params["monotone_constraints"] = _xgb_monotone_constraints_str(X.columns.tolist())
            return xgb.XGBRegressor(**params)

        raise ValueError(f"Backend no soportado: {self.backend}")

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Obtiene probabilidades del clasificador."""
        proba = self.clf_.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba

    def _predict_regression(self, X: pd.DataFrame) -> np.ndarray:
        """Obtiene predicción del regresor (en log1p)."""
        return self.reg_.predict(X)
