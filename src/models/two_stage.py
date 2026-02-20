"""
Modelo bi-etápico para demanda: (1) probabilidad de venta>0 (clasificación),
(2) unidades condicionadas a venta>0 (regresión). Predicción final = p_venta * max(0, unidades_pred).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from src.utils.metrics import regression_report, wmape_revenue


def _lgb_monotone_constraints_list(columns: List[str]) -> List[int]:
    return [-1 if c == "precio_unitario_usd" else 0 for c in columns]


@dataclass
class TwoStageDemandModel:
    clf_params: Optional[Dict] = None
    reg_params: Optional[Dict] = None
    threshold: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TwoStageDemandModel":
        y = y.values if hasattr(y, "values") else y
        y_orig = np.expm1(y)  # volver a escala original
        sale_flag = (y_orig > 0).astype(int)

        clf_params = self.clf_params or {
            "objective": "binary",
            "device": "gpu",
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 64,
            "random_state": 42,
            "verbose": -1,
        }
        mono = _lgb_monotone_constraints_list(X.columns.tolist())
        clf_params["monotone_constraints"] = mono
        self.clf_ = lgb.LGBMClassifier(**clf_params)
        self.clf_.fit(X, sale_flag)

        reg_params = self.reg_params or {
            "objective": "regression",
            "device": "gpu",
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 64,
            "random_state": 42,
            "verbose": -1,
        }
        reg_params["monotone_constraints"] = mono
        self.reg_ = lgb.LGBMRegressor(**reg_params)
        mask_pos = sale_flag == 1
        self.reg_.fit(X[mask_pos], y[mask_pos])  # en log1p
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_sale = self.clf_.predict_proba(X)[:, 1]
        y_log = self.reg_.predict(X)
        y_cond = np.expm1(y_log)
        return np.maximum(0, p_sale * y_cond)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, df_context: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        y_true = np.expm1(y.values if hasattr(y, "values") else y)
        y_pred = self.predict(X)
        extra = {}
        if df_context is not None:
            extra["WMAPE_revenue"] = wmape_revenue(df_context, y_true, y_pred)
        return regression_report(y_true, y_pred, extras=extra)

    def save(self, path_dir: str) -> None:
        import os
        os.makedirs(path_dir, exist_ok=True)
        joblib.dump(self.clf_, f"{path_dir}/stage1_clf.pkl")
        joblib.dump(self.reg_, f"{path_dir}/stage2_reg.pkl")

    @classmethod
    def load(cls, path_dir: str) -> "TwoStageDemandModel":
        import os
        m = cls()
        m.clf_ = joblib.load(f"{path_dir}/stage1_clf.pkl")
        m.reg_ = joblib.load(f"{path_dir}/stage2_reg.pkl")
        return m