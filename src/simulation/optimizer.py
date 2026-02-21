"""
Optimizador de precios por SKU-sucursal-día.

Sobre el grid de simulación generado por DemandSimulator, evalúa una función
objetivo con penalizaciones suaves y selecciona el precio óptimo por fila.

Función objetivo:
    score = α × Revenue - γ × Rev_base × |Δp/p_base| - λ × Rev_base × max(0, margin_min - margin_pct)

Donde:
    α: peso de ingreso (default 1.0)
    γ: penalización de cambio brusco (default 0.1)
    λ: penalización de violación de margen (default 5.0)

Restricciones hard:
    - price ∈ [0.70 × p_base, 1.30 × p_base]
    - price > costo_unitario

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-21
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.simulation.simulator import DemandSimulator


# Márgenes mínimos por clase (soft constraints)
DEFAULT_MARGIN_MINS = {
    "03CARN     ": 25.0,
    "05CHAR     ": 30.0,
    "08FRUV     ": 30.0,
}


@dataclass
class PriceOptimizer:
    """
    Optimizador de precios basado en grid search con penalizaciones suaves.
    """
    simulator: DemandSimulator
    alpha: float = 1.0      # peso de revenue
    gamma: float = 0.1      # penalización cambio brusco
    lam: float = 5.0        # penalización violación de margen
    margin_mins: Optional[Dict[str, float]] = None
    n_points: int = 50
    price_range: Tuple[float, float] = (0.70, 1.30)

    def __post_init__(self):
        if self.margin_mins is None:
            self.margin_mins = DEFAULT_MARGIN_MINS.copy()

    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimiza precios para un DataFrame completo.

        Args:
            df: DataFrame con features + metadata (producto_id, sucursal_id, clase, etc.)

        Returns:
            DataFrame con columnas de optimización por fila:
                producto_id, sucursal_id, fecha, clase,
                p_base, p_optimo, delta_price_pct,
                demand_base, demand_opt,
                revenue_base, revenue_opt, delta_revenue_pct,
                margin_base, margin_opt, margin_pct_base, margin_pct_opt,
                elasticity, score_base, score_opt
        """
        print(f"\n   Simulando grid ({self.n_points} puntos × {len(df):,} filas)...")
        grid = self.simulator.simulate_grid(df, self.n_points, self.price_range)

        print("   Calculando elasticidades...")
        elasticities = self.simulator.estimate_elasticity(df)

        print("   Evaluando función objetivo...")
        prices = grid["prices"]       # (n, k)
        demands = grid["demands"]     # (n, k)
        revenues = grid["revenues"]   # (n, k)
        margins = grid["margins"]     # (n, k)
        margin_pcts = grid["margin_pcts"]  # (n, k)
        p_base = grid["p_base"]       # (n,)
        cost_units = grid["cost_units"]  # (n,)

        n_rows = len(p_base)

        # Baseline: revenue con precio actual (primer predict del modelo)
        X_base, _ = self.simulator.prepare_features(df)
        demand_base = self.simulator.predict_demand(X_base)
        revenue_base = p_base * demand_base
        margin_base = (p_base - cost_units) * demand_base
        margin_pct_base = np.where(p_base > 0, (p_base - cost_units) / p_base * 100.0, 0.0)

        # Obtener margin_min por fila
        clase_values = df["clase"].values if "clase" in df.columns else np.full(n_rows, "")
        margin_min_arr = np.array([
            self.margin_mins.get(str(c).strip(), 0.0)
            if isinstance(c, str) else self.margin_mins.get(c, 0.0)
            for c in clase_values
        ])

        # Función objetivo: (n_rows, n_points)
        # score = α × Revenue - γ × Rev_base × |Δp/p| - λ × Rev_base × max(0, m_min - m_pct)
        delta_p_pct = np.abs(prices - p_base[:, None]) / np.maximum(p_base[:, None], 0.01)
        margin_deficit = np.maximum(margin_min_arr[:, None] - margin_pcts, 0.0)
        rev_base_2d = np.maximum(revenue_base[:, None], 0.01)

        scores = (
            self.alpha * revenues
            - self.gamma * rev_base_2d * delta_p_pct
            - self.lam * rev_base_2d * (margin_deficit / 100.0)  # normalize deficit
        )

        # Hard constraint: price > cost
        invalid = prices <= cost_units[:, None]
        scores[invalid] = -np.inf

        # Seleccionar argmax por fila
        best_idx = np.argmax(scores, axis=1)  # (n_rows,)
        row_idx = np.arange(n_rows)

        p_opt = prices[row_idx, best_idx]
        d_opt = demands[row_idx, best_idx]
        rev_opt = revenues[row_idx, best_idx]
        mar_opt = margins[row_idx, best_idx]
        mpct_opt = margin_pcts[row_idx, best_idx]
        score_opt = scores[row_idx, best_idx]

        # Score baseline (precio actual en el grid más cercano)
        base_grid_idx = np.argmin(np.abs(prices - p_base[:, None]), axis=1)
        score_base = scores[row_idx, base_grid_idx]

        # Construir resultado
        result = pd.DataFrame({
            "producto_id": df["producto_id"].values,
            "sucursal_id": df["sucursal_id"].values,
            "fecha": df["fecha"].values if "fecha" in df.columns else None,
            "clase": clase_values,
            "p_base": np.round(p_base, 2),
            "p_optimo": np.round(p_opt, 2),
            "delta_price_pct": np.round((p_opt - p_base) / np.maximum(p_base, 0.01) * 100, 2),
            "demand_base": np.round(demand_base, 4),
            "demand_opt": np.round(d_opt, 4),
            "delta_demand_pct": np.round(
                (d_opt - demand_base) / np.maximum(demand_base, 0.01) * 100, 2
            ),
            "revenue_base": np.round(revenue_base, 2),
            "revenue_opt": np.round(rev_opt, 2),
            "delta_revenue_pct": np.round(
                (rev_opt - revenue_base) / np.maximum(revenue_base, 0.01) * 100, 2
            ),
            "margin_base": np.round(margin_base, 2),
            "margin_opt": np.round(mar_opt, 2),
            "margin_pct_base": np.round(margin_pct_base, 2),
            "margin_pct_opt": np.round(mpct_opt, 2),
            "costo_unitario": np.round(cost_units, 4),
            "elasticity": np.round(elasticities, 4),
            "score_base": np.round(score_base, 2),
            "score_opt": np.round(score_opt, 2),
        })

        # Agregar bandas de confianza si hay cuantiles conformales
        if self.simulator.conformal_q90 > 0:
            result["demand_opt_lo_90"] = np.round(
                np.maximum(d_opt - self.simulator.conformal_q90, 0), 4
            )
            result["demand_opt_hi_90"] = np.round(
                d_opt + self.simulator.conformal_q90, 4
            )
        if self.simulator.conformal_q80 > 0:
            result["demand_opt_lo_80"] = np.round(
                np.maximum(d_opt - self.simulator.conformal_q80, 0), 4
            )
            result["demand_opt_hi_80"] = np.round(
                d_opt + self.simulator.conformal_q80, 4
            )

        print(f"   Optimización completa: {n_rows:,} filas")
        return result

    def optimize_whatif(
        self,
        df: pd.DataFrame,
        price_adjustments: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Simula un escenario what-if con ajustes fijos de precio por clase.

        Args:
            df: DataFrame con features + metadata
            price_adjustments: Dict clase → factor de ajuste (ej: {"03CARN": 1.10} = +10%)

        Returns:
            DataFrame con resultados del escenario what-if
        """
        X_base, _ = self.simulator.prepare_features(df)
        p_base = X_base["precio_unitario_usd"].values.copy()
        cost_units = self.simulator._get_cost_units(df)
        clase_values = df["clase"].values if "clase" in df.columns else np.full(len(df), "")

        # Aplicar ajustes por clase
        p_new = p_base.copy()
        for clase, factor in price_adjustments.items():
            mask = np.array([str(c).strip() == clase.strip() for c in clase_values])
            p_new[mask] = p_base[mask] * factor

        # Redondear y asegurar > costo
        p_new = np.round(p_new, 2)
        p_new = np.maximum(p_new, cost_units + 0.01)

        # Predecir con precios ajustados
        X_new = self.simulator.reprice_features(X_base, p_new, cost_units)
        d_base = self.simulator.predict_demand(X_base)
        d_new = self.simulator.predict_demand(X_new)

        result = pd.DataFrame({
            "producto_id": df["producto_id"].values,
            "sucursal_id": df["sucursal_id"].values,
            "clase": clase_values,
            "p_base": np.round(p_base, 2),
            "p_whatif": np.round(p_new, 2),
            "demand_base": np.round(d_base, 4),
            "demand_whatif": np.round(d_new, 4),
            "revenue_base": np.round(p_base * d_base, 2),
            "revenue_whatif": np.round(p_new * d_new, 2),
            "margin_base": np.round((p_base - cost_units) * d_base, 2),
            "margin_whatif": np.round((p_new - cost_units) * d_new, 2),
        })
        return result
