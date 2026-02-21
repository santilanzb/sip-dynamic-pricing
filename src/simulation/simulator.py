"""
Simulador de demanda para análisis what-if de precios.

Envuelve el modelo bietápico (TwoStageDemandModel) y permite:
- Modificar precio en un feature vector y re-predecir demanda
- Simular un grid de precios por fila (vectorizado)
- Estimar elasticidad precio-demanda por punto

Diseño vectorizado: usa numpy broadcasting para simular N precios × M filas
en batch, evitando loops de Python.

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-21
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.two_stage import TwoStageDemandModel


# Features que cambian al variar precio (las demás se mantienen frozen)
PRICE_SENSITIVE_FEATURES = [
    "precio_unitario_usd",
    "margen_pct",
    "precio_vs_historico",
    "precio_vs_clase",
    "precio_x_finsemana",
    "precio_x_perecedero",
]

# Columnas excluidas del modelo (metadata)
EXCLUDE_COLS = [
    "fecha", "producto_id", "sucursal_id", "target", "unidades",
    "ingreso_usd", "costo_usd", "margen_usd", "clase", "tasa_bcv", "rotacion",
]


def compute_cost_map(
    df: pd.DataFrame,
    winsorize_pct: Tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """
    Calcula costo unitario por producto-sucursal como media winsorizada
    de costo_usd / unidades.

    Returns:
        DataFrame con columnas [producto_id, sucursal_id, costo_unitario_usd]
    """
    tmp = df[["producto_id", "sucursal_id", "costo_usd", "unidades"]].copy()
    tmp = tmp[tmp["unidades"] > 0].copy()
    tmp["costo_unitario"] = tmp["costo_usd"] / tmp["unidades"]

    # Winsorize por producto-sucursal
    lo, hi = winsorize_pct

    def winsorize_group(g):
        q_lo = g["costo_unitario"].quantile(lo)
        q_hi = g["costo_unitario"].quantile(hi)
        g = g[(g["costo_unitario"] >= q_lo) & (g["costo_unitario"] <= q_hi)]
        return g["costo_unitario"].mean()

    cost_map = (
        tmp.groupby(["producto_id", "sucursal_id"])
        .apply(winsorize_group, include_groups=False)
        .rename("costo_unitario_usd")
        .reset_index()
    )
    return cost_map


@dataclass
class DemandSimulator:
    """
    Simulador de demanda basado en el modelo bietápico.

    Permite simular la respuesta de demanda a cambios de precio
    manteniendo todas las demás features constantes (ceteris paribus).
    """
    model: TwoStageDemandModel
    feature_cols: List[str]
    cost_map: pd.DataFrame  # [producto_id, sucursal_id, costo_unitario_usd]
    conformal_q90: float = 0.0  # cuantil conformal para bandas 90%
    conformal_q80: float = 0.0  # cuantil conformal para bandas 80%

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str,
        features_path: str,
        conformal_q90: float = 0.0,
        conformal_q80: float = 0.0,
    ) -> "DemandSimulator":
        """
        Carga el simulador desde artefactos en disco.

        Args:
            model_dir: directorio del modelo bietápico (e.g. models/two_stage/lgbm/)
            features_path: ruta a features.parquet
            conformal_q90: cuantil conformal 90% del entrenamiento
            conformal_q80: cuantil conformal 80% del entrenamiento
        """
        print("   Cargando modelo bietápico...")
        model = TwoStageDemandModel.load(model_dir)
        feature_cols = model.feature_names_

        print("   Calculando mapa de costos...")
        df = pd.read_parquet(features_path)
        cost_map = compute_cost_map(df)
        print(f"      {len(cost_map):,} pares producto-sucursal con costo")

        return cls(
            model=model,
            feature_cols=feature_cols,
            cost_map=cost_map,
            conformal_q90=conformal_q90,
            conformal_q80=conformal_q80,
        )

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separa features del modelo de metadata contextual.

        Returns:
            (X, ctx) donde X son las features del modelo y ctx es metadata
        """
        X = df[self.feature_cols].copy()
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = pd.Categorical(X[col]).codes
        X = X.fillna(-999)

        ctx_cols = [c for c in df.columns if c not in self.feature_cols]
        ctx = df[ctx_cols].copy() if ctx_cols else pd.DataFrame(index=df.index)
        return X, ctx

    def reprice_features(
        self,
        X: pd.DataFrame,
        new_prices: np.ndarray,
        cost_units: np.ndarray,
    ) -> pd.DataFrame:
        """
        Modifica las features sensibles al precio en batch.

        Args:
            X: DataFrame con features del modelo (N filas)
            new_prices: array (N,) con nuevos precios
            cost_units: array (N,) con costos unitarios

        Returns:
            X modificado con features de precio actualizadas
        """
        X = X.copy()
        p = np.asarray(new_prices, dtype=np.float64)
        c = np.asarray(cost_units, dtype=np.float64)

        X["precio_unitario_usd"] = p
        X["margen_pct"] = np.where(p > 0, (p - c) / p * 100.0, 0.0)

        if "precio_historico_producto" in X.columns:
            hist = X["precio_historico_producto"].values
            X["precio_vs_historico"] = np.where(hist > 0, p / hist, 1.0)

        if "precio_mean_clase" in X.columns:
            clase_mean = X["precio_mean_clase"].values
            X["precio_vs_clase"] = np.where(clase_mean > 0, p / clase_mean, 1.0)

        if "es_fin_semana" in X.columns:
            X["precio_x_finsemana"] = p * X["es_fin_semana"].values

        if "es_perecedero" in X.columns:
            X["precio_x_perecedero"] = p * X["es_perecedero"].values

        return X

    def predict_demand(self, X: pd.DataFrame) -> np.ndarray:
        """Predice demanda (unidades, escala original) para un batch de features."""
        return self.model.predict(X)

    def simulate_grid(
        self,
        df: pd.DataFrame,
        n_points: int = 50,
        price_range: Tuple[float, float] = (0.70, 1.30),
    ) -> Dict[str, np.ndarray]:
        """
        Simula un grid de precios para cada fila del DataFrame.

        Para cada fila, genera n_points precios en [lo×p_base, hi×p_base],
        predice demanda en cada punto, y calcula revenue/margin.

        Args:
            df: DataFrame con features + metadata (producto_id, sucursal_id, etc.)
            n_points: número de puntos en el grid
            price_range: (factor_min, factor_max) relativo al precio base

        Returns:
            Dict con arrays 2D (n_rows × n_points):
                - prices: grid de precios
                - demands: demanda predicha
                - revenues: price × demand
                - margins: (price - cost) × demand
                - margin_pcts: (price - cost) / price × 100
        """
        X_base, ctx = self.prepare_features(df)
        n_rows = len(X_base)
        p_base = X_base["precio_unitario_usd"].values.copy()

        # Merge costo unitario
        cost_units = self._get_cost_units(df)

        # Generar grid de precios: (n_rows, n_points)
        lo, hi = price_range
        # Fracciones del grid: [lo, ..., hi]
        fracs = np.linspace(lo, hi, n_points)  # (n_points,)
        # price_grid[i, j] = p_base[i] * fracs[j]
        price_grid = p_base[:, None] * fracs[None, :]  # (n_rows, n_points)

        # Asegurar precio > costo
        price_grid = np.maximum(price_grid, cost_units[:, None] + 0.01)

        # Redondear al centavo
        price_grid = np.round(price_grid, 2)

        # Simular demanda para cada punto del grid
        demands = np.zeros((n_rows, n_points), dtype=np.float64)

        # Batch predict: para cada punto del grid, crear X modificado y predecir
        # Optimización: procesar en bloques de puntos del grid
        for j in range(n_points):
            X_mod = self.reprice_features(X_base, price_grid[:, j], cost_units)
            demands[:, j] = self.predict_demand(X_mod)

        # Calcular revenue y margin
        revenues = price_grid * demands
        margins = (price_grid - cost_units[:, None]) * demands
        margin_pcts = np.where(
            price_grid > 0,
            (price_grid - cost_units[:, None]) / price_grid * 100.0,
            0.0,
        )

        return {
            "prices": price_grid,
            "demands": demands,
            "revenues": revenues,
            "margins": margins,
            "margin_pcts": margin_pcts,
            "p_base": p_base,
            "cost_units": cost_units,
        }

    def estimate_elasticity(
        self,
        df: pd.DataFrame,
        delta: float = 0.01,
    ) -> np.ndarray:
        """
        Estima elasticidad precio-demanda por fila usando diferencias finitas.

        ε = (ΔQ/Q) / (Δp/p) = (Q(p+δ) - Q(p-δ)) / (Q(p)) / (2δ)

        Args:
            df: DataFrame con features + metadata
            delta: variación porcentual para diferencia finita (0.01 = 1%)

        Returns:
            Array (n_rows,) con elasticidad por fila
        """
        X_base, _ = self.prepare_features(df)
        p_base = X_base["precio_unitario_usd"].values.copy()
        cost_units = self._get_cost_units(df)

        # Precio +delta y -delta
        p_up = p_base * (1.0 + delta)
        p_down = p_base * (1.0 - delta)
        p_down = np.maximum(p_down, cost_units + 0.01)

        # Demanda en cada punto
        X_up = self.reprice_features(X_base, p_up, cost_units)
        X_down = self.reprice_features(X_base, p_down, cost_units)
        d_base = self.predict_demand(X_base)
        d_up = self.predict_demand(X_up)
        d_down = self.predict_demand(X_down)

        # Elasticidad arco
        dp = p_up - p_down
        dq = d_up - d_down
        elasticity = np.where(
            (d_base > 0) & (dp > 0),
            (dq / d_base) / (dp / p_base),
            0.0,
        )
        return elasticity

    def _get_cost_units(self, df: pd.DataFrame) -> np.ndarray:
        """Obtiene costo unitario por fila, mergeando con cost_map."""
        tmp = df[["producto_id", "sucursal_id"]].copy().reset_index(drop=True)
        merged = tmp.merge(self.cost_map, on=["producto_id", "sucursal_id"], how="left")
        cost = merged["costo_unitario_usd"].values

        # Fallback: si no hay costo, usar estimación desde margen_pct
        if "precio_unitario_usd" in df.columns and "margen_pct" in df.columns:
            p = df["precio_unitario_usd"].values
            m = df["margen_pct"].values / 100.0
            fallback = p * (1.0 - m)
            mask = np.isnan(cost)
            cost[mask] = fallback[mask]

        cost = np.nan_to_num(cost, nan=0.0)
        return cost
