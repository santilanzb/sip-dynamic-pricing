"""
Cálculo de KPIs para el reporte de tesis.

A partir del output del optimizador (optimization_results), calcula 20 KPIs
factibles agrupados por categoría y/o SKU.

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-21
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def kpi_delta_revenue(df: pd.DataFrame, group_col: str = "clase") -> pd.DataFrame:
    """KPI 1: ΔRevenue por categoría."""
    g = df.groupby(group_col).agg(
        revenue_base=("revenue_base", "sum"),
        revenue_opt=("revenue_opt", "sum"),
        n=("revenue_base", "size"),
    ).reset_index()
    g["delta_revenue"] = g["revenue_opt"] - g["revenue_base"]
    g["delta_revenue_pct"] = (g["delta_revenue"] / g["revenue_base"].clip(lower=0.01)) * 100
    return g.round(2)


def kpi_delta_margin(df: pd.DataFrame, group_col: str = "clase") -> pd.DataFrame:
    """KPI 2: ΔMargin por categoría."""
    g = df.groupby(group_col).agg(
        margin_base=("margin_base", "sum"),
        margin_opt=("margin_opt", "sum"),
        margin_pct_base=("margin_pct_base", "mean"),
        margin_pct_opt=("margin_pct_opt", "mean"),
    ).reset_index()
    g["delta_margin"] = g["margin_opt"] - g["margin_base"]
    g["delta_margin_pct"] = (g["delta_margin"] / g["margin_base"].clip(lower=0.01)) * 100
    return g.round(2)


def kpi_price_change_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 3: Distribución de cambios de precio (bins)."""
    bins = [-30, -20, -10, -5, -1, 1, 5, 10, 20, 30]
    labels = [
        "-30 a -20%", "-20 a -10%", "-10 a -5%", "-5 a -1%",
        "Sin cambio", "+1 a +5%", "+5 a +10%", "+10 a +20%", "+20 a +30%",
    ]
    df = df.copy()
    df["bin"] = pd.cut(df["delta_price_pct"], bins=bins, labels=labels, include_lowest=True)
    return df.groupby(["clase", "bin"], observed=False).size().reset_index(name="count")


def kpi_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 4: Elasticidad precio-demanda por categoría."""
    g = df.groupby("clase").agg(
        elasticity_mean=("elasticity", "mean"),
        elasticity_median=("elasticity", "median"),
        elasticity_std=("elasticity", "std"),
        elasticity_p25=("elasticity", lambda x: x.quantile(0.25)),
        elasticity_p75=("elasticity", lambda x: x.quantile(0.75)),
        n=("elasticity", "size"),
    ).reset_index()
    return g.round(4)


def kpi_contribution_margin(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 5: Margen de contribución por producto (top SKUs)."""
    g = df.groupby(["producto_id", "clase"]).agg(
        revenue_opt=("revenue_opt", "sum"),
        margin_opt=("margin_opt", "sum"),
        n_days=("revenue_opt", "size"),
    ).reset_index()
    g["margin_contribution_pct"] = (g["margin_opt"] / g["revenue_opt"].clip(lower=0.01)) * 100
    return g.sort_values("margin_opt", ascending=False).round(2)


def kpi_ppv(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 6: Precio Promedio Ponderado de Venta (por volumen) por categoría."""
    df = df.copy()
    df["vol_base"] = df["demand_base"] * df["p_base"]
    df["vol_opt"] = df["demand_opt"] * df["p_optimo"]
    g = df.groupby("clase").agg(
        ppv_base=("vol_base", "sum"),
        demand_base_sum=("demand_base", "sum"),
        ppv_opt=("vol_opt", "sum"),
        demand_opt_sum=("demand_opt", "sum"),
    ).reset_index()
    g["ppv_base"] = g["ppv_base"] / g["demand_base_sum"].clip(lower=0.01)
    g["ppv_opt"] = g["ppv_opt"] / g["demand_opt_sum"].clip(lower=0.01)
    g["delta_ppv_pct"] = (g["ppv_opt"] - g["ppv_base"]) / g["ppv_base"].clip(lower=0.01) * 100
    return g[["clase", "ppv_base", "ppv_opt", "delta_ppv_pct"]].round(4)


def kpi_irp(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 7: Índice de Rentabilidad de Precios (margen_opt / margen al 30% del rango máximo)."""
    df = df.copy()
    # Margen potencial: si vendiéramos al precio máximo del rango (p_base × 1.30)
    df["margin_max_potential"] = (df["p_base"] * 1.30 - df["costo_unitario"]) * df["demand_base"]
    g = df.groupby("clase").agg(
        margin_opt=("margin_opt", "sum"),
        margin_max=("margin_max_potential", "sum"),
    ).reset_index()
    g["irp"] = (g["margin_opt"] / g["margin_max"].clip(lower=0.01)) * 100
    return g.round(2)


def kpi_price_acceptance_rate(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 8: Tasa de aceptación de precio (% donde demand_opt ≥ demand_base)."""
    df = df.copy()
    df["accepted"] = (df["demand_opt"] >= df["demand_base"] * 0.98).astype(int)  # 2% tolerancia
    g = df.groupby("clase").agg(
        n=("accepted", "size"),
        accepted=("accepted", "sum"),
    ).reset_index()
    g["acceptance_rate_pct"] = (g["accepted"] / g["n"]) * 100
    return g.round(2)


def kpi_price_conversion_rate(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 11/13: Tasa de conversión de precios (% ajustes que mejoran margen o volumen)."""
    df = df.copy()
    df["improved"] = (
        (df["margin_opt"] > df["margin_base"]) | (df["demand_opt"] > df["demand_base"])
    ).astype(int)
    g = df.groupby("clase").agg(
        n=("improved", "size"),
        improved=("improved", "sum"),
    ).reset_index()
    g["conversion_rate_pct"] = (g["improved"] / g["n"]) * 100
    return g.round(2)


def kpi_pareto_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 14: Análisis Pareto 80/20 — qué % de SKUs genera qué % de revenue."""
    sku_rev = df.groupby("producto_id")["revenue_base"].sum().sort_values(ascending=False)
    total_rev = sku_rev.sum()
    cumrev = sku_rev.cumsum() / total_rev * 100

    # Encontrar thresholds
    rows = []
    for pct in [50, 80, 90, 95]:
        n_skus = (cumrev <= pct).sum() + 1
        pct_skus = n_skus / len(sku_rev) * 100
        rows.append({
            "revenue_pct": pct,
            "n_skus": n_skus,
            "pct_skus": round(pct_skus, 2),
            "total_skus": len(sku_rev),
        })
    return pd.DataFrame(rows)


def kpi_margin_opportunity_ranking(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """KPI 16/29: Ranking de oportunidad de margen por SKU."""
    g = df.groupby(["producto_id", "clase"]).agg(
        margin_base=("margin_base", "sum"),
        margin_opt=("margin_opt", "sum"),
        revenue_base=("revenue_base", "sum"),
        revenue_opt=("revenue_opt", "sum"),
        p_base_mean=("p_base", "mean"),
        p_opt_mean=("p_optimo", "mean"),
        n_days=("margin_base", "size"),
    ).reset_index()
    g["margin_gap"] = g["margin_opt"] - g["margin_base"]
    g["margin_gap_pct"] = (g["margin_gap"] / g["margin_base"].clip(lower=0.01)) * 100
    return g.sort_values("margin_gap", ascending=False).head(top_n).round(2)


def kpi_rotation_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 19: Velocidad de rotación por categoría (días promedio entre ventas)."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    # Días únicos con venta por producto
    days_active = df.groupby(["producto_id", "clase"]).agg(
        n_days=("fecha", "nunique"),
        total_days=("fecha", lambda x: (x.max() - x.min()).days + 1),
    ).reset_index()
    days_active["rotation_days"] = days_active["total_days"] / days_active["n_days"].clip(lower=1)
    g = days_active.groupby("clase")["rotation_days"].agg(["mean", "median"]).reset_index()
    g.columns = ["clase", "rotation_days_mean", "rotation_days_median"]
    return g.round(2)


def kpi_demand_fulfillment(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 21: Índice de cumplimiento de demanda (demand_opt / demand_base)."""
    g = df.groupby("clase").agg(
        demand_base=("demand_base", "sum"),
        demand_opt=("demand_opt", "sum"),
    ).reset_index()
    g["fulfillment_index"] = g["demand_opt"] / g["demand_base"].clip(lower=0.01)
    return g.round(4)


def kpi_optimal_vs_actual_gap(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 23: Brecha precio óptimo vs actual por categoría."""
    g = df.groupby("clase").agg(
        delta_mean=("delta_price_pct", "mean"),
        delta_median=("delta_price_pct", "median"),
        delta_std=("delta_price_pct", "std"),
        pct_increase=("delta_price_pct", lambda x: (x > 1).mean() * 100),
        pct_decrease=("delta_price_pct", lambda x: (x < -1).mean() * 100),
        pct_unchanged=("delta_price_pct", lambda x: ((x >= -1) & (x <= 1)).mean() * 100),
    ).reset_index()
    return g.round(2)


def kpi_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 28: Datos para heatmap sensibilidad precio × día de semana."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    g = df.groupby(["clase", "dia_semana"]).agg(
        delta_revenue_mean=("delta_revenue_pct", "mean"),
        delta_price_mean=("delta_price_pct", "mean"),
        elasticity_mean=("elasticity", "mean"),
        n=("delta_revenue_pct", "size"),
    ).reset_index()
    return g.round(4)


def kpi_temporal_optimal(df: pd.DataFrame) -> pd.DataFrame:
    """KPI 31: Análisis temporal del precio óptimo (por mes)."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    g = df.groupby(["clase", "mes"]).agg(
        p_base_mean=("p_base", "mean"),
        p_opt_mean=("p_optimo", "mean"),
        delta_price_mean=("delta_price_pct", "mean"),
        revenue_base=("revenue_base", "sum"),
        revenue_opt=("revenue_opt", "sum"),
    ).reset_index()
    g["delta_revenue_pct"] = (
        (g["revenue_opt"] - g["revenue_base"]) / g["revenue_base"].clip(lower=0.01) * 100
    )
    return g.round(4)


def compute_all_kpis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calcula todos los KPIs factibles y retorna un dict name → DataFrame.
    """
    kpis = {}
    kpis["delta_revenue"] = kpi_delta_revenue(df)
    kpis["delta_margin"] = kpi_delta_margin(df)
    kpis["price_change_dist"] = kpi_price_change_distribution(df)
    kpis["elasticity"] = kpi_elasticity(df)
    kpis["contribution_margin"] = kpi_contribution_margin(df)
    kpis["ppv"] = kpi_ppv(df)
    kpis["irp"] = kpi_irp(df)
    kpis["price_acceptance"] = kpi_price_acceptance_rate(df)
    kpis["price_conversion"] = kpi_price_conversion_rate(df)
    kpis["pareto"] = kpi_pareto_analysis(df)
    kpis["margin_opportunity"] = kpi_margin_opportunity_ranking(df)
    kpis["optimal_vs_actual_gap"] = kpi_optimal_vs_actual_gap(df)
    kpis["heatmap_data"] = kpi_heatmap_data(df)
    kpis["demand_fulfillment"] = kpi_demand_fulfillment(df)

    if "fecha" in df.columns:
        kpis["rotation_velocity"] = kpi_rotation_velocity(df)
        kpis["temporal_optimal"] = kpi_temporal_optimal(df)

    return kpis


def kpis_global_summary(df: pd.DataFrame) -> Dict:
    """
    Resumen global de KPIs (un solo dict para JSON).
    """
    total_rev_base = df["revenue_base"].sum()
    total_rev_opt = df["revenue_opt"].sum()
    total_mar_base = df["margin_base"].sum()
    total_mar_opt = df["margin_opt"].sum()

    return {
        "total_revenue_base": round(total_rev_base, 2),
        "total_revenue_opt": round(total_rev_opt, 2),
        "delta_revenue": round(total_rev_opt - total_rev_base, 2),
        "delta_revenue_pct": round((total_rev_opt - total_rev_base) / max(total_rev_base, 0.01) * 100, 2),
        "total_margin_base": round(total_mar_base, 2),
        "total_margin_opt": round(total_mar_opt, 2),
        "delta_margin": round(total_mar_opt - total_mar_base, 2),
        "delta_margin_pct": round((total_mar_opt - total_mar_base) / max(total_mar_base, 0.01) * 100, 2),
        "avg_price_change_pct": round(df["delta_price_pct"].mean(), 2),
        "median_price_change_pct": round(df["delta_price_pct"].median(), 2),
        "pct_price_increased": round((df["delta_price_pct"] > 1).mean() * 100, 2),
        "pct_price_decreased": round((df["delta_price_pct"] < -1).mean() * 100, 2),
        "pct_price_unchanged": round(((df["delta_price_pct"] >= -1) & (df["delta_price_pct"] <= 1)).mean() * 100, 2),
        "avg_elasticity": round(df["elasticity"].mean(), 4),
        "median_elasticity": round(df["elasticity"].median(), 4),
        "n_rows": len(df),
        "n_products": df["producto_id"].nunique(),
        "n_branches": df["sucursal_id"].nunique(),
    }
