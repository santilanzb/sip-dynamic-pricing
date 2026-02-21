"""
Análisis contrafactual y visualizaciones para optimización de precios.

Genera:
- Escenarios what-if predefinidos
- Sensitivity sweep de γ (penalización de cambio)
- Visualizaciones (PNGs) para la tesis

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-21
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.optimizer import PriceOptimizer
from src.simulation.kpis import compute_all_kpis, kpis_global_summary


# =========================================================================
# What-if scenarios
# =========================================================================

WHATIF_SCENARIOS = {
    "carnes_+10%": {"03CARN": 1.10},
    "carnes_-10%": {"03CARN": 0.90},
    "fruver_+10%": {"08FRUV": 1.10},
    "fruver_-5%": {"08FRUV": 0.95},
    "charcu_+10%": {"05CHAR": 1.10},
    "all_+5%": {"03CARN": 1.05, "08FRUV": 1.05, "05CHAR": 1.05},
    "all_-5%": {"03CARN": 0.95, "08FRUV": 0.95, "05CHAR": 0.95},
    "carnes_+10%_fruver_-5%": {"03CARN": 1.10, "08FRUV": 0.95},
}


def run_whatif_scenarios(
    optimizer: PriceOptimizer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Ejecuta todos los escenarios what-if predefinidos."""
    results = []
    for name, adjustments in WHATIF_SCENARIOS.items():
        print(f"      Escenario: {name}")
        res = optimizer.optimize_whatif(df, adjustments)
        summary = {
            "scenario": name,
            "adjustments": str(adjustments),
            "revenue_base": res["revenue_base"].sum(),
            "revenue_whatif": res["revenue_whatif"].sum(),
            "margin_base": res["margin_base"].sum(),
            "margin_whatif": res["margin_whatif"].sum(),
        }
        summary["delta_revenue"] = summary["revenue_whatif"] - summary["revenue_base"]
        summary["delta_revenue_pct"] = (
            summary["delta_revenue"] / max(summary["revenue_base"], 0.01) * 100
        )
        summary["delta_margin"] = summary["margin_whatif"] - summary["margin_base"]
        summary["delta_margin_pct"] = (
            summary["delta_margin"] / max(summary["margin_base"], 0.01) * 100
        )
        results.append(summary)

    return pd.DataFrame(results).round(2)


# =========================================================================
# Sensitivity sweep
# =========================================================================

def sensitivity_sweep(
    simulator,
    df: pd.DataFrame,
    gamma_values: List[float] = None,
    alpha: float = 1.0,
    lam: float = 5.0,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Barrido de γ para ver trade-off revenue vs estabilidad de precios.
    """
    if gamma_values is None:
        gamma_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = []
    for g in gamma_values:
        print(f"      γ = {g}")
        opt = PriceOptimizer(
            simulator=simulator,
            alpha=alpha,
            gamma=g,
            lam=lam,
            n_points=n_points,
        )
        res = opt.optimize(df)
        results.append({
            "gamma": g,
            "total_revenue_opt": res["revenue_opt"].sum(),
            "total_revenue_base": res["revenue_base"].sum(),
            "delta_revenue_pct": (
                (res["revenue_opt"].sum() - res["revenue_base"].sum())
                / max(res["revenue_base"].sum(), 0.01) * 100
            ),
            "total_margin_opt": res["margin_opt"].sum(),
            "avg_abs_price_change_pct": res["delta_price_pct"].abs().mean(),
            "median_abs_price_change_pct": res["delta_price_pct"].abs().median(),
            "pct_unchanged": ((res["delta_price_pct"].abs() <= 1).mean() * 100),
        })

    return pd.DataFrame(results).round(4)


# =========================================================================
# Demand curve estimation (for representative SKUs)
# =========================================================================

def estimate_demand_curves(
    simulator,
    df: pd.DataFrame,
    n_skus_per_clase: int = 3,
    n_points: int = 30,
) -> pd.DataFrame:
    """
    Estima curvas de demanda D(p) para SKUs representativos por categoría.
    Selecciona los SKUs con más días de datos por categoría.
    """
    # Seleccionar SKUs representativos
    sku_counts = (
        df.groupby(["producto_id", "clase", "sucursal_id"])
        .size()
        .reset_index(name="n_days")
        .sort_values("n_days", ascending=False)
    )

    selected = []
    for clase in df["clase"].unique():
        top = sku_counts[sku_counts["clase"] == clase].head(n_skus_per_clase)
        selected.append(top)
    selected = pd.concat(selected)

    curves = []
    for _, row in selected.iterrows():
        mask = (
            (df["producto_id"] == row["producto_id"])
            & (df["sucursal_id"] == row["sucursal_id"])
        )
        sub = df[mask].head(1)  # Use one representative day
        if len(sub) == 0:
            continue

        grid = simulator.simulate_grid(sub, n_points=n_points)
        for j in range(n_points):
            curves.append({
                "producto_id": row["producto_id"],
                "sucursal_id": row["sucursal_id"],
                "clase": row["clase"],
                "price": grid["prices"][0, j],
                "demand": grid["demands"][0, j],
                "revenue": grid["revenues"][0, j],
                "margin": grid["margins"][0, j],
            })

    return pd.DataFrame(curves).round(4)


# =========================================================================
# Visualizations
# =========================================================================

def plot_price_change_distribution(df: pd.DataFrame, output_path: str):
    """Histograma de distribución de cambios de precio por categoría."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    clases = sorted(df["clase"].unique())

    for ax, clase in zip(axes, clases):
        sub = df[df["clase"] == clase]["delta_price_pct"]
        ax.hist(sub, bins=50, edgecolor="black", alpha=0.7, color="#2196F3")
        ax.axvline(0, color="red", linestyle="--", alpha=0.8)
        ax.set_title(f"{clase.strip()}", fontsize=14)
        ax.set_xlabel("Δ Precio (%)")
        ax.set_ylabel("Frecuencia")
        ax.text(0.95, 0.95, f"μ={sub.mean():.1f}%\nmd={sub.median():.1f}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Distribución de Cambios de Precio Recomendados", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_revenue_impact_by_clase(df: pd.DataFrame, output_path: str):
    """Barras de ΔRevenue y ΔMargin por categoría."""
    g = df.groupby("clase").agg(
        rev_base=("revenue_base", "sum"),
        rev_opt=("revenue_opt", "sum"),
        mar_base=("margin_base", "sum"),
        mar_opt=("margin_opt", "sum"),
    ).reset_index()
    g["delta_rev_pct"] = (g["rev_opt"] - g["rev_base"]) / g["rev_base"] * 100
    g["delta_mar_pct"] = (g["mar_opt"] - g["mar_base"]) / g["mar_base"] * 100
    g["clase_label"] = g["clase"].str.strip()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in g["delta_rev_pct"]]
    ax1.bar(g["clase_label"], g["delta_rev_pct"], color=colors, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Δ Revenue (%)", fontsize=14)
    ax1.set_ylabel("Cambio (%)")
    for i, v in enumerate(g["delta_rev_pct"]):
        ax1.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=11)

    colors2 = ["#4CAF50" if v >= 0 else "#F44336" for v in g["delta_mar_pct"]]
    ax2.bar(g["clase_label"], g["delta_mar_pct"], color=colors2, edgecolor="black")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Δ Margen (%)", fontsize=14)
    ax2.set_ylabel("Cambio (%)")
    for i, v in enumerate(g["delta_mar_pct"]):
        ax2.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=11)

    fig.suptitle("Impacto de la Optimización por Categoría", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_demand_curves(curves_df: pd.DataFrame, output_path: str):
    """Curvas de demanda D(p) por categoría."""
    clases = sorted(curves_df["clase"].unique())
    fig, axes = plt.subplots(1, len(clases), figsize=(6 * len(clases), 5))
    if len(clases) == 1:
        axes = [axes]

    for ax, clase in zip(axes, clases):
        sub = curves_df[curves_df["clase"] == clase]
        for pid in sub["producto_id"].unique():
            s = sub[sub["producto_id"] == pid]
            ax.plot(s["price"], s["demand"], label=str(pid)[:12], alpha=0.8)
        ax.set_title(f"{clase.strip()}", fontsize=14)
        ax.set_xlabel("Precio (USD)")
        ax.set_ylabel("Demanda (unidades)")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Curvas de Demanda Estimadas D(p)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sensitivity_gamma(sens_df: pd.DataFrame, output_path: str):
    """Frontera γ vs ΔRevenue y estabilidad de precios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sens_df["gamma"], sens_df["delta_revenue_pct"], "o-", color="#2196F3", linewidth=2)
    ax1.set_xlabel("γ (penalización cambio brusco)")
    ax1.set_ylabel("Δ Revenue (%)")
    ax1.set_title("Revenue Gain vs γ")
    ax1.grid(True, alpha=0.3)

    ax2.plot(sens_df["gamma"], sens_df["avg_abs_price_change_pct"], "o-", color="#FF9800", linewidth=2)
    ax2.set_xlabel("γ (penalización cambio brusco)")
    ax2.set_ylabel("Cambio Precio Promedio |Δp/p| (%)")
    ax2.set_title("Price Stability vs γ")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Análisis de Sensibilidad: Revenue vs Estabilidad", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(heatmap_df: pd.DataFrame, output_path: str):
    """Heatmap sensibilidad precio × día de semana por categoría."""
    days = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    clases = sorted(heatmap_df["clase"].unique())

    fig, axes = plt.subplots(1, len(clases), figsize=(6 * len(clases), 4))
    if len(clases) == 1:
        axes = [axes]

    for ax, clase in zip(axes, clases):
        sub = heatmap_df[heatmap_df["clase"] == clase].sort_values("dia_semana")
        if len(sub) < 7:
            continue
        vals = sub["delta_revenue_mean"].values[:7]
        ax.barh(range(7), vals, color=["#4CAF50" if v >= 0 else "#F44336" for v in vals])
        ax.set_yticks(range(7))
        ax.set_yticklabels(days)
        ax.set_xlabel("Δ Revenue Promedio (%)")
        ax.set_title(f"{clase.strip()}")
        ax.axvline(0, color="black", linewidth=0.5)

    fig.suptitle("Impacto de Revenue por Día de Semana", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto(df: pd.DataFrame, output_path: str):
    """Curva de Pareto 80/20."""
    sku_rev = df.groupby("producto_id")["revenue_base"].sum().sort_values(ascending=False)
    total_rev = sku_rev.sum()
    cumrev = sku_rev.cumsum() / total_rev * 100
    x = np.arange(1, len(cumrev) + 1) / len(cumrev) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, cumrev.values, color="#2196F3", linewidth=2)
    ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% Revenue")
    ax.axvline(20, color="orange", linestyle="--", alpha=0.7, label="20% SKUs")
    ax.fill_between(x, cumrev.values, alpha=0.15, color="#2196F3")
    ax.set_xlabel("% de SKUs (ordenados por revenue)")
    ax.set_ylabel("% Revenue Acumulado")
    ax.set_title("Análisis de Pareto: Concentración de Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_margin_opportunity(df: pd.DataFrame, output_path: str, top_n: int = 20):
    """Top N SKUs por oportunidad de margen."""
    g = df.groupby(["producto_id", "clase"]).agg(
        margin_gap=("margin_opt", "sum"),
        margin_base=("margin_base", "sum"),
    ).reset_index()
    g["margin_gap"] = g["margin_gap"] - g["margin_base"]
    top = g.sort_values("margin_gap", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#4CAF50" if str(c).strip() == "08FRUV" else
        "#F44336" if str(c).strip() == "03CARN" else "#FF9800"
        for c in top["clase"]
    ]
    y = range(len(top))
    ax.barh(y, top["margin_gap"].values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([str(p)[:15] for p in top["producto_id"]], fontsize=8)
    ax.set_xlabel("Δ Margen (USD)")
    ax.set_title(f"Top {top_n} SKUs: Mayor Oportunidad de Margen")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Fruver"),
        Patch(facecolor="#F44336", label="Carnes"),
        Patch(facecolor="#FF9800", label="Charcutería"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_visualizations(
    opt_results: pd.DataFrame,
    curves_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    output_dir: str,
):
    """Genera todas las visualizaciones y las guarda en output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("   Generando visualizaciones...")
    plot_price_change_distribution(opt_results, str(out / "price_change_distribution.png"))
    plot_revenue_impact_by_clase(opt_results, str(out / "revenue_impact_by_clase.png"))
    plot_pareto(opt_results, str(out / "pareto_80_20.png"))
    plot_margin_opportunity(opt_results, str(out / "margin_opportunity_ranking.png"))

    if len(curves_df) > 0:
        plot_demand_curves(curves_df, str(out / "demand_curves_by_clase.png"))

    if len(sens_df) > 0:
        plot_sensitivity_gamma(sens_df, str(out / "sensitivity_gamma.png"))

    if len(heatmap_df) > 0:
        plot_heatmap(heatmap_df, str(out / "heatmap_price_dayofweek.png"))

    print(f"   Visualizaciones guardadas en {out}/")
