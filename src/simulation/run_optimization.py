"""
Runner v2: Simulación Multi-Escenario con Dos Fases Temporales.

Fase 1 (Test — Out-of-sample):
  Ejecuta 4 escenarios sobre el test set (Jul–Dic 2025, 278K filas):
    - Conservador  (±5%,  γ=0.5)
    - Moderado     (±10%, γ=0.3) ← recomendación principal
    - Agresivo     (±15%, γ=0.1)
    - Extremo      (±30%, γ=0.1) — techo teórico

Fase 2 (Backtest — Contrafactual histórico):
  Ejecuta el escenario Moderado sobre Oct 2023–Sep 2025 (885K filas, 23 meses).
  NOTA: incluye datos de entrenamiento; es un análisis "¿qué habría pasado?"

Artefactos por escenario:
  - optimization_results.parquet / .csv
  - kpi_summary.json + kpis/*.csv
  - plots/ (visualizaciones)

Artefactos globales:
  - phase1/scenario_comparison.csv
  - phase1/plots/scenario_comparison.png
  - phase2/plots/monthly_timeseries.png

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-21
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── project paths ───────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR = str(PROJECT / "models" / "two_stage" / "lgbm")
FEATURES_PATH = str(PROJECT / "data" / "processed" / "features.parquet")
METADATA_PATH = str(PROJECT / "models" / "two_stage" / "two_stage_training_metadata.json")
OUTPUT_DIR = str(PROJECT / "output" / "simulation")

# ── constants ───────────────────────────────────────────────────────────
VALID_BRANCHES = ["SUC001", "SUC002", "SUC003", "SUC004"]
CONFORMAL_Q90_WIDTH = 10.906
CONFORMAL_Q80_WIDTH = 5.863


# ── scenario definitions ────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """Configuración de un escenario de optimización."""
    name: str
    price_range: Tuple[float, float]
    alpha: float
    gamma: float
    lam: float
    n_points: int
    description: str


SCENARIOS = [
    ScenarioConfig(
        name="conservador",
        price_range=(0.95, 1.05),
        alpha=1.0, gamma=0.5, lam=5.0, n_points=50,
        description="±5%, γ=0.5 — Ajustes incrementales dentro de la variación normal de precios",
    ),
    ScenarioConfig(
        name="moderado",
        price_range=(0.90, 1.10),
        alpha=1.0, gamma=0.3, lam=5.0, n_points=50,
        description="±10%, γ=0.3 — Ajustes moderados, recomendación principal para implementación",
    ),
    ScenarioConfig(
        name="agresivo",
        price_range=(0.85, 1.15),
        alpha=1.0, gamma=0.1, lam=5.0, n_points=50,
        description="±15%, γ=0.1 — Límite superior operativo. Para exploración de tesis",
    ),
    ScenarioConfig(
        name="extremo",
        price_range=(0.70, 1.30),
        alpha=1.0, gamma=0.1, lam=5.0, n_points=50,
        description="±30%, γ=0.1 — Techo teórico / stress test",
    ),
]


def _get_scenario(name: str) -> ScenarioConfig:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise ValueError(f"Escenario desconocido: {name}")


# ── helpers ─────────────────────────────────────────────────────────────

def run_single_scenario(
    optimizer_cls,
    simulator,
    df: pd.DataFrame,
    config: ScenarioConfig,
    output_dir: Path,
    run_whatif: bool = False,
    run_sensitivity: bool = False,
    run_curves: bool = False,
    phase_label: str = "",
):
    """
    Ejecuta un escenario completo y guarda todos los artefactos.

    Returns:
        (opt_results DataFrame, kpi_summary dict)
    """
    from src.simulation.kpis import compute_all_kpis, kpis_global_summary
    from src.simulation.counterfactual import (
        run_whatif_scenarios,
        sensitivity_sweep,
        estimate_demand_curves,
        generate_all_visualizations,
        plot_branch_breakdown,
    )

    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    label = f"[{phase_label} / {config.name}]" if phase_label else f"[{config.name}]"

    # ── Optimización ────────────────────────────────────────────────
    print(f"\n   {label} Optimizando (rango={config.price_range}, γ={config.gamma}, grid={config.n_points})...")
    opt = optimizer_cls(
        simulator=simulator,
        alpha=config.alpha,
        gamma=config.gamma,
        lam=config.lam,
        n_points=config.n_points,
        price_range=config.price_range,
    )

    t0 = time.time()
    results = opt.optimize(df)
    elapsed = time.time() - t0
    print(f"   {label} Completado en {elapsed:.1f}s — {len(results):,} filas")

    # ── Guardar resultados ──────────────────────────────────────────
    results.to_parquet(str(out / "optimization_results.parquet"), index=False)
    results.to_csv(str(out / "optimization_results.csv"), index=False)

    # ── KPIs ────────────────────────────────────────────────────────
    kpis = compute_all_kpis(results)
    summary = kpis_global_summary(results)

    kpi_dir = out / "kpis"
    kpi_dir.mkdir(parents=True, exist_ok=True)
    for name, kdf in kpis.items():
        kdf.to_csv(str(kpi_dir / f"{name}.csv"), index=False)

    with open(str(out / "kpi_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # ── Per-branch breakdown ────────────────────────────────────────
    branch_stats = results.groupby("sucursal_id").agg(
        revenue_base=("revenue_base", "sum"),
        revenue_opt=("revenue_opt", "sum"),
        margin_base=("margin_base", "sum"),
        margin_opt=("margin_opt", "sum"),
        avg_delta_price=("delta_price_pct", "mean"),
        median_delta_price=("delta_price_pct", "median"),
        avg_elasticity=("elasticity", "mean"),
        n_rows=("revenue_base", "size"),
        n_products=("producto_id", "nunique"),
    ).reset_index()
    branch_stats["delta_revenue_pct"] = (
        (branch_stats["revenue_opt"] - branch_stats["revenue_base"])
        / branch_stats["revenue_base"].clip(lower=0.01) * 100
    )
    branch_stats["delta_margin_pct"] = (
        (branch_stats["margin_opt"] - branch_stats["margin_base"])
        / branch_stats["margin_base"].clip(lower=0.01) * 100
    )
    branch_stats.round(2).to_csv(str(out / "branch_breakdown.csv"), index=False)

    # ── Per-category breakdown ──────────────────────────────────────
    clase_stats = results.groupby("clase").agg(
        revenue_base=("revenue_base", "sum"),
        revenue_opt=("revenue_opt", "sum"),
        margin_base=("margin_base", "sum"),
        margin_opt=("margin_opt", "sum"),
        demand_base=("demand_base", "sum"),
        demand_opt=("demand_opt", "sum"),
        avg_delta_price=("delta_price_pct", "mean"),
        avg_elasticity=("elasticity", "mean"),
        n_rows=("revenue_base", "size"),
    ).reset_index()
    clase_stats["delta_revenue_pct"] = (
        (clase_stats["revenue_opt"] - clase_stats["revenue_base"])
        / clase_stats["revenue_base"].clip(lower=0.01) * 100
    )
    clase_stats["delta_margin_pct"] = (
        (clase_stats["margin_opt"] - clase_stats["margin_base"])
        / clase_stats["margin_base"].clip(lower=0.01) * 100
    )
    clase_stats["fulfillment"] = (
        clase_stats["demand_opt"] / clase_stats["demand_base"].clip(lower=0.01)
    )
    clase_stats.round(4).to_csv(str(out / "clase_breakdown.csv"), index=False)

    # ── Visualizaciones por escenario ───────────────────────────────
    viz_dir = out / "plots"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Curves + sensitivity only when requested (Phase 1 moderado)
    curves_df = pd.DataFrame()
    sens_df = pd.DataFrame()
    heatmap_df = kpis.get("heatmap_data", pd.DataFrame())

    if run_curves:
        print(f"   {label} Estimando curvas de demanda...")
        curves_df = estimate_demand_curves(simulator, df, n_skus_per_clase=3, n_points=30)
        curves_df.to_csv(str(out / "demand_curves.csv"), index=False)

    if run_sensitivity:
        print(f"   {label} Sensitivity sweep γ...")
        n_sample = min(30_000, len(df))
        df_sample = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        sens_df = sensitivity_sweep(
            simulator=simulator, df=df_sample,
            gamma_values=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            alpha=config.alpha, lam=config.lam, n_points=config.n_points,
        )
        sens_df.to_csv(str(out / "sensitivity_gamma.csv"), index=False)

    if run_whatif:
        print(f"   {label} What-if scenarios...")
        whatif_df = run_whatif_scenarios(opt, df)
        whatif_df.to_csv(str(out / "whatif_scenarios.csv"), index=False)

    generate_all_visualizations(results, curves_df, sens_df, heatmap_df, str(viz_dir))
    plot_branch_breakdown(results, str(viz_dir / "branch_breakdown.png"),
                          scenario_name=f"{config.name.capitalize()} ({phase_label})")

    # ── Scenario metadata ───────────────────────────────────────────
    meta = {
        "scenario": config.name,
        "description": config.description,
        "phase": phase_label,
        "price_range": list(config.price_range),
        "alpha": config.alpha,
        "gamma": config.gamma,
        "lam": config.lam,
        "n_points": config.n_points,
        "n_rows": len(results),
        "n_products": int(results["producto_id"].nunique()),
        "n_branches": int(results["sucursal_id"].nunique()),
        "date_min": str(results["fecha"].min()) if "fecha" in results.columns else None,
        "date_max": str(results["fecha"].max()) if "fecha" in results.columns else None,
        "elapsed_seconds": round(elapsed, 1),
        "kpi_summary": summary,
    }
    with open(str(out / "scenario_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    return results, summary


def build_comparison_table(scenario_summaries: Dict[str, dict]) -> pd.DataFrame:
    """Construye tabla comparativa de escenarios."""
    rows = []
    for name, s in scenario_summaries.items():
        rows.append({
            "scenario": name,
            "delta_revenue_pct": s["delta_revenue_pct"],
            "delta_margin_pct": s["delta_margin_pct"],
            "total_revenue_base": s["total_revenue_base"],
            "total_revenue_opt": s["total_revenue_opt"],
            "delta_revenue_usd": s["total_revenue_opt"] - s["total_revenue_base"],
            "total_margin_base": s["total_margin_base"],
            "total_margin_opt": s["total_margin_opt"],
            "delta_margin_usd": s["total_margin_opt"] - s["total_margin_base"],
            "avg_price_change_pct": s["avg_price_change_pct"],
            "median_price_change_pct": s["median_price_change_pct"],
            "avg_abs_price_change_pct": abs(s["avg_price_change_pct"]),
            "pct_price_increased": s["pct_price_increased"],
            "pct_price_decreased": s["pct_price_decreased"],
            "pct_price_unchanged": s["pct_price_unchanged"],
            "avg_elasticity": s["avg_elasticity"],
            "n_rows": s["n_rows"],
            "n_products": s["n_products"],
        })
    return pd.DataFrame(rows).round(4)


# ── main ────────────────────────────────────────────────────────────────

def main():
    t_global = time.time()

    # Lazy imports
    from src.simulation.simulator import DemandSimulator
    from src.simulation.optimizer import PriceOptimizer
    from src.simulation.counterfactual import (
        plot_scenario_comparison,
        plot_scenario_by_clase,
        plot_phase2_monthly,
    )

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 0. Cargar modelo y datos
    # =================================================================
    print("=" * 72)
    print("  FASE 7 — SIMULACIÓN MULTI-ESCENARIO (v2)")
    print("=" * 72)

    sim = DemandSimulator.from_artifacts(
        model_dir=MODEL_DIR,
        features_path=FEATURES_PATH,
        conformal_q90=CONFORMAL_Q90_WIDTH / 2,
        conformal_q80=CONFORMAL_Q80_WIDTH / 2,
    )

    print("\n   Cargando features.parquet...")
    df_all = pd.read_parquet(FEATURES_PATH)
    df_all["fecha"] = pd.to_datetime(df_all["fecha"])
    df_all = df_all[df_all["sucursal_id"].isin(VALID_BRANCHES)].copy()
    df_all = df_all.sort_values(["fecha", "producto_id", "sucursal_id"]).reset_index(drop=True)
    print(f"      Total filas (4 sucursales): {len(df_all):,}")
    print(f"      Rango: {df_all['fecha'].min().date()} → {df_all['fecha'].max().date()}")

    # Training metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    test_size = meta["test_size"]

    # =================================================================
    # 1. FASE 1 — Test set (out-of-sample), 4 escenarios
    # =================================================================
    print(f"\n{'='*72}")
    print("  FASE 1 — TEST SET (OUT-OF-SAMPLE)")
    print(f"{'='*72}")

    df_test = df_all.tail(test_size).copy().reset_index(drop=True)
    print(f"   Filas: {len(df_test):,}")
    print(f"   Período: {df_test['fecha'].min().date()} → {df_test['fecha'].max().date()}")
    print(f"   Productos: {df_test['producto_id'].nunique():,}")
    print(f"   Sucursales: {sorted(df_test['sucursal_id'].unique())}")
    print(f"   Categorías: {sorted(df_test['clase'].str.strip().unique())}")

    phase1_dir = out_root / "phase1"
    phase1_summaries = {}
    phase1_clase_rows = []

    for sc in SCENARIOS:
        is_moderado = (sc.name == "moderado")
        sc_dir = phase1_dir / sc.name

        results, summary = run_single_scenario(
            optimizer_cls=PriceOptimizer,
            simulator=sim,
            df=df_test,
            config=sc,
            output_dir=sc_dir,
            run_whatif=is_moderado,
            run_sensitivity=is_moderado,
            run_curves=is_moderado,
            phase_label="Fase 1",
        )
        phase1_summaries[sc.name] = summary

        # Print summary
        print(f"\n   --- {sc.name.upper()} ---")
        print(f"       ΔRevenue: {summary['delta_revenue_pct']:+.2f}%  "
              f"(USD {summary['total_revenue_opt'] - summary['total_revenue_base']:+,.0f})")
        print(f"       ΔMargen:  {summary['delta_margin_pct']:+.2f}%  "
              f"(USD {summary['total_margin_opt'] - summary['total_margin_base']:+,.0f})")
        print(f"       Precio Δ promedio: {summary['avg_price_change_pct']:+.2f}%")
        print(f"       % subida / bajada / sin cambio: "
              f"{summary['pct_price_increased']:.1f}% / "
              f"{summary['pct_price_decreased']:.1f}% / "
              f"{summary['pct_price_unchanged']:.1f}%")

        # Collect per-clase data for cross-scenario viz
        clase_df = pd.read_csv(str(sc_dir / "clase_breakdown.csv"))
        for _, row in clase_df.iterrows():
            phase1_clase_rows.append({
                "scenario": sc.name.capitalize(),
                "clase": row["clase"],
                "clase_label": str(row["clase"]).strip(),
                "delta_revenue_pct": row["delta_revenue_pct"],
                "delta_margin_pct": row["delta_margin_pct"],
            })

    # ── Cross-scenario comparison ───────────────────────────────────
    print(f"\n{'='*72}")
    print("  COMPARACIÓN DE ESCENARIOS (FASE 1)")
    print(f"{'='*72}")

    comp_df = build_comparison_table(phase1_summaries)
    comp_df.to_csv(str(phase1_dir / "scenario_comparison.csv"), index=False)

    print(comp_df[[
        "scenario", "delta_revenue_pct", "delta_margin_pct",
        "avg_price_change_pct", "pct_price_increased", "pct_price_unchanged",
    ]].to_string(index=False))

    # Cross-scenario visualizations
    p1_plots = phase1_dir / "plots"
    p1_plots.mkdir(parents=True, exist_ok=True)
    plot_scenario_comparison(comp_df, str(p1_plots / "scenario_comparison.png"))
    plot_scenario_by_clase(phase1_clase_rows, str(p1_plots / "scenario_by_clase.png"))

    # =================================================================
    # 2. FASE 2 — Backtest histórico (Oct 2023 – Sep 2025)
    # =================================================================
    print(f"\n{'='*72}")
    print("  FASE 2 — BACKTEST HISTÓRICO (Oct 2023 – Sep 2025)")
    print(f"{'='*72}")

    mask_p2 = (df_all["fecha"] >= "2023-10-01") & (df_all["fecha"] <= "2025-09-30")
    df_p2 = df_all[mask_p2].copy().reset_index(drop=True)
    print(f"   Filas: {len(df_p2):,}")
    print(f"   Período: {df_p2['fecha'].min().date()} → {df_p2['fecha'].max().date()}")
    print(f"   Meses: {df_p2['fecha'].dt.to_period('M').nunique()}")
    print(f"   Productos: {df_p2['producto_id'].nunique():,}")
    print(f"   Sucursales: {sorted(df_p2['sucursal_id'].unique())}")
    print(f"   NOTA: Incluye datos de entrenamiento — análisis contrafactual")

    sc_moderado = _get_scenario("moderado")
    phase2_dir = out_root / "phase2" / "moderado"

    p2_results, p2_summary = run_single_scenario(
        optimizer_cls=PriceOptimizer,
        simulator=sim,
        df=df_p2,
        config=sc_moderado,
        output_dir=phase2_dir,
        run_whatif=True,
        run_sensitivity=False,
        run_curves=False,
        phase_label="Fase 2",
    )

    print(f"\n   --- FASE 2 MODERADO ---")
    print(f"       ΔRevenue: {p2_summary['delta_revenue_pct']:+.2f}%  "
          f"(USD {p2_summary['total_revenue_opt'] - p2_summary['total_revenue_base']:+,.0f})")
    print(f"       ΔMargen:  {p2_summary['delta_margin_pct']:+.2f}%  "
          f"(USD {p2_summary['total_margin_opt'] - p2_summary['total_margin_base']:+,.0f})")

    # Phase 2 monthly time series
    p2_plots = out_root / "phase2" / "plots"
    p2_plots.mkdir(parents=True, exist_ok=True)
    monthly_df = plot_phase2_monthly(p2_results, str(p2_plots / "monthly_timeseries.png"))
    if monthly_df is not None:
        monthly_df["mes_str"] = monthly_df["mes"].astype(str)
        monthly_df.drop(columns=["mes"], errors="ignore").to_csv(
            str(out_root / "phase2" / "monthly_breakdown.csv"), index=False
        )

    # =================================================================
    # 3. Resumen final
    # =================================================================
    elapsed_total = time.time() - t_global
    print(f"\n{'='*72}")
    print("  SIMULACIÓN COMPLETADA")
    print(f"{'='*72}")
    print(f"   Tiempo total: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"   Artefactos en: {out_root}/")
    print(f"\n   Fase 1 (Test, {len(df_test):,} filas):")
    for name, s in phase1_summaries.items():
        print(f"      {name:14s}  ΔRev={s['delta_revenue_pct']:+6.2f}%  "
              f"ΔMar={s['delta_margin_pct']:+6.2f}%  "
              f"ΔP={s['avg_price_change_pct']:+5.2f}%")
    print(f"\n   Fase 2 (Backtest, {len(df_p2):,} filas):")
    print(f"      moderado       ΔRev={p2_summary['delta_revenue_pct']:+6.2f}%  "
          f"ΔMar={p2_summary['delta_margin_pct']:+6.2f}%  "
          f"ΔP={p2_summary['avg_price_change_pct']:+5.2f}%")

    # Global metadata
    run_meta = {
        "version": "2.0",
        "timestamp": pd.Timestamp.now().isoformat(),
        "elapsed_total_seconds": round(elapsed_total, 1),
        "valid_branches": VALID_BRANCHES,
        "conformal_q90_halfwidth": CONFORMAL_Q90_WIDTH / 2,
        "conformal_q80_halfwidth": CONFORMAL_Q80_WIDTH / 2,
        "scenarios": [
            {
                "name": sc.name,
                "price_range": list(sc.price_range),
                "gamma": sc.gamma,
                "alpha": sc.alpha,
                "lam": sc.lam,
                "description": sc.description,
            }
            for sc in SCENARIOS
        ],
        "phase1": {
            "date_range": f"{df_test['fecha'].min().date()} to {df_test['fecha'].max().date()}",
            "n_rows": len(df_test),
            "n_products": int(df_test["producto_id"].nunique()),
            "summaries": phase1_summaries,
        },
        "phase2": {
            "scenario": "moderado",
            "date_range": f"{df_p2['fecha'].min().date()} to {df_p2['fecha'].max().date()}",
            "n_rows": len(df_p2),
            "n_products": int(df_p2["producto_id"].nunique()),
            "n_months": int(df_p2["fecha"].dt.to_period("M").nunique()),
            "note": "Incluye datos de entrenamiento — análisis contrafactual histórico",
            "summary": p2_summary,
        },
    }
    with open(str(out_root / "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
