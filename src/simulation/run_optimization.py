"""
Runner: Simulación y Optimización de Precios Dinámicos — Fase 7.

Orquesta todo el pipeline:
1. Carga modelo bietápico + features.parquet
2. Filtra sucursales válidas y período de test
3. Ejecuta optimización (α=1.0, γ=0.1, β=0.0)
4. Calcula KPIs
5. Ejecuta análisis contrafactual (what-if, sensitivity γ, curvas D(p))
6. Genera visualizaciones (PNGs)
7. Guarda artefactos en output/simulation/

Autores: Santiago Lanz, Diego Blanco
Fecha: 2026-02-21
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

# ── project paths ───────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR = str(PROJECT / "models" / "two_stage" / "lgbm")
FEATURES_PATH = str(PROJECT / "data" / "processed" / "features.parquet")
METADATA_PATH = str(PROJECT / "models" / "two_stage" / "two_stage_training_metadata.json")
OUTPUT_DIR = str(PROJECT / "output" / "simulation")

# ── constants ───────────────────────────────────────────────────────────
VALID_BRANCHES = ["SUC001", "SUC002", "SUC003", "SUC004"]

# Conformal prediction widths (from training metadata)
CONFORMAL_Q90_WIDTH = 10.906  # half-width = width/2
CONFORMAL_Q80_WIDTH = 5.863

# Optimization weights
ALPHA = 1.0   # revenue weight
GAMMA = 0.1   # price-change penalty
LAM = 5.0     # margin-violation penalty
N_POINTS = 50  # grid resolution


def main():
    t0 = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── imports (lazy to keep startup fast) ────────────────────────────
    from src.simulation.simulator import DemandSimulator
    from src.simulation.optimizer import PriceOptimizer
    from src.simulation.kpis import compute_all_kpis, kpis_global_summary
    from src.simulation.counterfactual import (
        run_whatif_scenarios,
        sensitivity_sweep,
        estimate_demand_curves,
        generate_all_visualizations,
    )

    # =================================================================
    # 1. Cargar modelo y datos
    # =================================================================
    print("=" * 70)
    print("FASE 7 — SIMULACIÓN Y OPTIMIZACIÓN DE PRECIOS")
    print("=" * 70)

    sim = DemandSimulator.from_artifacts(
        model_dir=MODEL_DIR,
        features_path=FEATURES_PATH,
        conformal_q90=CONFORMAL_Q90_WIDTH / 2,  # half-width for ±band
        conformal_q80=CONFORMAL_Q80_WIDTH / 2,
    )

    # =================================================================
    # 2. Filtrar datos: sucursales válidas + período test
    # =================================================================
    print("\n   Cargando features.parquet...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"      Total filas: {len(df):,}")

    # Filtrar sucursales
    df = df[df["sucursal_id"].isin(VALID_BRANCHES)].copy()
    print(f"      Filas tras filtro de sucursales: {len(df):,}")

    # Período test: usar mismo split que entrenamiento (últimos ~278K rows)
    # El test set corresponde a las fechas más recientes
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    test_size = meta["test_size"]

    # Ordenar por fecha para tomar el período test
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "producto_id", "sucursal_id"]).reset_index(drop=True)

    # Test = últimas test_size filas (split temporal)
    df_test = df.tail(test_size).copy().reset_index(drop=True)
    print(f"      Filas en test set: {len(df_test):,}")
    print(f"      Período: {df_test['fecha'].min().date()} → {df_test['fecha'].max().date()}")
    print(f"      Sucursales: {sorted(df_test['sucursal_id'].unique())}")
    print(f"      Categorías: {sorted(df_test['clase'].str.strip().unique())}")
    print(f"      Productos: {df_test['producto_id'].nunique():,}")

    # =================================================================
    # 3. Optimización
    # =================================================================
    print(f"\n{'='*70}")
    print("OPTIMIZACIÓN DE PRECIOS")
    print(f"   α={ALPHA}, γ={GAMMA}, λ={LAM}, grid={N_POINTS} puntos")
    print(f"{'='*70}")

    optimizer = PriceOptimizer(
        simulator=sim,
        alpha=ALPHA,
        gamma=GAMMA,
        lam=LAM,
        n_points=N_POINTS,
    )

    t1 = time.time()
    opt_results = optimizer.optimize(df_test)
    t_opt = time.time() - t1
    print(f"   Tiempo de optimización: {t_opt:.1f}s")

    # Guardar resultados detallados
    opt_results.to_parquet(str(out / "optimization_results.parquet"), index=False)
    opt_results.to_csv(str(out / "optimization_results.csv"), index=False)
    print(f"   Guardado: optimization_results.parquet ({len(opt_results):,} filas)")

    # =================================================================
    # 4. KPIs
    # =================================================================
    print(f"\n{'='*70}")
    print("CÁLCULO DE KPIs")
    print(f"{'='*70}")

    kpis = compute_all_kpis(opt_results)
    summary = kpis_global_summary(opt_results)

    # Guardar cada KPI como CSV
    kpi_dir = out / "kpis"
    kpi_dir.mkdir(parents=True, exist_ok=True)
    for name, kdf in kpis.items():
        kdf.to_csv(str(kpi_dir / f"{name}.csv"), index=False)

    # Guardar resumen global
    with open(str(out / "kpi_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"   KPIs guardados en {kpi_dir}/")
    print(f"\n   === RESUMEN GLOBAL ===")
    for k, v in summary.items():
        print(f"      {k}: {v}")

    # =================================================================
    # 5. Análisis contrafactual
    # =================================================================
    print(f"\n{'='*70}")
    print("ANÁLISIS CONTRAFACTUAL")
    print(f"{'='*70}")

    # 5a. What-if scenarios
    print("\n   [What-If Scenarios]")
    t2 = time.time()
    whatif_df = run_whatif_scenarios(optimizer, df_test)
    whatif_df.to_csv(str(out / "whatif_scenarios.csv"), index=False)
    print(f"   Guardado: whatif_scenarios.csv ({len(whatif_df)} escenarios)")
    print(f"   Tiempo: {time.time() - t2:.1f}s")
    print(whatif_df[["scenario", "delta_revenue_pct", "delta_margin_pct"]].to_string(index=False))

    # 5b. Sensitivity sweep γ
    print("\n   [Sensitivity Sweep γ]")
    t3 = time.time()
    # Sample for speed: random 10% of test for sensitivity sweep
    n_sample = min(30_000, len(df_test))
    df_sample = df_test.sample(n=n_sample, random_state=42).reset_index(drop=True)
    print(f"      Usando muestra de {n_sample:,} filas para sweep γ")

    sens_df = sensitivity_sweep(
        simulator=sim,
        df=df_sample,
        gamma_values=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        alpha=ALPHA,
        lam=LAM,
        n_points=N_POINTS,
    )
    sens_df.to_csv(str(out / "sensitivity_gamma.csv"), index=False)
    print(f"   Guardado: sensitivity_gamma.csv")
    print(f"   Tiempo: {time.time() - t3:.1f}s")
    print(sens_df.to_string(index=False))

    # 5c. Demand curves for representative SKUs
    print("\n   [Curvas de Demanda D(p)]")
    t4 = time.time()
    curves_df = estimate_demand_curves(sim, df_test, n_skus_per_clase=3, n_points=30)
    curves_df.to_csv(str(out / "demand_curves.csv"), index=False)
    print(f"   Guardado: demand_curves.csv ({len(curves_df)} puntos)")
    print(f"   Tiempo: {time.time() - t4:.1f}s")

    # =================================================================
    # 6. Visualizaciones
    # =================================================================
    print(f"\n{'='*70}")
    print("GENERACIÓN DE VISUALIZACIONES")
    print(f"{'='*70}")

    viz_dir = str(out / "plots")
    heatmap_df = kpis.get("heatmap_data", pd.DataFrame())
    generate_all_visualizations(opt_results, curves_df, sens_df, heatmap_df, viz_dir)

    # =================================================================
    # 7. Resumen final
    # =================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("SIMULACIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"   Tiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Artefactos en: {out}/")
    print(f"   Filas optimizadas: {len(opt_results):,}")
    print(f"   Δ Revenue: {summary['delta_revenue_pct']:+.2f}%")
    print(f"   Δ Margen:  {summary['delta_margin_pct']:+.2f}%")
    print(f"   Precio promedio Δ: {summary['avg_price_change_pct']:+.2f}%")

    # Guardar metadata de la ejecución
    run_meta = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "alpha": ALPHA,
        "gamma": GAMMA,
        "lam": LAM,
        "n_points": N_POINTS,
        "test_size": len(df_test),
        "valid_branches": VALID_BRANCHES,
        "conformal_q90_halfwidth": CONFORMAL_Q90_WIDTH / 2,
        "conformal_q80_halfwidth": CONFORMAL_Q80_WIDTH / 2,
        "elapsed_seconds": round(elapsed, 1),
        "kpi_summary": summary,
    }
    with open(str(out / "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
