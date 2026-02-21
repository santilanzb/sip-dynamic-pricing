"""
Estudio de ablación: impacto de features de competencia en el modelo de demanda.

Compara:
  Model A (baseline): LightGBM bietápico SIN features de competencia
  Model B (competition): LightGBM bietápico CON 7 features de competencia

Ambos modelos usan los mismos hiperparámetros (optimizados en Phase 4),
mismos splits temporales, y misma semilla.

Decision rule:
  - WMAPE mejora ≥ 0.5pp → Adoptar Model B
  - WMAPE mejora < 0.5pp → Mantener Model A (insensibilidad)
  - WMAPE empeora → Noise sintético degrada señal

Uso:
    python -m src.competition.ablation_study
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.two_stage import TwoStageDemandModel, _lgb_monotone_constraints
from src.utils.metrics import regression_report, group_metrics, wmape, mase_from_train_series

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output/competition")
FEATURES_PATH = Path("data/processed/features.parquet")

DEMAND_THRESHOLD = 1.0
SEED = 42

# Competition feature names (the 7 added in Phase 6)
COMPETITION_FEATURES = [
    "indice_gama_cat",
    "indice_plansuarez_cat",
    "competitividad_precio",
    "gap_precio_max_comp",
    "indice_mercado",
    "presion_competitiva",
    "volatilidad_mercado_7d",
]

# Best LightGBM params from Phase 4 Optuna optimization
BEST_LGBM_PARAMS = {
    "learning_rate": 0.03908294667479063,
    "num_leaves": 106,
    "min_child_samples": 51,
    "subsample": 0.6023052293389355,
    "colsample_bytree": 0.6547235997514183,
    "reg_alpha": 7.708035943595551,
    "reg_lambda": 3.6178661323367485e-07,
}

EXCLUDE_COLS = [
    "fecha", "producto_id", "sucursal_id", "target", "unidades",
    "ingreso_usd", "costo_usd", "margen_usd", "clase", "tasa_bcv", "rotacion",
]


def prepare_data(df, exclude_features=None):
    """Prepara features y target, opcionalmente excluyendo features específicas."""
    all_exclude = EXCLUDE_COLS.copy()
    if exclude_features:
        all_exclude.extend(exclude_features)

    feature_cols = [c for c in df.columns if c not in all_exclude]
    X = df[feature_cols].copy()
    y = df["target"].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.Categorical(X[col]).codes

    X = X.fillna(-999)
    return X, y, feature_cols


def build_model_params(feature_names):
    """Builds full LightGBM params from tuned hyperparameters."""
    mono = _lgb_monotone_constraints(feature_names)

    clf_params = {
        "objective": "binary", "device": "gpu",
        "n_estimators": 1000, "random_state": SEED, "n_jobs": -1, "verbose": -1,
        "monotone_constraints": mono,
        **BEST_LGBM_PARAMS,
    }
    reg_params = {
        "objective": "regression", "device": "gpu",
        "n_estimators": 1000, "random_state": SEED, "n_jobs": -1, "verbose": -1,
        "monotone_constraints": mono,
        **BEST_LGBM_PARAMS,
    }
    return clf_params, reg_params


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       test_df, train_df, feature_names, model_name):
    """Train a two-stage model and evaluate comprehensively."""
    clf_params, reg_params = build_model_params(feature_names)

    print(f"\n   Training {model_name}...")
    print(f"   Features: {len(feature_names)}")

    t0 = time.time()
    model = TwoStageDemandModel(
        backend="lightgbm",
        clf_params=clf_params,
        reg_params=reg_params,
        demand_threshold=DEMAND_THRESHOLD,
        calibrate=True,
    )
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0

    # Predict on test
    y_test_orig = np.expm1(y_test.values)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    # Full metrics
    metrics = regression_report(y_test_orig, y_pred)

    # MASE
    try:
        metrics["MASE"] = mase_from_train_series(
            train_df, test_df, y_test_orig, y_pred,
            group_cols=["producto_id", "sucursal_id"],
        )
    except Exception:
        metrics["MASE"] = float("nan")

    metrics["train_time_sec"] = train_time

    # Metrics by category
    by_cat = group_metrics(test_df, y_test_orig, y_pred, "clase")
    # Strip whitespace from clase values
    by_cat["clase"] = by_cat["clase"].astype(str).str.strip()

    # Metrics by sucursal
    by_suc = group_metrics(test_df, y_test_orig, y_pred, "sucursal_id")

    print(f"   {model_name} results:")
    print(f"     WMAPE: {metrics['WMAPE']:.2f}%")
    print(f"     SMAPE: {metrics['SMAPE']:.2f}%")
    print(f"     MAE:   {metrics['MAE']:.2f}")
    print(f"     R²:    {metrics['R2']:.4f}")
    print(f"     MASE:  {metrics['MASE']:.4f}")
    print(f"     Time:  {train_time:.1f}s")

    print(f"\n   By category:")
    for _, row in by_cat.iterrows():
        print(f"     {row['clase']}: WMAPE={row['WMAPE']:.2f}%, MAE={row['MAE']:.2f}")

    return {
        "metrics": metrics,
        "by_category": by_cat.to_dict(orient="records"),
        "by_sucursal": by_suc.to_dict(orient="records"),
        "model": model,
        "y_pred": y_pred,
        "feature_names": feature_names,
    }


def run_ablation():
    """Execute the full ablation study."""
    print("=" * 80)
    print("ESTUDIO DE ABLACIÓN — FEATURES DE COMPETENCIA")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n📂 Cargando datos...")
    df = pd.read_parquet(FEATURES_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"   Registros: {len(df):,}")

    # Check competition features exist
    comp_present = [c for c in COMPETITION_FEATURES if c in df.columns]
    comp_missing = [c for c in COMPETITION_FEATURES if c not in df.columns]
    if comp_missing:
        raise ValueError(f"Missing competition features: {comp_missing}")
    print(f"   Competition features presentes: {len(comp_present)}")

    # ── Temporal split ────────────────────────────────────────────────
    train_df = df[df["fecha"] <= "2024-12-31"]
    val_df = df[(df["fecha"] > "2024-12-31") & (df["fecha"] <= "2025-06-30")]
    test_df = df[df["fecha"] > "2025-06-30"]

    print(f"\n   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # ── Model A: WITHOUT competition features ─────────────────────────
    print("\n" + "=" * 80)
    print("MODEL A: BASELINE (sin features de competencia)")
    print("=" * 80)

    X_train_A, y_train_A, feat_A = prepare_data(train_df, exclude_features=COMPETITION_FEATURES)
    X_val_A, y_val_A, _ = prepare_data(val_df, exclude_features=COMPETITION_FEATURES)
    X_test_A, y_test_A, _ = prepare_data(test_df, exclude_features=COMPETITION_FEATURES)

    result_A = train_and_evaluate(
        X_train_A, y_train_A, X_val_A, y_val_A, X_test_A, y_test_A,
        test_df, train_df, feat_A, "Model_A (baseline)",
    )

    # ── Model B: WITH competition features ────────────────────────────
    print("\n" + "=" * 80)
    print("MODEL B: COMPETITION (con 7 features de competencia)")
    print("=" * 80)

    X_train_B, y_train_B, feat_B = prepare_data(train_df)
    X_val_B, y_val_B, _ = prepare_data(val_df)
    X_test_B, y_test_B, _ = prepare_data(test_df)

    result_B = train_and_evaluate(
        X_train_B, y_train_B, X_val_B, y_val_B, X_test_B, y_test_B,
        test_df, train_df, feat_B, "Model_B (competition)",
    )

    # ── Comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARACIÓN ABLACIÓN")
    print("=" * 80)

    wmape_A = result_A["metrics"]["WMAPE"]
    wmape_B = result_B["metrics"]["WMAPE"]
    delta_wmape = wmape_A - wmape_B  # positive = B is better

    print(f"\n   {'Métrica':<20} {'Model A':>12} {'Model B':>12} {'Δ (A→B)':>12}")
    print("   " + "─" * 58)

    for metric in ["WMAPE", "SMAPE", "MAE", "RMSE", "R2", "MASE"]:
        a = result_A["metrics"].get(metric, float("nan"))
        b = result_B["metrics"].get(metric, float("nan"))
        d = b - a
        # For WMAPE/SMAPE/MAE/RMSE: negative delta = improvement
        # For R2: positive delta = improvement
        if metric == "R2":
            arrow = "↑" if d > 0 else "↓"
        else:
            arrow = "↓" if d < 0 else "↑"
        print(f"   {metric:<20} {a:>12.4f} {b:>12.4f} {d:>+11.4f} {arrow}")

    # By category comparison
    print(f"\n   WMAPE por categoría:")
    cat_A = {r["clase"]: r["WMAPE"] for r in result_A["by_category"]}
    cat_B = {r["clase"]: r["WMAPE"] for r in result_B["by_category"]}
    for cat in sorted(set(cat_A) | set(cat_B)):
        a = cat_A.get(cat, float("nan"))
        b = cat_B.get(cat, float("nan"))
        d = b - a
        print(f"     {cat}: {a:.2f}% → {b:.2f}% (Δ={d:+.2f}pp)")

    # ── Decision ──────────────────────────────────────────────────────
    print(f"\n   WMAPE delta: {delta_wmape:+.2f}pp")

    if delta_wmape >= 0.5:
        decision = "ADOPT_MODEL_B"
        decision_text = (
            f"✅ WMAPE mejora {delta_wmape:.2f}pp (≥0.5pp threshold). "
            f"Adoptar Model B con features de competencia."
        )
    elif delta_wmape > 0:
        decision = "MARGINAL_IMPROVEMENT"
        decision_text = (
            f"⚖️ WMAPE mejora {delta_wmape:.2f}pp (<0.5pp threshold). "
            f"Mejora marginal. Mantener Model A; features de competencia aportan "
            f"señal limitada con datos sintéticos."
        )
    elif abs(delta_wmape) < 0.1:
        decision = "NO_EFFECT"
        decision_text = (
            f"📊 WMAPE prácticamente sin cambio ({delta_wmape:+.2f}pp). "
            f"Features de competencia sintéticas no aportan ni degradan el modelo."
        )
    else:
        decision = "REJECT_MODEL_B"
        decision_text = (
            f"⚠️ WMAPE empeora {abs(delta_wmape):.2f}pp. "
            f"Noise sintético degrada señal. Mantener Model A."
        )

    print(f"\n   Decisión: {decision}")
    print(f"   {decision_text}")

    # ── Feature importance of competition features in Model B ─────────
    print("\n   Importancia de features de competencia (Model B - regresor):")
    try:
        reg_imp = result_B["model"].reg_.feature_importances_
        fi_df = pd.DataFrame({
            "feature": result_B["feature_names"],
            "importance": reg_imp,
        }).sort_values("importance", ascending=False)

        total_imp = fi_df["importance"].sum()
        comp_fi = fi_df[fi_df["feature"].isin(COMPETITION_FEATURES)]
        comp_pct = comp_fi["importance"].sum() / total_imp * 100

        print(f"     Importancia total de competition features: {comp_pct:.2f}% del total")
        for _, row in comp_fi.iterrows():
            pct = row["importance"] / total_imp * 100
            rank = fi_df.index.get_loc(row.name) + 1 if row.name in fi_df.index else "?"
            print(f"     #{rank}: {row['feature']:<30s} {row['importance']:>8.0f} ({pct:.2f}%)")

        fi_df.to_csv(OUTPUT_DIR / "model_b_feature_importance.csv", index=False)
    except Exception as e:
        print(f"     Error: {e}")
        comp_pct = 0.0

    # ── Save results ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "decision": decision,
        "decision_text": decision_text,
        "delta_wmape_pp": round(delta_wmape, 4),
        "model_a": {
            "name": "LightGBM bietápico SIN competencia",
            "n_features": len(result_A["feature_names"]),
            "metrics": {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in result_A["metrics"].items()},
            "by_category": result_A["by_category"],
            "by_sucursal": result_A["by_sucursal"],
        },
        "model_b": {
            "name": "LightGBM bietápico CON competencia",
            "n_features": len(result_B["feature_names"]),
            "competition_features": COMPETITION_FEATURES,
            "competition_importance_pct": round(comp_pct, 4),
            "metrics": {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in result_B["metrics"].items()},
            "by_category": result_B["by_category"],
            "by_sucursal": result_B["by_sucursal"],
        },
        "hyperparameters": BEST_LGBM_PARAMS,
        "demand_threshold": DEMAND_THRESHOLD,
    }

    # JSON results
    with open(OUTPUT_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # CSV comparison
    rows = []
    for metric in ["WMAPE", "SMAPE", "MAE", "RMSE", "R2", "MASE", "MdAE", "MBE"]:
        a = result_A["metrics"].get(metric, float("nan"))
        b = result_B["metrics"].get(metric, float("nan"))
        rows.append({"metric": metric, "model_a": a, "model_b": b, "delta": b - a})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ablation_comparison.csv", index=False)

    print(f"\n💾 Resultados: {OUTPUT_DIR / 'ablation_results.json'}")
    print(f"💾 Comparación: {OUTPUT_DIR / 'ablation_comparison.csv'}")

    print("\n" + "=" * 80)
    print("✅ ABLACIÓN COMPLETADA")
    print("=" * 80)

    return output


if __name__ == "__main__":
    output = run_ablation()
