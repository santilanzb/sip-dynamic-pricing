"""
Generación de datos sintéticos de competencia.

Genera índices diarios de competencia por categoría para Gama y Plan Suárez,
utilizando los coeficientes calibrados y noise AR(1) temporalmente correlacionado.

Modelo:
  indice_comp(t, cat) = Coef(cat) × (1 + ε(t))
  ε(t) = φ × ε(t-1) + η(t),  η(t) ~ N(0, σ_η)

Donde:
  φ = 0.7 (AR1 persistence)
  σ = 0.025 (target stationary std)
  σ_η = σ × √(1 - φ²) (innovation std para alcanzar σ estacionaria)

Output:
  data/processed/competition_indices.parquet — índices diarios por categoría
  output/competition/validation_report.json  — reporte de validación

Uso:
    python -m src.competition.synthetic_generator
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────
DATA_EXT = Path("data/external")
DATA_PROC = Path("data/processed")
OUTPUT_DIR = Path("output/competition")

SEED = 42
AR1_PHI = 0.7       # AR(1) persistence parameter
AR1_SIGMA = 0.025   # Target stationary standard deviation of ε

CATEGORIES = ["03CARN", "05CHAR", "08FRUV"]
COMPETITORS = ["gama", "plansuarez"]


def load_coefficients() -> dict:
    """Carga coeficientes calibrados desde JSON."""
    with open(DATA_EXT / "competition_coefficients.json", "r") as f:
        data = json.load(f)
    return data["coefficients"]


def generate_ar1_noise(
    n_days: int,
    phi: float = AR1_PHI,
    sigma: float = AR1_SIGMA,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Genera serie de ruido AR(1) estacionaria.

    ε(t) = φ × ε(t-1) + η(t), donde η ~ N(0, σ_η)
    σ_η = σ × √(1 - φ²) para que la varianza estacionaria sea σ².
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    sigma_innovation = sigma * np.sqrt(1 - phi**2)
    eps = np.zeros(n_days)
    eps[0] = rng.normal(0, sigma)  # Initialize from stationary distribution

    for t in range(1, n_days):
        eps[t] = phi * eps[t - 1] + rng.normal(0, sigma_innovation)

    return eps


def generate_competition_indices() -> pd.DataFrame:
    """
    Genera índices diarios de competencia por categoría.

    Para cada combinación (competidor, categoría), genera una serie temporal
    de índices = Coef × (1 + ε_AR1(t)).

    Returns DataFrame con columnas:
      fecha, categoria, indice_gama, indice_plansuarez
    """
    coefficients = load_coefficients()
    rng = np.random.default_rng(SEED)

    # Get date range from fact_ventas
    fv = pd.read_parquet(DATA_PROC / "fact_ventas.parquet", columns=["fecha"])
    date_min = fv["fecha"].min()
    date_max = fv["fecha"].max()
    dates = pd.date_range(date_min, date_max, freq="D")
    n_days = len(dates)

    print("=" * 70)
    print("GENERACIÓN DE ÍNDICES SINTÉTICOS DE COMPETENCIA")
    print("=" * 70)
    print(f"  Rango: {date_min.date()} — {date_max.date()} ({n_days} días)")
    print(f"  AR(1): φ={AR1_PHI}, σ={AR1_SIGMA}")
    print(f"  Seed: {SEED}")
    print()

    rows = []
    noise_series = {}  # For validation

    for cat in CATEGORIES:
        # Generate independent AR(1) noise for each competitor × category
        noise_gama = generate_ar1_noise(n_days, rng=rng)
        noise_ps = generate_ar1_noise(n_days, rng=rng)

        coef_gama = coefficients["gama"][cat]
        coef_ps = coefficients["plansuarez"][cat]

        for i, date in enumerate(dates):
            idx_gama = coef_gama * (1 + noise_gama[i])
            idx_ps = coef_ps * (1 + noise_ps[i])

            rows.append({
                "fecha": date,
                "categoria": cat,
                "indice_gama": round(idx_gama, 6),
                "indice_plansuarez": round(idx_ps, 6),
            })

        # Store for validation
        noise_series[f"gama_{cat}"] = noise_gama
        noise_series[f"plansuarez_{cat}"] = noise_ps

        print(f"  {cat}:")
        print(f"    Gama:         coef={coef_gama:.4f}, índice medio="
              f"{np.mean([coef_gama * (1 + e) for e in noise_gama]):.4f}")
        print(f"    Plan Suárez:  coef={coef_ps:.4f}, índice medio="
              f"{np.mean([coef_ps * (1 + e) for e in noise_ps]):.4f}")

    df = pd.DataFrame(rows)

    # Save
    out_path = DATA_PROC / "competition_indices.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n💾 Índices guardados: {out_path}")
    print(f"   Shape: {df.shape}")

    return df, noise_series


def validate_indices(df: pd.DataFrame, noise_series: dict) -> dict:
    """
    Validación estadística de los índices generados.

    Tests:
      1. Media del índice ≈ coeficiente calibrado
      2. Std del noise ≈ σ target (0.025)
      3. Autocorrelación lag-1 ≈ φ (0.7)
      4. KS test: noise vs N(0, σ) — distribución estacionaria
      5. Stationarity: no trend significativo
    """
    coefficients = load_coefficients()
    report = {"timestamp": datetime.now().isoformat(), "tests": {}}

    print("\n" + "=" * 70)
    print("VALIDACIÓN DE ÍNDICES SINTÉTICOS")
    print("=" * 70)

    for comp in COMPETITORS:
        for cat in CATEGORIES:
            key = f"{comp}_{cat}"
            noise = noise_series[key]
            coef = coefficients[comp][cat]

            sub = df[(df["categoria"] == cat)]
            col = f"indice_{comp}"
            indices = sub[col].values

            # ── Test 1: Mean index ≈ coefficient ──────────────────────
            mean_idx = np.mean(indices)
            mean_err = abs(mean_idx - coef) / coef * 100

            # ── Test 2: Noise std ≈ target σ ──────────────────────────
            noise_std = np.std(noise)
            std_err = abs(noise_std - AR1_SIGMA) / AR1_SIGMA * 100

            # ── Test 3: Autocorrelation lag-1 ≈ φ ─────────────────────
            autocorr = np.corrcoef(noise[:-1], noise[1:])[0, 1]
            phi_err = abs(autocorr - AR1_PHI) / AR1_PHI * 100

            # ── Test 4: KS test vs N(0, σ) ────────────────────────────
            ks_stat, ks_pval = stats.kstest(noise, "norm", args=(0, AR1_SIGMA))

            # ── Test 5: No trend (linear regression slope ≈ 0) ────────
            x = np.arange(len(noise))
            slope, _, _, _, _ = stats.linregress(x, noise)

            status = "PASS" if (mean_err < 1 and std_err < 20 and phi_err < 15) else "WARN"

            label = "Gama" if comp == "gama" else "Plan Suárez"
            print(f"\n  {label} — {cat} [{status}]:")
            print(f"    Media índice:  {mean_idx:.4f} (target: {coef:.4f}, err: {mean_err:.2f}%)")
            print(f"    Noise σ:       {noise_std:.4f} (target: {AR1_SIGMA}, err: {std_err:.1f}%)")
            print(f"    Autocorr(1):   {autocorr:.3f} (target: {AR1_PHI}, err: {phi_err:.1f}%)")
            print(f"    KS test:       stat={ks_stat:.4f}, p={ks_pval:.4f}")
            print(f"    Trend slope:   {slope:.2e} (≈0 = no trend)")

            report["tests"][key] = {
                "coefficient": coef,
                "mean_index": round(mean_idx, 6),
                "mean_error_pct": round(mean_err, 4),
                "noise_std": round(noise_std, 6),
                "noise_std_error_pct": round(std_err, 2),
                "autocorr_lag1": round(autocorr, 4),
                "autocorr_error_pct": round(phi_err, 2),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "trend_slope": round(slope, 8),
                "status": status,
            }

    # Summary
    all_pass = all(t["status"] == "PASS" for t in report["tests"].values())
    report["overall_status"] = "ALL_PASS" if all_pass else "HAS_WARNINGS"

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"Overall: {report['overall_status']}")
    print(f"💾 Reporte: {report_path}")

    return report


if __name__ == "__main__":
    df, noise_series = generate_competition_indices()
    report = validate_indices(df, noise_series)
