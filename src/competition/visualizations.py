"""
Visualizaciones para Fase 6: Datos de Competencia.

Genera 5 figuras para documentación y presentación del estudio competitivo:
  1. Coeficientes de precio por categoría (scraped vs expert)
  2. Series temporales de índices sintéticos (muestra 3 meses)
  3. Heatmap de correlación: competition features vs features originales
  4. Ablación WMAPE: Model A vs Model B (global + por categoría)
  5. Feature importance de competition features en Model B

Uso:
    python -m src.competition.visualizations
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────
COEFF_PATH = Path("data/external/competition_coefficients.json")
INDICES_PATH = Path("data/processed/competition_indices.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
ABLATION_PATH = Path("output/competition/ablation_results.json")
FI_PATH = Path("output/competition/model_b_feature_importance.csv")
PLOT_DIR = Path("output/competition/plots")

# ── Style ─────────────────────────────────────────────────────────────────
CAT_LABELS = {"03CARN": "Carnes", "05CHAR": "Charcutería", "08FRUV": "Fruver"}
COMP_COLORS = {"Gama": "#2196F3", "Plan Suárez": "#FF9800"}
MODEL_COLORS = {"Model A\n(sin comp.)": "#78909C", "Model B\n(con comp.)": "#26A69A"}

COMPETITION_FEATURES = [
    "indice_gama_cat", "indice_plansuarez_cat", "competitividad_precio",
    "gap_precio_max_comp", "indice_mercado", "presion_competitiva",
    "volatilidad_mercado_7d",
]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def load_coefficients():
    with open(COEFF_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ablation():
    with open(ABLATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Plot 1: Coeficientes de precio — Scraped vs Expert
# =========================================================================
def plot_coefficients(coeff_data):
    """Barplot agrupado: coeficientes calibrados vs estimaciones expertas."""
    cats = ["03CARN", "05CHAR", "08FRUV"]
    x = np.arange(len(cats))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))

    # Scraped (blended) coefficients
    gama_scraped = [coeff_data["coefficients"]["gama"][c] for c in cats]
    ps_scraped = [coeff_data["coefficients"]["plansuarez"][c] for c in cats]

    # Expert estimates
    gama_expert = [coeff_data["expert_estimates"]["gama"][c] for c in cats]
    ps_expert = [coeff_data["expert_estimates"]["plansuarez"][c] for c in cats]

    bars1 = ax.bar(x - 1.5 * width, gama_scraped, width, label="Gama (calibrado)",
                   color="#2196F3", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x - 0.5 * width, gama_expert, width, label="Gama (experto)",
                   color="#2196F3", alpha=0.4, edgecolor="#2196F3", linewidth=1,
                   linestyle="--", hatch="//")
    bars3 = ax.bar(x + 0.5 * width, ps_scraped, width, label="Plan Suárez (calibrado)",
                   color="#FF9800", edgecolor="white", linewidth=0.5)
    bars4 = ax.bar(x + 1.5 * width, ps_expert, width, label="Plan Suárez (experto)",
                   color="#FF9800", alpha=0.4, edgecolor="#FF9800", linewidth=1,
                   linestyle="--", hatch="//")

    # Value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            pct = (h - 1) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"+{pct:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(-0.4, 1.002, "Emporium = 1.0", fontsize=7.5, color="gray", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in cats])
    ax.set_ylabel("Ratio Precio vs Emporium")
    ax.set_title("Coeficientes de Precio Competitivo por Categoría\n(Calibrado con scraping vs Estimación experta)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0.95, max(max(gama_scraped), max(ps_scraped)) + 0.08)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_coeficientes_precio.png")
    plt.close(fig)
    print("   ✅ 01_coeficientes_precio.png")


# =========================================================================
# Plot 2: Series temporales de índices sintéticos
# =========================================================================
def plot_time_series():
    """Time series of synthetic competition indices (3-month window)."""
    idx = pd.read_parquet(INDICES_PATH)
    idx["fecha"] = pd.to_datetime(idx["fecha"])

    # Sample: Oct-Dec 2025 (last 3 months)
    start = pd.Timestamp("2025-10-01")
    end = pd.Timestamp("2025-12-31")
    sample = idx[(idx["fecha"] >= start) & (idx["fecha"] <= end)]

    cats = ["03CARN", "05CHAR", "08FRUV"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    for i, cat in enumerate(cats):
        ax = axes[i]
        sub = sample[sample["categoria"] == cat].sort_values("fecha")

        ax.plot(sub["fecha"], sub["indice_gama"], color="#2196F3",
                linewidth=1.2, label="Gama", alpha=0.9)
        ax.plot(sub["fecha"], sub["indice_plansuarez"], color="#FF9800",
                linewidth=1.2, label="Plan Suárez", alpha=0.9)

        # Coefficient reference lines
        coeff = load_coefficients()["coefficients"]
        ax.axhline(y=coeff["gama"][cat], color="#2196F3", linestyle="--",
                    linewidth=0.7, alpha=0.5)
        ax.axhline(y=coeff["plansuarez"][cat], color="#FF9800", linestyle="--",
                    linewidth=0.7, alpha=0.5)
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)

        ax.set_ylabel("Índice")
        ax.set_title(f"{CAT_LABELS[cat]}", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

        # Y-axis formatting
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Índices Sintéticos de Competencia — Oct-Dic 2025\n"
                 "(Líneas punteadas = coeficiente calibrado medio)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_series_temporales_indices.png")
    plt.close(fig)
    print("   ✅ 02_series_temporales_indices.png")


# =========================================================================
# Plot 3: Heatmap de correlación
# =========================================================================
def plot_correlation_heatmap():
    """Correlation heatmap: competition features vs key original features."""
    df = pd.read_parquet(FEATURES_PATH)

    # Select key original features for comparison
    original_features = [
        "precio_unitario_usd", "precio_var_1d", "precio_var_7d",
        "precio_vs_clase", "precio_vs_historico",
        "unidades_lag_1", "unidades_mean_7d", "unidades_mean_30d",
        "margen_pct", "tiene_promocion",
    ]
    # Filter to existing columns
    original_features = [f for f in original_features if f in df.columns]
    comp_features = [f for f in COMPETITION_FEATURES if f in df.columns]

    all_feats = comp_features + original_features
    corr = df[all_feats].corr()

    # Extract just competition vs original cross-correlation
    cross_corr = corr.loc[comp_features, original_features]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        cross_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Correlación de Pearson", "shrink": 0.8},
        annot_kws={"size": 8},
    )

    # Clean labels
    comp_labels = [
        "Índ. Gama", "Índ. Plan Suárez", "Competitividad",
        "Gap Máx.", "Índ. Mercado", "Presión Comp.", "Volatilidad 7d",
    ]
    orig_labels = [
        "Precio USD", "ΔPrecio 1d", "ΔPrecio 7d",
        "Precio vs Clase", "Precio vs Hist.",
        "Demanda lag1", "Demanda media 7d", "Demanda media 30d",
        "Margen %", "Promoción",
    ]
    ax.set_yticklabels(comp_labels[:len(comp_features)], rotation=0, fontsize=8.5)
    ax.set_xticklabels(orig_labels[:len(original_features)], rotation=35, ha="right",
                       fontsize=8.5)
    ax.set_title("Correlación: Features de Competencia vs Features Originales")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_correlacion_competencia.png")
    plt.close(fig)
    print("   ✅ 03_correlacion_competencia.png")


# =========================================================================
# Plot 4: Ablación WMAPE — Model A vs Model B
# =========================================================================
def plot_ablation(ablation_data):
    """Barplot: WMAPE comparison global + by category."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                    gridspec_kw={"width_ratios": [1, 1.5]})

    # — Left: Global metrics —
    metrics_to_show = ["WMAPE", "SMAPE", "MAE"]
    x = np.arange(len(metrics_to_show))
    width = 0.3

    vals_a = [ablation_data["model_a"]["metrics"][m] for m in metrics_to_show]
    vals_b = [ablation_data["model_b"]["metrics"][m] for m in metrics_to_show]

    bars_a = ax1.bar(x - width / 2, vals_a, width, label="Model A (sin comp.)",
                     color="#78909C", edgecolor="white")
    bars_b = ax1.bar(x + width / 2, vals_b, width, label="Model B (con comp.)",
                     color="#26A69A", edgecolor="white")

    for bars in [bars_a, bars_b]:
        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.2f}" if h > 5 else f"{h:.3f}"
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                     fmt, ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_show)
    ax1.set_ylabel("Valor")
    ax1.set_title("Métricas Globales")
    ax1.legend(fontsize=8)

    # — Right: WMAPE by category —
    cats = ["03CARN", "05CHAR", "08FRUV"]
    cat_a = {r["clase"]: r["WMAPE"] for r in ablation_data["model_a"]["by_category"]}
    cat_b = {r["clase"]: r["WMAPE"] for r in ablation_data["model_b"]["by_category"]}

    x2 = np.arange(len(cats))
    wa = [cat_a.get(c, 0) for c in cats]
    wb = [cat_b.get(c, 0) for c in cats]

    bars_ca = ax2.bar(x2 - width / 2, wa, width, label="Model A (sin comp.)",
                      color="#78909C", edgecolor="white")
    bars_cb = ax2.bar(x2 + width / 2, wb, width, label="Model B (con comp.)",
                      color="#26A69A", edgecolor="white")

    # Value labels + delta annotations
    for i, c in enumerate(cats):
        delta = wb[i] - wa[i]
        color = "#26A69A" if delta < 0 else "#EF5350"
        ax2.annotate(f"Δ={delta:+.2f}pp",
                     xy=(x2[i], max(wa[i], wb[i]) + 0.15),
                     ha="center", fontsize=8, fontweight="bold", color=color)

    for bars in [bars_ca, bars_cb]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([CAT_LABELS[c] for c in cats])
    ax2.set_ylabel("WMAPE (%)")
    ax2.set_title("WMAPE por Categoría")
    ax2.legend(fontsize=8)

    # Global delta annotation
    delta_global = ablation_data["delta_wmape_pp"]
    fig.suptitle(
        f"Estudio de Ablación: Impacto de Features de Competencia\n"
        f"ΔWMAPE global: {delta_global:+.2f}pp — Decisión: {ablation_data['decision']}",
        fontsize=11, y=1.03,
    )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_ablacion_wmape.png")
    plt.close(fig)
    print("   ✅ 04_ablacion_wmape.png")


# =========================================================================
# Plot 5: Feature Importance — Competition features en Model B
# =========================================================================
def plot_feature_importance():
    """Horizontal barplot: all features ranked, competition highlighted."""
    fi = pd.read_csv(FI_PATH)
    fi = fi.sort_values("importance", ascending=True)  # ascending for horizontal bars
    total = fi["importance"].sum()
    fi["pct"] = fi["importance"] / total * 100

    # Top 25 + all competition features
    top_n = 25
    top_features = fi.tail(top_n).copy()

    # Ensure all competition features are included even if outside top 25
    comp_fi = fi[fi["feature"].isin(COMPETITION_FEATURES)]
    extra = comp_fi[~comp_fi["feature"].isin(top_features["feature"])]
    if len(extra) > 0:
        top_features = pd.concat([extra, top_features]).drop_duplicates(subset="feature")
        top_features = top_features.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 8))

    colors = [
        "#26A69A" if f in COMPETITION_FEATURES else "#B0BEC5"
        for f in top_features["feature"]
    ]

    bars = ax.barh(range(len(top_features)), top_features["pct"].values,
                   color=colors, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values, fontsize=8)
    ax.set_xlabel("Importancia (%)")
    ax.set_title("Feature Importance — Model B (Regresor LightGBM)\n"
                 "Features de competencia resaltadas en verde")

    # Percentage labels
    for i, (bar, pct) in enumerate(zip(bars, top_features["pct"].values)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=7)

    # Competition total annotation
    comp_total = comp_fi["pct"].sum()
    ax.text(0.97, 0.02,
            f"Total competencia: {comp_total:.1f}%\n({len(COMPETITION_FEATURES)} features)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color="#26A69A",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E0F2F1", edgecolor="#26A69A"))

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_feature_importance_competencia.png")
    plt.close(fig)
    print("   ✅ 05_feature_importance_competencia.png")


# =========================================================================
# Main
# =========================================================================
def generate_all_plots():
    """Generate all Phase 6 competition visualizations."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VISUALIZACIONES — FASE 6: DATOS DE COMPETENCIA")
    print("=" * 70)

    coeff_data = load_coefficients()
    ablation_data = load_ablation()

    print("\n📊 Generando visualizaciones...")

    plot_coefficients(coeff_data)
    plot_time_series()
    plot_correlation_heatmap()
    plot_ablation(ablation_data)
    plot_feature_importance()

    print(f"\n💾 Plots guardados en: {PLOT_DIR}/")
    print("=" * 70)
    print("✅ VISUALIZACIONES COMPLETADAS")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_plots()
