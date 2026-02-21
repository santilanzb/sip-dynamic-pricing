"""
Matching de productos scrapeados con catálogo Emporium y cálculo de coeficientes.

Pipeline:
  1. Carga datos scrapeados (Gama, Plan Suárez) y catálogo Emporium (dim_producto)
  2. Limpieza y normalización de nombres
  3. Fuzzy matching por categoría (rapidfuzz, threshold ≥ 65)
  4. Cálculo de price ratios por producto matched
  5. Agregación a coeficientes por categoría
  6. Validación contra estimaciones expertas
  7. Exporta competition_coefficients.json + match details CSV

Uso:
    python -m src.competition.match_products
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_EXT = Path("data/external")
DATA_PROC = Path("data/processed")
OUTPUT_DIR = Path("output/competition")

# ── Expert estimates (from competitive intelligence) ───────────────────────
EXPERT_ESTIMATES = {
    "gama": {
        "03CARN": 1.00,   # "Precios similares en carnes"
        "05CHAR": 1.175,  # "15-20% más caro" → midpoint 17.5%
        "08FRUV": 1.30,   # "30%+ más caro"
    },
    "plansuarez": {
        "03CARN": 1.10,   # "~10% más caro uniforme"
        "05CHAR": 1.10,
        "08FRUV": 1.10,
    },
}

# ── Keywords to filter out irrelevant search results ──────────────────────
# Products that appear in search but don't belong to the category
EXCLUDE_KEYWORDS = {
    "03CARN": [
        "alimento p/", "comida p/", "whiskas", "dog", "gato", "perro",
        "qualidy", "dog chow", "cat chow", "arena", "croqueta",
        "papel hig", "detergente", "jabon", "shampoo", "crema",
        "arcos dental", "cepillo", "enjuague", "pasta dental",
        "arroz", "harina", "aceite", "salsa", "pasta", "conserva",
        "palmito", "pastel", "arepa", "tequeño", "empanada",
        "huevos", "gallina",
    ],
    "05CHAR": [
        "alimento p/", "whiskas", "dog", "gato", "perro",
        "papel hig", "detergente", "jabon",
        "arcos dental", "cepillo",
        "arepa", "tequeño", "empanada",
        "leche", "mantequilla", "yogurt", "jugo",
    ],
    "08FRUV": [
        "alimento p/", "whiskas", "dog", "gato", "perro",
        "papel hig", "detergente", "jabon",
        "arcos dental", "cepillo",
        "salsa", "pasta", "conserva", "ketchup", "mayonesa",
        "aceite", "vinagre", "mccormick", "especia",
        "jugo", "refresco", "cerveza", "agua mineral",
        "passata", "osole", "pure",
    ],
}

# ── Keywords that MUST be present for category relevance ───────────────────
CATEGORY_KEYWORDS = {
    "03CARN": [
        "pollo", "carne", "res", "cerdo", "cochino", "pernil", "chuleta",
        "bistec", "solomo", "molida", "muslo", "pechuga", "ala ", "alas",
        "higado", "milanesa", "lomito", "costilla", "filete", "lomo",
        "entero", "deshuesado", "corte", "kg",
    ],
    "05CHAR": [
        "queso", "jamon", "jamón", "mortadela", "salchicha", "chorizo",
        "salami", "tocineta", "pavo", "embutido", "mozzarella",
        "paisa", "blanco", "amarillo", "pechuga", "paleta", "lechon",
        "lechón", "pierna",
    ],
    "08FRUV": [
        "plátano", "platano", "papa", "tomate", "cebolla", "aguacate",
        "cambur", "zanahoria", "limón", "limon", "pimenton", "pimentón",
        "ajo", "papaya", "lechuga", "brocoli", "brócoli", "manzana",
        "naranja", "fruta", "verdura", "hortaliza", "fresca", "bandeja",
        "kg", "kilo", "500 gr", "500g", "1kg", "guayaba", "piña", "melon",
        "pepino", "repollo", "apio", "cilantro", "perejil", "yuca",
    ],
}


def _normalize(text: str) -> str:
    """Normaliza texto para matching: lowercase, sin acentos, sin puntuación."""
    if not text:
        return ""
    # Lowercase
    text = text.lower().strip()
    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove weight indicators for better matching
    text = re.sub(r"\b\d+[.,]?\d*\s*(kg|gr|g|ml|lt|l|und|un|cc)\b", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_weight_kg(text: str) -> Optional[float]:
    """
    Extrae peso en KG del nombre del producto.

    Patrones reconocidos:
      '300 GR' / '300GR' / '300G' → 0.3 kg
      '1 KG' / '1KG' → 1.0 kg
      '1,500 KG' / '1.5 KG' → 1.5 kg
      '!500! GR' → 0.5 kg (formato Plan Suárez)
    Returns None si no se detecta peso.
    """
    if not text:
        return None
    text = text.upper().strip()

    # Plan Suárez format: !500! GR
    m = re.search(r"!(\d+)!\s*(?:GR|G)\b", text)
    if m:
        return float(m.group(1)) / 1000.0

    # Standard: "1,500 KG" or "1.5 KG" or "2 KG"
    m = re.search(r"(\d+[.,]?\d*)\s*KG\b", text)
    if m:
        val = m.group(1).replace(",", ".")
        return float(val)

    # Grams: "300 GR" or "300G" or "500 GR"
    m = re.search(r"(\d+)\s*(?:GR|G)\b", text)
    if m:
        return float(m.group(1)) / 1000.0

    # "X UN" or "X UND" — not weight-based, skip
    return None


def _normalize_price_per_kg(
    price: float,
    product_name: str,
    emporium_name: str,
) -> tuple[float, str]:
    """
    Normaliza precios a per-KG cuando hay mismatch de unidades.

    Si el competidor vende por paquete (300g) y Emporium vende por KG,
    escala el precio del competidor a precio/KG equivalente.

    Returns:
        (normalized_price, adjustment_note)
    """
    comp_weight = _extract_weight_kg(product_name)
    emp_weight = _extract_weight_kg(emporium_name)

    # Case 1: Emporium is per-KG (name contains "KG" or no weight = assumed per-KG)
    emp_is_per_kg = (
        "KG" in emporium_name.upper()
        or emp_weight is None  # Many Emporium products don't specify (implicit KG)
    )

    # Case 2: Competitor has explicit weight in grams/kg
    if comp_weight and comp_weight < 0.95 and emp_is_per_kg:
        # Competitor is per-package, Emporium is per-KG → scale up
        normalized = price / comp_weight
        return normalized, f"scaled {comp_weight:.3f}kg→1kg"

    if comp_weight and comp_weight > 1.05 and emp_is_per_kg:
        # Competitor sells >1kg package, Emporium per-KG → scale down
        normalized = price / comp_weight
        return normalized, f"scaled {comp_weight:.3f}kg→1kg"

    # Case 3: Both have explicit weights — normalize both to per-KG
    if comp_weight and emp_weight and abs(comp_weight - emp_weight) > 0.05:
        normalized = price / comp_weight
        return normalized, f"scaled {comp_weight:.3f}kg→1kg (emp={emp_weight:.3f}kg)"

    return price, "no_adjustment"


def _is_relevant(product_name: str, category: str) -> bool:
    """Filtra productos irrelevantes para la categoría."""
    name_lower = product_name.lower()

    # Check exclusion keywords
    for kw in EXCLUDE_KEYWORDS.get(category, []):
        if kw.lower() in name_lower:
            return False

    # Check category relevance — at least one keyword must match
    cat_keywords = CATEGORY_KEYWORDS.get(category, [])
    if cat_keywords:
        return any(kw.lower() in name_lower for kw in cat_keywords)

    return True


def load_and_clean_scraped(competitor: str) -> pd.DataFrame:
    """Carga y limpia datos scrapeados de un competidor."""
    if competitor == "gama":
        path = DATA_EXT / "gama_prices_raw.csv"
        df = pd.read_csv(path)
        df = df.rename(columns={"nombre": "nombre_comp", "precio_usd": "precio_comp_usd"})
    elif competitor == "plansuarez":
        path = DATA_EXT / "plansuarez_prices_raw.csv"
        df = pd.read_csv(path)
        df = df.rename(columns={"nombre": "nombre_comp", "precio_usd": "precio_comp_usd"})
    else:
        raise ValueError(f"Competidor desconocido: {competitor}")

    # Basic cleaning
    df["nombre_comp"] = df["nombre_comp"].astype(str).str.strip()
    df["nombre_norm"] = df["nombre_comp"].apply(_normalize)
    df["precio_comp_usd"] = pd.to_numeric(df["precio_comp_usd"], errors="coerce")
    df = df.dropna(subset=["precio_comp_usd"])
    df = df[df["precio_comp_usd"] > 0]

    # Filter out irrelevant products
    n_before = len(df)
    df["is_relevant"] = df.apply(
        lambda r: _is_relevant(r["nombre_comp"], r["categoria_emporium"]), axis=1
    )
    df = df[df["is_relevant"]].drop(columns=["is_relevant"])
    n_after = len(df)
    print(f"  {competitor}: {n_before} → {n_after} productos relevantes "
          f"(filtrados {n_before - n_after} irrelevantes)")

    return df


def load_emporium_catalog() -> pd.DataFrame:
    """
    Carga catálogo Emporium con precios RECIENTES (último mes de fact_ventas).

    Usar precios recientes en vez de promedios históricos es crítico porque
    los precios de Emporium han subido ~10-16% respecto al promedio histórico
    (inflación + ajustes BCV), y los competidores se scrapearon hoy.
    """
    dp = pd.read_parquet(DATA_PROC / "dim_producto.parquet")
    dp["clase"] = dp["clase"].str.strip()
    dp["descripcion_norm"] = dp["descripcion"].apply(_normalize)

    # ── Extract recent prices from fact_ventas ────────────────────────
    fv = pd.read_parquet(DATA_PROC / "fact_ventas.parquet")
    max_date = fv["fecha"].max()
    cutoff = max_date - pd.Timedelta(days=30)
    recent = fv[fv["fecha"] >= cutoff]

    # Median price per product in recent period (robust to outlier days)
    recent_prices = (
        recent.groupby("producto_id")["precio_unitario_usd"]
        .median()
        .reset_index()
        .rename(columns={"precio_unitario_usd": "precio_medio_usd"})
    )

    print(f"  Emporium precios recientes: {cutoff.date()} — {max_date.date()} "
          f"({len(recent):,} transacciones)")

    # Merge recent prices into catalog (replace old precio_medio_usd)
    dp = dp.drop(columns=["precio_medio_usd"], errors="ignore")
    dp = dp.merge(recent_prices, on="producto_id", how="inner")
    dp["precio_medio_usd"] = pd.to_numeric(dp["precio_medio_usd"], errors="coerce")
    dp = dp[dp["precio_medio_usd"] > 0]

    print(f"  Emporium catálogo: {len(dp)} productos con precio reciente > 0")
    for cat in ["03CARN", "05CHAR", "08FRUV"]:
        sub = dp[dp["clase"] == cat]
        n = len(sub)
        med = sub["precio_medio_usd"].median()
        print(f"    {cat}: {n} productos, precio mediano ${med:.2f}")

    return dp


def fuzzy_match_products(
    comp_df: pd.DataFrame,
    emp_df: pd.DataFrame,
    competitor: str,
    threshold: int = 65,
) -> pd.DataFrame:
    """
    Fuzzy match productos del competidor con catálogo Emporium.

    Matching se hace dentro de cada categoría para evitar cross-category matches.
    Usa token_sort_ratio para robustez ante reordenamiento de palabras.
    """
    matches = []
    categories = comp_df["categoria_emporium"].unique()

    for cat in sorted(categories):
        comp_cat = comp_df[comp_df["categoria_emporium"] == cat]
        emp_cat = emp_df[emp_df["clase"] == cat]

        if len(emp_cat) == 0:
            print(f"    ⚠️ No hay productos Emporium en {cat}")
            continue

        # Build choices list
        emp_names = emp_cat["descripcion_norm"].tolist()
        emp_indices = emp_cat.index.tolist()

        for _, comp_row in comp_cat.iterrows():
            query = comp_row["nombre_norm"]
            if not query or len(query) < 3:
                continue

            # Find best match using token_sort_ratio (handles word reordering)
            result = process.extractOne(
                query,
                emp_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
            )

            if result is None:
                continue

            best_match, score, idx_in_list = result
            emp_idx = emp_indices[idx_in_list]
            emp_row = emp_cat.loc[emp_idx]

            # ── Weight normalization ──────────────────────────────
            raw_price = comp_row["precio_comp_usd"]
            normalized_price, price_note = _normalize_price_per_kg(
                raw_price,
                comp_row["nombre_comp"],
                emp_row["descripcion"],
            )

            # Calculate price ratio using normalized price
            ratio = normalized_price / emp_row["precio_medio_usd"]

            matches.append({
                "competidor": competitor,
                "categoria": cat,
                "nombre_comp": comp_row["nombre_comp"],
                "nombre_comp_norm": query,
                "nombre_emporium": emp_row["descripcion"],
                "nombre_emporium_norm": best_match,
                "match_score": score,
                "precio_comp_raw_usd": round(raw_price, 4),
                "precio_comp_norm_usd": round(normalized_price, 4),
                "precio_emporium_usd": round(emp_row["precio_medio_usd"], 4),
                "price_adjustment": price_note,
                "ratio": round(ratio, 4),
                "search_term": comp_row.get("search_term", ""),
            })

    df_matches = pd.DataFrame(matches)

    if len(df_matches) > 0:
        # Remove duplicates: keep best match per competitor product
        df_matches = df_matches.sort_values("match_score", ascending=False)
        df_matches = df_matches.drop_duplicates(subset=["competidor", "nombre_comp"], keep="first")

        print(f"    {competitor}: {len(df_matches)} matches (threshold={threshold})")
        for cat in sorted(df_matches["categoria"].unique()):
            n = len(df_matches[df_matches["categoria"] == cat])
            print(f"      {cat}: {n} matches")
    else:
        print(f"    ⚠️ {competitor}: 0 matches encontrados")

    return df_matches


def calculate_coefficients(all_matches: pd.DataFrame) -> dict:
    """
    Calcula coeficientes de competencia por categoría por competidor.

    Pipeline:
      1. Pre-filter: only keep ratios in [0.3, 3.0] (sane economic bounds)
      2. IQR outlier removal (1.5×IQR)
      3. Require minimum 3 valid products; otherwise fallback to expert estimate
      4. Use median (robust to remaining outliers)
      5. Blend with expert estimate when scraped confidence is low
    """
    coefficients = {}

    for competitor in all_matches["competidor"].unique():
        comp_data = all_matches[all_matches["competidor"] == competitor]
        coefficients[competitor] = {}

        for cat in ["03CARN", "05CHAR", "08FRUV"]:
            cat_data = comp_data[comp_data["categoria"] == cat].copy()
            expert = EXPERT_ESTIMATES.get(competitor, {}).get(cat, 1.10)

            # Pre-filter: economically sane ratios only [0.3, 3.0]
            sane = cat_data[(cat_data["ratio"] >= 0.3) & (cat_data["ratio"] <= 3.0)]

            if len(sane) < 3:
                coefficients[competitor][cat] = {
                    "coef": round(expert, 4),
                    "source": "expert_estimate",
                    "n_products": len(sane),
                    "n_raw": len(cat_data),
                    "note": f"Insuficientes matches sanos (<3), usando estimación experta: {expert}",
                    "expert_estimate": expert,
                    "diff_from_expert_pp": 0.0,
                }
                continue

            ratios = sane["ratio"]

            # IQR outlier removal (1.5× for tighter control)
            q1 = ratios.quantile(0.25)
            q3 = ratios.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            ratios_clean = ratios[(ratios >= lower) & (ratios <= upper)]

            if len(ratios_clean) < 3:
                ratios_clean = ratios  # Keep all sane ratios if IQR too aggressive

            n_outliers = len(cat_data) - len(ratios_clean)

            scraped_coef = float(ratios_clean.median())

            # Confidence-weighted blend:
            # - High confidence (n ≥ 10, low σ): use scraped
            # - Low confidence (n < 5 or high σ): blend 50/50 with expert
            n_clean = len(ratios_clean)
            std = float(ratios_clean.std()) if n_clean > 1 else 1.0

            if n_clean >= 10 and std < 0.3:
                blend_weight = 0.85  # High confidence → mostly scraped
            elif n_clean >= 5:
                blend_weight = 0.65  # Medium confidence
            else:
                blend_weight = 0.40  # Low confidence → lean on expert

            blended_coef = round(blend_weight * scraped_coef + (1 - blend_weight) * expert, 4)

            diff_from_expert = round((blended_coef - expert) * 100, 2)

            coefficients[competitor][cat] = {
                "coef": blended_coef,
                "coef_scraped_raw": round(scraped_coef, 4),
                "blend_weight": blend_weight,
                "source": "blended",
                "n_products": n_clean,
                "n_outliers_removed": n_outliers,
                "mean": round(float(ratios_clean.mean()), 4),
                "median": round(scraped_coef, 4),
                "std": round(std, 4),
                "min": round(float(ratios_clean.min()), 4),
                "max": round(float(ratios_clean.max()), 4),
                "expert_estimate": expert,
                "diff_from_expert_pp": diff_from_expert,
            }

    return coefficients


def build_output_json(coefficients: dict, all_matches: pd.DataFrame) -> dict:
    """Construye el JSON final de coeficientes."""
    # Extract simple coefficient values for easy consumption
    coef_simple = {}
    for competitor, cats in coefficients.items():
        coef_simple[competitor] = {}
        for cat, info in cats.items():
            coef_simple[competitor][cat] = info["coef"]

    n_matched = {}
    for competitor in all_matches["competidor"].unique():
        n_matched[competitor] = int(len(all_matches[all_matches["competidor"] == competitor]))

    output = {
        "coefficients": coef_simple,
        "detailed_stats": coefficients,
        "metadata": {
            "scrape_date": datetime.now().strftime("%Y-%m-%d"),
            "bcv_rate": float(pd.read_csv(DATA_EXT / "plansuarez_prices_raw.csv")["bcv_rate"].iloc[0]),
            "n_products_matched": n_matched,
            "match_threshold": 65,
            "aggregation_method": "median",
            "outlier_removal": "3×IQR",
        },
        "expert_estimates": EXPERT_ESTIMATES,
    }
    return output


def print_comparison_report(coefficients: dict):
    """Imprime reporte comparativo de coeficientes vs expertos."""
    print("\n" + "=" * 80)
    print("REPORTE DE COEFICIENTES DE COMPETENCIA")
    print("=" * 80)

    for competitor in ["gama", "plansuarez"]:
        label = "Gama" if competitor == "gama" else "Plan Suárez"
        print(f"\n{'─' * 40}")
        print(f"  {label}")
        print(f"{'─' * 40}")

        cats = coefficients.get(competitor, {})
        for cat in ["03CARN", "05CHAR", "08FRUV"]:
            info = cats.get(cat, {})
            coef = info.get("coef", "N/A")
            source = info.get("source", "unknown")
            n = info.get("n_products", 0)
            expert = info.get("expert_estimate", "N/A")
            diff = info.get("diff_from_expert_pp", "N/A")
            scraped_raw = info.get("coef_scraped_raw", None)
            blend_w = info.get("blend_weight", None)

            pct = f"{(coef - 1) * 100:+.1f}%" if isinstance(coef, float) else "N/A"
            exp_pct = f"{(expert - 1) * 100:+.1f}%" if isinstance(expert, float) else "N/A"

            print(f"  {cat}:")
            print(f"    Coef FINAL:        {coef:.4f} ({pct})" if isinstance(coef, float) else f"    Coef: {coef}")
            if scraped_raw is not None:
                print(f"    Coef scraped raw:  {scraped_raw:.4f} ({(scraped_raw-1)*100:+.1f}%)")
                print(f"    Blend:             {blend_w:.0%} scraped + {1-blend_w:.0%} expert")
            print(f"    Expert estimate:   {expert:.4f} ({exp_pct})" if isinstance(expert, float) else f"    Expert: {expert}")
            print(f"    Δ vs expert:       {diff:+.2f}pp" if isinstance(diff, float) else f"    Δ: {diff}")
            print(f"    N matches:         {n} [{source}]")

            if isinstance(info.get("std"), float):
                print(f"    Rango:             [{info['min']:.2f} — {info['max']:.2f}], σ={info['std']:.3f}")


def run_matching_pipeline():
    """Ejecuta pipeline completo de matching y cálculo de coeficientes."""
    print("=" * 80)
    print("PIPELINE DE MATCHING Y CÁLCULO DE COEFICIENTES")
    print("=" * 80)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n📁 Cargando datos...")
    emp_df = load_emporium_catalog()

    print()
    gama_df = load_and_clean_scraped("gama")
    ps_df = load_and_clean_scraped("plansuarez")

    # ── 2. Fuzzy matching ─────────────────────────────────────────────
    print("\n🔍 Fuzzy matching...")
    gama_matches = fuzzy_match_products(gama_df, emp_df, "gama", threshold=65)
    ps_matches = fuzzy_match_products(ps_df, emp_df, "plansuarez", threshold=65)

    all_matches = pd.concat([gama_matches, ps_matches], ignore_index=True)
    print(f"\n  Total matches: {len(all_matches)}")

    # ── 3. Save match details ─────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    match_path = OUTPUT_DIR / "product_matches.csv"
    all_matches.to_csv(match_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Matches guardados: {match_path}")

    # ── 4. Calculate coefficients ─────────────────────────────────────
    print("\n📊 Calculando coeficientes...")
    coefficients = calculate_coefficients(all_matches)

    # ── 5. Print comparison report ────────────────────────────────────
    print_comparison_report(coefficients)

    # ── 6. Build and save JSON ────────────────────────────────────────
    output_json = build_output_json(coefficients, all_matches)

    json_path = DATA_EXT / "competition_coefficients.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Coeficientes guardados: {json_path}")

    # ── 7. Summary stats ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    for comp in ["gama", "plansuarez"]:
        label = "Gama" if comp == "gama" else "Plan Suárez"
        coefs = output_json["coefficients"].get(comp, {})
        print(f"\n  {label}:")
        for cat in ["03CARN", "05CHAR", "08FRUV"]:
            c = coefs.get(cat, "N/A")
            if isinstance(c, float):
                print(f"    {cat}: {c:.4f} ({(c-1)*100:+.1f}%)")
            else:
                print(f"    {cat}: {c}")

    return output_json, all_matches


if __name__ == "__main__":
    output_json, all_matches = run_matching_pipeline()
