"""
Scraper de precios de Plan Suárez via Playwright local.

Navega las categorías relevantes de plansuarez.com (OpenCart-based)
y extrae precios en Bs, convirtiéndolos a USD via BCV.

Adaptado de la arquitectura del scraper de Gama (Playwright + search/category
navigation + product element extraction).

Uso:
    python -m src.competition.scrape_plansuarez
"""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────
BCV_RATE = float(os.getenv("BCV_RATE_CURRENT", "402.3343"))
BASE_URL = "https://www.plansuarez.com"

# Búsquedas por categoría — mismo enfoque que Gama: buscar productos específicos
# Plan Suárez tiene barra de búsqueda en su sitio OpenCart
SEARCH_TERMS = {
    "03CARN": [
        "pollo entero",
        "carne molida",
        "solomo",
        "muslo pollo",
        "chuleta",
        "costillitas",
        "pernil",
        "pechuga pollo",
        "higado",
        "milanesa",
        "bistec",
        "carne premium",
        "cochino",
        "lomito",
        "pollo deshuesado",
    ],
    "05CHAR": [
        "queso",
        "jamon",
        "pechuga pavo",
        "mortadela",
        "salchicha",
        "tocineta",
        "chorizo",
        "salami",
        "mozzarella",
        "queso amarillo",
        "queso blanco",
        "queso paisa",
        "embutido",
        "pierna lechon",
        "paleta",
    ],
    "08FRUV": [
        "platano",
        "papa",
        "tomate",
        "cebolla",
        "aguacate",
        "cambur",
        "zanahoria",
        "limon",
        "pimenton",
        "ajo",
        "papaya",
        "lechuga",
        "brocoli",
        "manzana",
        "naranja",
    ],
}


def _parse_bs_price(text: str) -> Optional[float]:
    """
    Parsea precio en Bs desde texto de Plan Suárez.

    Plan Suárez usa formato US: coma=miles, punto=decimal.
    Ejemplos:
        'Bs.348.91'    -> 348.91
        'Bs.3,525.39'  -> 3525.39
        'Bs. 1,044.56' -> 1044.56
    """
    if not text:
        return None
    try:
        match = re.search(r"Bs\.?\s*([\d.,]+)", text)
        if not match:
            return None
        num_str = match.group(1)
        # Formato US: coma = miles, punto = decimal
        num_str = num_str.replace(",", "")
        return float(num_str)
    except (ValueError, AttributeError):
        return None


def scrape_plansuarez(
    output_path: str = "data/external/plansuarez_prices_raw.csv",
    bcv_rate: Optional[float] = None,
    headless: bool = True,
) -> pd.DataFrame:
    """
    Scrape Plan Suárez usando Playwright local con enfoque de búsqueda.

    Mismo patrón que Gama: buscar producto por producto, extraer resultados.
    """
    from playwright.sync_api import sync_playwright

    rate = bcv_rate or BCV_RATE
    all_products = []
    total_searches = sum(len(v) for v in SEARCH_TERMS.values())

    print("=" * 70)
    print("SCRAPING PLAN SUÁREZ — plansuarez.com")
    print("=" * 70)
    print(f"  BCV Rate: {rate:.4f} Bs/USD")
    print(f"  Búsquedas totales: {total_searches}")
    print(f"  Headless: {headless}")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        # Load homepage first to establish session/cookies
        print("🌐 Cargando página principal...")
        try:
            page.goto(BASE_URL, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            print("   ✅ Página cargada")
        except Exception as e:
            print(f"   ⚠️ Timeout parcial en homepage: {e}")

        search_num = 0
        for categoria, terms in SEARCH_TERMS.items():
            print(f"\n📦 Categoría: {categoria}")
            for term in terms:
                search_num += 1
                print(f"  [{search_num}/{total_searches}] Buscando: '{term}'...", end=" ")

                try:
                    products = _search_and_extract(page, term, categoria, rate)
                    print(f"→ {len(products)} productos")
                    all_products.extend(products)
                except Exception as e:
                    print(f"→ ERROR: {e}")

                # Delay between searches
                time.sleep(2)

        browser.close()

    # Create DataFrame
    df = pd.DataFrame(all_products)

    if len(df) == 0:
        print("\n⚠️ No se obtuvieron resultados del scraping.")
        return df

    # Deduplicate by product name
    df = df.drop_duplicates(subset=["nombre"], keep="first")
    df = df.dropna(subset=["precio_usd"])
    df = df[df["precio_usd"] > 0]

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"✅ Scraping completado: {len(df)} productos guardados")
    print(f"   Por categoría:")
    for cat in sorted(df["categoria_emporium"].unique()):
        n = len(df[df["categoria_emporium"] == cat])
        print(f"     {cat}: {n} productos")
    print(f"   BCV Rate: {rate:.4f}")
    print(f"   Archivo: {output_path}")
    print("=" * 70)

    return df


def _search_and_extract(page, search_term: str, categoria: str, bcv_rate: float) -> list[dict]:
    """
    Navega a la URL de búsqueda y extrae productos.

    OpenCart search URL pattern: /index.php?route=product/search&search=TERM
    """
    products = []

    # Navigate directly to search URL (avoids needing to find/click search box)
    search_url = f"{BASE_URL}/index.php?route=product/search&search={search_term}"
    try:
        page.goto(search_url, timeout=20000, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
    except Exception:
        # Partial load is OK
        pass

    # OpenCart product listing selectors
    product_selectors = [
        ".product-layout",
        ".product-thumb",
        ".product-layout .product-thumb",
        ".product-grid > div",
        ".product-list > div",
    ]

    elements = []
    for sel in product_selectors:
        try:
            elements = page.query_selector_all(sel)
            if elements:
                break
        except Exception:
            continue

    # Fallback: look for any element containing price text
    if not elements:
        try:
            elements = page.query_selector_all(".row .col-lg-4, .row .col-md-4, .row .col-sm-6")
        except Exception:
            pass

    for el in elements:
        try:
            product = _extract_product(el, categoria, search_term, bcv_rate)
            if product:
                products.append(product)
        except Exception:
            continue

    return products


def _extract_product(element, categoria: str, search_term: str, bcv_rate: float) -> Optional[dict]:
    """Extract a single product's data from a DOM element."""

    # ── Name ──────────────────────────────────────────────────────
    nombre = None
    for sel in ["h4 a", ".name a", ".caption a", "a.product-name", "h4", ".name", "a[href*='product']"]:
        try:
            name_el = element.query_selector(sel)
            if name_el:
                text = (name_el.inner_text() or "").strip()
                if text and len(text) > 3:
                    nombre = text
                    break
        except Exception:
            continue

    # Fallback: image alt text
    if not nombre:
        try:
            img = element.query_selector("img")
            if img:
                alt = img.get_attribute("alt") or ""
                if alt.strip() and len(alt.strip()) > 3:
                    nombre = alt.strip()
        except Exception:
            pass

    if not nombre:
        return None

    # ── Price ─────────────────────────────────────────────────────
    precio_bs = None
    precio_raw = ""

    # Try specific price selectors first
    for sel in [".price-new", ".price", "p.price", "span.price-new", ".special-price"]:
        try:
            price_el = element.query_selector(sel)
            if price_el:
                text = (price_el.inner_text() or "").strip()
                parsed = _parse_bs_price(text)
                if parsed and parsed > 0:
                    precio_bs = parsed
                    precio_raw = text
                    break
        except Exception:
            continue

    # Fallback: search full text for Bs pattern
    if not precio_bs:
        try:
            full_text = (element.inner_text() or "")
            # Find ALL Bs prices in text, take the first valid one
            matches = re.findall(r"Bs\.?\s*([\d.,]+)", full_text)
            for m in matches:
                try:
                    val = float(m.replace(",", ""))
                    if val > 0:
                        precio_bs = val
                        precio_raw = f"Bs.{m}"
                        break
                except ValueError:
                    continue
        except Exception:
            pass

    if not precio_bs:
        return None

    # ── URL ────────────────────────────────────────────────────────
    url = ""
    try:
        link = element.query_selector("a[href*='product']")
        if link:
            url = link.get_attribute("href") or ""
    except Exception:
        pass

    return {
        "nombre": nombre,
        "precio_bs": round(precio_bs, 2),
        "precio_usd": round(precio_bs / bcv_rate, 4),
        "precio_raw": precio_raw,
        "categoria_emporium": categoria,
        "search_term": search_term,
        "url": url,
        "bcv_rate": bcv_rate,
        "scraped_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    df = scrape_plansuarez(headless=True)
