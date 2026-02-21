"""
Scraper de precios de Automercados Gama via Apify.

Usa el actor santilanzb/gama-supermarket-scraper para buscar productos
en gamaenlinea.com y extraer precios en USD (Ref).

Uso:
    python -m src.competition.scrape_gama
"""

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from apify_client import ApifyClient
from dotenv import load_dotenv
import os

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────
ACTOR_ID = "chsVPkexFOOXf8Yg6"  # santilanzb/gama-supermarket-scraper
MAX_ITEMS_PER_SEARCH = 5
DELAY_BETWEEN_SEARCHES = 3  # segundos entre búsquedas para no saturar

# Productos a buscar por categoría (términos genéricos para máximo match)
SEARCH_TERMS = {
    "03CARN": [
        "pollo entero",
        "carne premium",
        "solomo molido",
        "muslo pollo",
        "chuleta ahumada",
        "chuleta cochino",
        "milanesa res",
        "filet pechuga",
        "higado pollo",
        "delicias pollo",
        "pollo deshuesado",
        "costilla res",
        "pechuga pollo",
        "pernil",
        "carne molida",
    ],
    "05CHAR": [
        "queso rallado",
        "queso mozzarella",
        "queso merideño",
        "queso santa barbara",
        "jamon",
        "pechuga pavo",
        "tocineta",
        "paleta cerdo",
        "pierna lechon",
        "queso pasteurizado",
        "mortadela",
        "salchicha",
        "queso paisa",
        "queso especial",
        "chorizo",
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


def _parse_gama_price(raw_price: str) -> Optional[float]:
    """
    Parsea precio Gama en formato 'Ref. X,XX' o 'Ref. X.XXX,XX' a float USD.

    Ejemplos:
        'Ref. 3,96'     -> 3.96
        'Ref. 11,13'    -> 11.13
        'Ref. 1.234,56' -> 1234.56
    """
    if not raw_price or not isinstance(raw_price, str):
        return None
    try:
        # Eliminar prefijo "Ref." y espacios
        cleaned = raw_price.replace("Ref.", "").strip()
        # Formato venezolano: punto = separador de miles, coma = decimal
        cleaned = cleaned.replace(".", "").replace(",", ".")
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def scrape_gama(
    output_path: str = "data/external/gama_prices_raw.csv",
    api_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ejecuta búsquedas en Gama y guarda los resultados.

    Args:
        output_path: Ruta de salida para el CSV
        api_token: Token de Apify (o se lee de .env)

    Returns:
        DataFrame con columnas: search_term, categoria_emporium, nombre,
        precio_ref, precio_iva, precio_total, categoria_gama, url
    """
    token = api_token or os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("APIFY_API_TOKEN no encontrado en .env ni en parámetros")

    client = ApifyClient(token)
    all_results = []
    total_searches = sum(len(v) for v in SEARCH_TERMS.values())

    print("=" * 70)
    print("SCRAPING GAMA EN LÍNEA — gamaenlinea.com")
    print("=" * 70)
    print(f"  Búsquedas totales: {total_searches}")
    print(f"  Max items por búsqueda: {MAX_ITEMS_PER_SEARCH}")
    print()

    search_num = 0
    for categoria, terms in SEARCH_TERMS.items():
        print(f"📦 Categoría: {categoria}")
        for term in terms:
            search_num += 1
            print(f"  [{search_num}/{total_searches}] Buscando: '{term}'...", end=" ")

            try:
                run_input = {
                    "search": term,
                    "maxItems": MAX_ITEMS_PER_SEARCH,
                }
                run = client.actor(ACTOR_ID).call(run_input=run_input)

                items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                print(f"→ {len(items)} resultados")

                for item in items:
                    # Parse price: "Ref. 3,96" → 3.96 USD
                    raw_price = item.get("price", "")
                    precio_usd = _parse_gama_price(raw_price)

                    all_results.append({
                        "search_term": term,
                        "categoria_emporium": categoria,
                        "nombre": item.get("name", item.get("title", "")),
                        "precio_raw": raw_price,
                        "precio_usd": precio_usd,
                        "categoria_gama": item.get("category", item.get("categoria", "")),
                        "url": item.get("url", ""),
                        "availability": item.get("availability", ""),
                        "scraped_at": item.get("scrapedAt", ""),
                    })

            except Exception as e:
                print(f"→ ERROR: {e}")

            # Delay entre búsquedas
            if search_num < total_searches:
                time.sleep(DELAY_BETWEEN_SEARCHES)

        print()

    # Crear DataFrame
    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("⚠️ No se obtuvieron resultados del scraping.")
        return df

    # Eliminar duplicados por nombre (mismo producto encontrado con distintos terms)
    df = df.drop_duplicates(subset=["nombre"], keep="first")

    # Filtrar productos sin precio
    df = df.dropna(subset=["precio_usd"])
    df = df[df["precio_usd"] > 0]

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print(f"✅ Scraping completado: {len(df)} productos guardados")
    print(f"   Por categoría:")
    for cat in df["categoria_emporium"].unique():
        n = len(df[df["categoria_emporium"] == cat])
        print(f"     {cat}: {n} productos")
    print(f"   Archivo: {output_path}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = scrape_gama()
