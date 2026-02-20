"""
Genera reportes comparativos a partir de models/training_metadata_gpu.json
Crea: reports/models/comparative.csv con métricas principales (incl. WMAPE_revenue si está)
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

META = Path("models/training_metadata_gpu.json")
OUT_DIR = Path("reports/models")

if not META.exists():
    print("No metadata found:", META)
    raise SystemExit(0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
meta = json.loads(META.read_text(encoding="utf-8"))
rows = []
for name, res in meta.get("results", {}).items():
    m = res.get("metrics", {})
    rows.append({"model": name, **m})

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "comparative.csv", index=False)
print("Report written to", OUT_DIR / "comparative.csv")