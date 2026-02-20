"""
Quality checks rigurosos para garantizar que los datos están "impecables".
- Validaciones de esquema y tipos
- Nulos, duplicados, rangos y consistencia
- Verificación de alineación de lags (no leakage)
- Drift entre splits (PSI)
- Reporte JSON + CSV para auditoría
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.metrics import psi_by_feature


REQUIRED_COLUMNS = [
    "fecha", "producto_id", "sucursal_id", "unidades", "target",
    # algunas features clave que deben existir si se corrió features.py
    "unidades_lag_7", "unidades_mean_7d",
]


def _pass(v: bool) -> str:
    return "PASS" if v else "FAIL"


def run_quality_checks(
    features_path: str = "data/processed/features.parquet",
    fact_path: Optional[str] = "data/processed/fact_ventas.parquet",
    output_dir: str = "reports/data_quality",
    train_end: str = "2024-12-31",
    val_end: str = "2025-06-30",
) -> Dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datasets
    df = pd.read_parquet(features_path)
    fact = None
    if fact_path and Path(fact_path).exists():
        fact = pd.read_parquet(fact_path)

    # Checks
    report: Dict = {"totals": {"rows": int(len(df)), "cols": int(len(df.columns))}}

    # 1) Esquema
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["schema"] = {"required_missing": missing, "status": _pass(len(missing) == 0)}

    # 2) Tipos básicos
    type_issues = {}
    expected_types = {
        "fecha": "datetime64[ns]",
        "producto_id": "id",   # aceptar int u object (códigos alfanuméricos)
        "sucursal_id": "id",
        "unidades": "float",
        "target": "float",
    }
    for c, t in expected_types.items():
        if c in df.columns:
            if c == "fecha":
                ok = pd.api.types.is_datetime64_any_dtype(df[c])
            elif t == "id":
                ok = pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
            elif t == "float":
                ok = pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])
            else:
                ok = True
            if not ok:
                type_issues[c] = str(df[c].dtype)
    report["types"] = {"mismatches": type_issues, "status": _pass(len(type_issues) == 0)}

    # 3) Nulos
    nulls = df.isna().mean().sort_values(ascending=False)
    # Reglas: columnas críticas (unidades_lag_7, unidades_mean_7d) deben tener <=0.1% nulos; umbral general 10%
    crit_ok = True
    for c in ["unidades_lag_7", "unidades_mean_7d"]:
        if c in df.columns and nulls.get(c, 0.0) > 0.001:
            crit_ok = False
    report["nulls"] = {"top_null_cols": nulls.head(20).to_dict(), "status": _pass(crit_ok and nulls.max() <= 0.10)}

    # 4) Duplicados por llave natural (producto, sucursal, fecha)
    key = ["producto_id", "sucursal_id", "fecha"]
    dups = int(df.duplicated(subset=key).sum())
    report["duplicates"] = {"count": dups, "status": _pass(dups == 0)}

    # 5) Rangos
    range_status = True
    range_notes: List[str] = []
    if "unidades" in df.columns:
        if (df["unidades"] < 0).any():
            range_notes.append("unidades negativas")
            range_status = False
    if "precio_unitario_usd" in df.columns:
        if (df["precio_unitario_usd"] <= 0).any():
            range_notes.append("precio_unitario_usd <= 0")
            range_status = False
    if "tasa_bcv" in df.columns:
        if (df["tasa_bcv"] <= 0).any():
            range_notes.append("tasa_bcv <= 0")
            range_status = False
    if "tiene_promocion" in df.columns:
        bad = ~df["tiene_promocion"].isin([0, 1]).any()
        if bad:
            range_notes.append("tiene_promocion fuera de {0,1}")
            range_status = False
    report["ranges"] = {"notes": range_notes, "status": _pass(range_status)}

    # 6) Alineación de lags (consistencia temporal, sin leakage)
    try:
        # Para evitar romper la secuencia temporal, muestreamos grupos (producto,sucursal)
        if all(col in df.columns for col in ["unidades", "unidades_lag_7", "producto_id", "sucursal_id", "fecha"]):
            # Seleccionar hasta 200 grupos aleatorios
            grp_keys = df[["producto_id", "sucursal_id"]].drop_duplicates()
            grp_keys = grp_keys.sample(min(200, len(grp_keys)), random_state=42)
            sample = df.merge(grp_keys, on=["producto_id", "sucursal_id"], how="inner")
            sample = sample.sort_values(["producto_id", "sucursal_id", "fecha"]).copy()
            check = sample.groupby(["producto_id", "sucursal_id"])["unidades"].shift(7)
            eq_rate = float((np.isclose(check.values, sample["unidades_lag_7"].values, equal_nan=True)).mean() * 100)
        else:
            eq_rate = float("nan")
        report["lags_alignment"] = {"lag7_match_pct": eq_rate, "status": _pass((not np.isnan(eq_rate)) and eq_rate >= 95.0)}
    except Exception as e:
        report["lags_alignment"] = {"error": str(e), "status": "WARN"}

    # 7) Drift entre splits (PSI)
    try:
        df["fecha"] = pd.to_datetime(df["fecha"])
        train = df[df["fecha"] <= train_end]
        test = df[df["fecha"] > val_end]
        # PSI sólo sobre features usadas por el modelo (mismo criterio que prepare_data)
        exclude = {"fecha", "producto_id", "sucursal_id", "target", "unidades", "ingreso_usd", "costo_usd", "margen_usd", "clase", "tasa_bcv", "rotacion"}
        feature_cols = [c for c in df.columns if c not in exclude]
        psi_df = psi_by_feature(train, test, feature_cols)
        psi_df.to_csv(out_dir / "psi_by_feature.csv", index=False)
        # Thresholds de referencia: <0.1 estable, 0.1-0.25 alerta, >0.25 drift
        worst = float(psi_df["psi"].max()) if len(psi_df) else float(0.0)
        report["psi"] = {"max_psi": worst, "status": _pass(worst < 0.25)}
    except Exception as e:
        report["psi"] = {"error": str(e), "status": "WARN"}

    # Guardar
    with open(out_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Resumen CSV
    flat_rows = []
    for sec, val in report.items():
        if isinstance(val, dict):
            flat_rows.append({"check": sec, **{k: (str(v) if not isinstance(v, (int, float, str)) else v) for k, v in val.items()}})
    pd.DataFrame(flat_rows).to_csv(out_dir / "summary.csv", index=False)

    print("\nDATA QUALITY REPORT ->", out_dir)
    return report


if __name__ == "__main__":
    run_quality_checks()
