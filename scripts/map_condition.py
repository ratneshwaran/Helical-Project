from __future__ import annotations

import argparse
import logging
from typing import Optional

import anndata as ad
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("map_condition")


def guess_condition_column(adata: ad.AnnData) -> Optional[str]:
    candidates = [
        "condition",
        "diagnosis",
        "group",
        "status",
        "disease",
        "cohort",
        "label",
    ]
    cols = [str(c) for c in adata.obs.columns]
    for c in candidates:
        if c in cols:
            return c
    return None


def map_value(v: str) -> Optional[str]:
    v = str(v).lower()
    if any(k in v for k in ["control", "healthy", "ctrl", "non-als", "normal"]):
        return "healthy"
    if any(k in v for k in ["als", "sals", "fals"]):
        return "als"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Map an obs column to standardized 'condition' labels: healthy/als")
    ap.add_argument("--input", required=True, help="Input .h5ad path")
    ap.add_argument("--output", default=None, help="Output .h5ad path (default: overwrite input)")
    ap.add_argument("--column", default=None, help="Obs column to map; if omitted, auto-detect")
    args = ap.parse_args()

    a = ad.read_h5ad(args.input)
    col = args.column or guess_condition_column(a)
    if col is None:
        logger.warning("No suitable obs column found to map condition. Skipping.")
        out = args.output or args.input
        a.write_h5ad(out)
        return

    mapped = a.obs[col].astype(str).map(map_value)
    if mapped.notna().sum() == 0:
        logger.warning("Condition mapping produced all-NaN. Leaving dataset unchanged.")
    else:
        a.obs["condition"] = mapped
        logger.info("Mapped '%s' -> 'condition' with %d/%d non-NaN values", col, mapped.notna().sum(), len(mapped))

    out = args.output or args.input
    a.write_h5ad(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()


