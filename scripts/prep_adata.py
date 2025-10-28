from __future__ import annotations

import argparse
import logging
from pathlib import Path

import scanpy as sc

from utils.io import ensure_dirs, read_raw, write_adata


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prep_adata")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare AnnData baseline from raw inputs")
    p.add_argument("--input", required=True, help="Path to raw input (.h5ad, 10x dir, or .mtx)")
    p.add_argument("--output", required=True, help="Output .h5ad path under data/adata/")
    p.add_argument("--region", default="BA4", help="Region to subset if 'region' column exists")
    p.add_argument("--min_genes", type=int, default=200)
    p.add_argument("--min_cells", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    logger.info("Reading raw input from %s", args.input)
    adata = read_raw(args.input)

    # Basic QC
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    # Optional subset by region
    if "region" in adata.obs.columns:
        before = adata.n_obs
        adata = adata[adata.obs["region"].astype(str) == str(args.region)].copy()
        logger.info("Filtered by region %s: %d -> %d cells", args.region, before, adata.n_obs)
    else:
        logger.warning("No 'region' column in obs; proceeding without region subsetting.")

    # Normalize and log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Preserve counts already in layers["counts"] from read_raw
    out_path = Path(args.output)
    logger.info("Writing baseline AnnData to %s", out_path)
    write_adata(adata, out_path)


if __name__ == "__main__":
    main()


