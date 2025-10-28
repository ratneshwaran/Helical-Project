from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Set

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


logger = logging.getLogger(__name__)


DATA_DIRS = {
    "raw": Path("data/raw"),
    "adata": Path("data/adata"),
    "embeddings": Path("data/embeddings"),
    "figs": Path("data/figs"),
}


def ensure_dirs() -> None:
    """Create required data subdirectories if missing."""
    for p in DATA_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _read_10x_dir(path: Path) -> ad.AnnData:
    """Read a 10x directory using scanpy.

    Args:
        path: Directory containing 10x files.

    Returns:
        AnnData object.
    """
    return sc.read_10x_mtx(path, var_names="gene_symbols", cache=True)


def read_raw(path: str) -> ad.AnnData:
    """Read raw data from a path to .h5ad, .mtx(+features/barcodes), or 10x folder.

    Args:
        path: Input path (file or directory).

    Returns:
        AnnData with counts in layers["counts"].
    """
    p = Path(path)
    if p.is_dir():
        adata = _read_10x_dir(p)
    else:
        if p.suffix == ".h5ad":
            adata = ad.read_h5ad(p)
        elif p.suffix == ".mtx" or p.suffixes[-2:] == [".mtx", ".gz"]:
            # Expect features/genes and barcodes alongside
            matrix = sc.read_mtx(p)
            dir_ = p.parent
            features = None
            for cand in ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]:
                if (dir_ / cand).exists():
                    features = pd.read_csv(dir_ / cand, sep="\t", header=None)
                    break
            barcodes = None
            for cand in ["barcodes.tsv.gz", "barcodes.tsv"]:
                if (dir_ / cand).exists():
                    barcodes = pd.read_csv(dir_ / cand, sep="\t", header=None)
                    break
            if features is None or barcodes is None:
                raise FileNotFoundError("Missing features.tsv/genes.tsv or barcodes.tsv next to .mtx")
            var_names = features.iloc[:, 1] if features.shape[1] > 1 else features.iloc[:, 0]
            obs_names = barcodes.iloc[:, 0]
            adata = ad.AnnData(X=matrix.X)
            adata.var_names = var_names.astype(str).values
            adata.obs_names = obs_names.astype(str).values
        else:
            raise ValueError(f"Unsupported file type: {p}")

    # Preserve raw counts in layers["counts"]
    adata.layers["counts"] = adata.X.copy()
    # Standardize gene symbol key
    if "gene_symbol" not in adata.var.columns:
        adata.var["gene_symbol"] = adata.var_names
    return adata


def write_adata(adata: ad.AnnData, path: str | Path) -> None:
    """Write AnnData to path, creating directories as needed.

    Args:
        adata: AnnData object to save.
        path: Output .h5ad path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)


def map_gene_symbols_to_vocab(adata: ad.AnnData, vocab_set: Set[str]) -> ad.AnnData:
    """Filter/rename genes to match a tokenizer vocabulary.

    Keeps genes in the provided `vocab_set`. Non-matching genes are dropped.

    Args:
        adata: Input AnnData.
        vocab_set: Allowed gene symbols.

    Returns:
        New AnnData subset to genes present in `vocab_set`.
    """
    if "gene_symbol" not in adata.var.columns:
        adata.var["gene_symbol"] = adata.var_names
    mask = adata.var["gene_symbol"].astype(str).isin(vocab_set)
    if mask.sum() == 0:
        logger.warning("No genes matched the provided vocabulary; returning empty AnnData subset.")
    adata_filt = adata[:, mask].copy()
    return adata_filt


