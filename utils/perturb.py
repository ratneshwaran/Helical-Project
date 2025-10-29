from __future__ import annotations

import logging
from typing import Dict, List, Literal

import anndata as ad
import numpy as np
import scanpy as sc


logger = logging.getLogger(__name__)


def _validate_genes(adata: ad.AnnData, genes: List[str], gene_symbol_key: str = "gene_symbol") -> List[str]:
    if gene_symbol_key not in adata.var.columns:
        adata.var[gene_symbol_key] = adata.var_names
    symbols = adata.var[gene_symbol_key].astype(str)
    present = [g for g in genes if g in set(symbols)]
    missing = [g for g in genes if g not in set(symbols)]
    if missing:
        logger.warning("Missing genes skipped: %s", ", ".join(missing))
    return present


def _get_gene_indices(adata: ad.AnnData, genes: List[str], gene_symbol_key: str = "gene_symbol") -> np.ndarray:
    symbols = adata.var[gene_symbol_key].astype(str)
    index_map = {g: i for i, g in enumerate(symbols)}
    return np.array([index_map[g] for g in genes], dtype=int)


def knock_rank(
    adata: ad.AnnData,
    genes: List[str],
    direction: Literal["up", "down"],
    delta_percentile: float = 0.15,
    gene_symbol_key: str = "gene_symbol",
) -> ad.AnnData:
    """Rank-based perturbation via local swaps.

    Non-destructive to `adata.X`. Computes per-cell gene ranks (descending expression)
    and shifts target genes by a fraction of rank positions. Stores resulting rank
    indices in `adata.obsm["rank_matrix"]` (same shape as X with integer ranks).

    Args:
      adata: Input AnnData. Uses `layers["counts"]` if present for ranking.
      genes: Target gene symbols.
      direction: "up" (increase rank) or "down" (decrease rank).
      delta_percentile: Fraction of the gene list length to shift.
      gene_symbol_key: Column name in `adata.var` for gene symbols.

    Returns:
      A copy of AnnData with `obsm["rank_matrix"]` added.
    """
    present = _validate_genes(adata, genes, gene_symbol_key)
    if len(present) == 0:
        logger.warning("No valid genes provided for rank perturbation.")
        adata_out = adata.copy()
        adata_out.obsm["rank_matrix"] = None
        return adata_out

    X = adata.layers.get("counts", adata.X)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    n_cells, n_genes = X.shape

    # Precompute original ranks (descending)
    order = np.argsort(-X, axis=1)
    ranks = np.empty_like(order)
    for i in range(n_cells):
        ranks[i, order[i]] = np.arange(n_genes)

    shift = int(max(1, round(delta_percentile * n_genes)))
    shift = min(shift, n_genes - 1)
    gene_idx = _get_gene_indices(adata, present, gene_symbol_key)

    ranks_new = ranks.copy()
    for i in range(n_cells):
        for g in gene_idx:
            r = ranks_new[i, g]
            if direction == "up":
                r_new = max(0, r - shift)
            else:
                r_new = min(n_genes - 1, r + shift)
            # swap positions to maintain a permutation
            if r_new != r:
                g_at_target = order[i, r_new]
                order[i, r], order[i, r_new] = order[i, r_new], order[i, r]
                ranks_new[i, g] = r_new
                ranks_new[i, g_at_target] = r

    adata_out = adata.copy()
    adata_out.obsm["rank_matrix"] = ranks_new
    return adata_out


def knock_expr(
    adata: ad.AnnData,
    genes: List[str],
    direction: Literal["up", "down"],
    log2fc: float = 1.0,
    propagate: bool = True,
    k_neighbors: int = 20,
    gene_symbol_key: str = "gene_symbol",
) -> ad.AnnData:
    """Expression-based perturbation with optional propagation in kNN gene graph.

    Multiplies expression of target genes by 2^(±log2fc), then renormalizes per cell
    to preserve library size. If propagate=True, spreads a fraction of the change
    to k-nearest neighbor genes (in gene-gene space based on correlation).

    Args:
      adata: Input AnnData. Uses `layers["counts"]` if present; otherwise `X`.
      genes: Target gene symbols.
      direction: "up" or "down".
      log2fc: Log2 fold-change magnitude.
      propagate: Whether to propagate effects to neighbor genes.
      k_neighbors: Number of neighbors for propagation.
      gene_symbol_key: Column name for gene symbols.

    Returns:
      A new AnnData with perturbed `X` and `layers["counts"]` preserved as original counts.
    """
    present = _validate_genes(adata, genes, gene_symbol_key)
    if len(present) == 0:
        logger.warning("No valid genes provided for expr perturbation.")
        return adata.copy()

    X_counts = adata.layers.get("counts", adata.X)
    X_counts = X_counts.toarray() if hasattr(X_counts, "toarray") else np.asarray(X_counts)
    X = X_counts.astype(float).copy()

    factor = 2.0 ** (log2fc if direction == "up" else -log2fc)
    idx = _get_gene_indices(adata, present, gene_symbol_key)
    X[:, idx] *= factor

    if propagate and X.shape[1] > 1:
        # gene-gene correlation graph
        # Avoid degenerate vars by adding small epsilon
        eps = 1e-9
        gene_corr = np.corrcoef((X + eps).T)
        np.fill_diagonal(gene_corr, 0.0)
        for g in idx:
            nbrs = np.argsort(-np.abs(gene_corr[g]))[:k_neighbors]
            # Decay weights by rank
            weights = np.linspace(0.5, 0.1, num=len(nbrs))
            for w, j in zip(weights, nbrs):
                X[:, j] *= (1.0 - w) + w * factor

    # Renormalize per cell to original library sizes
    libsize_orig = X_counts.sum(axis=1, keepdims=True)
    libsize_new = X.sum(axis=1, keepdims=True) + 1e-12
    X = X * (libsize_orig / libsize_new)

    adata_out = adata.copy()
    adata_out.X = X
    # Ensure downstream tokenization sees perturbed expression by updating counts layer
    # This preserves library size (renormalized above)
    try:
        adata_out.layers["counts"] = X
    except Exception:
        # Fallback if layers dict missing
        adata_out.layers["counts"] = X
    return adata_out


def batch_perturb(
    adata: ad.AnnData,
    gene_sets: Dict[str, List[str]],
    mode: Literal["rank", "expr"],
    **kwargs,
) -> Dict[str, ad.AnnData]:
    """Apply perturbations across multiple gene sets.

    Args:
      adata: Input AnnData.
      gene_sets: Mapping name -> list of genes.
      mode: "rank" or "expr".
      **kwargs: Passed to `knock_rank` or `knock_expr`.

    Returns:
      Dict of name -> perturbed AnnData copies.
    """
    out = {}
    for name, genes in gene_sets.items():
        if mode == "rank":
            out[name] = knock_rank(adata, genes=genes, **kwargs)
        elif mode == "expr":
            out[name] = knock_expr(adata, genes=genes, **kwargs)
        else:
            raise ValueError("mode must be 'rank' or 'expr'")
    return out


