from __future__ import annotations

import numpy as np
import anndata as ad

from utils.perturb import knock_rank, knock_expr
from utils.geneformer_helpers import GFEmbedder


def _toy_adata(n_cells: int = 5, n_genes: int = 10) -> ad.AnnData:
    X = np.random.poisson(5, size=(n_cells, n_genes)).astype(float)
    var_names = np.array([f"G{i}" for i in range(n_genes)], dtype=object)
    adata = ad.AnnData(X=X)
    adata.layers["counts"] = X.copy()
    adata.var_names = var_names
    adata.var["gene_symbol"] = adata.var_names
    return adata


def test_rank_bounds():
    adata = _toy_adata()
    pert = knock_rank(adata, genes=["G1"], direction="up", delta_percentile=0.5)
    ranks = pert.obsm["rank_matrix"]
    assert ranks.min() >= 0 and ranks.max() < adata.n_vars


def test_expr_renorm():
    adata = _toy_adata()
    lib_before = adata.layers["counts"].sum(axis=1)
    pert = knock_expr(adata, genes=["G2"], direction="up", log2fc=2.0, propagate=False)
    lib_after = pert.X.sum(axis=1)
    np.testing.assert_allclose(lib_before, lib_after, rtol=1e-5, atol=1e-5)


def test_gfembedder_smoke_offline():
    import os
    os.environ["GF_OFFLINE"] = "1"
    emb = GFEmbedder(device="cpu")
    tokens = [[1, 2, 3], [4, 5]]
    out = emb.get_cls_embeddings(tokens, batch_size=2)
    assert out.shape[0] == 2 and out.shape[1] == 768


