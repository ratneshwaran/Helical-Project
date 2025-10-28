from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np

from utils.perturb import batch_perturb
from utils.geneformer_helpers import GFEmbedder
from utils.plotting import umap_2d, plot_umap


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_embeddings")


ALS_GENES = ["C9orf72", "SOD1", "TARDBP", "FUS", "TBK1", "NEK1"]


def main() -> None:
    base = Path("data/adata/baseline.h5ad")
    if not base.exists():
        raise FileNotFoundError("data/adata/baseline.h5ad not found. Run prep first.")
    adata = ad.read_h5ad(base)

    # Map from 'Condition' if standardized 'condition' is missing
    if "condition" not in adata.obs.columns:
        if "Condition" in adata.obs.columns:
            def _map_val(v: str) -> str | None:
                v = str(v).lower()
                if any(k in v for k in ["control", "healthy", "ctrl", "non-als", "normal"]):
                    return "healthy"
                if "als" in v:
                    return "als"
                return None
            adata.obs["condition"] = adata.obs["Condition"].map(_map_val)
        else:
            raise ValueError("baseline.h5ad is missing obs['condition'] and 'Condition'. Please map it first.")

    healthy = adata[adata.obs["condition"].astype(str) == "healthy"].copy()
    als = adata[adata.obs["condition"].astype(str) == "als"].copy()
    if healthy.n_obs == 0 or als.n_obs == 0:
        raise ValueError("Empty healthy or ALS subset after condition mapping.")

    up_sets = {f"{g}_up": [g] for g in ALS_GENES}
    down_sets = {f"{g}_down": [g] for g in ALS_GENES}

    logger.info("Perturbing healthy (ALS-like) with rank-up ...")
    pert_rank_up = batch_perturb(healthy, up_sets, mode="rank", direction="up", delta_percentile=0.15)
    logger.info("Perturbing ALS (rescue) with rank-down ...")
    pert_rank_down = batch_perturb(als, down_sets, mode="rank", direction="down", delta_percentile=0.15)

    emb_dir = Path("data/embeddings"); emb_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("data/figs"); fig_dir.mkdir(parents=True, exist_ok=True)

    embedder = GFEmbedder()

    logger.info("Embedding healthy and ALS baselines ...")
    healthy_tokens = embedder.adata_to_rank_tokens(healthy, top_k=4096)
    healthy_emb = embedder.get_cls_embeddings(healthy_tokens, batch_size=16)
    np.savez(emb_dir / "healthy_base.npz", arr=healthy_emb)

    als_tokens = embedder.adata_to_rank_tokens(als, top_k=4096)
    als_emb = embedder.get_cls_embeddings(als_tokens, batch_size=16)
    np.savez(emb_dir / "als_base.npz", arr=als_emb)

    logger.info("Embedding perturbations ...")
    for name, adx in pert_rank_up.items():
        toks = embedder.adata_to_rank_tokens(adx, top_k=4096)
        emb = embedder.get_cls_embeddings(toks, batch_size=16)
        np.savez(emb_dir / f"healthy_{name}.npz", arr=emb)
    for name, adx in pert_rank_down.items():
        toks = embedder.adata_to_rank_tokens(adx, top_k=4096)
        emb = embedder.get_cls_embeddings(toks, batch_size=16)
        np.savez(emb_dir / f"als_{name}.npz", arr=emb)

    # Snapshot figure
    stack = np.vstack([healthy_emb, als_emb])
    labels = ["healthy"] * len(healthy_emb) + ["als"] * len(als_emb)
    pts = umap_2d(stack)
    plot_umap(pts, labels, "Task2: Healthy vs ALS (baseline)", str(fig_dir / "task2_embedding_snapshot.png"))
    logger.info("Saved embeddings and snapshot figure.")


if __name__ == "__main__":
    main()


