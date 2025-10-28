from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import (
    delta_to_healthy,
    wasserstein1d_along_pc,
    knn_overlap_fraction,
    silhouette_scores_by_label,
)
from utils.plotting import umap_2d, plot_umap, plot_centroid_shifts


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_metrics")


def main() -> None:
    emb_dir = Path("data/embeddings")
    fig_dir = Path("data/figs"); fig_dir.mkdir(parents=True, exist_ok=True)

    healthy_p = emb_dir / "healthy_base.npz"
    als_p = emb_dir / "als_base.npz"
    if not (healthy_p.exists() and als_p.exists()):
        raise FileNotFoundError("Missing healthy_base.npz or als_base.npz. Run embeddings first.")

    healthy = np.load(healthy_p)["arr"]
    als = np.load(als_p)["arr"]

    pert_embeddings = {}
    for p in sorted(emb_dir.glob("healthy_*_up.npz")):
        pert_embeddings[p.stem.replace("healthy_", "")] = np.load(p)["arr"]
    for p in sorted(emb_dir.glob("als_*_down.npz")):
        pert_embeddings[p.stem.replace("als_", "")] = np.load(p)["arr"]

    rows = []
    for name, emb_als_pert in pert_embeddings.items():
        d_health = delta_to_healthy(als, emb_als_pert, healthy)
        w1d = wasserstein1d_along_pc(als, healthy)
        w1d_after = wasserstein1d_along_pc(emb_als_pert, healthy)
        knn_gain = knn_overlap_fraction(healthy, emb_als_pert, k=15) - knn_overlap_fraction(healthy, als, k=15)
        sil_base = silhouette_scores_by_label(np.vstack([healthy, als]), ["healthy"]*len(healthy)+["als"]*len(als))
        sil_after = silhouette_scores_by_label(np.vstack([healthy, emb_als_pert]), ["healthy"]*len(healthy)+["als_pert"]*len(emb_als_pert))
        rows.append({
            "perturbation": name,
            "delta_to_healthy": d_health,
            "wasserstein_before": w1d,
            "wasserstein_after": w1d_after,
            "knn_overlap_gain": knn_gain,
            "silhouette_before": sil_base,
            "silhouette_after": sil_after,
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(fig_dir / "task3_metrics.csv", index=False)

    # Plots
    pool = [healthy, als] + list(pert_embeddings.values())
    labels = (["healthy"]*len(healthy) + ["als"]*len(als) +
              sum([[k]*len(v) for k, v in pert_embeddings.items()], []))
    X = np.vstack(pool)
    pts = umap_2d(X)
    plot_umap(pts, labels, "Task3: UMAP pooled", str(fig_dir / "task3_umap.png"))
    plot_centroid_shifts(healthy, als, pert_embeddings, "Task3: Centroid shifts", str(fig_dir / "task3_centroid_shifts.png"))
    logger.info("Saved metrics and figures to data/figs.")


if __name__ == "__main__":
    main()


