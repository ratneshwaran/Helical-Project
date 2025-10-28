from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP


def umap_2d(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    pts = reducer.fit_transform(X)
    return pts


def plot_umap(points: np.ndarray, labels: Iterable, title: str, outpath: str) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=list(labels), s=8, linewidth=0)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_centroid_shifts(
    emb_healthy: np.ndarray,
    emb_als: np.ndarray,
    emb_als_pert_dict: Dict[str, np.ndarray],
    title: str,
    outpath: str,
) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    # UMAP of pooled embeddings
    names = ["healthy", "als"] + list(emb_als_pert_dict.keys())
    X = np.vstack([emb_healthy, emb_als] + list(emb_als_pert_dict.values()))
    pts = umap_2d(X)

    # Compute centroids in UMAP space
    n_h = len(emb_healthy)
    n_a = len(emb_als)
    idx = 0
    pts_h = pts[idx : idx + n_h]; idx += n_h
    pts_a = pts[idx : idx + n_a]; idx += n_a
    centroids = {"healthy": pts_h.mean(axis=0), "als": pts_a.mean(axis=0)}
    for k, v in emb_als_pert_dict.items():
        pts_k = pts[idx : idx + len(v)]; idx += len(v)
        centroids[k] = pts_k.mean(axis=0)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=pts[:, 0], y=pts[:, 1], s=6, color="lightgray", linewidth=0)
    # Plot centroids and arrows
    for name, c in centroids.items():
        plt.scatter(c[0], c[1], s=60, label=name)
    for name in emb_als_pert_dict.keys():
        c_from = centroids["als"]
        c_to = centroids[name]
        plt.arrow(c_from[0], c_from[1], c_to[0] - c_from[0], c_to[1] - c_from[1],
                  head_width=0.2, length_includes_head=True, color="tab:blue", alpha=0.7)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def barplot_scores(dict_name_to_score: Dict[str, float], title: str, outpath: str) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    items = sorted(dict_name_to_score.items(), key=lambda x: x[1], reverse=True)
    names, scores = zip(*items) if items else ([], [])
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(names), y=list(scores))
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


