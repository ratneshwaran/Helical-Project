from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def centroid(X: np.ndarray) -> np.ndarray:
    """Compute centroid of points (rows)."""
    return np.mean(X, axis=0, keepdims=True)


def delta_to_healthy(emb_als_base: np.ndarray, emb_als_pert: np.ndarray, emb_healthy: np.ndarray) -> float:
    """Change in distance to healthy centroid (negative = improvement)."""
    c_healthy = centroid(emb_healthy)
    d_base = np.mean(np.linalg.norm(emb_als_base - c_healthy, axis=1))
    d_pert = np.mean(np.linalg.norm(emb_als_pert - c_healthy, axis=1))
    return d_pert - d_base


def wasserstein1d_along_pc(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Project on first principal component of Xa ∪ Xb, compute 1D Wasserstein."""
    X = np.vstack([Xa, Xb])
    pc = PCA(n_components=1, random_state=42).fit(X)
    a1 = pc.transform(Xa)[:, 0]
    b1 = pc.transform(Xb)[:, 0]
    return float(wasserstein_distance(a1, b1))


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, len(X) - 1), algorithm="auto")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    return idx


def knn_overlap_fraction(X_ref: np.ndarray, X_query: np.ndarray, k: int = 15) -> float:
    """Fraction overlap in kNN between reference and query sets.

    This computes the average fraction of neighbors that are shared when comparing
    kNN within the union space. Here we approximate by comparing neighbor sets of
    X_query to those of X_ref when both are pooled.
    """
    X = np.vstack([X_ref, X_query])
    idx = _knn_indices(X, k)
    n_ref = len(X_ref)
    overlaps = []
    for i in range(n_ref, len(X)):
        nbrs_query = set(idx[i])
        nbrs_ref = set(idx[i - n_ref])
        # Remove self indices
        nbrs_query.discard(i)
        nbrs_ref.discard(i - n_ref)
        inter = len(nbrs_query & nbrs_ref)
        denom = max(1, len(nbrs_query | nbrs_ref))
        overlaps.append(inter / denom)
    return float(np.mean(overlaps)) if overlaps else 0.0


def silhouette_scores_by_label(X: np.ndarray, labels: Iterable) -> float:
    labels = np.array(list(labels))
    if len(set(labels)) < 2:
        return 0.0
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return 0.0


def composite_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted sum over components.

    Missing components count as 0.
    """
    total = 0.0
    for k, w in weights.items():
        total += w * float(components.get(k, 0.0))
    return float(total)


