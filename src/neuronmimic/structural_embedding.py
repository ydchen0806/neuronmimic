"""Exact directed embedding function reused from BioFly kc_weighted_structure.py."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

def _embedding(adjacency: sparse.csr_matrix, dimensions: int, random_seed: int) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    if adjacency.nnz == 0 or n_nodes < 3:
        raise ValueError("cannot embed an empty or too-small directed graph")
    outgoing = np.asarray(adjacency.sum(axis=1)).ravel()
    incoming = np.asarray(adjacency.sum(axis=0)).ravel()
    left = sparse.diags(np.divide(1.0, np.sqrt(outgoing), out=np.zeros_like(outgoing), where=outgoing > 0.0))
    right = sparse.diags(np.divide(1.0, np.sqrt(incoming), out=np.zeros_like(incoming), where=incoming > 0.0))
    normalized = left @ adjacency @ right
    k = min(max(2, int(dimensions)), n_nodes - 1)
    try:
        vectors_left, singular, vectors_right = randomized_svd(
            normalized,
            n_components=k,
            n_iter=7,
            random_state=random_seed,
            flip_sign=True,
        )
    except Exception as error:
        raise RuntimeError(f"directed spectral embedding failed: {error}") from error
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    vectors_left = vectors_left[:, order]
    vectors_right = vectors_right[order, :].T
    scale = np.sqrt(np.maximum(singular, 0.0))
    combined = np.concatenate((vectors_left * scale, vectors_right * scale), axis=1)
    return StandardScaler().fit_transform(combined)
