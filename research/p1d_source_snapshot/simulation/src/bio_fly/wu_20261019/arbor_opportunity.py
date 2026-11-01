"""Contact-independent cable quadrature with certified point-to-segment distance.

Integration direction describes geometry only, never synaptic polarity.
"""
import numpy as np
from scipy.spatial import cKDTree


def subdivide(vertices, edges, branches, step=1.):
    if step <= 0:
        raise ValueError('step must be positive')
    vertices = np.asarray(vertices, float)
    edges = np.asarray(edges, int)
    branches = np.asarray(branches)
    seg = vertices[edges]
    lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    keep = np.flatnonzero(lengths > 0)
    counts = np.ceil(lengths[keep] / step).astype(int)
    row = np.repeat(keep, counts)
    count = np.repeat(counts, counts)
    offsets = np.r_[0, np.cumsum(counts)[:-1]] if len(counts) else np.array([], int)
    local = np.arange(counts.sum()) - np.repeat(offsets, counts)
    delta = seg[row, 1] - seg[row, 0]
    a = seg[row, 0] + delta * (local / count)[:, None]
    b = seg[row, 0] + delta * ((local + 1) / count)[:, None]
    return dict(a=a, b=b, midpoint=(a+b)/2, weight=lengths[row]/count,
                original_edge=row, branch_id=branches[row])


def projection_distances(points, a, b):
    """Broadcastable distances to closed segments, including degenerate ones."""
    ab = b-a
    den = np.sum(ab*ab, axis=-1)
    numerator = np.sum((points-a)*ab, axis=-1)
    fraction = np.divide(numerator, den,
                         out=np.zeros_like(numerator), where=den > 0)
    closest = a + np.clip(fraction, 0, 1)[..., None]*ab
    return np.linalg.norm(points-closest, axis=-1)


class SegmentIndex:
    def __init__(self, a, b):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.n = len(self.a)
        self.tree = cKDTree((self.a+self.b)/2) if self.n else None
        self.half = float(np.linalg.norm(self.b-self.a, axis=1).max()/2) if self.n else 0.

    def distances(self, points, initial_k=8):
        points = np.asarray(points, float)
        if not self.n:
            return np.full(len(points), np.nan)
        result = np.full(len(points), np.inf)
        active = np.arange(len(points))
        k = min(max(1, initial_k), self.n)
        while len(active):
            mid_dist, ix = self.tree.query(points[active], k=k)
            mid_dist = np.asarray(mid_dist).reshape(len(active), k)
            ix = np.asarray(ix).reshape(len(active), k)
            exact = projection_distances(points[active, None, :], self.a[ix], self.b[ix]).min(axis=1)
            result[active] = exact
            # Every unseen segment has distance >= kth midpoint distance - max half length.
            certified = (mid_dist[:, -1] >= exact + self.half + 1e-12) | (k == self.n)
            active = active[~certified]
            k = min(self.n, max(k+1, 2*k))
        return result


def cable_kernels(points, weights, target):
    """Undefined empty source/target returns NA; modelling must add missing flags."""
    if not len(points) or not target.n or np.sum(weights) <= 0:
        return np.full(3, np.nan), np.nan
    d = target.distances(points)
    scores = [np.average(np.exp(-d*d/(2*s*s)), weights=weights) for s in [1., 2., 5.]]
    return np.asarray(scores), float(np.average(d, weights=weights))
