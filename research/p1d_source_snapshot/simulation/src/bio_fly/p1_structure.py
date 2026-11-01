"""P1-v2 primitives: pair isolation and contact-free geometry, no label promotion."""
from __future__ import annotations
import hashlib
import numpy as np


def pair_uniform(ids, pre, post, seed=20261030):
    """Order-invariant identity hash; every direction/contact of a pair shares a draw."""
    tokens = np.array([int.from_bytes(hashlib.blake2b(str(i).encode(), digest_size=8).digest(), 'little') for i in ids], dtype=np.uint64)
    if len(np.unique(tokens)) != len(tokens):
        raise ValueError('duplicate ID or hash collision')
    a, b = tokens[np.asarray(pre)], tokens[np.asarray(post)]
    lo, hi = np.minimum(a,b), np.maximum(a,b)
    with np.errstate(over='ignore'):
        x = lo ^ (hi * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(seed)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / 2**53


def pair_split(ids, pre, post, seed=20261030):
    u=pair_uniform(ids,pre,post,seed)
    return np.where(u<.70,0,np.where(u<.85,1,2)).astype(np.int8)


def tree_features(d):
    """Only vertices, tree edges, branch IDs and skeleton paths; ignore contacts."""
    v=np.asarray(d['vertices'],float); e=np.asarray(d['edges'],int)
    if not np.isfinite(v).all() or v.shape[1]!=3:
        raise ValueError('invalid geometry')
    length=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)
    if length.sum()<=0: raise ValueError('zero cable length')
    mid=(v[e[:,0]]+v[e[:,1]])/2
    center=np.average(mid,axis=0,weights=length)
    delta=mid-center
    cov=(delta*length[:,None]).T@delta/length.sum()
    eigen=np.maximum(np.linalg.eigvalsh(cov),0)
    degree=np.bincount(e.ravel(),minlength=len(v))
    path=np.asarray(d['path_from_soma_nearest_um'],float)
    finite=path[np.isfinite(path)]
    out={'cable_length_um':length.sum(),'n_branch_ids':len(set(d['branch_per_edge'])),
         'n_terminals':int((degree==1).sum()),'n_branchpoints':int((degree>2).sum()),
         'edge_length_median_um':np.median(length),'edge_length_p95_um':np.quantile(length,.95),
         'path_median_um':np.median(finite),'path_p95_um':np.quantile(finite,.95)}
    for j,axis in enumerate('xyz'):
        out[f'centroid_{axis}_um']=center[j]
        out[f'extent_{axis}_um']=np.ptp(v[:,j])
        out[f'principal_sd_{j}_um']=np.sqrt(eigen[j])
    return out


def fuse_blocks(graph, soma, tree, soma_weight, tree_weight):
    """Squared distance = graph distance + weighted soma/tree distances."""
    blocks=[graph/np.sqrt(graph.shape[1])]
    if soma_weight: blocks.append(np.sqrt(soma_weight)*soma)
    if tree_weight: blocks.append(np.sqrt(tree_weight)*tree/np.sqrt(tree.shape[1]))
    return np.column_stack(blocks)
