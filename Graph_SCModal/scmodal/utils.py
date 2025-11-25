import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import umap
from annoy import AnnoyIndex
from scmodal.model import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.sparse import issparse


def acquire_pairs(X, Y, k=30, metric='angular', build_trees_n_trees=10, densify_threshold=50_000_000):
    """
    Find mutual nearest-neighbor pairs between X and Y using Annoy.
    Works with dense numpy arrays and scipy sparse matrices.

    Parameters
    ----------
    X, Y : array-like or scipy.sparse
        Shape: (n_samples, n_features)
    k : int
        Number of neighbors to retrieve.
    metric : str
        Annoy metric ('angular', 'euclidean', 'manhattan', 'hamming', 'dot').
    build_trees_n_trees : int
        Number of trees to build for Annoy index.
    densify_threshold : int
        If X or Y (after densification) would allocate more than this many elements,
        warn/raise to avoid accidental OOM. Set to None to ignore.

    Returns
    -------
    mnn_mat : numpy.ndarray, shape (n_X, n_Y)
        Binary matrix where mnn_mat[i, j] == 1 if i and j are mutual nearest neighbors.
    """
    # Validate shapes (works for sparse too)
    nX, f = X.shape
    nY, f2 = Y.shape
    if f != f2:
        raise ValueError("X and Y must have same number of features (columns)")

    # Helper: convert to dense (numpy) safely
    def to_dense(arr, name):
        if issparse(arr):
            # memory check (number of elements)
            n_elements = arr.shape[0] * arr.shape[1]
            if densify_threshold is not None and n_elements > densify_threshold:
                raise MemoryError(
                    f"Trying to densify {name} with {n_elements} elements — exceeds threshold {densify_threshold}. "
                    "Reduce batch size or increase densify_threshold."
                )
            return arr.toarray().astype(np.float32)
        else:
            return np.asarray(arr, dtype=np.float32)

    Xd = to_dense(X, "X")
    Yd = to_dense(Y, "Y")

    # Build Annoy indices
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)

    # Annoy needs lists of floats (use float32)
    for i in range(nX):
        t1.add_item(i, Xd[i].tolist())
    for i in range(nY):
        t2.add_item(i, Yd[i].tolist())

    t1.build(build_trees_n_trees)
    t2.build(build_trees_n_trees)

    # For each X row, get top-k neighbours from Y
    # NOTE: get_nns_by_vector returns indices (len <= k)
    sorted_from_X = [t2.get_nns_by_vector(
        Xd[i].tolist(), k) for i in range(nX)]
    mnn_mat = np.zeros((nX, nY), dtype=int)
    for i, neighs in enumerate(sorted_from_X):
        if len(neighs) > 0:
            mnn_mat[i, neighs] = 1

    # For each Y row, get top-k neighbours from X
    sorted_from_Y = [t1.get_nns_by_vector(
        Yd[j].tolist(), k) for j in range(nY)]
    tmp = np.zeros_like(mnn_mat)
    for j, neighs in enumerate(sorted_from_Y):
        if len(neighs) > 0:
            tmp[neighs, j] = 1

    # mutual nearest neighbors
    mnn_mat = (mnn_mat & tmp).astype(int)
    return mnn_mat


def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count)
              for i in idx]
    return labels


def compute_umap(adata, rep=None):
    import umap

    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    if rep is None:
        X_umap = reducer.fit_transform(adata.X)
    else:
        X_umap = reducer.fit_transform(adata.obsm[rep])

    adata.obsm['X_umap'] = X_umap
