import logging

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def align_path(path):
    path = np.array(path)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0
    for g, next_g in zip(path, path[1:]):
        rmsd, next_g[:] = align_geom(g, next_g)
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_geom(ref_geom, geom):
    center = np.mean(ref_geom, axis=0)
    ref2 = ref_geom - center
    geom2 = geom - np.mean(geom, axis=0)
    cov = np.dot(geom2.T, ref2)
    v, sv, w = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    new_geom = np.dot(geom2, u) + center
    rmsd = np.sqrt(np.mean((new_geom - ref_geom) ** 2))
    return rmsd, new_geom


ATOMIC_RADIUS = dict(H=0.31,
                     He=0.28,
                     Li=1.28,
                     Be=0.96,
                     B=0.84,
                     C=0.76,
                     N=0.71,
                     O=0.66,
                     F=0.57,
                     Ne=0.58,
                     Na=1.66,
                     Mg=1.41,
                     Al=1.21,
                     Si=1.11,
                     P=1.07,
                     S=1.05,
                     Cl=1.02,
                     Ar=1.06)


def get_bond_list(geom,
                  atoms=None,
                  threshold=4,
                  min_neighbors=4,
                  snapshots=30,
                  bond_threshold=1.8,
                  enforce=()):
    # Type casting and value checks on input parameters
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        # If there is only one geometry or it is flattened, promote to 3d
        geom = geom.reshape(1, -1, 3)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    # Determine which images to be used to determine distances
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
    # Get neighbor list for included geometry and merge them
    rij_set = set(enforce)
    for image in images:
        tree = KDTree(geom[image])
        pairs = tree.query_pairs(threshold)
        rij_set.update(pairs)
        bonded = tree.query_pairs(bond_threshold)
        neighbors = {i: {i} for i in range(geom.shape[1])}
        for i, j in bonded:
            neighbors[i].add(j)
            neighbors[j].add(i)
        for i, j in bonded:
            for ni in neighbors[i]:
                for nj in neighbors[j]:
                    if ni != nj:
                        pair = tuple(sorted([ni, nj]))
                        if pair not in rij_set:
                            rij_set.add(pair)
    rij_list = sorted(rij_set)
    # Check neighbor count to make sure `min_neighbors` is satisfied
    count = np.zeros(geom.shape[1], dtype=int)
    for i, j in rij_list:
        count[i] += 1
        count[j] += 1
    for idx, ct in enumerate(count):
        if ct < min_neighbors:
            _, neighbors = tree.query(geom[-1, idx], k=min_neighbors + 1)
            for i in neighbors:
                if i == idx:
                    continue
                pair = tuple(sorted([i, idx]))
                if pair in rij_set:
                    continue
                else:
                    rij_set.add(pair)
                    rij_list.append(pair)
                    count[i] += 1
                    count[idx] += 1
    if atoms is None:
        re = np.full(len(rij_list), 2.0)
    else:
        radius = np.array([ATOMIC_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms])
        re = np.array([radius[i] + radius[j] for i, j in rij_list])
    logger.debug("Pair list contain %d pairs", len(rij_list))
    return rij_list, re


def compute_rij(geom, rij_list):
    n_rij = len(rij_list)
    rij = np.zeros(n_rij)
    b_mat = np.zeros((n_rij, len(geom), 3))
    for idx, (i, j) in enumerate(rij_list):
        d_vec = geom[i] - geom[j]
        rij[idx] = r = np.sqrt(d_vec[0] * d_vec[0] +
                               d_vec[1] * d_vec[1] +
                               d_vec[2] * d_vec[2])
        grad = d_vec / r
        b_mat[idx, i] = grad
        b_mat[idx, j] = -grad
    return rij, b_mat


def compute_wij(geom, rij_list, func):
    geom = np.asarray(geom).reshape(-1, 3)
    n_rij = len(rij_list)
    rij, b_mat = compute_rij(geom, rij_list)
    wij, d_wdr = func(rij)
    for idx, grad in enumerate(d_wdr):
        b_mat[idx] *= grad
    return wij, b_mat.reshape(n_rij, -1)


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    def scaler(x):
        ratio = x / re
        val1 = np.exp(alpha * (1 - ratio))
        val2 = beta / ratio
        d_val = -alpha / re * val1 - val2 / x
        return val1 + val2, d_val

    return scaler


def elu_scaler(re=2, alpha=2, beta=0.01):
    def scaler(x):
        val1 = (1 - x / re) * alpha + 1
        d_val = np.full(x.shape, -alpha / re)
        large = x > re
        v1l = np.exp(alpha * (1 - x[large] / re))
        val1[large] = v1l
        d_val[large] = -alpha / re * v1l
        val2 = beta * re / x
        return val1 + val2, d_val - val2 / x

    return scaler
