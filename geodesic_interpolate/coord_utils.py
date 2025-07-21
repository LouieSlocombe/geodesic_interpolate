import logging

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def align_path(path):
    path = np.array(path)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0.0
    for g, next_g in zip(path, path[1:]):
        rmsd, aligned_geom = align_geom(g, next_g)
        next_g[:] = aligned_geom
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_geom(ref_geom, geom):
    center = np.mean(ref_geom, axis=0)
    ref_geom_centered = ref_geom - center
    geom_centered = geom - np.mean(geom, axis=0)

    cov = np.dot(geom_centered.T, ref_geom_centered)
    v, _, w = np.linalg.svd(cov)

    if np.linalg.det(v) * np.linalg.det(w) < 0.0:
        v[:, -1] *= -1

    rotation_matrix = np.dot(v, w)
    aligned_geom = np.dot(geom_centered, rotation_matrix) + center
    rmsd = np.sqrt(np.mean((aligned_geom - ref_geom) ** 2))

    return rmsd, aligned_geom


def get_bond_list(geom,
                  atoms=None,
                  threshold=4.0,
                  min_neighbors=4,
                  snapshots=30,
                  bond_threshold=1.8,
                  enforce=()):
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        geom = geom.reshape(1, -1, 3)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
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
        # Use covalent_radii from ase.data
        atom_numbers = [atomic_numbers[atom.capitalize()] for atom in atoms]
        radius = np.array([covalent_radii[num] for num in atom_numbers])
        re = np.array([radius[i] + radius[j] for i, j in rij_list])
    logger.debug("Pair list contain %d pairs", len(rij_list))
    return rij_list, re


def compute_rij(geom, rij_list):
    n_rij = len(rij_list)
    rij = np.zeros(n_rij)
    b_mat = np.zeros((n_rij, len(geom), 3))

    for idx, (i, j) in enumerate(rij_list):
        d_vec = geom[i] - geom[j]
        r = np.linalg.norm(d_vec)
        rij[idx] = r
        grad = d_vec / r
        b_mat[idx, i] = grad
        b_mat[idx, j] = -grad

    return rij, b_mat


def compute_wij(geom, rij_list, func):
    geom = np.asarray(geom).reshape(-1, 3)
    rij, b_mat = compute_rij(geom, rij_list)
    wij, d_wdr = func(rij)
    b_mat *= d_wdr[:, None, None]
    return wij, b_mat.reshape(len(rij_list), -1)


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    def scaler(x):
        ratio = x / re
        val1 = np.exp(alpha * (1.0 - ratio))
        val2 = beta / ratio
        d_val = (-alpha * val1 / re) - (val2 / x)
        return val1 + val2, d_val

    return scaler


def elu_scaler(re=2.0, alpha=2.0, beta=0.01):
    def scaler(x):
        val1 = np.where(x > re, np.exp(alpha * (1.0 - x / re)), (1.0 - x / re) * alpha + 1.0)
        d_val = np.where(x > re, -alpha / re * np.exp(alpha * (1.0 - x / re)), -alpha / re)
        val2 = beta * re / x
        return val1 + val2, d_val - val2 / x

    return scaler
