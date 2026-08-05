"""Coordinate utilities used by the interpolation program.

Geodesic interpolation works in a redundant internal coordinate system built from
scaled inter-atomic distances.  This module provides the pieces needed to set that
system up: alignment of geometries and paths (Kabsch), selection of the atom pairs
that make up the coordinates, evaluation of those coordinates together with their
Cartesian derivatives (the Wilson B matrix), and the scaling functions that define
the metric.
"""
import logging
from typing import Callable
from typing import List, Tuple, Optional

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def align_path(path: np.ndarray) -> tuple[float, np.ndarray]:
    """Rotate and translate the images of a path to minimise RMSD movement along it.

    The first image is shifted so its geometric centre sits at the origin, and each
    following image is aligned against the one before it, which leaves every image
    centred on the origin as well.

    Args:
        path: Sequence of geometries, of shape ``(n_images, n_atoms, 3)``.

    Returns:
        max_rmsd: Largest RMSD between any two adjacent images after alignment.
        path: The aligned path.  This is a copy, so the input is left untouched.
    """
    path = np.array(path)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0.0
    for g, next_g in zip(path, path[1:]):
        rmsd, aligned_geom = align_geom(g, next_g)
        next_g[:] = aligned_geom
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_geom(ref_geom: np.ndarray, geom: np.ndarray) -> tuple[float, np.ndarray]:
    """Find the rigid-body motion that maximally overlaps one geometry with another.

    Implemented with the Kabsch algorithm: both geometries are centred, the SVD of
    their covariance matrix gives the optimal rotation, and the result is moved back
    onto the centre of the reference.

    Args:
        ref_geom: The reference geometry to rotate towards.
        geom: The geometry to be rotated and shifted.

    Returns:
        rmsd: Root-mean-squared difference between the rotated geometry and the
            reference.
        aligned_geom: The rotated geometry that maximally overlaps the reference.
    """
    center = np.mean(ref_geom, axis=0)
    ref_geom_centered = ref_geom - center
    geom_centered = geom - np.mean(geom, axis=0)

    cov = np.dot(geom_centered.T, ref_geom_centered)
    v, _, w = np.linalg.svd(cov)

    # A negative determinant means the SVD produced a reflection rather than a
    # rotation.  Flipping the least significant axis gives a proper rotation.
    if np.linalg.det(v) * np.linalg.det(w) < 0.0:
        v[:, -1] *= -1

    rotation_matrix = np.dot(v, w)
    aligned_geom = np.dot(geom_centered, rotation_matrix) + center
    rmsd = np.sqrt(np.mean((aligned_geom - ref_geom) ** 2))

    return rmsd, aligned_geom


def get_bond_list(geom: np.ndarray,
                  atoms: Optional[List[str]] = None,
                  threshold: float = 4.0,
                  min_neighbors: int = 4,
                  snapshots: int = 30,
                  bond_threshold: float = 1.8,
                  enforce: Tuple[Tuple[int, int], ...] = ()) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Get the list of atom pairs that define the internal coordinate system.

    Samples images from the path and collects every pair of atoms that comes within
    ``threshold`` of each other in any of them.  Pairs linked by three or fewer bonds
    are always included, as are the pairs given in ``enforce``, and further pairs are
    added for any atom left with fewer than ``min_neighbors`` neighbours.

    Args:
        geom: A single geometry or a path of them.  Anything that is not already
            ``(n_images, n_atoms, 3)`` is promoted to that shape.
        atoms: Element symbols, used to look up covalent radii.  When omitted, every
            pair is given a nominal equilibrium distance of 2.0 Angstrom instead.
        threshold: Distance cut-off for including a pair in the coordinate system.
        min_neighbors: Minimum number of pairs each atom must take part in.  Atoms
            below this get their nearest neighbours added.
        snapshots: Maximum number of images to sample.  Keeps the cost down when the
            path is long and the atoms numerous.
        bond_threshold: Distance below which two atoms count as bonded, used to work
            out which pairs are within three bonds of each other.
        enforce: Pairs to include regardless of how far apart the atoms are.

    Returns:
        rij_list: Sorted list of the ``(i, j)`` atom pairs making up the coordinates.
        re: Equilibrium distance for each pair, taken as the sum of the two covalent
            radii.
    """
    # Type casting and value checks on the input parameters
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        # A single geometry, or a flattened one, is promoted to 3d
        geom = geom.reshape(1, -1, 3)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    # Always look at both end points, plus a random selection of the images between
    # them, so that a long path costs no more to analyse than a short one
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
    # Build the neighbour list for each sampled image and merge them together
    rij_set = set(enforce)
    for image in images:
        tree = KDTree(geom[image])
        pairs = tree.query_pairs(threshold)
        rij_set.update(pairs)
        # Anything within three bonds of each other is included whatever the
        # distance: take each bonded pair and join up their neighbours
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
    # Count how many pairs each atom appears in, so `min_neighbors` can be checked
    count = np.zeros(geom.shape[1], dtype=int)
    for i, j in rij_list:
        count[i] += 1
        count[j] += 1
    # Top up any under-connected atom with its nearest neighbours.  This reuses the
    # KD-tree left over from the sampling loop, which is the last image visited.
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
        atom_numbers = [atomic_numbers[atom.capitalize()] for atom in atoms]
        radius = np.array([covalent_radii[num] for num in atom_numbers])
        re = np.array([radius[i] + radius[j] for i, j in rij_list])
    logger.debug("Pair list contain %d pairs", len(rij_list))
    return rij_list, re


def compute_rij(geom: np.ndarray,
                rij_list: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a list of inter-atomic distances and their Cartesian derivatives.

    Args:
        geom: Cartesian geometry of all the atoms, shape ``(n_atoms, 3)``.
        rij_list: Indices of the atom pairs to evaluate.

    Returns:
        rij: The distance for each pair.
        b_mat: Wilson B matrix, of shape ``(n_pairs, n_atoms, 3)``, holding the
            Cartesian gradient of every distance.
    """
    n_rij = len(rij_list)
    rij = np.zeros(n_rij)
    b_mat = np.zeros((n_rij, len(geom), 3))

    for idx, (i, j) in enumerate(rij_list):
        d_vec = geom[i] - geom[j]
        r = np.linalg.norm(d_vec)
        rij[idx] = r
        # A distance only depends on its own two atoms, and moving one of them is
        # the exact opposite of moving the other
        grad = d_vec / r
        b_mat[idx, i] = grad
        b_mat[idx, j] = -grad

    return rij, b_mat


def compute_wij(geom: np.ndarray,
                rij_list: List[Tuple[int, int]],
                func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate a list of scaled distances and their Cartesian derivatives.

    Same as `compute_rij`, except each distance is passed through a scaling function
    which sets the metric of the internal coordinates.

    Args:
        geom: Cartesian geometry of all the atoms.  Flattened input is accepted and
            reshaped to ``(n_atoms, 3)``.
        rij_list: Indices of the atom pairs to evaluate.
        func: Scaling function returning both the scaled value and its derivative
            with respect to the raw distance.  Must broadcast over arrays.

    Returns:
        wij: The scaled distance for each pair.
        b_mat: Cartesian gradients of the scaled distances, with the atom and
            component axes flattened together so scipy.optimize can use it directly.
    """
    geom = np.asarray(geom).reshape(-1, 3)
    rij, b_mat = compute_rij(geom, rij_list)
    wij, d_wdr = func(rij)
    # Chain rule: scale each pair's gradient by dw/dr for that pair
    b_mat *= d_wdr[:, None, None]
    return wij, b_mat.reshape(len(rij_list), -1)


def morse_scaler(re: float = 1.5, alpha: float = 1.7, beta: float = 0.01) -> Callable[
    [np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build a scaling function based on a Morse potential.

    The returned function takes an inter-nuclear distance and gives back the scaled
    distance along with its derivative with respect to the unscaled one.  A small
    ``beta / r`` tail is added so that the coordinate keeps responding to atoms that
    are far apart, where the exponential has already decayed away.

    Args:
        re: Equilibrium distance.  Usually the per-pair array returned by
            `get_bond_list` rather than a single number.
        alpha: Decay constant of the exponential.  Larger values are more localised,
            which tracks a sharp energy landscape better, while smaller values have
            longer range and give smoother paths from few images.
        beta: Weight of the long-range tail.
    """

    def scaler(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ratio = x / re
        val1 = np.exp(alpha * (1.0 - ratio))
        val2 = beta / ratio
        d_val = (-alpha * val1 / re) - (val2 / x)
        return val1 + val2, d_val

    return scaler


def elu_scaler(re: float = 2.0, alpha: float = 2.0, beta: float = 0.01) -> Callable[
    [np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build a scaling function with an exponential tail and a linear core.

    Shaped like an ELU: beyond ``re`` the scaled distance decays exponentially, as in
    `morse_scaler`, while below ``re`` it continues linearly along the tangent at
    ``re`` instead of blowing up.  A ``beta * re / r`` tail is added for the same
    reason as in the Morse case.

    The returned function takes an inter-nuclear distance and gives back the scaled
    distance along with its derivative with respect to the unscaled one.

    Args:
        re: Distance at which the behaviour switches from linear to exponential.
        alpha: Decay constant of the exponential, which also sets the linear slope.
        beta: Weight of the long-range tail.
    """

    def scaler(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        val1 = np.where(x > re, np.exp(alpha * (1.0 - x / re)), (1.0 - x / re) * alpha + 1.0)
        d_val = np.where(x > re, -alpha / re * np.exp(alpha * (1.0 - x / re)), -alpha / re)
        val2 = beta * re / x
        return val1 + val2, d_val - val2 / x

    return scaler
