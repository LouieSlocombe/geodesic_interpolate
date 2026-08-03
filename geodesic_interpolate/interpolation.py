"""Simplified geodesic interpolation module.

Uses geodesic lengths as the criterion for adding bisection points until the image
count matches what was asked for.  The result is only a starting guess: a following
round of geodesic smoothing is needed to get the final path.
"""
import logging
from typing import Any, List

import numpy as np
from scipy.optimize import least_squares

from .coord_utils import align_geom, align_path
from .coord_utils import get_bond_list, compute_wij, morse_scaler
from .geodesic import Geodesic

logger = logging.getLogger(__name__)


def _mid_point(atoms: Any,
               geom1: np.ndarray,
               geom2: np.ndarray,
               tol: float = 1e-2,
               nudge: float = 0.01,
               threshold: float = 4.0) -> np.ndarray:
    """Find the geometry whose internal coordinates sit closest to the average of two others.

    A least-squares minimisation against the average of the two end points, run twice,
    starting from just beside either end point.  DON'T USE THE CARTESIAN AVERAGE AS THE
    GUESS, THINGS WILL BLOW UP.  The two runs are then compared by local geodesic length
    and the shorter one wins.

    The point produced here need not join smoothly onto either end point; it only has to
    be a good enough starting guess for the smoothing that follows.

    Random nudges are added to the starting geometry, so repeated runs need not converge
    to the same answer — for larger systems they essentially never will.  Running several
    times and keeping the best result is therefore worthwhile.

    Args:
        atoms: Atom symbols, used to look up covalent radii.
        geom1, geom2: Cartesian geometries of the two end points.
        tol: Convergence tolerance for the least-squares minimisation.
        nudge: Size of the random nudge added to the starting geometry.  Helps to turn
            up different solutions, and to break symmetry when the optimal path does.
        threshold: Distance cut-off for including an atom pair in the coordinates.

    Returns:
        The optimised mid-point, bisecting the two end points in internal coordinates.
    """
    geom1, geom2 = np.array(geom1), np.array(geom2)
    add_pair: set = set()
    geom_list: list[np.ndarray] = [geom1, geom2]

    # The outer loop makes sure the coordinate system is large enough.  The interpolated
    # point can bring atom pairs into contact that are far apart at both end points,
    # which would let them collide unnoticed.  Including every pair would blow up for
    # large molecules, so the compromise is to start from a screened list, add any pair
    # that comes into contact, and redo the minimisation until the coordinate system and
    # the interpolated geometry agree.
    while True:
        rij_list, re = get_bond_list(geom_list, threshold=threshold + 1.0, enforce=add_pair)
        scaler = morse_scaler(alpha=0.7, re=re)
        w = (compute_wij(geom1, rij_list, scaler)[0] + compute_wij(geom2, rij_list, scaler)[0]) / 2
        d_min: float = np.inf
        x_min: np.ndarray | None = None
        friction: float = 0.1 / np.sqrt(geom1.shape[0])

        # The inner loop minimises from either end point in turn as the starting guess
        for coef in [0.02, 0.98]:
            x0: np.ndarray = (geom1 * coef + geom2 * (1 - coef)).ravel() + nudge * np.random.random_sample(geom1.size)
            logger.debug("Starting least-squares minimization of bisection point at %7.2f.", coef)
            # Residuals are the difference from the target internals, plus a friction
            # term holding the geometry near where it started
            result = least_squares(
                lambda x: np.concatenate([compute_wij(x, rij_list, scaler)[0] - w, (x - x0) * friction]),
                x0,
                lambda x: np.vstack([compute_wij(x, rij_list, scaler)[1], np.identity(x.size) * friction]),
                ftol=tol,
                gtol=tol,
            )
            x_mid: np.ndarray = result["x"].reshape(-1, 3)
            # Rebuild the pair list including the new point and check for fresh contacts
            new_rij, _ = get_bond_list(geom_list + [x_mid], threshold=threshold, min_neighbors=0)
            extras = set(new_rij) - set(rij_list)

            if extras:
                logger.info("  Screened pairs came into contact. Adding reference point.")
                # Widen the coordinate system and start the minimisation over
                geom_list.append(x_mid)
                add_pair.update(extras)
                break

            # Score this candidate by locally smoothing the three-image path through it
            smoother = Geodesic(atoms,
                                [geom1, x_mid, geom2],
                                scaler=0.7,
                                threshold=threshold,
                                log_level=logging.DEBUG,
                                friction=1)
            smoother.compute_displacements()
            width = max(np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2])
            dist = width + smoother.length

            logger.debug("  Trial path length: %8.3f after %d iterations", dist, result["nfev"])
            if dist < d_min:
                d_min, x_min = dist, smoother.path[1]
        else:
            # Both starting guesses finished without new atom pairs, so the coordinate
            # system held up and the minimisation is done
            break

    return x_min


def redistribute(atoms: Any, geoms: List[np.ndarray], n_images: int, tol: float = 1e-2) -> List[np.ndarray]:
    """Add or remove images so the path has the requested number of them.

    If there are too few, new points are added by bisecting the largest RMSD gap.  If
    there are too many, images are dropped one at a time, each time choosing the one
    whose removal leaves the shortest merged segment.

    Args:
        atoms: Atom symbols, used to look up covalent radii.
        geoms: Geometries of the original path.
        n_images: The desired number of images.
        tol: Convergence tolerance for the bisection.

    Returns:
        An aligned path with the correct number of images.
    """
    _, geoms = align_path(geoms)
    geoms = list(geoms)

    # Add bisection points if there are too few images
    while len(geoms) < n_images:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i: int = int(np.argmax(dists))
        logger.info(
            "Inserting image between %d and %d with Cartesian RMSD %10.3f. New length: %d",
            max_i, max_i + 1, dists[max_i], len(geoms) + 1
        )
        insertion: np.ndarray = _mid_point(atoms, geoms[max_i], geoms[max_i + 1], tol)
        _, insertion = align_geom(geoms[max_i], insertion)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms)[1])

    # Remove points if there are too many images
    while len(geoms) > n_images:
        # Each distance here spans two segments, so the smallest one marks the image
        # that can be dropped with the least disruption
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i: int = int(np.argmin(dists))
        logger.info(
            "Removing image %d. Cartesian RMSD of merged section %10.3f",
            min_i + 1, dists[min_i]
        )
        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])

    return geoms
