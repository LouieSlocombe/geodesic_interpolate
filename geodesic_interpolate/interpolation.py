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
    geom1, geom2 = np.array(geom1), np.array(geom2)
    add_pair: set = set()
    geom_list: list[np.ndarray] = [geom1, geom2]

    while True:
        rij_list, re = get_bond_list(geom_list, threshold=threshold + 1.0, enforce=add_pair)
        scaler = morse_scaler(alpha=0.7, re=re)
        w = (compute_wij(geom1, rij_list, scaler)[0] + compute_wij(geom2, rij_list, scaler)[0]) / 2
        d_min: float = np.inf
        x_min: np.ndarray | None = None
        friction: float = 0.1 / np.sqrt(geom1.shape[0])

        for coef in [0.02, 0.98]:
            x0: np.ndarray = (geom1 * coef + geom2 * (1 - coef)).ravel() + nudge * np.random.random_sample(geom1.size)
            logger.debug("Starting least-squares minimization of bisection point at %7.2f.", coef)
            result = least_squares(
                lambda x: np.concatenate([compute_wij(x, rij_list, scaler)[0] - w, (x - x0) * friction]),
                x0,
                lambda x: np.vstack([compute_wij(x, rij_list, scaler)[1], np.identity(x.size) * friction]),
                ftol=tol,
                gtol=tol,
            )
            x_mid: np.ndarray = result["x"].reshape(-1, 3)
            new_rij, _ = get_bond_list(geom_list + [x_mid], threshold=threshold, min_neighbors=0)
            extras = set(new_rij) - set(rij_list)

            if extras:
                logger.info("  Screened pairs came into contact. Adding reference point.")
                geom_list.append(x_mid)
                add_pair.update(extras)
                break

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
            break

    return x_min


def redistribute(atoms: Any, geoms: List[np.ndarray], n_images: int, tol: float = 1e-2) -> List[np.ndarray]:
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
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i: int = int(np.argmin(dists))
        logger.info(
            "Removing image %d. Cartesian RMSD of merged section %10.3f",
            min_i + 1, dists[min_i]
        )
        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])

    return geoms
