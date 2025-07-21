import logging

import numpy as np
from scipy.optimize import least_squares

from .coord_utils import get_bond_list, compute_wij, morse_scaler, align_geom, align_path
from .geodesic import Geodesic

logger = logging.getLogger(__name__)


def _mid_point(atoms, geom1, geom2, tol=1e-2, nudge=0.01, threshold=4):
    # Process the initial geometries, construct coordinate system and obtain average internals
    geom1, geom2 = np.array(geom1), np.array(geom2)
    add_pair = set()
    geom_list = [geom1, geom2]
    # This loop is for ensuring a sufficient large coordinate system.  The interpolated point may
    # have atom pairs in contact that are far away at both end-points, which may cause collision.
    # One can include all atom pairs, but this may blow up for large molecules.  Here the compromise
    # is to use a screened list of atom pairs first, then add more if additional atoms come into
    # contant, then rerun the minimization until the coordinate system is consistant with the
    # interpolated geometry
    while True:
        rijlist, re = get_bond_list(geom_list, threshold=threshold + 1, enforce=add_pair)
        scaler = morse_scaler(alpha=0.7, re=re)
        w1, _ = compute_wij(geom1, rijlist, scaler)
        w2, _ = compute_wij(geom2, rijlist, scaler)
        w = (w1 + w2) / 2
        d_min, x_min = np.inf, None
        friction = 0.1 / np.sqrt(geom1.shape[0])

        def target_func(X):
            """Squared difference with reference w0"""
            wx, dwdR = compute_wij(X, rijlist, scaler)
            delta_w = wx - w
            val, grad = 0.5 * np.dot(delta_w, delta_w), np.einsum('i,ij->j', delta_w, dwdR)
            logger.info("val=%10.3f  ", val)
            return val, grad

        # The inner loop performs minimization using either end-point as the starting guess.
        for coef in [0.02, 0.98]:
            x0 = (geom1 * coef + (1 - coef) * geom2).ravel()
            x0 += nudge * np.random.random_sample(x0.shape)
            logger.debug('Starting least-squares minimization of bisection point at %7.2f.', coef)
            result = least_squares(
                lambda x: np.concatenate([compute_wij(x, rijlist, scaler)[0] - w, (x - x0) * friction]), x0,
                lambda x: np.vstack([compute_wij(x, rijlist, scaler)[1], np.identity(x.size) * friction]), ftol=tol,
                gtol=tol)
            x_mid = result['x'].reshape(-1, 3)
            # Take the interpolated geometry, construct new pair list and check for new contacts
            new_list = geom_list + [x_mid]
            new_rij, _ = get_bond_list(new_list, threshold=threshold, min_neighbors=0)
            extras = set(new_rij) - set(rijlist)
            if extras:
                logger.info('  Screened pairs came into contact. Adding reference point.')
                # Update pair list then go back to the minimization loop if new contacts are found
                geom_list = new_list
                add_pair |= extras
                break
            # Perform local geodesic optimization for the new image.
            smoother = Geodesic(atoms, [geom1, x_mid, geom2], 0.7, threshold=threshold, log_level=logging.DEBUG,
                                friction=1)
            smoother.compute_displacements()
            width = max([np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2]])
            dist, x_mid = width + smoother.length, smoother.path[1]
            logger.debug('  Trial path length: %8.3f after %d iterations', dist, result['nfev'])
            if dist < d_min:
                d_min, x_min = dist, x_mid
        else:  # Both starting guesses finished without new atom pairs.  Minimization successful
            break
    return x_min


def redistribute(atoms, geoms, nimages, tol=1e-2):
    _, geoms = align_path(geoms)
    geoms = list(geoms)
    # If there are too few images, add bisection points
    while len(geoms) < nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i = np.argmax(dists)
        logger.info("Inserting image between %d and %d with Cartesian RMSD %10.3f.  New length:%d",
                    max_i, max_i + 1, dists[max_i], len(geoms) + 1)
        insertion = _mid_point(atoms, geoms[max_i], geoms[max_i + 1], tol)
        _, insertion = align_geom(geoms[max_i], insertion)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms)[1])
    # If there are too many images, remove points
    while len(geoms) > nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i = np.argmin(dists)
        logger.info("Removing image %d.  Cartesian RMSD of merged section %10.3f",
                    min_i + 1, dists[min_i])
        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])
    return geoms
