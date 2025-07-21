import logging

import numpy as np

from .geodesic import Geodesic

logger = logging.getLogger(__name__)

from .fileio import from_ase_atoms, to_ase_atoms, read_xyz, write_xyz
from .interpolation import redistribute


def geodesic_interpolate(
        atoms,
        n_images=17,
        output="interpolated.xyz",
        tol=2e-3,
        max_iter=15,
        micro_iter=20,
        scaling=1.7,
        friction=1e-2,
        dist_cutoff=3.0,
        logging_level="INFO",
        seed=42,
):
    np.random.seed(seed)
    logging.basicConfig(format="[%(module)-12s]%(message)s", level=logging_level)
    if isinstance(atoms, list):
        symbols, geometries = from_ase_atoms(atoms)
    elif isinstance(atoms, str):
        symbols, geometries = read_xyz(atoms)
    else:
        raise TypeError("Input must be an ASE Atoms object or a filename.")

    if len(geometries) < 2:
        raise ValueError("Need at least two initial geometries.")

    raw = redistribute(symbols, geometries, n_images, tol=tol * 5)
    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction)

    # If the number of symbols is greater than 35, use sweep smoothing
    sweep = len(symbols) > 35

    if sweep:
        smoother.sweep(tol=tol, max_iter=max_iter, micro_iter=micro_iter)
    else:
        smoother.smooth(tol=tol, max_iter=max_iter)

    if isinstance(atoms, list):
        return to_ase_atoms(symbols, smoother.path)
    else:
        write_xyz(output, symbols, smoother.path)
        return None
