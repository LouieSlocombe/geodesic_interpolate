import logging

import numpy as np

from .geodesic import Geodesic

logger = logging.getLogger(__name__)

from .fileio import from_ase_atoms, to_ase_atoms, read_xyz, write_xyz
from .interpolation import redistribute


def interpolate(
        atoms,
        n_images=17,
        sweep=None,
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
    # set a random seed for reproducibility
    np.random.seed(seed)

    # Setup logging based on designated logging level
    logging.basicConfig(format="[%(module)-12s]%(message)s", level=logging_level)

    # Check if the input is an ASE Atoms object or a filename.
    if isinstance(atoms, list):
        # If it's an ASE Atoms object, convert it to symbols and geometry.
        symbols, geometries = from_ase_atoms(atoms)
    elif isinstance(atoms, str):
        # If it's a filename, read the geometries from the file.
        symbols, geometries = read_xyz(atoms)
    else:
        raise TypeError("Input must be an ASE Atoms object or a filename.")

    if len(geometries) < 2:
        raise ValueError("Need at least two initial geometries.")

    raw = redistribute(symbols, geometries, n_images, tol=tol * 5)

    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction)
    if sweep is None:
        sweep = len(symbols) > 35
    try:
        if sweep:
            smoother.sweep(tol=tol, max_iter=max_iter, micro_iter=micro_iter)
        else:
            smoother.smooth(tol=tol, max_iter=max_iter)
    except ValueError as e:
        logger.error(f"Error during smoothing: {e}")

    if isinstance(atoms, list):
        return to_ase_atoms(symbols, smoother.path)
    else:
        write_xyz(output, symbols, smoother.path)
        return None
