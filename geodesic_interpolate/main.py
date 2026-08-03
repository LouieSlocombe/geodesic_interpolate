"""Top-level driver tying the interpolation and smoothing stages together."""
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
    """Interpolate a reaction path between two or more geometries.

    Runs the two stages in turn: `redistribute` builds a raw path with the requested
    number of images, then `Geodesic` smooths it into a geodesic under the internal
    coordinate metric.

    Input and output mirror each other.  Given ASE Atoms objects the interpolated path
    comes back as Atoms objects; given a filename it is written to `output` instead.

    Args:
        atoms: Either a list of ASE Atoms objects, or the name of an XYZ file holding
            the end points.  Only the first and last geometries need be meaningful,
            but intermediate ones are used if present.
        n_images: Number of images in the interpolated path.
        output: XYZ file to write to.  Only used when `atoms` is a filename.
        tol: Convergence tolerance for the smoothing.
        max_iter: Maximum number of iterations, or sweeps in the sweeping case.
        micro_iter: Micro-iterations per image, used only when sweeping.
        scaling: Alpha parameter of the Morse scaler setting the coordinate metric.
        friction: Weight of the friction term regularising the optimization step size.
        dist_cutoff: Distance cut-off for building the internal coordinates.
        logging_level: Logging level for the progress output.
        seed: Seed for the random nudges and image sampling, so runs reproduce.  The
            bisection is stochastic, and without a fixed seed larger systems will not
            give the same path twice.

    Returns:
        The interpolated path as a list of ASE Atoms objects when `atoms` was a list of
        them, otherwise None, with the path written to `output`.

    Raises:
        TypeError: If `atoms` is neither a list of ASE Atoms nor a filename.
        ValueError: If fewer than two geometries are supplied.
    """
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

    # A looser tolerance is enough for the raw path, which only has to be a decent
    # starting guess for the smoothing that follows
    raw = redistribute(symbols, geometries, n_images, tol=tol * 5)
    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction)

    # Optimizing the whole path at once is faster, but scipy's optimizers slow down
    # badly as the system grows, so past this size sweep one image at a time instead
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
