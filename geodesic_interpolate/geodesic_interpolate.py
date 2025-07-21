import logging

import numpy as np

from .fileio import read_xyz, write_xyz
from .geodesic import Geodesic
from .interpolation import redistribute

logger = logging.getLogger(__name__)


def interpolate(
        filename,
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
        save_raw=None,
        seed=42,
):
    # set a random seed for reproducibility
    np.random.seed(seed)

    # Setup logging based on designated logging level
    logging.basicConfig(format="[%(module)-12s]%(message)s", level=logging_level)

    # Read the initial geometries.
    symbols, geometries = read_xyz(filename)
    logger.info('Loaded %d geometries from %s', len(geometries), filename)
    if len(geometries) < 2:
        raise ValueError("Need at least two initial geometries.")

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(symbols, geometries, n_images, tol=tol * 5)
    if save_raw is not None:
        write_xyz(save_raw, symbols, raw)

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction)
    if sweep is None:
        sweep = len(symbols) > 35
    try:
        if sweep:
            smoother.sweep(tol=tol, max_iter=max_iter, micro_iter=micro_iter)
        else:
            smoother.smooth(tol=tol, max_iter=max_iter)
    finally:
        # Save the smoothed path to output file.
        logging.info('Saving final path to file %s', output)
        write_xyz(output, symbols, smoother.path)
