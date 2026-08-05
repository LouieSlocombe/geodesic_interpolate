"""Geodesic smoothing.

Minimises the length of a reaction path measured with a redundant internal coordinate
metric, but optimises the Cartesian coordinates directly.  Working in Cartesians avoids
the feasibility problems that come with redundant internals, where an arbitrary set of
internal coordinate values need not correspond to any real geometry.
"""
import logging

import numpy as np
from scipy.optimize import least_squares

from .coord_utils import align_path, get_bond_list, morse_scaler, compute_wij

logger = logging.getLogger(__name__)


class Geodesic(object):
    """Optimizer that finds a geodesic in redundant internal coordinates.

    The heart of it is the path length in the internal metric: the length of each
    segment is measured through its midpoint, so that the target function is a sum of
    squared displacements and can be handed straight to a least-squares optimizer.
    """

    def __init__(self,
                 atoms,
                 path,
                 scaler=1.7,
                 threshold=3.0,
                 min_neighbors=4,
                 log_level=logging.INFO,
                 friction=1e-3):
        """Initialise the interpolater.

        Args:
            atoms: Atom symbols, used to look up covalent radii.
            path: Initial geometries of the path, of shape ``(n_images, n_atoms, 3)``.
            scaler: Either the alpha parameter for the Morse scaler, or an explicit
                scaling function.  Small values have long range and make it easier to
                get smooth paths out of few images; larger values usually give better
                energetics because they represent the sharp energy landscape better.
            threshold: Distance cut-off for building the inter-nuclear distance
                coordinates.  Atoms linked by three or fewer bonds are added too,
                whatever their distance.
            min_neighbors: Minimum number of neighbours each atom must have in the
                atom pair list.
            log_level: Logging level to report progress at.
            friction: Weight of the friction term in the target function, which keeps
                the optimizer from taking steps large enough to blow the path up.

        Raises:
            ValueError: If the path is not three dimensional.
        """
        rmsd0, self.path = align_path(path)
        logger.log(log_level, "Maximum RMSD change in initial path: %10.2f", rmsd0)
        if self.path.ndim != 3:
            raise ValueError('The path to be interpolated must have 3 dimensions')
        self.n_images, self.n_atoms, _ = self.path.shape
        # Construct coordinates
        self.rij_list, self.re = get_bond_list(path, atoms, threshold=threshold, min_neighbors=min_neighbors)
        if isinstance(scaler, float):
            self.scaler = morse_scaler(re=self.re, alpha=1.7)
        else:
            self.scaler = scaler
        self.n_rij = len(self.rij_list)
        self.friction = friction
        # Initalize interal storages for mid points, internal coordinates and B matrices.
        # `None` marks a value as unknown, so it is only ever computed when needed.
        logger.log(log_level, "Performing geodesic smoothing")
        logger.log(log_level, "  Images: %4d  Atoms %4d Rijs %6d", self.n_images, self.n_atoms, len(self.rij_list))
        self.n_eval = 0
        self.w = [None] * len(path)
        self.dw_dR = [None] * len(path)
        self.X_mid = [None] * (len(path) - 1)
        self.w_mid = [None] * (len(path) - 1)
        self.dwdR_mid = [None] * (len(path) - 1)
        self.displacements = self.grad = self.segment = None
        self.conv_path = []

    def update_intc(self):
        """Fill in any internal coordinates and derivatives currently marked unknown.

        Missing entries are flagged with `None` in the internal storage; this finds
        them, works out the midpoint geometries where needed, and evaluates the
        coordinates and their gradients.  Everything already known is left alone, so
        nothing is evaluated twice.
        """
        for i, X in enumerate(self.path):
            if self.w[i] is None:
                self.w[i], self.dw_dR[i] = compute_wij(X, self.rij_list, self.scaler)
        for i, (X0, X1) in enumerate(zip(self.path, self.path[1:])):
            if self.w_mid[i] is None:
                Xm = (X0 + X1) / 2
                self.X_mid[i] = Xm
                self.w_mid[i], self.dwdR_mid[i] = compute_wij(Xm, self.rij_list, self.scaler)

    def update_geometry(self, X, start, end):
        """Move a segment of the path, invalidating everything that depended on it.

        The internal coordinates, derivatives and midpoints of the affected images are
        reset to unknown so that `update_intc` recomputes them.  Note that moving
        images ``start:end`` also invalidates the midpoint just before ``start``.

        Args:
            X: New Cartesian coordinates for the segment, flattened or otherwise.
            start, end: The section of the path being replaced.

        Returns:
            True if the geometry actually changed, False if `X` was already the
            current geometry and nothing needed invalidating.
        """
        X = X.reshape(self.path[start:end].shape)
        if np.array_equal(X, self.path[start:end]):
            return False
        self.path[start:end] = X
        self.w[start:end] = [None] * (end - start)
        self.w_mid[start - 1:end] = [None] * (end - start + 1)
        return True

    def compute_displacements(self, start=1, end=-1, dx=None, friction=1e-3):
        """Compute the displacement vectors along a section of the path, and its length.

        Each segment is split at its midpoint and measured in two halves, which is what
        makes the length a sum of squares.  The friction term is appended to the same
        vector so the least-squares optimizer sees it as extra residuals.

        Args:
            start, end: Section of the path to measure.  A negative ``end`` counts back
                from the last image.
            dx: Displacement of the segment from its reference geometry.  When given,
                it enters the target function scaled by ``friction``.
            friction: Weight of the friction term.

        Sets `self.length`, `self.displacements` and `self.disps0` (the displacements
        without the friction residuals).
        """
        if end < 0:
            end += self.n_images
        self.update_intc()
        # Calculate displacement vectors in each segment, and the total length
        vecs_l = [wm - wl for wl, wm in zip(self.w[start - 1:end], self.w_mid[start - 1:end])]
        vecs_r = [wr - wm for wr, wm in zip(self.w[start:end + 1], self.w_mid[start - 1:end])]
        self.length = np.sum(np.linalg.norm(vecs_l, axis=1)) + np.sum(np.linalg.norm(vecs_r, axis=1))
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from initial geometry.  friction term
        self.displacements = np.concatenate(vecs_l + vecs_r + [trans])
        self.disps0 = self.displacements[:len(vecs_l) * 2]

    def compute_disp_grad(self, start, end, friction=1e-3):
        """Compute derivatives of the displacement vectors with respect to Cartesians.

        Moving one image changes the two half-segments on either side of it directly,
        and the two midpoints it shares with its neighbours at half the rate, which is
        where the factors of a half come from.  The friction residuals contribute a
        diagonal block at the bottom of the matrix.

        Args:
            start, end: Section of the path being differentiated.
            friction: Weight of the friction term, matching `compute_displacements`.

        Sets `self.grad` and `self.grad0` (the block excluding friction).
        """
        # Calculate derivatives of displacement vectors with respect to image Cartesians
        l = end - start + 1
        self.grad = np.zeros((l * 2 * self.n_rij + 3 * (end - start) * self.n_atoms, (end - start) * 3 * self.n_atoms))
        self.grad0 = self.grad[:l * 2 * self.n_rij]
        grad_shape = (l, self.n_rij, end - start, 3 * self.n_atoms)
        grad_l = self.grad[:l * self.n_rij].reshape(grad_shape)
        grad_r = self.grad[l * self.n_rij:l * self.n_rij * 2].reshape(grad_shape)
        for i, image in enumerate(range(start, end)):
            dmid1 = self.dwdR_mid[image - 1] / 2
            dmid2 = self.dwdR_mid[image] / 2
            grad_l[i + 1, :, i, :] = dmid2 - self.dw_dR[image]
            grad_l[i, :, i, :] = dmid1
            grad_r[i + 1, :, i, :] = -dmid2
            grad_r[i, :, i, :] = self.dw_dR[image] - dmid1
        for idx in range((end - start) * 3 * self.n_atoms):
            self.grad[l * self.n_rij * 2 + idx, idx] = friction

    def compute_target_func(self, X=None, start=1, end=-1, log_level=logging.INFO, x0=None, friction=1e-3):
        """Compute the vectorised target function used for least-squares minimisation.

        Args:
            X: Geometry to evaluate at.  If it matches the geometry already stored for
                this segment, the whole evaluation is skipped.
            start, end: Section of the path being optimised.
            log_level: Logging level for the per-iteration progress line.
            x0: Reference geometry the friction term pulls back towards.  Defaults to
                no pull at all.
            friction: Weight of the friction term.

        Sets `self.optimality`, the infinity norm of the gradient of the length, which
        is what convergence is judged on.
        """
        if end < 0:
            end += self.n_images
        if X is not None and self.segment == (start, end) and not self.update_geometry(X, start, end):
            return

        self.segment = (start, end)
        dx = np.zeros(self.path[start:end].size) if x0 is None else self.path[start:end].ravel() - x0.ravel()
        self.compute_displacements(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        self.optimality = np.linalg.norm(np.einsum('i,i...', self.displacements, self.grad), ord=np.inf)
        logger.log(log_level, "Iteration %3d: Length %10.3f |dL|=%7.3e", self.n_eval, self.length, self.optimality)
        self.conv_path.append(self.path[1].copy())
        self.n_eval += 1

    def target_func(self, X, **kwargs):
        """Residuals for the optimizer.

        Wraps `compute_target_func`, which skips the work if the geometry has not moved
        since the last call.
        """
        self.compute_target_func(X, **kwargs)
        return self.displacements

    def target_deriv(self, X, **kwargs):
        """Jacobian for the optimizer.

        Wraps `compute_target_func`, which skips the work if the geometry has not moved
        since the last call.  Paired with `target_func`, this means each geometry is
        only ever evaluated once even though scipy asks for value and Jacobian
        separately.
        """
        self.compute_target_func(X, **kwargs)
        return self.grad

    def smooth(self,
               tol=1e-3,
               max_iter=50,
               start=1,
               end=-1,
               log_level=logging.INFO,
               friction=None,
               xref=None):
        """Minimise the path length as a single function of all the image coordinates.

        In principle this is very efficient, but it can get costly for large systems
        with many images, in which case `sweep` is the better option.

        Args:
            tol: Convergence tolerance on the optimality, i.e. the uniform gradient of
                the target function.
            max_iter: Maximum number of iterations to run.
            start, end: Section of the path to optimise.
            log_level: Logging level during the optimization.
            friction: Weight of the friction term.  Defaults to the value given to the
                constructor.
            xref: Reference geometry for the friction term.  Defaults to the starting
                geometry of the segment.

        Returns:
            The optimized path.  This is also stored in `self.path`.
        """

        X0 = np.array(self.path[start:end]).ravel()
        if xref is None:
            xref = X0
        self.displacements = self.grad = self.segment = None
        logger.log(log_level, "  Degree of freedoms %6d: ", len(X0))
        if friction is None:
            friction = self.friction
        # Configure the keyword arguments that will be sent to the target function.
        kwargs = dict(start=start, end=end, log_level=log_level, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)  # Compute length and optimality
        if self.optimality > tol:
            # `soft_l1` keeps a single badly placed image from dominating the fit
            result = least_squares(self.target_func, X0, self.target_deriv, ftol=tol, gtol=tol,
                                   max_nfev=max_iter, kwargs=kwargs, loss='soft_l1')
            self.update_geometry(result['x'], start, end)
            logger.log(log_level, "Smoothing converged after %d iterations", result['nfev'])
        else:
            logger.log(log_level, "Skipping smoothing: path already optimal.")
        rmsd, self.path = align_path(self.path)
        logger.log(log_level, "Final path length: %12.5f  Max RMSD in path: %10.2f", self.length, rmsd)
        return self.path

    def sweep(self, tol=1e-3, max_iter=50, micro_iter=20, start=1, end=-1):
        """Minimise the path length one image at a time, sweeping back and forth.

        Less efficient per iteration than `smooth`, but it scales far more kindly with
        system size given how slow scipy's optimizers get on large problems.  It also
        allows finer control, and makes it cheap to skip images that are already close
        to optimal.

        Args:
            tol: Convergence tolerance on the optimality, i.e. the uniform gradient of
                the target function.
            max_iter: Maximum number of sweeps through the path.
            micro_iter: Number of micro-iterations spent optimizing each image.
            start, end: Section of the path to optimise.

        Returns:
            The optimized path.  This is also stored in `self.path`.
        """
        if end < 0:
            end = self.n_images + end
        self.neval = 0
        images = range(start, end)
        logger.info("  Degree of freedoms %6d: ", (end - start) * 3 * self.n_atoms)
        # Microiteration convergence tolerances are adjusted on the fly based on level of convergence.
        curr_tol = tol * 10
        self.compute_displacements()  # Compute and print the initial path length
        logger.info("  Initial length: %8.3f", self.length)
        for iteration in range(max_iter):
            max_dL = 0
            X0 = self.path.copy()
            for i in images[:-1]:  # Use self.smooth() to optimize individual images
                # Each image is pulled back towards the midpoint of its neighbours,
                # with heavy friction on the first sweep to keep the initial guess
                # from being thrown around
                xmid = (self.path[i - 1] + self.path[i + 1]) * 0.5
                self.smooth(curr_tol, max_iter=min(micro_iter, iteration + 6),
                            start=i, end=i + 1, log_level=logging.DEBUG,
                            friction=self.friction if iteration else 0.1,
                            xref=xmid)
                max_dL = max(max_dL, self.optimality)
            self.compute_displacements()  # Compute final length after sweep
            logger.info("Sweep %3d: L=%7.2f dX=%7.2e tol=%7.3e dL=%7.3e",
                        iteration, self.length, np.linalg.norm(self.path - X0), curr_tol, max_dL)
            if max_dL < tol:  # Check for convergence.
                logger.info("Optimization converged after %d iteartions", iteration)
                break
            curr_tol = max(tol * 0.5, max_dL * 0.2)  # Adjust micro-iteration threshold
            images = list(reversed(images))  # Alternate sweeping direction.
        else:
            logger.info("Optimization not converged after %d iteartions", iteration)
        rmsd, self.path = align_path(self.path)
        logger.info("Final path length: %12.5f  Max RMSD in path: %10.2f", self.length, rmsd)
        return self.path
