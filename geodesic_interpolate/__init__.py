"""Geodesic interpolation for reaction paths.

Builds a smooth initial guess for a reaction path between two geometries by finding
the shortest path under a metric of scaled inter-atomic distances.  `geodesic_interpolate`
is the entry point; the XYZ and ASE helpers are re-exported for convenience.
"""
from importlib.metadata import PackageNotFoundError, version

from .fileio import from_ase_atoms, read_xyz, to_ase_atoms, write_xyz
from .main import geodesic_interpolate

try:
    __version__ = version("geodesic_interpolate")
except PackageNotFoundError:  # Running from a source checkout that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "from_ase_atoms",
    "geodesic_interpolate",
    "read_xyz",
    "to_ase_atoms",
    "write_xyz",
]
