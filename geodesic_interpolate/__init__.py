"""Geodesic interpolation for reaction paths.

Builds a smooth initial guess for a reaction path between two geometries by finding
the shortest path under a metric of scaled inter-atomic distances.  `geodesic_interpolate`
is the entry point; the XYZ and ASE helpers are re-exported for convenience.
"""
from .main import geodesic_interpolate
from .fileio import from_ase_atoms, to_ase_atoms, read_xyz, write_xyz
