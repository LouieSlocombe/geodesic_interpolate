"""File IO utilities, and conversion to and from ASE Atoms objects."""
import os

import numpy as np
from ase import Atoms

Filename = str | os.PathLike


def from_ase_atoms(atoms: list[Atoms]) -> tuple[list[str], list[np.ndarray]]:
    """Split a list of ASE Atoms objects into symbols and coordinates.

    Args:
        atoms: Frames of the path.  All are assumed to hold the same atoms in the same
            order, so the symbols are taken from the first one.

    Returns:
        atom_names: Element symbols of all the atoms.
        coords: Cartesian coordinates for every frame.
    """
    atom_names = atoms[0].get_chemical_symbols()
    coords: list[np.ndarray] = []
    for atom in atoms:
        coords.append(np.array(atom.get_positions()))
    return atom_names, coords


def to_ase_atoms(atoms: list[str], coords: np.ndarray | list[np.ndarray]) -> list[Atoms]:
    """Rebuild a list of ASE Atoms objects from symbols and coordinates.

    Args:
        atoms: Element symbols of all the atoms.
        coords: Cartesian coordinates, of shape ``(n_images, n_atoms, 3)``.  A single
            frame of shape ``(n_atoms, 3)`` is accepted too.

    Returns:
        One ASE Atoms object per frame.
    """
    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]  # Add a new axis for single frame
    return [Atoms(symbols=atoms, positions=frame) for frame in coords]


def read_xyz(filename: Filename) -> tuple[list[str], list[np.ndarray]]:
    """Read an XYZ file and return the atom names and coordinates.

    Args:
        filename: Name of the XYZ data file.  It may hold any number of frames, and
            blank lines between them are ignored.

    Returns:
        atom_names: Element symbols of all the atoms, read from the last frame.
        coords: Cartesian coordinates for every frame.

    Raises:
        ValueError: If the file is empty or does not parse as XYZ.
    """
    coords: list[np.ndarray] = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            if not line.strip():  # Blank line between or after frames
                continue
            try:
                n_atoms = int(line)  # Read number of atoms
                next(f)  # Skip over comments
                atom_names: list[str] = []
                geom = np.zeros((n_atoms, 3), float)
                for i in range(n_atoms):
                    line = next(f).split()
                    atom_names.append(line[0])
                    geom[i] = line[1:4]  # Numpy auto-converts str to float
            except (TypeError, ValueError, OSError, IndexError, StopIteration) as err:
                raise ValueError('Incorrect XYZ file format') from err
            coords.append(geom)
    if not coords:
        raise ValueError("File is empty")
    return atom_names, coords


def write_xyz(filename: Filename, atoms: list[str], coords: np.ndarray | list[np.ndarray]) -> None:
    """Write atom names and coordinate data to an XYZ file.

    Args:
        filename: Name of the XYZ data file to write.
        atoms: Iterable of atom names.
        coords: Coordinates, of shape ``(n_images, n_atoms, 3)``.  A single frame of
            shape ``(n_atoms, 3)`` is accepted too.
    """
    # Not `np.atleast_3d`, which would turn a single frame into `(n_atoms, 3, 1)`
    coords = np.asarray(coords, dtype=float).reshape(-1, len(atoms), 3)
    with open(filename, 'w', encoding='utf-8') as f:
        for i, X in enumerate(coords):
            f.write(f"{len(atoms)}\n")
            f.write(f"Frame {i}\n")
            f.writelines(
                f" {a:3} {Xa[0]:21.12f} {Xa[1]:21.12f} {Xa[2]:21.12f}\n"
                for a, Xa in zip(atoms, X)
            )
