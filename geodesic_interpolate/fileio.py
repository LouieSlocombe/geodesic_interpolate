"""File IO utilities, and conversion to and from ASE Atoms objects."""
from typing import List, Tuple
from typing import Union

import numpy as np
from ase import Atoms


def from_ase_atoms(atoms: List[Atoms]) -> Tuple[List[str], List[np.ndarray]]:
    """Split a list of ASE Atoms objects into symbols and coordinates.

    Args:
        atoms: Frames of the path.  All are assumed to hold the same atoms in the same
            order, so the symbols are taken from the first one.

    Returns:
        atom_names: Element symbols of all the atoms.
        coords: Cartesian coordinates for every frame.
    """
    atom_names = atoms[0].get_chemical_symbols()
    coords: List[np.ndarray] = []
    for atom in atoms:
        coords.append(np.array(atom.get_positions()))
    return atom_names, coords


def to_ase_atoms(atoms: List[str], coords: Union[np.ndarray, List[np.ndarray]]) -> List[Atoms]:
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


def read_xyz(filename: str) -> Tuple[List[str], List[np.ndarray]]:
    """Read an XYZ file and return the atom names and coordinates.

    Args:
        filename: Name of the XYZ data file.  It may hold any number of frames.

    Returns:
        atom_names: Element symbols of all the atoms, read from the last frame.
        coords: Cartesian coordinates for every frame.

    Raises:
        ValueError: If the file is empty or does not parse as XYZ.
    """
    coords: List[np.ndarray] = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                n_atoms = int(line)  # Read number of atoms
                next(f)  # Skip over comments
                atom_names: List[str] = []
                geom = np.zeros((n_atoms, 3), float)
                for i in range(n_atoms):
                    line = next(f).split()
                    atom_names.append(line[0])
                    geom[i] = line[1:4]  # Numpy auto-converts str to float
            except (TypeError, IOError, IndexError, StopIteration):
                raise ValueError('Incorrect XYZ file format')
            coords.append(geom)
    if not coords:
        raise ValueError("File is empty")
    return atom_names, coords


def write_xyz(filename: str, atoms: List[str], coords: Union[np.ndarray, List[np.ndarray]]) -> None:
    """Write atom names and coordinate data to an XYZ file.

    Args:
        filename: Name of the XYZ data file to write.
        atoms: Iterable of atom names.
        coords: Coordinates, of shape ``(n_images, n_atoms, 3)``.
    """
    with open(filename, 'w') as f:
        for i, X in enumerate(np.atleast_3d(coords)):
            f.write(f"{len(atoms)}\n")
            f.write(f"Frame {i}\n")
            f.writelines(
                f" {a:3} {Xa[0]:21.12f} {Xa[1]:21.12f} {Xa[2]:21.12f}\n"
                for a, Xa in zip(atoms, X)
            )
