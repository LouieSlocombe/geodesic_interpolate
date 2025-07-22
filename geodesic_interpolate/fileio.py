from typing import List, Tuple
from typing import Union

import numpy as np
from ase import Atoms


def from_ase_atoms(atoms: List[Atoms]) -> Tuple[List[str], List[np.ndarray]]:
    atom_names = atoms[0].get_chemical_symbols()
    coords: List[np.ndarray] = []
    for atom in atoms:
        coords.append(np.array(atom.get_positions()))
    return atom_names, coords


def to_ase_atoms(atoms: List[str], coords: Union[np.ndarray, List[np.ndarray]]) -> List[Atoms]:
    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]  # Add a new axis for single frame
    return [Atoms(symbols=atoms, positions=frame) for frame in coords]


def read_xyz(filename: str) -> Tuple[List[str], List[np.ndarray]]:
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
    with open(filename, 'w') as f:
        for i, X in enumerate(np.atleast_3d(coords)):
            f.write(f"{len(atoms)}\n")
            f.write(f"Frame {i}\n")
            f.writelines(
                f" {a:3} {Xa[0]:21.12f} {Xa[1]:21.12f} {Xa[2]:21.12f}\n"
                for a, Xa in zip(atoms, X)
            )
