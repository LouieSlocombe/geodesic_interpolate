import numpy as np


def read_xyz(filename):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                n_atoms = int(line)  # Read number of atoms
                next(f)  # Skip over comments
                atom_names = []
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


def write_xyz(filename, atoms, coords):
    n_atoms = len(atoms)
    with open(filename, 'w') as f:
        for i, X in enumerate(np.atleast_3d(coords)):
            f.write("%d\n" % n_atoms)
            f.write("Frame %d\n" % i)
            for a, Xa in zip(atoms, X):
                f.write(" {:3} {:21.12f} {:21.12f} {:21.12f}\n".format(a, *Xa))
