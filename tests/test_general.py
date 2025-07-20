import os

import numpy as np
from ase.io import read
from scipy.spatial import KDTree

import geodesic_interpolate as gi


def atoms_equal(atoms1, atoms2, tol=1e-2):
    if len(atoms1) != len(atoms2):
        print("non equal lengths:", len(atoms1), len(atoms2))
        return False
    if not np.all(atoms1.get_atomic_numbers() == atoms2.get_atomic_numbers()):
        print("non equal atomic numbers:", atoms1.get_atomic_numbers(), atoms2.get_atomic_numbers())
        return False
    coords1 = atoms1.get_positions()
    coords2 = atoms2.get_positions()
    return np.allclose(coords1, coords2, atol=tol)


def atoms_list_equal(list1, list2, tol=1e-1):
    if len(list1) != len(list2):
        print("Non-equal list lengths:", len(list1), len(list2))
        return False
    for i, (a1, a2) in enumerate(zip(list1, list2)):
        if len(a1) != len(a2):
            print(f"Non-equal atom counts at index {i}: {len(a1)} vs {len(a2)}")
            return False
        if not np.all(a1.get_atomic_numbers() == a2.get_atomic_numbers()):
            print(f"Non-equal atomic numbers at index {i}: {a1.get_atomic_numbers()} vs {a2.get_atomic_numbers()}")
            return False
        if not np.allclose(a1.get_positions(), a2.get_positions(), atol=tol):
            print(f"Non-equal coordinates at index {i}")
            return False
    return True


def atoms_list_bond_lengths_equal(list1, list2, cutoff=2.0, tol=1e-3):
    """Check if two lists of ASE Atoms objects have the same bond lengths."""
    if len(list1) != len(list2):
        print("Non-equal list lengths:", len(list1), len(list2))
        return False
    for i, (a1, a2) in enumerate(zip(list1, list2)):
        pos1 = a1.get_positions()
        pos2 = a2.get_positions()
        tree1 = KDTree(pos1)
        tree2 = KDTree(pos2)
        pairs1 = np.array(sorted(tree1.query_pairs(cutoff)))
        pairs2 = np.array(sorted(tree2.query_pairs(cutoff)))
        if not np.array_equal(pairs1, pairs2):
            print(f"Non-equal bond pairs at index {i}")
            return False
        bonds1 = np.linalg.norm(pos1[pairs1[:, 0]] - pos1[pairs1[:, 1]], axis=1)
        bonds2 = np.linalg.norm(pos2[pairs2[:, 0]] - pos2[pairs2[:, 1]], axis=1)
        if not np.allclose(bonds1, bonds2, atol=tol):
            print(f"Non-equal bond lengths at index {i}")
            return False
    return True


def test_case_ch():
    for i in range(1, 50):
        gi.interpolate("data/H+CH4_CH3+H2.xyz")
        atoms = read("interpolated.xyz", index=':')
        # view(atoms)
        atoms_ref = read("data/H+CH4_CH3+H2_interpolated.xyz", index=':')
        # Check the end points are equal
        assert atoms_list_bond_lengths_equal([atoms[0], atoms[-1]], [atoms_ref[0], atoms_ref[-1]], tol=1e-4)
        # assert atoms_list_equal(atoms, atoms_ref)

        os.remove("interpolated.xyz")


def test_case_diels_alder():
    # DielsAlder.xyz
    pass


def test_case_trp_cage_unfold():
    # TrpCage_unfold.xyz
    pass


def test_case_collagen():
    # collagen.xyz
    pass


def test_case_calcium_binding():
    # calcium_binding.xyz
    pass
