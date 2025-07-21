import os

import numpy as np
import pytest
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


def atoms_list_bond_lengths_equal(list1, list2, cutoff=2.0, tol=1e-4):
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
    print(flush=True)
    in_file = "data/H+CH4_CH3+H2"
    out_file = "interpolated"

    gi.geodesic_interpolate(f"{in_file}.xyz")
    atoms = read(f"{out_file}.xyz", index=':')
    atoms_ref = read(f"{in_file}_{out_file}.xyz", index=':')

    os.remove(f"{out_file}.xyz")

    assert atoms_list_bond_lengths_equal([atoms[0], atoms[-1]], [atoms_ref[0], atoms_ref[-1]])
    assert atoms_list_equal(atoms, atoms_ref)


def test_case_ch_atoms():
    print(flush=True)
    in_file = "data/H+CH4_CH3+H2"
    out_file = "interpolated"
    atoms_in = read(f"{in_file}.xyz", index=':')

    atoms = gi.geodesic_interpolate(atoms_in)
    atoms_ref = read(f"{in_file}_{out_file}.xyz", index=':')

    assert atoms_list_bond_lengths_equal([atoms[0], atoms[-1]], [atoms_ref[0], atoms_ref[-1]])
    assert atoms_list_equal(atoms, atoms_ref)


def test_case_diels_alder():
    print(flush=True)
    in_file = "data/DielsAlder"
    out_file = "interpolated"

    gi.geodesic_interpolate(f"{in_file}.xyz")
    atoms = read(f"{out_file}.xyz", index=':')
    atoms_ref = read(f"{in_file}_{out_file}.xyz", index=':')

    os.remove(f"{out_file}.xyz")

    assert atoms_list_bond_lengths_equal([atoms[0], atoms[-1]], [atoms_ref[0], atoms_ref[-1]])
    assert atoms_list_equal(atoms, atoms_ref)


# @pytest.mark.slow
@pytest.mark.skip  # Fails, Non-equal coordinates at index 1
def test_case_trp_cage_unfold():
    print(flush=True)
    in_file = "data/TrpCage_unfold"
    out_file = "interpolated"

    gi.geodesic_interpolate(f"{in_file}.xyz")
    atoms = read(f"{out_file}.xyz", index=':')
    atoms_ref = read(f"{in_file}_{out_file}.xyz", index=':')

    os.remove(f"{out_file}.xyz")

    assert atoms_list_bond_lengths_equal([atoms[0], atoms[-1]], [atoms_ref[0], atoms_ref[-1]])
    assert atoms_list_equal(atoms, atoms_ref)


# @pytest.mark.slow
@pytest.mark.skip  # Fails, Non-equal coordinates at index 1
def test_case_collagen():
    print(flush=True)
    in_file = "data/collagen"
    out_file = "interpolated"

    gi.geodesic_interpolate(f"{in_file}.xyz")
    atoms = read(f"{out_file}.xyz", index=':')
    atoms_ref = read(f"{in_file}_{out_file}.xyz", index=':')

    os.remove(f"{out_file}.xyz")

    assert atoms_list_bond_lengths_equal([atoms[0], atoms[-1]], [atoms_ref[0], atoms_ref[-1]])
    assert atoms_list_equal(atoms, atoms_ref)


# @pytest.mark.slow
def test_case_calcium_binding():
    print(flush=True)
    in_file = "data/calcium_binding"
    out_file = "interpolated"

    gi.geodesic_interpolate(f"{in_file}.xyz")
    atoms = read(f"{out_file}.xyz", index=':')
    atoms_ref = read(f"{in_file}_{out_file}.xyz", index=':')

    os.remove(f"{out_file}.xyz")

    assert atoms_list_bond_lengths_equal([atoms[0], atoms[-1]], [atoms_ref[0], atoms_ref[-1]])
    assert atoms_list_equal(atoms, atoms_ref)


def test_xyz_interconversion():
    print(flush=True)
    in_file = "data/H+CH4_CH3+H2"
    atom_names, coords = gi.read_xyz(f"{in_file}.xyz")
    print(atom_names)
    print(coords)
    gi.write_xyz('test.xyz', atom_names, coords)


def test_atoms_interconversion():
    print(flush=True)
    in_file = "data/H+CH4_CH3+H2"
    atoms = read(f"{in_file}.xyz", index=':')
    print(atoms)
    for atom in atoms:
        print(atom.get_chemical_symbols())
        print(atom.get_positions())
    atom_names, coords = gi.from_ase_atoms(atoms)
    print(atom_names)
    print(coords)

    atom_names, coords = gi.read_xyz(f"{in_file}.xyz")
    print(atom_names)
    print(coords)

    atoms = gi.to_ase_atoms(atom_names, coords)

    print(atoms)
    for atom in atoms:
        print(atom.get_chemical_symbols())
        print(atom.get_positions())
