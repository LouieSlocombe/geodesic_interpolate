import os

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic
from ase.mep import NEB
from ase.optimize.fire import FIRE as QuasiNewton
from ase.visualize import view
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


# @pytest.mark.skip  # Fails, Non-equal coordinates at index 1
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


# @pytest.mark.skip  # Fails, Non-equal coordinates at index 1
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


def test_ase_compare():
    # Set the number of images you want.
    nimages = 15

    # Some algebra to determine surface normal and the plane of the surface.
    d3 = [2, 1, 1]
    a1 = np.array([0, 1, 1])
    d1 = np.cross(a1, d3)
    a2 = np.array([0, -1, 1])
    d2 = np.cross(a2, d3)

    # Create the slab.
    slab = FaceCenteredCubic(
        directions=[d1, d2, d3], size=(2, 1, 2), symbol='Pt', latticeconstant=3.9
    )

    # Add some vacuum to the slab.
    uc = slab.get_cell()
    uc[2] += [0.0, 0.0, 10.0]  # There are ten layers of vacuum.
    uc = slab.set_cell(uc, scale_atoms=False)

    # Some positions needed to place the atom in the correct place.
    x1 = 1.379
    x2 = 4.137
    x3 = 2.759
    y1 = 0.0
    y2 = 2.238
    z1 = 7.165
    z2 = 6.439

    # Add the adatom to the list of atoms and set constraints of surface atoms.
    slab += Atoms('N', [((x2 + x1) / 2, y1, z1 + 1.5)])
    # mask = [atom.symbol == 'Pt' for atom in slab]
    # slab.set_constraint(FixAtoms(mask=mask))

    # Optimise the initial state: atom below step.
    initial = slab.copy()
    initial.calc = EMT()
    relax = QuasiNewton(initial)
    relax.run(fmax=0.05)

    # Optimise the final state: atom above step.
    slab[-1].position = (x3, y2 + 1.0, z2 + 3.5)
    final = slab.copy()
    final.calc = EMT()
    relax = QuasiNewton(final)
    relax.run(fmax=0.05)

    # Create a list of images for interpolation.
    images = [initial]
    for i in range(nimages - 2):
        images.append(initial.copy())

    for image in images:
        image.calc = EMT()

    images.append(final)

    images = gi.geodesic_interpolate([images[0], images[-1]], n_images=nimages)  # 155
    for image in images:
        image.calc = EMT()
    #
    # view(images)

    # Carry out idpp interpolation.
    neb = NEB(images)
    # neb.interpolate('idpp')  # 176
    # neb.idpp_interpolate() #54
    # neb.interpolate() # 23 173

    # Run NEB calculation.
    qn = QuasiNewton(neb)
    qn.run(fmax=0.05)


def test_ase_ethane():
    nimages = 15

    # Optimise molecule.
    initial = molecule('C2H6')
    initial.calc = EMT()
    relax = QuasiNewton(initial)
    relax.run(fmax=0.05)

    # Create final state.
    final = initial.copy()
    final.positions[2:5] = initial.positions[[3, 4, 2]]

    # Generate blank images.
    images = [initial]

    for i in range(nimages - 2):
        images.append(initial.copy())

    for image in images:
        image.calc = EMT()

    images.append(final)

    images = gi.geodesic_interpolate([images[0], images[-1]], n_images=nimages)  # 1
    for image in images:
        image.calc = EMT()

    # Run linear interpolation.
    neb = NEB(images)  # 66
    # neb.interpolate() # 46
    # neb.idpp_interpolate() # 7
    # neb.interpolate('idpp') # 7

    # Run NEB calculation.
    qn = QuasiNewton(neb)
    qn.run(fmax=0.05)

    view(neb.images)
