"""Regression tests for geodesic interpolation.

The reference paths in `data` are converged optimisation results rather than analytic
answers, so the file-based cases check that a fresh run lands back on them: bond lengths
at the two fixed end points, and then every image's coordinates.

Data is resolved relative to this file, and every output goes to a `tmp_path`, so the
tests run from any working directory and leave nothing behind.
"""
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic
from ase.mep import NEB
from ase.optimize.fire import FIRE as QuasiNewton
from scipy.spatial import KDTree

import geodesic_interpolate as gi

DATA = Path(__file__).parent / "data"

# The three protein-scale systems dominate the runtime, so they are marked for deselection
CASES = [
    "H+CH4_CH3+H2",
    "DielsAlder",
    pytest.param("TrpCage_unfold", marks=pytest.mark.slow),
    pytest.param("collagen", marks=pytest.mark.slow),
    pytest.param("calcium_binding", marks=pytest.mark.slow),
]


def assert_bond_lengths_equal(path, reference, cutoff=2.0, tol=1e-4):
    """Assert two paths agree on which atoms are bonded and how long those bonds are.

    Bond lengths survive the rigid-body motion that path alignment applies, so this holds
    even when the two paths have been rotated into different orientations.

    Args:
        path, reference: The two paths to compare, as lists of ASE Atoms.
        cutoff: Distance below which two atoms count as bonded.
        tol: Largest bond length difference to accept, in Angstrom.
    """
    assert len(path) == len(reference), f"{len(path)} images against {len(reference)}"
    for i, (image, ref) in enumerate(zip(path, reference, strict=True)):
        pos, ref_pos = image.get_positions(), ref.get_positions()
        pairs = np.array(sorted(KDTree(pos).query_pairs(cutoff)))
        ref_pairs = np.array(sorted(KDTree(ref_pos).query_pairs(cutoff)))
        assert np.array_equal(pairs, ref_pairs), f"different bonded pairs in image {i}"
        bonds = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
        ref_bonds = np.linalg.norm(ref_pos[ref_pairs[:, 0]] - ref_pos[ref_pairs[:, 1]], axis=1)
        assert np.allclose(bonds, ref_bonds, atol=tol), (
            f"bond lengths differ in image {i} by up to {np.abs(bonds - ref_bonds).max():.2e}"
        )


def assert_paths_equal(path, reference, tol=1e-1):
    """Assert two paths hold the same atoms in the same places.

    Args:
        path, reference: The two paths to compare, as lists of ASE Atoms.
        tol: Largest coordinate deviation to accept, in Angstrom.
    """
    assert len(path) == len(reference), f"{len(path)} images against {len(reference)}"
    for i, (image, ref) in enumerate(zip(path, reference, strict=True)):
        assert image.get_chemical_symbols() == ref.get_chemical_symbols(), f"different atoms in image {i}"
        deviation = np.abs(image.get_positions() - ref.get_positions()).max()
        assert deviation < tol, f"image {i} is off the reference by {deviation:.3f} A"


@pytest.mark.parametrize("case", CASES)
def test_interpolate_from_file(case, tmp_path):
    """A filename in, an XYZ path out, matching the stored reference."""
    output = tmp_path / "interpolated.xyz"
    gi.geodesic_interpolate(DATA / f"{case}.xyz", output=output)

    path = read(output, index=':')
    reference = read(DATA / f"{case}_interpolated.xyz", index=':')
    assert_bond_lengths_equal([path[0], path[-1]], [reference[0], reference[-1]])
    assert_paths_equal(path, reference)


def test_interpolate_from_atoms():
    """ASE Atoms in, ASE Atoms out, with no file involved on either side."""
    end_points = read(DATA / "H+CH4_CH3+H2.xyz", index=':')

    path = gi.geodesic_interpolate(end_points)

    assert all(isinstance(image, Atoms) for image in path)
    reference = read(DATA / "H+CH4_CH3+H2_interpolated.xyz", index=':')
    assert_bond_lengths_equal([path[0], path[-1]], [reference[0], reference[-1]])
    assert_paths_equal(path, reference)


def test_interpolation_is_reproducible(tmp_path):
    """The bisection is stochastic, so `seed` is what makes a run repeatable."""
    first, second = tmp_path / "first.xyz", tmp_path / "second.xyz"

    gi.geodesic_interpolate(DATA / "H+CH4_CH3+H2.xyz", output=first)
    gi.geodesic_interpolate(DATA / "H+CH4_CH3+H2.xyz", output=second)

    assert first.read_text() == second.read_text()


def test_xyz_round_trip(tmp_path):
    """Coordinates written to an XYZ file come back unchanged."""
    atom_names, coords = gi.read_xyz(DATA / "H+CH4_CH3+H2.xyz")
    output = tmp_path / "round_trip.xyz"

    gi.write_xyz(output, atom_names, coords)
    names_back, coords_back = gi.read_xyz(output)

    assert names_back == atom_names
    assert np.allclose(coords_back, coords)


def test_ase_round_trip():
    """Coordinates converted to ASE Atoms and back come back unchanged."""
    atom_names, coords = gi.read_xyz(DATA / "H+CH4_CH3+H2.xyz")

    images = gi.to_ase_atoms(atom_names, coords)
    names_back, coords_back = gi.from_ase_atoms(images)

    assert len(images) == len(coords)
    assert names_back == atom_names
    assert np.allclose(coords_back, coords)


def test_read_xyz_rejects_truncated_file(tmp_path):
    """A frame that ends early is a format error, not a silently short geometry."""
    truncated = tmp_path / "truncated.xyz"
    truncated.write_text("3\ncomment\n C 0.0 0.0 0.0\n")  # Claims three atoms, holds one

    with pytest.raises(ValueError, match="Incorrect XYZ file format"):
        gi.read_xyz(truncated)


def test_read_xyz_rejects_empty_file(tmp_path):
    """A file with no frames in it is an error rather than an empty path."""
    empty = tmp_path / "empty.xyz"
    empty.write_text("")

    with pytest.raises(ValueError, match="File is empty"):
        gi.read_xyz(empty)


def test_periodic_input_keeps_its_cell():
    """A periodic system keeps its cell, and comes back in the frame that cell describes."""
    end_points = read(DATA / "H+CH4_CH3+H2.xyz", index=':')
    molecular = gi.geodesic_interpolate(end_points, n_images=5)

    for end_point in end_points:
        end_point.set_cell([12.0, 13.0, 14.0])
        end_point.set_pbc(True)
        end_point.set_tags(range(len(end_point)))
    path = gi.geodesic_interpolate(end_points, n_images=5)

    for image in path:
        assert np.allclose(image.get_cell(), end_points[0].get_cell())
        assert image.pbc.all()
        assert list(image.get_tags()) == list(range(len(end_points[0])))
    # Without the frame being restored the path would be centred on the origin instead,
    # leaving the atoms sitting outside the cell they claim to be in
    assert np.allclose(path[0].get_positions(), end_points[0].get_positions(), atol=1e-6)
    # Restoring the frame is one rigid motion applied to every image, so it moves the path
    # without deforming it: same path, different frame
    assert_bond_lengths_equal(path, molecular, tol=1e-10)


def test_non_periodic_input_is_left_in_the_optimizer_frame():
    """The frame is only restored for periodic input, so molecules are unaffected."""
    end_points = read(DATA / "H+CH4_CH3+H2.xyz", index=':')

    path = gi.geodesic_interpolate(end_points, n_images=5)

    assert path[0].get_cell().rank == 0
    assert not path[0].pbc.any()
    assert np.allclose(path[0].get_positions().mean(axis=0), 0.0, atol=1e-8)


def test_constraints_are_carried_over_without_moving_atoms():
    """A constraint has to survive for the NEB, without ASE applying it to the path."""
    end_points = read(DATA / "H+CH4_CH3+H2.xyz", index=':')
    unconstrained = gi.geodesic_interpolate(end_points, n_images=5)

    # The interpolation only ever reads symbols and positions, so constraining an atom
    # must not change the path it produces
    for end_point in end_points:
        end_point.set_constraint(FixAtoms(indices=[0]))
    constrained = gi.geodesic_interpolate(end_points, n_images=5)

    assert all(isinstance(image.constraints[0], FixAtoms) for image in constrained)
    # `set_positions` applies constraints by default, which would pin atom 0 of every
    # image to where the first end point had it rather than where the path puts it
    assert_paths_equal(constrained, unconstrained, tol=1e-12)


def test_to_ase_atoms_rejects_mismatched_template():
    """A template for a different system would silently override the requested symbols."""
    atom_names, coords = gi.read_xyz(DATA / "H+CH4_CH3+H2.xyz")
    template = Atoms(symbols=['Ar'] * len(atom_names), positions=coords[0])

    with pytest.raises(ValueError, match="same system"):
        gi.to_ase_atoms(atom_names, coords, template=template)


# Both NEB tests are smoke tests: they check the path ASE is given, then that ASE can
# relax it.  The slab keeps its cell and boundary conditions through the interpolation,
# so these run periodically, as the slab is meant to be.

@pytest.mark.slow
def test_neb_on_slab_adatom():
    """Interpolate an adatom hopping across a Pt step, then relax the path with NEB."""
    n_images = 15

    # Algebra determining the surface normal and the plane of the surface
    d3 = [2, 1, 1]
    d1 = np.cross(np.array([0, 1, 1]), d3)
    d2 = np.cross(np.array([0, -1, 1]), d3)
    slab = FaceCenteredCubic(directions=[d1, d2, d3], size=(2, 1, 2), symbol='Pt', latticeconstant=3.9)

    cell = slab.get_cell()
    cell[2] += [0.0, 0.0, 10.0]  # Ten layers of vacuum above the slab
    slab.set_cell(cell, scale_atoms=False)

    # Positions placing the adatom below, and then above, the step
    x1, x2, x3 = 1.379, 4.137, 2.759
    y1, y2 = 0.0, 2.238
    z1, z2 = 7.165, 6.439

    slab += Atoms('N', [((x2 + x1) / 2, y1, z1 + 1.5)])
    initial = slab.copy()
    initial.calc = EMT()
    QuasiNewton(initial).run(fmax=0.05)

    slab[-1].position = (x3, y2 + 1.0, z2 + 3.5)
    final = slab.copy()
    final.calc = EMT()
    QuasiNewton(final).run(fmax=0.05)

    images = gi.geodesic_interpolate([initial, final], n_images=n_images)

    assert len(images) == n_images
    assert images[0].get_chemical_symbols() == initial.get_chemical_symbols()
    assert all(np.isfinite(image.get_positions()).all() for image in images)

    for image in images:
        image.calc = EMT()
    QuasiNewton(NEB(images)).run(fmax=0.05)


@pytest.mark.slow
def test_neb_on_ethane_rotation():
    """Interpolate a methyl rotation in ethane, then relax the path with NEB."""
    n_images = 15

    initial = molecule('C2H6')
    initial.calc = EMT()
    QuasiNewton(initial).run(fmax=0.05)

    # Permute three hydrogens to rotate one methyl group onto itself
    final = initial.copy()
    final.positions[2:5] = initial.positions[[3, 4, 2]]

    images = gi.geodesic_interpolate([initial, final], n_images=n_images)

    assert len(images) == n_images
    assert images[0].get_chemical_symbols() == initial.get_chemical_symbols()
    assert all(np.isfinite(image.get_positions()).all() for image in images)

    for image in images:
        image.calc = EMT()
    QuasiNewton(NEB(images)).run(fmax=0.05)
