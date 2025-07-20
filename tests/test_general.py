from ase.io import read
from ase.visualize import view

import geodesic_interpolate as gi


def test_case_ch():
    gi.interpolate("data/H+CH4_CH3+H2.xyz")

    # atoms = read("data/H+CH4_CH3+H2.xyz", index=':')
    # view(atoms)

    # atoms = read("data/H+CH4_CH3+H2_interpolated.xyz", index=':')
    # view(atoms)

    atoms = read("interpolated.xyz", index=':')
    view(atoms)

    pass


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
