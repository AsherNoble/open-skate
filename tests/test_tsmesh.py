"""The True Skate mesh decoder, checked by SHAPE rather than by parsing.

A parse test passes on garbage: 4720 bytes divides evenly by both 16 and 20,
so "the arithmetic works" proves nothing about the layout. These assert
properties a wrong decode has no reason to satisfy -- UVs inside [0, 1], a
wheel that is a surface of revolution, a deck that is a long flat plate --
which is what actually caught the interleaved-vs-two-stream mistake.
"""
import numpy as np
import pytest

from pathlib import Path

from opensk.assets.tsmesh import (TopologyNotDecoded, UnsupportedMesh,
                                  read_from_ipa, triangles)
from opensk.assets.tsdeck import from_ipa

IPA = Path.home() / "Projects/Robotics & hardware/TrueSkate.ipa"
SUPPORTED = ("deck_bottom.bin", "oldschool_deck_bottom.bin",
             "wheel.bin", "oldschool_wheels.bin")

pytestmark = pytest.mark.skipif(
    not IPA.exists(), reason="needs the user's own copy of the game bundle")


@pytest.mark.parametrize("name", SUPPORTED)
def test_uvs_land_in_the_unit_square(name):
    """The check that discriminates the layout.

    Positions and UVs are separate streams, not interleaved. Read the
    interleaved way, UVs bleed into the position columns and every axis ends
    up sharing one range; read correctly, the UVs are texture coordinates.
    """
    mesh = read_from_ipa(IPA, name)
    assert np.isfinite(mesh.positions).all()
    assert -0.05 <= mesh.uvs.min() and mesh.uvs.max() <= 1.05


@pytest.mark.parametrize("name", ("wheel.bin", "oldschool_wheels.bin"))
def test_the_wheel_is_a_surface_of_revolution(name):
    p = read_from_ipa(IPA, name).positions.astype(float)
    ext = np.sort(p.max(0) - p.min(0))[::-1]
    # Two axes are the diameter and agree; the third is the narrower width.
    assert abs(ext[0] - ext[1]) / ext[0] < 0.05
    assert ext[2] < 0.95 * ext[0]

    q = p - p.mean(0)
    axis = np.linalg.svd(q, full_matrices=False)[2][2]      # the thin axis
    r = np.linalg.norm(q - np.outer(q @ axis, axis), axis=1)
    rim = r > np.percentile(r, 70)
    assert r[rim].std() / r[rim].mean() < 0.10


def test_the_deck_is_a_long_flat_plate():
    p = read_from_ipa(IPA, "deck_bottom.bin").positions.astype(float)
    length, width, height = np.sort(p.max(0) - p.min(0))[::-1]
    assert 0.22 < width / length < 0.28          # a popsicle, not a longboard
    assert height / length < 0.10                # a deck, not a box


def test_topology_is_refused_rather_than_invented():
    """Two wheels of different tessellation both gave exactly 110 triangles.

    A statistic that does not move when the input changes is a broken
    instrument, so the index buffer is not interpreted at all.
    """
    with pytest.raises(TopologyNotDecoded):
        triangles(read_from_ipa(IPA, "wheel.bin"))


@pytest.mark.parametrize("name", ("deck.bin", "truck.bin"))
def test_unsupported_containers_refuse_rather_than_guess(name):
    with pytest.raises(UnsupportedMesh):
        read_from_ipa(IPA, name)


def test_scale_agrees_from_two_independent_anchors():
    """Assume 8.0 in wide -> the length must come out a standard deck length.

    Nothing ties these together in the data, so agreement is real evidence
    about the unit rather than a restatement of the assumption.
    """
    d = from_ipa(IPA)
    assert abs(d.length - 0.8128) < 0.010        # 32 in, within 1 cm
    assert 0.038 < d.unit_m < 0.042
