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
# Every mesh in the bundle the decoder claims. `deck.bin` and `truck.bin`
# are SKDE/SKTR and used to be refused as "multi-part containers whose header
# length is unknown"; they carry two index buffers over one shared vertex
# pool, and locating the vertex block by the end-of-file rule needs no header
# knowledge at all.
SUPPORTED = ("deck_bottom.bin", "oldschool_deck_bottom.bin",
             "grip_tape.bin", "edge_top.bin", "edge_bottom.bin",
             "deck.bin", "oldschool_deck.bin", "oldschool_trucks.bin",
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


@pytest.mark.parametrize("name", ("grip_tape2.bin", "edge_top3.bin",
                                  "oldschool_grip_tape2.bin"))
def test_the_files_we_still_cannot_read_refuse_rather_than_guess(name):
    with pytest.raises(UnsupportedMesh):
        read_from_ipa(IPA, name)


@pytest.mark.parametrize("name", SUPPORTED + ("truck.bin",))
def test_every_index_lands_inside_the_vertex_pool(name):
    """The check that actually pins the vertex block down.

    A wrong offset has to satisfy two constraints at once -- the block must
    end EXACTLY at end of file, and every index must be a valid vertex -- and
    the old walk failed the second: it read the second index buffer's length
    as a vertex count, giving "indices reaching 1251 against a claimed 945".
    945 was the first buffer's length; 1251 is nv - 1.
    """
    mesh = read_from_ipa(IPA, name)
    assert mesh.strip.min() >= 0
    assert mesh.strip.max() < len(mesh.positions)
    assert sum(b - a for a, b in mesh.parts) == len(mesh.strip)


@pytest.mark.parametrize("name", ("deck.bin", "oldschool_deck.bin",
                                  "truck.bin", "oldschool_trucks.bin"))
def test_the_deck_and_truck_containers_carry_two_index_buffers(name):
    """Two draw calls over one vertex pool. This is what makes them
    'multi-part', and it is not a longer header."""
    assert len(read_from_ipa(IPA, name).parts) == 2


def test_deck_bin_agrees_with_the_surfaces_it_was_never_compared_to():
    """`deck.bin` is the whole deck and was decoded long after the profile
    was measured off `edge_*` and `grip_tape`. It is a free cross-check."""
    whole = read_from_ipa(IPA, "deck.bin").positions.astype(float)
    rim = read_from_ipa(IPA, "edge_top.bin").positions.astype(float)
    a = np.sort(np.ptp(whole, axis=0))[::-1]
    b = np.sort(np.ptp(rim, axis=0))[::-1]
    assert abs(a[0] - b[0]) / b[0] < 0.005, "length disagrees"
    assert abs(a[1] - b[1]) / b[1] < 0.010, "width disagrees"


def test_the_board_meshes_have_unit_square_uvs():
    """The check that discriminates two-stream from interleaved.

    It used to live inside `decode` and refuse the file. That rejected
    `truck.bin`, which decodes correctly and reaches 1.272 because its metal
    texture tiles -- so it moved here, where it can stay strict about the
    files it actually applies to. Do not weaken it: under the interleaved
    reading these all fail.
    """
    for name in SUPPORTED:
        assert read_from_ipa(IPA, name).uvs_in_unit_square, name


def test_scale_agrees_from_two_independent_anchors():
    """Assume 8.0 in wide -> the length must come out a standard deck length.

    Nothing ties these together in the data, so agreement is real evidence
    about the unit rather than a restatement of the assumption.
    """
    d = from_ipa(IPA)
    assert abs(d.length - 0.8128) < 0.010        # 32 in, within 1 cm
    assert 0.038 < d.unit_m < 0.042
