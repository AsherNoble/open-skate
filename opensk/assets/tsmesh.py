"""Decoder for True Skate's mesh files (`res/*.bin`).

The format was recovered by inspection, and the layout is not a guess: the
game's own SPIR-V vertex shaders (`res/shaders/*.vert.spv`) declare exactly two
vertex inputs --

    location 0  a_v4Position   vec4<f32>
    location 1  a_v2TexCoord   vec2<f32>

-- and a vec4 position fed from a 3-component buffer is the ordinary case. So a
vertex is 12 bytes of position plus 8 bytes of UV.

**They are two separate streams, not interleaved.** That distinction is the
whole difficulty: 20 bytes per vertex divides the block either way, so the
arithmetic cannot tell them apart, and reading interleaved silently mixes UVs
into the position columns. The tell was that every position axis then shared
one range, and the fix was checked the only way that discriminates -- decoded
UVs must land in [0, 1], which they do on all four supported files and do not
under the interleaved reading.

Layout, after a per-magic header:

    (u32 n_idx, u16[n_idx])  ONE OR MORE index buffers
    u32                      vertex count
    f32[nv][3]               positions
    f32[nv][2]               UVs

**All four magics use this, `SKDE` and `SKTR` included.** They were previously
refused as "multi-part containers whose header length is unknown", which was
half right: their headers ARE longer (0x3C and 0x30), and they carry TWO index
buffers over ONE shared vertex pool -- two draw calls, two materials, one mesh.
The earlier reading walked them as repeated (indices, vertices) pairs, so it
took the second buffer's index count for a vertex count and landed in the
middle of the index data. That is what produced "indices reaching vertex 1251
against a claimed 945": 945 was the FIRST buffer's length, and 1251 is one
less than the true vertex count of 1252.

Finding the vertex block needs no header knowledge at all. Read index buffers
until the u32 that follows one is a vertex count whose block ends EXACTLY at
end of file. That rule parses every mesh in the bundle, and on every one of
them the indices then land inside the vertex count -- a joint constraint a
wrong offset has no way to satisfy.

Four files still refuse (`grip_tape2`, `edge_top3` and their `oldschool_`
twins). They have some further structure, and nothing here needs them.
"""
from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Header length per magic: magic, version, a pad word, then a per-magic block
# of floats (a bounding box and two dequantisation scales) before the first
# index count. Validated end to end on every file of each magic -- indices in
# range, vertex block ending exactly at EOF, and the decoded cloud the right
# shape.
HEADER_BYTES = {b"OMSH": 0x0C, b"SKWH": 0x28, b"SKTR": 0x30, b"SKDE": 0x3C}

POS_BYTES, UV_BYTES = 12, 8
VERTEX_BYTES = POS_BYTES + UV_BYTES

IPA_MEMBER = "Payload/True Skate.app/res/{}"


class UnsupportedMesh(ValueError):
    """The file is a mesh we cannot yet decode correctly."""


@dataclass(frozen=True)
class TSMesh:
    """One decoded mesh.

    `positions` and `uvs` are VALIDATED. `strip` is the raw index buffer and
    its meaning is NOT yet established -- see `triangles()`.
    """

    magic: str
    positions: np.ndarray      # (nv, 3) float32, game units
    uvs: np.ndarray            # (nv, 2) float32
    strip: np.ndarray          # (n_idx,) int32, all index buffers concatenated
    parts: tuple               # (start, stop) into `strip`, one per buffer

    @property
    def extent(self) -> np.ndarray:
        return self.positions.max(axis=0) - self.positions.min(axis=0)

    @property
    def uvs_in_unit_square(self) -> bool:
        """Whether the UVs are untiled.

        This used to be a hard refusal inside `decode`, as the check that
        discriminates the two-stream vertex layout from the interleaved one --
        and it is still exactly that, which is why `tests/test_tsmesh.py`
        asserts it on the board's own meshes and must go on doing so. But it
        is a property of a FILE, not of the format: `truck.bin` decodes
        correctly and reaches 1.272 because its metal texture tiles. Refusing
        it there rejected a good decode, so the check moved to where it can
        stay strict about the files it actually applies to.
        """
        return bool(np.isfinite(self.uvs).all()
                    and self.uvs.min() >= -0.05 and self.uvs.max() <= 1.05)


class TopologyNotDecoded(NotImplementedError):
    """The index buffer's meaning is not established. See `triangles()`."""


def triangles(mesh: TSMesh) -> np.ndarray:
    """NOT AVAILABLE -- the index buffer's meaning is not yet established.

    Read as a triangle strip, the decoded meshes are not manifold: almost every
    edge is used by ONE triangle rather than two (`wheel.bin` gives 236 edges
    used once against 47 used twice). Read as a triangle list, `oldschool_
    wheels.bin` yields 330 edges all used once -- i.e. 110 triangles that share
    no vertices at all. Neither is a surface.

    A second tell: two wheels of different tessellation (236 vs 330 vertices)
    both produced exactly 110 triangles. A statistic that does not move when
    the input changes is a broken instrument, not a result.

    So this raises. Positions and UVs are usable now (see `tsdeck.py`, which
    measures the deck's outline from the point cloud alone and needs no
    topology); triangles are not, and inventing them would put a made-up
    surface into the model wearing the authority of a decode.
    """
    raise TopologyNotDecoded(
        "index buffer semantics unknown; positions/UVs are validated, "
        "topology is not")


def decode(data: bytes) -> TSMesh:
    """Decode one `res/*.bin` mesh, or raise `UnsupportedMesh`.

    Structure is validated three ways, and a wrong offset cannot satisfy all
    three at once: the vertex block must end exactly at end of file, every
    index must be inside the vertex count, and the positions must be finite
    and bounded. UV range is NOT checked here -- see `uvs_in_unit_square`.
    """
    magic = data[:4]
    if magic not in HEADER_BYTES:
        raise UnsupportedMesh(f"unknown mesh magic {magic!r}")

    off = HEADER_BYTES[magic]
    spans: list[tuple[int, int]] = []
    parts: list[tuple[int, int]] = []
    while True:
        if off + 8 > len(data):
            raise UnsupportedMesh(
                "ran out of file looking for the vertex block; this magic's "
                "header length or its part structure is not what we assume")
        n_idx = struct.unpack_from("<I", data, off)[0]
        nxt = off + 4 + 2 * n_idx
        if nxt + 4 > len(data):
            raise UnsupportedMesh("index block runs past end of file")
        nv = struct.unpack_from("<I", data, nxt)[0]
        spans.append((off + 4, n_idx))
        if nxt + 4 + VERTEX_BYTES * nv == len(data):
            break
        off = nxt

    base = nxt + 4
    pos = np.frombuffer(data, "<f4", count=nv * 3, offset=base).reshape(nv, 3)
    uv = np.frombuffer(data, "<f4", count=nv * 2,
                       offset=base + POS_BYTES * nv).reshape(nv, 2)
    if not np.isfinite(pos).all() or np.abs(pos).max() > 1.0e4:
        raise UnsupportedMesh("positions are not finite and bounded")

    strips, cursor = [], 0
    for start, n_idx in spans:
        s = np.frombuffer(data, "<u2", count=n_idx, offset=start).astype(np.int32)
        if n_idx and s.max() >= nv:
            raise UnsupportedMesh(
                f"index {s.max()} out of range for {nv} vertices")
        strips.append(s)
        parts.append((cursor, cursor + n_idx))
        cursor += n_idx

    return TSMesh(magic.decode(), pos, uv,
                  np.concatenate(strips) if strips else np.zeros(0, np.int32),
                  tuple(parts))


def read_from_ipa(ipa: Path, name: str) -> TSMesh:
    """Decode `res/<name>` straight out of the .ipa, without extracting it."""
    with zipfile.ZipFile(ipa) as z:
        return decode(z.read(IPA_MEMBER.format(name)))
