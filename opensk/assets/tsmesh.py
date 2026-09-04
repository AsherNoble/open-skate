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

    u32          index count
    u16[n_idx]   triangle-STRIP indices, degenerate-stitched
    u32          vertex count
    f32[nv][3]   positions
    f32[nv][2]   UVs

`SKDE` (deck top) and `SKTR` (truck) are NOT supported. They are multi-part
containers whose header length differs -- `deck.bin` carries indices reaching
vertex 1251 while the count read at the guessed offset says 945, which is
proof the offset is wrong rather than a reason to pick a bigger number.
`decode` raises rather than returning a plausible-looking point cloud, because
a wrong decode here would look like a shape and quietly become the board.
"""
from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Only magics whose decode has been VALIDATED end to end (UVs in range, indices
# in range, file consumed exactly, and the decoded shape is the right shape).
HEADER_BYTES = {b"OMSH": 0x0C, b"SKWH": 0x28}
UNSUPPORTED = {b"SKDE": "deck top", b"SKTR": "truck"}

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
    uvs: np.ndarray            # (nv, 2) float32, in [0, 1]
    strip: np.ndarray          # (n_idx,) int32, raw index buffer

    @property
    def extent(self) -> np.ndarray:
        return self.positions.max(axis=0) - self.positions.min(axis=0)


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
    """Decode one `res/*.bin` mesh, or raise `UnsupportedMesh`."""
    magic = data[:4]
    if magic in UNSUPPORTED:
        raise UnsupportedMesh(
            f"{magic.decode()} ({UNSUPPORTED[magic]}) is a multi-part container "
            "whose header length is not yet known; refusing to guess")
    if magic not in HEADER_BYTES:
        raise UnsupportedMesh(f"unknown mesh magic {magic!r}")

    off = HEADER_BYTES[magic]
    positions, uvs, strips, base = [], [], [], 0
    while off < len(data):
        n_idx = struct.unpack_from("<I", data, off)[0]
        idx_end = off + 4 + 2 * n_idx
        if idx_end + 4 > len(data):
            raise UnsupportedMesh("index block runs past end of file")
        nv = struct.unpack_from("<I", data, idx_end)[0]
        end = idx_end + 4 + VERTEX_BYTES * nv
        if end > len(data):
            raise UnsupportedMesh("vertex block runs past end of file")

        strip = np.frombuffer(data, "<u2", count=n_idx, offset=off + 4)
        if n_idx and strip.max() >= nv:
            raise UnsupportedMesh(
                f"index {strip.max()} out of range for {nv} vertices")
        pos = np.frombuffer(data, "<f4", count=nv * 3,
                            offset=idx_end + 4).reshape(nv, 3)
        uv = np.frombuffer(data, "<f4", count=nv * 2,
                           offset=idx_end + 4 + POS_BYTES * nv).reshape(nv, 2)
        # The check that actually discriminates the two-stream layout from the
        # interleaved one. Do not weaken it.
        if not np.isfinite(uv).all() or uv.min() < -0.05 or uv.max() > 1.05:
            raise UnsupportedMesh(
                f"UVs outside [0,1] ({uv.min():.3f}..{uv.max():.3f}); the "
                "vertex layout is not what this decoder assumes")

        positions.append(pos)
        uvs.append(uv)
        strips.append(strip.astype(np.int32) + base)
        base += nv
        off = end

    if off != len(data):
        raise UnsupportedMesh("trailing bytes after the last part")
    return TSMesh(magic.decode(), np.concatenate(positions),
                  np.concatenate(uvs), np.concatenate(strips))


def read_from_ipa(ipa: Path, name: str) -> TSMesh:
    """Decode `res/<name>` straight out of the .ipa, without extracting it."""
    with zipfile.ZipFile(ipa) as z:
        return decode(z.read(IPA_MEMBER.format(name)))
