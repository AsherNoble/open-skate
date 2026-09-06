"""Small RGB goldens for the procedural Blender-directed appearance pass."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from opensk.pose.render import SceneRenderer, gl_backend_unavailable
from opensk.sim.appearance import APPEARANCE_PRESETS
from opensk.sim.core import SkateSim
from opensk.sim.model.parks import PARKS


GOLDEN = Path(__file__).with_name("golden")


def _render(name: str, park: str = "plaza") -> tuple[np.ndarray, np.ndarray]:
    sim = SkateSim(park=PARKS[park], appearance=name)
    sim.reset(seed=0)
    sim.step(400)
    try:
        renderer = SceneRenderer.for_mode(sim, "training")
        return renderer.render(), renderer.board_pixels()
    except Exception as exc:
        if gl_backend_unavailable(exc):
            pytest.skip(f"MuJoCo GL backend unavailable: {exc}")
        raise


def test_unrelated_renderer_errors_cannot_be_classified_as_gl_unavailable():
    assert not gl_backend_unavailable(RuntimeError("renderer implementation bug"))


def test_renderer_implementation_errors_are_raised_not_skipped(monkeypatch):
    def broken_renderer(*_args, **_kwargs):
        raise RuntimeError("deliberate renderer implementation bug")

    monkeypatch.setattr(SceneRenderer, "for_mode", broken_renderer)
    with pytest.raises(RuntimeError, match="deliberate renderer implementation bug"):
        _render("day")


@pytest.mark.parametrize("appearance", APPEARANCE_PRESETS)
def test_training_rgb_matches_visual_golden(appearance):
    actual, mask = _render(appearance)
    expected_bgr = cv2.imread(str(GOLDEN / f"appearance_{appearance}.png"))
    assert expected_bgr is not None, "visual golden is missing"
    expected = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)
    assert actual.shape == expected.shape == (128, 64, 3)

    # Pixel-exact on the reference renderer; tolerant enough for small driver
    # differences while still catching lost materials, lights or obstacles.
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert error.mean() < 7.0
    assert np.quantile(error, 0.95) < 18.0

    actual_edges = cv2.Canny(actual, 45, 110) > 0
    expected_edges = cv2.Canny(expected, 45, 110) > 0
    overlap = 2 * np.logical_and(actual_edges, expected_edges).sum()
    edge_dice = overlap / max(actual_edges.sum() + expected_edges.sum(), 1)
    assert edge_dice > 0.72

    # Composition is tested from the dedicated mask, never inferred from RGB:
    # the complete board is in-frame and lives in the lower portion.
    ys, xs = np.nonzero(mask)
    assert xs.min() > 0 and xs.max() < mask.shape[1] - 1
    assert ys.min() > 0 and ys.max() < mask.shape[0] - 1
    assert ys.mean() / mask.shape[0] > 0.58


def test_silhouette_is_independent_of_rgb_appearance():
    masks = [_render(name)[1] for name in APPEARANCE_PRESETS]
    assert masks[0].any()
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])


def test_every_park_renders_useful_training_pixels():
    """Exercise the real CPU renderer for all collision environments.

    Kept as one test so a genuinely unavailable backend produces one explicit
    skip, while any failure after context creation still fails the suite.
    """
    for park in PARKS:
        rgb, mask = _render("day", park)
        assert rgb.shape == (128, 64, 3)
        assert mask.shape == (128, 64)
        assert mask.any()
        assert float(rgb.std()) > 8.0
