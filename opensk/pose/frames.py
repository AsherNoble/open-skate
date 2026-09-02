"""Load the real-gameplay corpus: frames plus the gesture that produced them.

This is the only corpus that contains Δenvironment, so it is the ground truth
the physics is fitted against. Each sample directory holds `frame_NNN.png` at
~10 fps plus a `meta.json` carrying the exact gesture that was executed and the
frame timestamps relative to the gesture's start.

Non-gameplay frames (park menu, replay UI, editor) are excluded using the rig's
own `vision.gameplay_filter`, imported rather than reimplemented so the two
projects cannot drift about what counts as gameplay. This corpus carries no
`.menu` markers — that pass was only ever run over `sls_traces` — so the filter
has to run here, and it matters: menu screens are present and they defeat board
segmentation completely.
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass

import cv2
import numpy as np

RIG = pathlib.Path("/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI")
RIG_SRC = RIG / "src"
CORPUS = RIG / "data/self_labeled_traces"


def _gameplay_filter():
    if str(RIG_SRC) not in sys.path:
        sys.path.insert(0, str(RIG_SRC))
    from trueskate_ai.vision import gameplay_filter
    return gameplay_filter


@dataclass(frozen=True)
class Sample:
    path: pathlib.Path
    waypoints: np.ndarray      # (n, 2) normalised screen coords
    duration: float
    easing_power: float
    frame_times: np.ndarray    # (n,) seconds from gesture start
    frame_paths: list[pathlib.Path]
    park: str | None
    spin_active: bool

    def recipe(self) -> dict:
        """The gesture, in the schema `TouchModel.run` consumes."""
        return {"gestures": [{"points": self.waypoints.tolist(),
                              "duration": self.duration,
                              "easing_power": self.easing_power}],
                "delays": []}

    def frame(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.frame_paths[i]))


def load_sample(d: pathlib.Path) -> Sample | None:
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    m = json.loads(meta_path.read_text())
    wp = m.get("waypoints")
    if not wp or len(wp) < 2:
        return None  # params-vector samples carry no explicit waypoints
    frames = sorted(d.glob("frame_*.png"))
    times = np.asarray(m.get("frame_times", []), dtype=float)
    if not len(frames) or len(frames) != len(times):
        return None
    return Sample(d, np.asarray(wp, float), float(m["duration"]),
                  float(m.get("easing_power", 1.0)), times, frames,
                  m.get("park"), bool(m.get("spin_active", False)))


def iter_samples(root: pathlib.Path = CORPUS):
    for d in sorted(root.glob("*/sample_*")):
        s = load_sample(d)
        if s is not None:
            yield s


def gameplay_flags(sample: Sample) -> np.ndarray:
    """Per-frame boolean: real gameplay, rather than a menu or the editor."""
    gf = _gameplay_filter()
    from PIL import Image
    out = []
    for p in sample.frame_paths:
        try:
            out.append(bool(gf.is_gameplay_frame(Image.open(p))))
        except Exception:
            out.append(False)
    return np.asarray(out)
