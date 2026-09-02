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
import os
import pathlib
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# Location of the sibling TrueSkate-AI checkout, which owns the gesture schema
# and the capture corpus. Overridable so the path is not baked in.
RIG = pathlib.Path(os.environ.get(
    "TRUESKATE_AI_ROOT",
    "/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI"))
RIG_SRC = RIG / "src"
CORPUS = RIG / "data/self_labeled_traces"


def _gameplay_filter():
    if str(RIG_SRC) not in sys.path:
        sys.path.insert(0, str(RIG_SRC))
    from trueskate_ai.vision import gameplay_filter
    return gameplay_filter


@dataclass(frozen=True, eq=False)
class Sample:
    path: pathlib.Path
    waypoints: np.ndarray      # (n, 2) normalised screen coords
    duration: float
    easing_power: float
    frame_times: np.ndarray    # (n,) seconds from gesture start
    frame_paths: list[pathlib.Path]
    park: str | None
    spin_active: bool
    params: list | None = None          # CMA-ES vector, for recipe samples
    video: pathlib.Path | None = None   # frames.mp4, when frames are packed
    _decoded: list | None = None        # lazily decoded video frames

    def recipe(self) -> dict:
        """The gesture, in the schema `TouchModel.run` consumes.

        Recipe-kind samples store the flat CMA-ES parameter vector rather than
        explicit waypoints, and are decoded with the rig's own
        `unpack_gesture_params` so the layout has exactly one definition -- it
        infers the slot count and the spin block from the vector length alone.
        """
        if self.params is not None:
            if str(RIG_SRC) not in sys.path:
                sys.path.insert(0, str(RIG_SRC))
            import numpy as _np
            from trueskate_ai.rl.cmaes.action_param import unpack_gesture_params
            return unpack_gesture_params(_np.asarray(self.params, dtype=float))
        return {"gestures": [{"points": self.waypoints.tolist(),
                              "duration": self.duration,
                              "easing_power": self.easing_power}],
                "delays": []}

    def frame(self, i: int) -> np.ndarray:
        """Frame `i`, from PNGs or from a packed video.

        The dense aligner writes a single `frames.mp4` per sample instead of
        `frame_NNN.png`, so both layouts exist in the corpora on disk. Video
        frames are decoded sequentially and cached, because seeking per frame
        is far slower than a single pass and callers read most of a sample.
        """
        if self.video is None:
            return cv2.imread(str(self.frame_paths[i]))
        if self._decoded is None:
            cap = cv2.VideoCapture(str(self.video))
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            object.__setattr__(self, "_decoded", frames)
        return self._decoded[i]


def load_sample(d: pathlib.Path) -> Sample | None:
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    m = json.loads(meta_path.read_text())
    wp = m.get("waypoints")
    params = m.get("params")
    if (not wp or len(wp) < 2) and not params:
        return None
    times = np.asarray(m.get("frame_times", []), dtype=float)
    if not len(times):
        return None

    wp_arr = np.asarray(wp, float) if wp else np.zeros((0, 2))
    dur = float(m.get("duration", 0.0))
    ease = float(m.get("easing_power", 1.0))
    park, spin = m.get("park"), bool(m.get("spin_active", False))

    video = d / "frames.mp4"
    if m.get("frames_format") == "mp4" or video.exists():
        # One entry per frame so len(frame_paths) still reports frame count.
        return Sample(d, wp_arr, dur, ease, times, [video] * len(times),
                      park, spin, params, video)

    frames = sorted(d.glob("frame_*.png"))
    if not len(frames) or len(frames) != len(times):
        return None
    return Sample(d, wp_arr, dur, ease, times, frames, park, spin, params)


def iter_samples(root: pathlib.Path = CORPUS):
    """Every loadable sample under `root`, whatever the nesting.

    Layouts differ between corpora: the self-labelled traces are
    `session/sample_NNN`, while the XCTest collector adds a park level
    (`session/park/sample_NNNNNN`). Globbing on meta.json finds both rather
    than encoding one depth.
    """
    for meta in sorted(root.glob("**/meta.json")):
        s = load_sample(meta.parent)
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
