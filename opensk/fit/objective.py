"""Compare a simulated gesture against the real frames it was captured from.

The comparison happens in IMAGE SPACE, not pose space. Simulate the gesture
with candidate physics, render the board's silhouette at each real frame's
timestamp, and score the overlap against the segmented real silhouette.

Why not recover per-frame 6-DOF pose first and compare trajectories? Because
that inversion is the hard part, and it is not needed. A thin, end-symmetric
plate has a badly multimodal silhouette objective: measured directly, the true
pose scores IoU 1.0000 while a CMA-ES search from a warm start returned 0.84
median with 58 deg median rotation error, at 4.4 s per frame — around 13 hours
for the corpus, and wrong. Matching whole trajectories instead lets the physics
supply temporal coherence for free, and searches the 26 parameters we actually
want rather than six per frame.

WHAT THIS CAN AND CANNOT CONSTRAIN. The chase camera is locked to the board, so
translating the board translates the camera identically and the silhouette does
not change at all (measured: IoU 1.0000 after a 2 m translation). Board
translation is therefore invisible to this objective, and so are its causes.
What it does constrain is rotation and height — flip rate, shuvit rate, pop
height, landing attitude — which is the bulk of trick dynamics. Recovering
translation needs camera egomotion from the static background, which is not
solved here. Do not read a good score as evidence that translation is right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose.frames import Sample, gameplay_flags
from ..pose.render import SceneRenderer
from ..pose.segment import board_mask, plausible
from ..sim.camera import FollowCamera, board_yaw
from ..sim.core import SkateSim
from ..sim.params import SkateParams
from ..sim.touch import TouchModel


@dataclass(frozen=True)
class SampleScore:
    iou: float                 # mean over scored frames
    per_frame: np.ndarray
    n_scored: int
    n_frames: int

    @property
    def loss(self) -> float:
        return 1.0 - self.iou


def real_masks(sample: Sample, height: int, width: int
               ) -> tuple[list[np.ndarray | None], np.ndarray]:
    """Segmented board silhouettes for a sample, at render resolution.

    Frames that are menus, or where segmentation finds nothing board-shaped,
    come back as None and are skipped rather than scored as zero — scoring
    them would reward a simulation that put the board off-screen.
    """
    import cv2

    flags = gameplay_flags(sample)
    out: list[np.ndarray | None] = []
    for i, ok in enumerate(flags):
        if not ok:
            out.append(None)
            continue
        frame = sample.frame(i)
        m = board_mask(frame)
        if not plausible(m, frame.shape[0]):
            out.append(None)
            continue
        small = cv2.resize(m.mask.astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(small)
    return out, flags


def score_sample(sample: Sample, params: SkateParams | None = None, *,
                 height: int = 224, targets=None,
                 sim: SkateSim | None = None,
                 renderer: SceneRenderer | None = None) -> SampleScore:
    """Simulate `sample`'s gesture and score silhouette overlap per frame.

    Replay conventions are the capture's, not ours: the board is reset to the
    anchor and settled, and there is NO push — `collect_self_labeled_traces.py`
    resets every gesture and never pushes. Replaying with a push carries the
    board out from under the gesture entirely.
    """
    params = params or SkateParams()
    sim = sim or SkateSim(params)
    renderer = renderer or SceneRenderer(sim, height=height)
    if targets is None:
        targets, _ = real_masks(sample, renderer.height, renderer.width)

    sim.reset(seed=0)
    sim.step(200)                       # settle at the anchor, as the rig does
    touch = TouchModel(sim)

    # Frame timestamps are relative to the gesture's start, and the earliest is
    # already positive, so the sim starts at the gesture and is sampled forward.
    times = np.asarray(sample.frame_times, dtype=float)
    want = [(t, i) for i, t in enumerate(times) if targets[i] is not None]
    if not want:
        return SampleScore(0.0, np.zeros(0), 0, len(times))
    want.sort()

    dt = params.timestep
    ious = np.zeros(len(want))
    t_sim = 0.0
    # Run the gesture and the settle in one pass, sampling as timestamps pass.
    schedule = touch.run_iter(sample.recipe(), push=False,
                              settle=max(0.0, float(want[-1][0]) + 0.05))
    k = 0
    for st in schedule:
        t_sim += dt
        while k < len(want) and t_sim >= want[k][0]:
            cam = FollowCamera(params)
            cam.reset(st.pos, board_yaw(st.quat))
            m = renderer.board_pixels(cam)
            tgt = targets[want[k][1]]
            union = np.count_nonzero(m | tgt)
            ious[k] = 0.0 if union == 0 else np.count_nonzero(m & tgt) / union
            k += 1
        if k >= len(want):
            break

    return SampleScore(float(ious[:k].mean()) if k else 0.0,
                       ious[:k], int(k), len(times))
