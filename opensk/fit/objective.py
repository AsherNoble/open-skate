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
    static_iou: float = 0.0    # same frames, board left at its initial pose

    @property
    def loss(self) -> float:
        return 1.0 - self.iou

    @property
    def gain(self) -> float:
        """Overlap earned ABOVE a board that never moves.

        This is the number that matters. Raw overlap is dominated by the board
        simply being roughly where it started: fitting it directly drove
        `touch_gain` from 600 to 76 and still scored 0.6424 on held-out data
        against 0.6444 for an inert board -- the optimiser's best strategy was
        to stop moving. Scoring the gain makes inertness worth exactly zero, so
        only reproducing the real motion can earn credit.
        """
        return self.iou - self.static_iou


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
                 renderer: SceneRenderer | None = None,
                 time_offset: float = 0.0) -> SampleScore:
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

    # Reference silhouette: the board where the gesture found it. Rendered once
    # per sample, since holding still is parameter-independent once deck
    # geometry is fixed (and deck geometry is measured, never fitted).
    st0 = sim.state()
    cam0 = FollowCamera(params)
    cam0.reset(st0.pos, board_yaw(st0.quat))
    static_mask = renderer.board_pixels(cam0)

    touch = TouchModel(sim)

    # Frame timestamps are relative to the gesture's start, and the earliest is
    # already positive, so the sim starts at the gesture and is sampled forward.
    # `time_offset` shifts the sim clock relative to the frame timestamps, to
    # absorb command->pixel latency: the delay between the host issuing a W3C
    # action and the board actually moving on screen. The rig documents this
    # offset as UNCALIBRATED and assumed zero; if it is really 100-200 ms then
    # at 30 fps every trajectory is compared against frames 3-6 too early, and
    # even correct physics would score no better than not moving.
    times = np.asarray(sample.frame_times, dtype=float) - float(time_offset)
    want = [(t, i) for i, t in enumerate(times) if targets[i] is not None]
    if not want:
        return SampleScore(0.0, np.zeros(0), 0, len(times), 0.0)
    want.sort()

    dt = params.timestep
    ious = np.zeros(len(want))
    statics = np.zeros(len(want))
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
            su = np.count_nonzero(static_mask | tgt)
            statics[k] = 0.0 if su == 0 else np.count_nonzero(static_mask & tgt) / su
            k += 1
        if k >= len(want):
            break

    return SampleScore(float(ious[:k].mean()) if k else 0.0,
                       ious[:k], int(k), len(times),
                       float(statics[:k].mean()) if k else 0.0)
