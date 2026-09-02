"""Recover the board's 6-DOF pose from a frame, by analysis-by-synthesis.

Render the board at a candidate pose, compare its silhouette to the segmented
real one, and search for the pose that maximises overlap. No keypoint network
and no training data: silhouettes carry no texture, so there is no gap between
how our untextured MuJoCo deck looks and how True Skate's textured one does.

THE CAMERA MUST BE FIXED, NOT FOLLOWING. This is not a detail. With the chase
camera tracking a candidate pose, translating the board two metres leaves the
rendered silhouette bit-identical (IoU 1.0000) — the camera moves with it, so
translation is perfectly unobservable. Held fixed, 0.3 m of translation drops
IoU to 0.41. Every caller therefore supplies the camera for the frame; the
camera's own motion has to be recovered from the static background, which is
a separate problem this module deliberately does not solve.

What a fixed camera does buy is all six degrees of freedom at once: position
against the background, and orientation and height against the silhouette.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("Open Skate needs mujoco") from exc

from ..sim.camera import FollowCamera
from ..sim.core import SkateSim
from ..sim.params import SkateParams
from .render import SceneRenderer


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> wxyz quaternion.

    Axis-angle rather than Euler angles because it has no gimbal lock and no
    wrap discontinuity, both of which would put false local optima in the
    search exactly where tricks live.
    """
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(r, dtype=float) / theta
    s = np.sin(0.5 * theta)
    return np.array([np.cos(0.5 * theta), *(axis * s)])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q[0] < 0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    theta = 2.0 * np.arccos(w)
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    return q[1:] / s * theta


@dataclass(frozen=True)
class PoseFit:
    pos: np.ndarray      # (3,) world
    quat: np.ndarray     # (4,) wxyz
    iou: float
    evals: int

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.pos, quat_to_rotvec(self.quat)])


class PoseFitter:
    """Fit board pose to a target silhouette under a FIXED camera."""

    def __init__(self, params: SkateParams | None = None, height: int = 224):
        # Renders at reduced resolution: the search runs hundreds of renders
        # per frame and IoU is insensitive to fine detail at this scale.
        self.params = params or SkateParams()
        self.sim = SkateSim(self.params)
        self.sim.reset(seed=0)
        self.renderer = SceneRenderer(self.sim, height=height)
        self.height = height
        self.width = self.renderer.width

    # -- forward model -----------------------------------------------------

    def render_mask(self, pos, rotvec, camera: FollowCamera) -> np.ndarray:
        d = self.sim.data
        d.qpos[0:3] = pos
        d.qpos[3:7] = rotvec_to_quat(np.asarray(rotvec, dtype=float))
        # Trucks and wheels are left at their neutral joint angles: they are a
        # few pixels each at this scale and fitting them would add four badly
        # observed degrees of freedom.
        d.qpos[7:] = 0.0
        mujoco.mj_forward(self.sim.model, d)
        return self.renderer.board_pixels(camera)

    @staticmethod
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        union = np.count_nonzero(a | b)
        if union == 0:
            return 0.0
        return np.count_nonzero(a & b) / union

    # -- search ------------------------------------------------------------

    def fit(self, target: np.ndarray, camera: FollowCamera,
            seed_pos: np.ndarray, seed_rotvec: np.ndarray, *,
            sigma_pos: float = 0.10, sigma_rot: float = 0.5,
            evals: int = 300, cma_seed: int = 0) -> PoseFit:
        """Maximise silhouette IoU against `target` from a seeded pose.

        `target` must already be at this fitter's render resolution.
        Seeding from the previous frame matters: at 10 fps the board barely
        moves between frames except through a trick, so a warm start both
        speeds the search and keeps it in the right basin.
        """
        import cma

        x0 = np.concatenate([np.asarray(seed_pos, float),
                             np.asarray(seed_rotvec, float)])
        stds = [sigma_pos] * 3 + [sigma_rot] * 3
        es = cma.CMAEvolutionStrategy(
            list(x0), 1.0,
            {"seed": cma_seed + 1, "maxfevals": evals, "verbose": -9,
             "CMA_stds": stds,
             "bounds": [[-50, -50, 0.0, -20, -20, -20],
                        [50, 50, 4.0, 20, 20, 20]]})
        n = 0
        while not es.stop():
            xs = es.ask()
            losses = []
            for x in xs:
                m = self.render_mask(x[:3], x[3:], camera)
                losses.append(1.0 - self.iou(m, target))
                n += 1
            es.tell(xs, losses)
        best = np.array(es.result.xbest)
        return PoseFit(best[:3], rotvec_to_quat(best[3:]),
                       1.0 - float(es.result.fbest), n)


def fixed_camera(params: SkateParams, target_xy_yaw) -> FollowCamera:
    """A camera pinned to a known viewpoint, which never follows.

    `target_xy_yaw` is ((x, y, z), yaw) of the point the camera looks at.
    """
    cam = FollowCamera(params)
    pos, yaw = target_xy_yaw
    cam.reset(np.asarray(pos, dtype=float), float(yaw))
    return cam


# Rotation increments to seed from the previous frame's pose.
#
# Not random: at 10 fps a board mid-trick can turn most of a revolution between
# frames, and it turns about its OWN axes -- flips about the long axis, shuvits
# about the normal. Quarter-turn increments about those two therefore cover the
# real jumps, and they double as the symmetry traps a flat end-symmetric deck
# falls into (a half turn about either axis leaves the outline nearly
# unchanged). Random restarts alone do not find these: from a cold start with
# arbitrary target orientations the search returned IoU 0.55-0.93 where the
# true pose scores 1.0000 -- a search failure, not an ambiguity.
_LONG = np.array([1.0, 0.0, 0.0])
_NORMAL = np.array([0.0, 0.0, 1.0])
_LATERAL = np.array([0.0, 1.0, 0.0])

_QUARTER_TURNS = tuple(
    axis * ang
    for axis in (_LONG, _NORMAL, _LATERAL)
    for ang in (np.pi / 2, np.pi, 3 * np.pi / 2)
)
_SEED_INCREMENTS = (np.zeros(3),) + _QUARTER_TURNS


def compose_rotvec(a, b) -> np.ndarray:
    """Rotation vector of (rot a) applied after (rot b)."""
    from ..sim.state import quat_mul
    return quat_to_rotvec(quat_mul(rotvec_to_quat(np.asarray(a, float)),
                                   rotvec_to_quat(np.asarray(b, float))))


def body_increment(seed_rotvec, increment) -> np.ndarray:
    """Apply `increment` in the BOARD's frame, not the world's."""
    return compose_rotvec(seed_rotvec, increment)


def fit_multistart(fitter: "PoseFitter", target, camera, seed_pos, seed_rotvec,
                   *, evals_per_start: int = 200, extra_random: int = 2,
                   rng_seed: int = 0, **kw) -> PoseFit:
    """Fit from the previous pose plus quarter-turn increments; keep best IoU.

    Selection by IoU is sound because the objective is sharply discriminative:
    the correct basin reaches ~1.0 while symmetry traps plateau well below it.
    """
    rng = np.random.default_rng(rng_seed)
    seeds = [body_increment(seed_rotvec, inc) for inc in _SEED_INCREMENTS]
    for _ in range(extra_random):
        seeds.append(compose_rotvec(seed_rotvec, rng.normal(0, 0.6, 3)))

    best, total = None, 0
    for i, r0 in enumerate(seeds):
        res = fitter.fit(target, camera, seed_pos, r0,
                         evals=evals_per_start, cma_seed=rng_seed * 97 + i, **kw)
        total += res.evals
        if best is None or res.iou > best.iou:
            best = res
        if best.iou > 0.985:      # already essentially exact; stop paying
            break
    return PoseFit(best.pos, best.quat, best.iou, total)
