"""The second half of the parity gate: do the two backends AGREE ON OUTCOMES?

Trajectory parity past the first contact is not achievable and is the wrong
thing to demand -- MJX and MuJoCo C resolve simultaneous contacts differently
and a flipping board amplifies that chaotically (see `parity.py` for the
measured divergence). What must hold instead is weaker and more useful: over a
population of gestures, the DISTRIBUTION of what the board does has to be the
same. That is the property the fitted parameters depend on. If a gesture that
pops 0.4 m on CPU pops 0.05 m on GPU, every fitted number is void on GPU, no
matter how well the two agree in free flight.

Outcomes here are computed FROM POSE ALONE -- position and orientation per
substep -- because that is all both backends expose identically. The CPU
`State` also carries wheel contacts, but using them on one side and not the
other would compare two different measurements and call the difference physics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.state import quat_conj, quat_mul


@dataclass(frozen=True)
class PoseOutcome:
    """What the board did, from its pose trajectory. No trick names."""
    roll_deg: float      # accumulated turn about the deck's long axis
    yaw_deg: float       # accumulated turn about the deck's normal
    peak_height: float   # metres above the resting ride height
    air_s: float         # time spent clear of the ground
    displacement: float  # furthest travelled from the start, in plan


# Clearance above the resting height that counts as airborne. The wheels are
# spheres of a few centimetres, so this is comfortably above contact jitter and
# comfortably below any real pop.
AIR_CLEARANCE_M = 0.01


def pose_outcome(pos: np.ndarray, quat: np.ndarray, timestep: float,
                 rest_z: float | None = None) -> PoseOutcome:
    """(T,3) positions and (T,4) wxyz quaternions -> what happened.

    Rotation is accumulated per step and projected onto the deck's own axes, so
    it survives a board that rolls and yaws at once; summing per-step turns
    rather than comparing first to last is what makes a full 360 read as 360
    instead of 0.
    """
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    rest = float(pos[0, 2] if rest_z is None else rest_z)

    roll = yaw = 0.0
    for i in range(1, len(quat)):
        d = quat_mul(quat[i], quat_conj(quat[i - 1]))
        n = float(np.linalg.norm(d[1:]))
        if n <= 1e-12:
            continue
        ang = 2.0 * np.arctan2(n, d[0])
        axis = d[1:] / n
        # into the deck's frame at this instant
        local = _mat(quat[i]).T @ axis
        roll += ang * local[0]
        yaw += ang * local[2]

    height = pos[:, 2] - rest
    return PoseOutcome(
        roll_deg=float(np.degrees(roll)),
        yaw_deg=float(np.degrees(yaw)),
        peak_height=float(height.max()),
        air_s=float((height > AIR_CLEARANCE_M).sum() * timestep),
        displacement=float(np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1).max()),
    )


def _mat(q):
    from ..sim.state import quat_to_mat
    return quat_to_mat(q)


# --- comparing two populations -------------------------------------------

@dataclass(frozen=True)
class FieldAgreement:
    name: str
    cpu_mean: float
    mjx_mean: float
    cpu_sd: float
    mjx_sd: float
    ks: float           # Kolmogorov-Smirnov statistic between the two samples
    rank_corr: float    # Spearman correlation, gesture by gesture


FIELDS = ("roll_deg", "yaw_deg", "peak_height", "air_s", "displacement")


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic, without pulling in scipy."""
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else 0.0


def compare(cpu: list[PoseOutcome], mjx: list[PoseOutcome]) -> list[FieldAgreement]:
    """Per-field agreement between two populations of outcomes.

    Two different questions, deliberately kept apart:
      * KS asks whether the two backends produce the same DISTRIBUTION -- the
        property that has to hold for fitted parameters to transfer.
      * Spearman asks whether they ORDER the same gestures the same way, which
        is what an optimiser searching over gestures actually consumes. A
        backend can match the distribution while shuffling which gesture is
        which, and that would still break gesture search.
    """
    out = []
    for f in FIELDS:
        a = np.array([getattr(o, f) for o in cpu], dtype=float)
        b = np.array([getattr(o, f) for o in mjx], dtype=float)
        out.append(FieldAgreement(f, float(a.mean()), float(b.mean()),
                                  float(a.std()), float(b.std()),
                                  _ks(a, b), _spearman(a, b)))
    return out


def format_table(rows: list[FieldAgreement]) -> str:
    head = (f"{'field':<14}{'cpu mean':>11}{'mjx mean':>11}{'cpu sd':>10}"
            f"{'mjx sd':>10}{'KS':>7}{'rho':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.name:<14}{r.cpu_mean:>11.3f}{r.mjx_mean:>11.3f}"
                     f"{r.cpu_sd:>10.3f}{r.mjx_sd:>10.3f}{r.ks:>7.2f}{r.rank_corr:>7.2f}")
    return "\n".join(lines)


# --- drivers --------------------------------------------------------------

def random_recipes(n: int, seed: int = 0) -> list[dict]:
    """A population of gestures spanning the flick regime.

    Deliberately wide: a parity check on gestures that all do nothing would
    pass trivially. These range from a light nudge to a violent tail press.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        start = np.array([rng.uniform(0.30, 0.70), rng.uniform(0.45, 0.75)])
        step = rng.normal(0, 0.06, 2)
        pts = [start.tolist(), (start + step).tolist(), (start + 2 * step).tolist()]
        out.append({"gestures": [{"points": pts,
                                  "duration": float(rng.uniform(0.06, 0.35)),
                                  "easing_power": float(rng.uniform(0.5, 2.8))}],
                    "delays": []})
    return out


def cpu_outcomes(recipes: list[dict], params=None,
                 seconds: float | None = None) -> list[PoseOutcome]:
    """Run recipes through the reference simulator, pose only.

    The settle is chosen per recipe so every episode spans exactly the same
    wall time as the MJX one. Comparing a 1.2 s trajectory against a 2.3 s one
    would show a difference in the outcomes that is purely a difference in how
    long each side was allowed to run.
    """
    from ..sim.core import SkateSim
    from ..sim.params import SkateParams
    from ..sim.touch import TouchModel
    from .rollout import EPISODE_SECONDS

    params = params or SkateParams()
    seconds = EPISODE_SECONDS if seconds is None else seconds
    out = []
    for rec in recipes:
        span = max(g["duration"] for g in rec["gestures"])
        sim = SkateSim(params)
        sim.reset(seed=0)
        sim.step(200)
        rest = float(sim.state().pos[2])
        touch = TouchModel(sim)
        pos, quat = [], []
        for st in touch.run_iter(rec, push=False, settle=seconds - span):
            pos.append(st.pos.copy())
            quat.append(st.quat.copy())
        out.append(pose_outcome(np.array(pos), np.array(quat),
                                params.timestep, rest_z=rest))
    return out


def mjx_outcomes(recipes: list[dict], params=None, n_steps: int | None = None,
                 n_slots: int = 2) -> list[PoseOutcome]:
    """Run the same recipes through the batched MJX rollout, pose only.

    Batched in one `vmap`, which is also the only way this is worth running:
    on the reference path each recipe is a separate serial episode.
    """
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from ..sim.params import SkateParams
    from .parity import make_mjx
    from .rollout import episode_length, gesture_arrays, rollout

    params = params or SkateParams()
    mx, d0, cpu = make_mjx(params)
    step = jax.jit(lambda dd: mjx.step(mx, dd))
    for _ in range(200):
        d0 = step(d0)
    rest = float(np.asarray(d0.qpos)[2])

    n = n_steps or episode_length(params)
    arrays = [gesture_arrays(r, n_slots) for r in recipes]
    P = jnp.asarray(np.stack([a[0] for a in arrays]))
    S = jnp.asarray(np.stack([a[1] for a in arrays]))
    T = jnp.asarray(np.stack([a[2] for a in arrays]))

    run = jax.jit(jax.vmap(lambda p, s, t: rollout(
        mx, cpu.model, params, cpu.deck_bid, sorted(cpu._deck_gids),
        d0, p, s, t, n, n_slots)))
    res = run(P, S, T)
    pos = np.asarray(res.pos)
    quat = np.asarray(res.quat)
    return [pose_outcome(pos[i], quat[i], params.timestep, rest_z=rest)
            for i in range(len(recipes))]
