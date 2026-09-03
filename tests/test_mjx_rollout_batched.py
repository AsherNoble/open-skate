"""The batch-major rollout must be the same simulator as the env-major one.

Two nestings of the same physics: `vmap` over per-episode `scan`s, versus one
batched `Data` stepped inside a `scan`. The second exists only because MJX's
render context cannot be traced through a `vmap`, so if the two disagree the
pixel path is simulating a different game from the one that was fitted.

Agreement is judged on OUTCOME DISTRIBUTIONS, not trajectories -- the same
standard `test_mjx_outcomes.py` sets, and for the same reason. Different
arithmetic ordering means different float rounding, and a board in continuous
ground contact amplifies that chaotically: measured 4.4e-2 m apart by the last
frame in float64, from 2e-21 at frame 0.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

N_GESTURES = 24


def test_the_frame_grid_matches_the_env_major_path():
    """68 frames of 17 substeps. If these drift apart the two stop comparable."""
    from opensk.mjx.rollout import episode_length, frame_indices
    from opensk.mjx.rollout_batched import frames_and_substeps
    from opensk.sim.params import SkateParams

    p = SkateParams()
    n_frames, substeps = frames_and_substeps(p)
    assert (n_frames, substeps) == (68, 17)
    assert n_frames == len(frame_indices(episode_length(p), p))


@pytest.fixture(scope="module")
def both():
    """(env outcomes, batch outcomes, env frame-0 pos, batch frame-0 pos).

    Both trajectories come from one fixture because compiling either costs more
    than everything else in this file put together.
    """
    import jax.numpy as jnp
    from mujoco import mjx

    from opensk.mjx.outcomes import pose_outcome, random_recipes
    from opensk.mjx.parity import make_mjx
    from opensk.mjx.rollout import (episode_length, frame_indices,
                                    gesture_arrays, rollout)
    from opensk.mjx.rollout_batched import frames_and_substeps, rollout_batched
    from opensk.sim.params import SkateParams

    p = SkateParams()
    mx, d0, cpu = make_mjx(p)
    step = jax.jit(lambda d: mjx.step(mx, d))
    for _ in range(200):
        d0 = step(d0)
    rest = float(np.asarray(d0.qpos)[2])

    recipes = random_recipes(N_GESTURES, seed=1)
    arrays = [gesture_arrays(r, 2, xp=jnp) for r in recipes]
    P = jnp.stack([a[0] for a in arrays])
    S = jnp.stack([a[1] for a in arrays])
    T = jnp.stack([a[2] for a in arrays])
    db = jax.tree.map(lambda x: jnp.broadcast_to(x, (len(recipes),) + x.shape), d0)

    rb = rollout_batched(mx, cpu.model, p, cpu.deck_bid, sorted(cpu._deck_gids),
                         db, P, S, T)
    bpos = np.asarray(rb.pos).transpose(1, 0, 2)
    bquat = np.asarray(rb.quat).transpose(1, 0, 2)
    _, substeps = frames_and_substeps(p)
    batch = [pose_outcome(bpos[i], bquat[i], p.timestep * substeps, rest_z=rest)
             for i in range(len(recipes))]

    n = episode_length(p)
    idx = frame_indices(n, p)
    run = jax.jit(jax.vmap(lambda a, b, c: rollout(
        mx, cpu.model, p, cpu.deck_bid, sorted(cpu._deck_gids), d0,
        a, b, c, n, 2)))
    re = run(P, S, T)
    epos = np.asarray(re.pos)[:, idx]
    equat = np.asarray(re.quat)[:, idx]
    env = [pose_outcome(epos[i], equat[i], p.timestep * substeps, rest_z=rest)
           for i in range(len(recipes))]
    return env, batch, epos[:, 0], bpos[:, 0]


def test_the_two_nestings_agree_on_outcomes(both):
    from opensk.mjx.outcomes import compare

    for r in compare(both[0], both[1]):
        assert r.ks < 0.25, f"{r.name}: distributions differ, KS {r.ks:.2f}"
        assert r.rank_corr > 0.85, f"{r.name}: ranking differs, rho {r.rank_corr:.2f}"


def test_the_first_frame_is_identical(both):
    """Frame 0, before contact has had time to amplify any rounding difference.

    This is the check that caught the real bug: the env-major scan emits state
    AFTER stepping, so its frame k is substep 17k+1. Sampling the batch-major
    path at 17k left the two a substep apart, which looked exactly like
    numerical divergence and grew to 4.4e-2 m by the last frame.
    """
    _, _, env_pos0, batch_pos0 = both
    assert np.abs(env_pos0 - batch_pos0).max() < 1e-6
