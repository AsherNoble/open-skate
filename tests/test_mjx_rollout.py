"""One gesture episode as a compiled scan, and its shape contract.

The episode spans the same 2.3 s as a capture window, so a simulated rollout
and an expert demonstration are the same shaped object -- 68-69 frames at
30 fps, one gesture. Keeping that alignment is what lets sim rollouts and real
demonstrations feed the same world model.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")


def test_episode_matches_the_capture_window():
    from opensk.mjx.rollout import episode_length, frame_indices
    from opensk.sim.params import SkateParams

    p = SkateParams()
    n = episode_length(p)
    frames = frame_indices(n, p)
    assert n == 1150
    assert 66 <= len(frames) <= 70, f"{len(frames)} frames; captures have 68-69"


def test_gesture_arrays_pad_without_becoming_live():
    """A padded slot must never fire, or a one-slot gesture grows a phantom finger."""
    import jax.numpy as jnp

    from opensk.mjx.rollout import gesture_arrays

    recipe = {"gestures": [{"points": [[0.5, 0.6], [0.5, 0.7], [0.5, 0.8]],
                            "duration": 0.13, "easing_power": 2.2}], "delays": []}
    pts, seg, t0 = gesture_arrays(recipe, n_slots=2, xp=jnp)
    assert pts.shape == (2, 3, 2) and seg.shape == (2, 3) and t0.shape == (2,)
    assert float(t0[0]) == 0.0
    assert float(t0[1]) > 1e5, "padded slot must start beyond any episode"


def test_rollout_runs_and_is_deterministic():
    import jax.numpy as jnp
    from mujoco import mjx

    from opensk.mjx.parity import make_mjx
    from opensk.mjx.rollout import episode_length, gesture_arrays, rollout
    from opensk.sim.params import SkateParams

    p = SkateParams()
    mx, d0, cpu = make_mjx(p)
    step = jax.jit(lambda dd: mjx.step(mx, dd))
    for _ in range(50):
        d0 = step(d0)

    recipe = {"gestures": [{"points": [[0.5, 0.62], [0.5, 0.66], [0.5, 0.70]],
                            "duration": 0.12, "easing_power": 2.0}], "delays": []}
    pts, seg, t0 = gesture_arrays(recipe, n_slots=2, xp=jnp)
    n = 120  # short: this checks wiring and determinism, not physics
    run = jax.jit(lambda d, P, S, T: rollout(
        mx, cpu.model, p, cpu.deck_bid, sorted(cpu._deck_gids), d, P, S, T, n))

    a = run(d0, pts, seg, t0)
    b = run(d0, pts, seg, t0)
    assert a.pos.shape == (n, 3) and a.quat.shape == (n, 4)
    assert np.array_equal(np.asarray(a.pos), np.asarray(b.pos))
    assert np.isfinite(np.asarray(a.pos)).all(), "rollout produced non-finite state"
