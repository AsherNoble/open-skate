"""The training environment's contract: shapes, bounds, determinism, honesty.

The thing most worth protecting here is that an action is a DEVICE gesture.
The moment the action space drifts from the phone's, everything learned in
simulation stops being executable on hardware, and the project loses its point.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from opensk.rl.action import action_dim, decode, to_recipe
from opensk.sim.gesture_spec import X_BOUND_MIN, Y_BOUND_MAX, Y_BOUND_MIN


def test_every_action_decodes_to_a_gesture_the_phone_can_execute():
    """Bounds are the device's. An out-of-bounds point is not a gesture."""
    rng = np.random.default_rng(0)
    for v in rng.normal(0, 4.0, (200, action_dim(2))):   # deliberately extreme
        rec = to_recipe(v)
        for gest in rec["gestures"]:
            for x, y in gest["points"]:
                assert X_BOUND_MIN <= x <= 1.0
                assert Y_BOUND_MIN <= y <= Y_BOUND_MAX
            assert 0.05 <= gest["duration"] <= 0.80
            assert 0.3 <= gest["easing_power"] <= 3.0
        assert all(-0.25 <= d <= 0.60 for d in rec["delays"])


def test_decode_is_the_same_under_numpy_and_jax():
    """The env decodes under jax; the round-trip to a phone recipe under numpy.

    If those two disagree, the gesture that was searched for is not the gesture
    that gets executed.
    """
    import jax.numpy as jnp

    v = np.random.default_rng(1).normal(0, 1.5, action_dim(2))
    a = decode(v, 2, xp=np)
    b = decode(jnp.asarray(v), 2, xp=jnp)
    for x, y in zip(a, b):
        assert np.allclose(np.asarray(x), np.asarray(y), atol=1e-5)


@pytest.fixture(scope="module")
def env():
    from opensk.rl.env import GestureEnv
    return GestureEnv()


def test_step_returns_a_rollout_shaped_like_a_capture(env):
    """68 frames, one gesture -- the same object shape as an expert demo."""
    ep = env.step(env.sample_actions(4, seed=0))
    assert ep.pos.shape == (4, len(env.frames), 3)
    assert ep.quat.shape == (4, len(env.frames), 4)
    assert 66 <= len(env.frames) <= 70
    for f in (ep.roll_deg, ep.yaw_deg, ep.peak_height, ep.air_s,
              ep.displacement, ep.valid):
        assert np.asarray(f).shape == (4,)


def test_the_same_actions_give_the_same_episodes(env):
    """A world model trained on a non-deterministic env learns the noise."""
    a = env.sample_actions(3, seed=7)
    first, second = env.step(a), env.step(a)
    # equal_nan, because an unstable episode legitimately carries NaN and
    # NaN != NaN would make this test fail for a reason that is not
    # non-determinism.
    assert np.array_equal(np.asarray(first.pos), np.asarray(second.pos),
                          equal_nan=True)
    assert np.array_equal(np.asarray(first.quat), np.asarray(second.quat),
                          equal_nan=True)


def test_unstable_episodes_are_reported_not_hidden(env):
    """The uncapped ground shove can destabilise the solver -- on CPU too.

    `valid` is the environment saying so out loud. A trainer handed a silent
    NaN learns from it; a trainer handed a flag can drop the sample.
    """
    ep = env.step(env.sample_actions(8, seed=0))
    valid = np.asarray(ep.valid)
    peak = np.asarray(ep.peak_height)
    disp = np.asarray(ep.displacement)
    for i in range(len(valid)):
        physical = (np.isfinite(peak[i]) and np.isfinite(disp[i])
                    and peak[i] < 3.0 and disp[i] < 40.0)
        assert bool(valid[i]) == physical
