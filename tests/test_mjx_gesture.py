"""The JAX gesture and camera port must equal the CPU reference exactly.

These are not approximations of the CPU path -- they are the same arithmetic
written against an `xp` namespace. Any drift here changes the environment's
action space, and the action space matching the device byte-for-byte is the
single property that makes anything learned in simulation transferable.
"""
import numpy as np
import pytest

from opensk.mjx.gesture import (board_yaw, camera_ray, path_position, schedule,
                                segment_durations_ms)
from opensk.sim.camera import FollowCamera
from opensk.sim.camera import board_yaw as cpu_board_yaw
from opensk.sim.gesture_spec import GesturePath
from opensk.sim.gesture_spec import segment_durations_ms as cpu_segment_durations
from opensk.sim.params import SkateParams

CASES = [(2, 350, 1.0), (2, 350, 1.8), (2, 350, 0.4), (3, 500, 2.0),
         (2, 30, 3.0), (4, 800, 0.3), (3, 47, 2.7)]


@pytest.mark.parametrize("n,ms,power", CASES)
def test_segment_durations_match_cpu(n, ms, power):
    """Whole-millisecond truncation and the easing_power==1 even split."""
    assert [int(x) for x in segment_durations_ms(n, float(ms), power)] == \
        cpu_segment_durations(n, ms, power)


def test_path_position_matches_cpu():
    pts = np.array([[0.5, 0.6], [0.45, 0.5], [0.6, 0.3]])
    cpu = GesturePath(pts, 0.35, 1.8)
    P, seg_t, dur = schedule(pts, 0.35, 1.8)
    assert float(dur) == pytest.approx(cpu.duration)
    for t in np.linspace(0.0, 0.4, 25):
        assert np.allclose(path_position(P, seg_t, t), cpu.position_at(t), atol=0)


def test_camera_ray_matches_cpu():
    p = SkateParams()
    cam = FollowCamera(p)
    pos = np.array([0.3, -0.1, 0.09])
    quat = np.array([0.966, 0.0, 0.0, 0.259])
    cam.reset(pos, cpu_board_yaw(quat))
    assert float(board_yaw(quat)) == pytest.approx(cpu_board_yaw(quat))
    for nx in (0.2, 0.5, 0.8):
        for ny in (0.35, 0.6, 0.85):
            o_cpu, d_cpu = cam.ray(nx, ny)
            o, d = camera_ray(nx, ny, pos, board_yaw(quat), p)
            assert np.allclose(o, o_cpu, atol=0)
            assert np.allclose(d, d_cpu, atol=0)


def test_runs_under_jit_and_vmap():
    """The reason for the xp namespace: batch over thousands of environments."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    p = SkateParams()
    fn = jax.jit(jax.vmap(
        lambda nx, ny, pos, quat: camera_ray(nx, ny, pos, board_yaw(quat, xp=jnp),
                                             p, xp=jnp)))
    B = 256
    o, d = fn(jnp.full((B,), 0.5), jnp.full((B,), 0.6),
              jnp.tile(jnp.array([0.0, 0.0, 0.09]), (B, 1)),
              jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (B, 1)))
    assert o.shape == (B, 3) and d.shape == (B, 3)
    assert np.allclose(np.linalg.norm(np.asarray(d), axis=1), 1.0, atol=1e-6)
