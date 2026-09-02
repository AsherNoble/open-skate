"""Determinism is a hard requirement, not a nicety.

The MJX port, batched world-model rollouts and CMA-ES system identification
all assume that replaying the same gesture reproduces the same trajectory
exactly. If that drifts, sysid is fitting noise.
"""
import numpy as np

from opensk.sim.core import SkateSim
from opensk.sim.params import SkateParams
from opensk.sim.state import VECTOR_DIM


def _rollout(seed: int, n: int = 5000) -> np.ndarray:
    sim = SkateSim(SkateParams())
    sim.reset(seed=seed, speed=2.5)
    rng = np.random.default_rng(seed)
    for i in range(n):
        if i % 250 < 60:  # intermittent off-centre presses, like a gesture
            f = rng.uniform(-40, 40, size=3)
            sim.apply_force(f, sim.body_point([-0.3, 0.04, 0.006]))
        sim.step()
    return np.concatenate([sim.data.qpos, sim.data.qvel])


def test_bitwise_identical_across_instances():
    assert np.array_equal(_rollout(3), _rollout(3))


def test_different_seeds_diverge():
    """Guards against the test passing because forces are silently ignored."""
    assert not np.array_equal(_rollout(3), _rollout(4))


def test_observation_vector_dim():
    sim = SkateSim()
    assert sim.reset(seed=0).to_vector().shape == (VECTOR_DIM,)
