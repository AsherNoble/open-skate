"""The observable state of the board.

Deliberately a plain, flat, NumPy-only value type. Nothing here touches
MuJoCo, so `game/` and `fit/` can be tested against synthetic states with no
physics in the loop, and the same struct can be produced by an MJX rollout
later without either consumer noticing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Order matters and is part of the observation contract.
WHEELS = ("front_wheel_l", "front_wheel_r", "rear_wheel_l", "rear_wheel_r")


@dataclass(frozen=True)
class State:
    t: float
    pos: np.ndarray            # (3,) deck centre, world
    quat: np.ndarray           # (4,) wxyz, deck orientation
    linvel: np.ndarray         # (3,) world
    angvel: np.ndarray         # (3,) world
    steer: np.ndarray          # (2,) front, rear truck angle (rad)
    wheel_spin: np.ndarray     # (4,) rad/s, WHEELS order
    wheel_contact: np.ndarray  # (4,) bool, WHEELS order
    deck_contact: bool         # any deck geom touching anything

    @property
    def airborne(self) -> bool:
        return not self.wheel_contact.any() and not self.deck_contact

    @property
    def rolling(self) -> bool:
        return bool(self.wheel_contact.all())

    def basis(self) -> np.ndarray:
        """Deck frame as columns (long, lateral, normal) in world coords."""
        return quat_to_mat(self.quat)

    def to_vector(self) -> np.ndarray:
        """Flat observation vector. Layout is fixed; append only."""
        return np.concatenate([
            self.pos, self.quat, self.linvel, self.angvel,
            self.steer, self.wheel_spin,
            self.wheel_contact.astype(np.float64),
            np.array([float(self.deck_contact)]),
        ])


VECTOR_LAYOUT = (
    ("pos", 3), ("quat", 4), ("linvel", 3), ("angvel", 3),
    ("steer", 2), ("wheel_spin", 4), ("wheel_contact", 4), ("deck_contact", 1),
)
VECTOR_DIM = sum(n for _, n in VECTOR_LAYOUT)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_delta(q_prev: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    """Rotation taking q_prev to q_next, expressed in q_prev's own frame.

    Trick naming integrates these per step rather than differencing Euler
    angles, which wrap and would turn a 360 flip into a 0.
    """
    d = quat_mul(quat_conj(q_prev), q_next)
    return d if d[0] >= 0 else -d  # keep the short way round
