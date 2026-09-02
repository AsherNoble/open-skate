"""The game camera — part of the physics model, not the renderer.

A gesture in True Skate is a path in *screen* space. Where that path lands on
the board depends entirely on where the camera is, so the camera sits inside
the fitted parameter vector alongside friction and bushing stiffness. Get the
camera wrong and every fitted force is wrong in a way no amount of tuning the
board can fix.

Screen coordinates follow GESTURES.md exactly: normalised [0, 1], origin at
the top-left, y increasing downward.
"""
from __future__ import annotations

import math

import numpy as np

from .params import SkateParams


class FollowCamera:
    """Chase camera: behind and above the board, yawing to follow it.

    The follow is first-order lagged (`cam_follow_tau`). That lag is not
    cosmetic — during a trick the board rotates far faster than the camera
    does, so the lag changes where a mid-gesture screen point projects, and
    therefore changes the forces. It is a fitted parameter.
    """

    def __init__(self, params: SkateParams):
        self.p = params
        self._target = np.zeros(3)
        self._yaw = 0.0
        self._init = False

    def _aim(self, board_pos: np.ndarray, yaw: float) -> np.ndarray:
        """Where the camera points: ahead of the board, not at it."""
        lead = self.p.cam_lead_m
        return np.asarray(board_pos, dtype=np.float64) + np.array(
            [lead * math.cos(yaw), lead * math.sin(yaw), 0.0])

    def reset(self, board_pos: np.ndarray, board_yaw: float) -> None:
        self._yaw = float(board_yaw)
        self._target = self._aim(board_pos, self._yaw)
        self._init = True

    def update(self, board_pos: np.ndarray, board_yaw: float, dt: float) -> None:
        if not self._init:
            self.reset(board_pos, board_yaw)
            return
        tau = self.p.cam_follow_tau
        a = 1.0 if tau <= 1e-6 else 1.0 - math.exp(-dt / tau)
        self._target += a * (self._aim(board_pos, board_yaw) - self._target)
        # Shortest-arc yaw blend, so passing through +/-pi doesn't whipround.
        self._yaw += a * math.atan2(math.sin(board_yaw - self._yaw),
                                    math.cos(board_yaw - self._yaw))

    # -- frame -------------------------------------------------------------

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, forward) unit vectors of the camera, in world coords."""
        pitch = math.radians(self.p.cam_pitch_deg)
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        forward = np.array([cy * cp, sy * cp, sp])
        right = np.array([-sy, cy, 0.0])
        # right x up = forward, so up = forward x right. Getting this
        # backwards flips the screen vertically while leaving projection
        # round-trips self-consistent, so it survives the obvious test.
        up = np.cross(forward, right)
        return right, up, forward / np.linalg.norm(forward)

    @property
    def position(self) -> np.ndarray:
        """Sits `cam_distance` back along the view ray from the board.

        Aiming at the target is what makes the camera identifiable: with an
        independent height the fit could trade height against pitch without
        changing the image.
        """
        _, _, forward = self.basis()
        return self._target - self.p.cam_distance * forward

    @property
    def elevation_deg(self) -> float:
        return self.p.cam_pitch_deg

    @property
    def azimuth_deg(self) -> float:
        return math.degrees(self._yaw)

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    # -- projection --------------------------------------------------------

    def ray(self, nx: float, ny: float) -> tuple[np.ndarray, np.ndarray]:
        """Normalised screen point -> (origin, unit direction) world ray."""
        right, up, forward = self.basis()
        tan_v = math.tan(0.5 * math.radians(self.p.cam_fov_deg))
        tan_h = tan_v * self.p.screen_aspect
        # y is measured downward from the top of the screen, hence the flip.
        sx = (nx - 0.5) * 2.0 * tan_h
        sy = (0.5 - ny) * 2.0 * tan_v
        d = forward + sx * right + sy * up
        return self.position, d / np.linalg.norm(d)

    def point_at_depth(self, nx: float, ny: float, depth: float) -> np.ndarray:
        """Where a screen point sits on the view-parallel plane at `depth`.

        This is what makes dragging feel like dragging: the fingertip stays at
        the depth of whatever it grabbed, so lateral screen motion becomes
        lateral world motion rather than motion toward or away from the camera.
        """
        _, _, forward = self.basis()
        o, d = self.ray(nx, ny)
        denom = float(d @ forward)
        if abs(denom) < 1e-9:
            return o + d * depth
        return o + d * (depth / denom)

    def depth_of(self, world_point: np.ndarray) -> float:
        _, _, forward = self.basis()
        return float((np.asarray(world_point) - self.position) @ forward)

    def project(self, world_point: np.ndarray) -> tuple[float, float] | None:
        """World point -> normalised screen point, or None if behind camera."""
        right, up, forward = self.basis()
        rel = np.asarray(world_point, dtype=np.float64) - self.position
        z = float(rel @ forward)
        if z <= 1e-6:
            return None
        tan_v = math.tan(0.5 * math.radians(self.p.cam_fov_deg))
        tan_h = tan_v * self.p.screen_aspect
        nx = 0.5 + (float(rel @ right) / z) / (2.0 * tan_h)
        ny = 0.5 - (float(rel @ up) / z) / (2.0 * tan_v)
        return nx, ny


def board_yaw(quat: np.ndarray) -> float:
    """Heading of the deck's long axis, from a wxyz quaternion."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
