"""Render Open Skate through the same camera model the touch system uses.

Analysis-by-synthesis compares a render against a real frame, so the render
must come from `FollowCamera` — not from some separate viewing camera that
happens to look similar. If the two ever diverge, fitted camera parameters
stop meaning anything for where a touch lands, which is the only reason the
camera is in the parameter vector at all.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..sim.camera import FollowCamera, board_yaw
from ..sim.core import SkateSim

TRAINING_MODE = "training"
PREVIEW_MODE = "preview"
LOCAL_PREVIEW_HEIGHT = 812
RENDER_MODES = (TRAINING_MODE, PREVIEW_MODE)


class SceneRenderer:
    """Offscreen renderer driving a MuJoCo free camera from `FollowCamera`.

    MuJoCo's `mjvCamera` always looks at its lookat point, which is exactly the
    convention `FollowCamera` was reparameterised to, so the two agree by
    construction rather than by coincidence.
    """

    def __init__(self, sim: SkateSim, height: int = 448, width: int | None = None):
        if width is None:
            width = int(round(height * sim.params.screen_aspect))
        self.sim, self.height, self.width = sim, height, width
        self._renderer = mujoco.Renderer(sim.model, height=height, width=width)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._mask_option = mujoco.MjvOption()
        self._mask_option.geomgroup[:] = 0
        self._mask_option.geomgroup[1] = 1
        # MuJoCo's fovy is a model-level visual setting, not a camera field.
        sim.model.vis.global_.fovy = sim.params.cam_fov_deg

    @classmethod
    def for_mode(cls, sim: SkateSim, mode: str = TRAINING_MODE) -> "SceneRenderer":
        """Construct one of the two supported output modes.

        Training remains exactly ``params.render_width x params.render_height``
        (64x128 by default).  Preview scales the same fitted projection and
        portrait aspect to device-like local inspection resolution.  It does
        not alter FOV, lead, distance, pitch or any other camera geometry.
        """
        if mode == TRAINING_MODE:
            return cls(sim, height=sim.params.render_height,
                       width=sim.params.render_width)
        if mode == PREVIEW_MODE:
            return cls(sim, height=LOCAL_PREVIEW_HEIGHT)
        raise ValueError(f"unknown render mode {mode!r}; choose one of: "
                         f"{', '.join(RENDER_MODES)}")

    def render(self, camera: FollowCamera | None = None, *,
               scene_option: mujoco.MjvOption | None = None) -> np.ndarray:
        cam = camera
        if cam is None:
            cam = FollowCamera(self.sim.params)
            st = self.sim.state()
            cam.reset(st.pos, board_yaw(st.quat))
        self.sim.model.vis.global_.fovy = self.sim.params.cam_fov_deg
        self._cam.lookat[:] = cam.target
        self._cam.distance = self.sim.params.cam_distance
        self._cam.elevation = cam.elevation_deg
        self._cam.azimuth = cam.azimuth_deg
        self._renderer.update_scene(self.sim.data, self._cam,
                                    scene_option=scene_option)
        return self._renderer.render()

    def board_pixels(self, camera: FollowCamera | None = None) -> np.ndarray:
        """Boolean mask of the board, by rendering with segmentation ids.

        Exact, unlike thresholding a colour render: it needs no assumption
        about how the board's brightness compares to the ground, which is the
        very assumption that makes segmenting the REAL frames delicate.
        """
        self._renderer.enable_segmentation_rendering()
        try:
            self.render(camera, scene_option=self._mask_option)
            seg = self._renderer.render()[:, :, 0]
        finally:
            self._renderer.disable_segmentation_rendering()
        # The VISUAL outline, not the collision boxes: the silhouette this is
        # compared against is a picture of the real deck, which is a popsicle.
        return np.isin(seg, list(self.sim._deck_visual_gids))
