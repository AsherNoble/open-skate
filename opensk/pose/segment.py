"""Segment the board out of a real True Skate frame.

Analysis-by-synthesis needs a target to compare a render against, and the
cheapest reliable target in these frames is the board's silhouette: the deck is
a dark, strongly elongated object on a bright, low-texture floor, held near the
centre of the frame by the chase camera.

Silhouettes are used rather than appearance on purpose. Our MuJoCo board will
never look like True Skate's textured deck, so any pixel-level comparison would
be dominated by a texture gap we cannot close. A mask has no texture to differ.

Frames carry game UI at the top and bottom which must be excluded, or the dark
UI bar wins the "largest dark blob" contest outright.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import cv2
import numpy as np

# Fractions of frame height to exclude. The top cut clears not just the UI bar
# (rewind/record buttons) but the dark skyline above the horizon, which is a
# far bigger dark region than the board and wins outright if left in. The
# bottom cut clears the score dial and home indicator.
UI_TOP = 0.30
UI_BOTTOM = 0.80
# The chase camera holds the board in a narrow central band, so the search is
# restricted to it. This matters more than it looks: in the darker parks the
# board is not among the darkest pixels of the whole frame, and a threshold
# over the whole play area locks onto shadow near the horizon instead.
ROI_X0, ROI_X1 = 0.15, 0.95


@dataclass(frozen=True)
class BoardMask:
    mask: np.ndarray        # bool, full frame size
    centroid: np.ndarray    # (2,) pixels, (x, y)
    angle_deg: float        # long-axis orientation, 0 = up the screen
    length_px: float        # long-axis extent
    width_px: float         # short-axis extent
    area_px: float
    confidence: float       # [0, 1] elongation x centrality x plausible size

    @property
    def elongation(self) -> float:
        return self.length_px / max(self.width_px, 1e-6)


def board_mask(frame_bgr: np.ndarray, *, dark_percentile: float = 12.0
               ) -> BoardMask | None:
    """Largest plausible dark elongated blob in the play area.

    The threshold is a percentile of the play area's own brightness rather than
    a fixed value, so it survives the large lighting differences between parks
    (the glasshouse and Munich captures are far brighter than SLS Newark).
    """
    h, w = frame_bgr.shape[:2]
    y0, y1 = int(UI_TOP * h), int(UI_BOTTOM * h)
    x0, x1 = int(ROI_X0 * w), int(ROI_X1 * w)
    roi = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    thresh = np.percentile(roi, dark_percentile)
    dark = (roi < thresh).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        return None

    roi_h, roi_w = roi.shape
    best, best_score = None, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        # The deck is a substantial object; anything tiny is texture or shadow.
        if area < 0.0015 * roi_h * roi_w:
            continue
        bx, by = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # A blob touching the top of the play area is the skyline bleeding
        # through, not the board; one spanning most of the width is scenery.
        if by <= 1 or bw > 0.60 * roi_w or bh > 0.75 * roi_h:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        length, width = max(rw, rh), min(rw, rh)
        if width < 1e-6:
            continue
        elong = length / width
        if elong < 1.4:
            continue
        centrality = 1.0 - min(1.0, abs(cx - roi_w / 2) / (roi_w / 2))
        # Score by how board-shaped and board-placed the blob is. The earlier
        # version keyed on an arbitrary target area fraction, which scored the
        # typical board at 0.03 and so inverted any threshold applied to it.
        elong_ok = math.exp(-((elong - 2.6) / 1.1) ** 2)
        size_ok = math.exp(-((math.log(max(area, 1.0) / (0.021 * roi_h * roi_w))) / 0.9) ** 2)
        score = elong_ok * size_ok * (0.35 + 0.65 * centrality)
        if score > best_score:
            # minAreaRect's angle refers to the wider side; normalise so the
            # angle always describes the LONG axis, measured from screen-up.
            long_ang = ang if rw >= rh else ang + 90.0
            long_ang = (long_ang + 90.0) % 180.0 - 90.0
            full = np.zeros((h, w), dtype=bool)
            full[y0:y1, x0:x1] = labels == i
            best = BoardMask(full, np.array([cx + x0, cy + y0]), float(long_ang),
                             float(length), float(width), float(area),
                             float(min(score, 1.0)))
            best_score = score
    return best


# --- shared silhouette features -------------------------------------------
# Used on BOTH real frames and rendered masks, so camera calibration compares
# like with like. Every feature is normalised by frame height, so it is
# invariant to the 828x1792 capture vs whatever size we render at.

FEATURE_NAMES = ("length", "width", "cy", "taper")


def mask_features(mask: np.ndarray) -> np.ndarray | None:
    """(length, width, cy, taper), all normalised. None if the mask is empty.

    `taper` is the ratio of the silhouette's width across its far third to its
    width across its near third. It is the perspective cue that separates
    field of view from camera distance: dollying back and zooming in preserves
    the board's size on screen but flattens its taper.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size < 30:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    length, width = max(rw, rh), min(rw, rh)

    y0, y1 = ys.min(), ys.max()
    span = max(y1 - y0, 1)
    third = span / 3.0
    far = mask[y0:int(y0 + third)]
    near = mask[int(y1 - third):y1 + 1]
    far_w = far.sum(axis=1).mean() if far.size else 0.0
    near_w = near.sum(axis=1).mean() if near.size else 0.0
    taper = far_w / near_w if near_w > 1e-6 else 0.0

    return np.array([length / h, width / h, ys.mean() / h, taper])


def plausible(m: "BoardMask", frame_h: int) -> bool:
    """Geometric sanity gate on a segmented board.

    Kept separate from `confidence` so the numeric score can be tuned without
    silently changing which frames are admitted to a calibration.
    """
    if m is None:
        return False
    return (0.09 <= m.length_px / frame_h <= 0.45
            and 1.8 <= m.elongation <= 5.0
            and 0.25 <= m.centroid[0] / (frame_h * 0.4621) <= 0.80)
