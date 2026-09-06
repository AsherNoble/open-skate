"""Render the neutral appearance presets through the fitted chase camera.

Example::

    python -m opensk.pose.preview --output-dir results/visual_direction

The command is intentionally local and CPU-renderer based.  It does not use
Modal, paid compute, Blender at runtime, or generated imagery.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..sim.appearance import APPEARANCE_PRESETS
from ..sim.core import SkateSim
from ..sim.model.parks import PARKS
from .render import PREVIEW_MODE, RENDER_MODES, SceneRenderer


def render_presets(output_dir: str | Path, *, mode: str = PREVIEW_MODE,
                   prefix: str = "after", park: str = "plaza",
                   position: tuple[float, float] = (0.0, 0.0)) -> list[Path]:
    """Render every preset from one identical physical state."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in APPEARANCE_PRESETS:
        sim = SkateSim(park=PARKS[park], appearance=name)
        sim.reset(seed=0, pos=position)
        sim.step(400)
        rgb = SceneRenderer.for_mode(sim, mode).render()
        path = target / f"{prefix}_{name}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"could not write {path}")
        paths.append(path)
    return paths


def render_matrix(output_dir: str | Path, *, mode: str = PREVIEW_MODE,
                  prefix: str = "sample",
                  parks: tuple[str, ...] = tuple(PARKS)) -> list[Path]:
    """Render the complete park/appearance domain matrix in stable order."""
    paths = []
    for park in parks:
        paths.extend(render_presets(output_dir, mode=mode,
                                    prefix=f"{prefix}_{park}", park=park))
    return paths


def write_contact_sheet(paths: list[Path], output: str | Path, *,
                        columns: int = 3) -> Path:
    """Arrange already-rendered samples without rescaling their pixels."""
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if not images or any(image is None for image in images):
        raise OSError("could not read every contact-sheet input")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("contact-sheet inputs must have identical dimensions")
    rows = []
    for start in range(0, len(images), columns):
        row = images[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(images[0]))
        rows.append(cv2.hconcat(row))
    sheet = cv2.vconcat(rows)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), sheet):
        raise OSError(f"could not write {target}")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/visual_direction")
    parser.add_argument("--mode", choices=RENDER_MODES, default=PREVIEW_MODE)
    parser.add_argument("--prefix", default="after")
    parser.add_argument("--park", choices=sorted(PARKS), default="plaza")
    parser.add_argument("--all-parks", action="store_true",
                        help="render the complete 3x3 park/appearance matrix")
    parser.add_argument("--contact-sheet", default=None,
                        help="optional path for an unscaled contact sheet")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.all_parks:
        paths = render_matrix(args.output_dir, mode=args.mode,
                              prefix=args.prefix)
    else:
        paths = render_presets(args.output_dir, mode=args.mode,
                               prefix=args.prefix, park=args.park,
                               position=(args.x, args.y))
    for path in paths:
        print(path)
    if args.contact_sheet:
        print(write_contact_sheet(paths, args.contact_sheet))


if __name__ == "__main__":  # pragma: no cover - exercised as a local tool
    main()
