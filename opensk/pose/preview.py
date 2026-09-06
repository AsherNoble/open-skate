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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/visual_direction")
    parser.add_argument("--mode", choices=RENDER_MODES, default=PREVIEW_MODE)
    parser.add_argument("--prefix", default="after")
    parser.add_argument("--park", choices=sorted(PARKS), default="plaza")
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    args = parser.parse_args(argv)
    for path in render_presets(args.output_dir, mode=args.mode,
                               prefix=args.prefix, park=args.park,
                               position=(args.x, args.y)):
        print(path)


if __name__ == "__main__":  # pragma: no cover - exercised as a local tool
    main()
