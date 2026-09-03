"""Find out what the MJX-Warp batch renderer actually offers, on real hardware.

Pixels are the observation the plan chose, and the renderer is the risk that
could sink that choice: at B=1024 an episode batch is ~70K frames, and if
rendering costs more than the physics it erases the 64x margin measured in
`results/THROUGHPUT.md`.

MJX-Warp needs NVIDIA, so none of this can be checked on the M2 -- including
the API surface. This probe runs first and reports what exists, rather than
guessing at an interface and writing a benchmark against it.

    modal run bench/render_probe_modal.py
"""
from __future__ import annotations

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("open-skate-render-probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libglx-mesa0", "libosmesa6")
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx[warp]", "numpy")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(_ROOT / "opensk"), remote_path="/root/src/opensk")
)


@app.function(image=image, gpu="A10G", timeout=1800)
def probe() -> dict:
    import importlib

    out = {}
    import inspect

    # The batch renderer lives in mujoco.mjx itself, not in a separate
    # mujoco_warp package -- `mujoco-mjx[warp]` installs only warp-lang, and
    # `import mujoco_warp` fails. render_with_segmentation is the interesting
    # one: masks are the cheap way past the appearance gap.
    from mujoco import mjx
    for name in ("create_render_context", "render", "render_with_segmentation",
                 "get_rgb"):
        fn = getattr(mjx, name)
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        print(f"--- mjx.{name}{inspect.signature(fn)}", flush=True)
        for line in doc[:24]:
            print("    " + line, flush=True)
        out[name] = str(inspect.signature(fn))
    return out


@app.local_entrypoint()
def main():
    probe.remote()
