"""Does rendering erase the 64x physics margin? The measurement that decides it.

`results/THROUGHPUT.md` records 959K episodes/hour of physics at B=1024 on an
A10G. An episode is 68 frames, so a 1024-wide batch is ~70K frames per call.
If the renderer cannot produce those in the ~4 s the physics takes, pixels are
not the observation at scale and segmentation masks become the plan rather
than the fallback.

Two constraints found by probing the real API on hardware, neither of them
guessable from the docs:

  * The batch renderer is in `mujoco.mjx` itself -- `create_render_context`,
    `render`, `render_with_segmentation`, `get_rgb`. `mujoco-mjx[warp]`
    installs only `warp-lang`; there is no `mujoco_warp` package to import.
  * **The render context CANNOT be used under `vmap`.** Its own docstring says
    so: nworld is hardcoded because Warp allocates arrays JAX cannot see. The
    physics rollout is env-major (`vmap` over independent episodes), so the
    render path has to be batch-major instead -- one batched `Data`, stepped
    and rendered as a whole. That is a real restructuring, and measuring the
    renderer first says whether it is worth doing.

Rendering only. Physics cost is already known and is not re-measured here.

    modal run bench/render_throughput_modal.py --worlds 64,256,1024
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("open-skate-render-throughput")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libglx-mesa0", "libosmesa6")
    # `mujoco-mjx[warp]` installs warp-lang only, and mjx.render then raises
    # "render only implemented for MuJoCo Warp". The Warp BACKEND is a separate
    # package that is not on PyPI, so it comes from source.
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx[warp]", "numpy")
    .apt_install("git")
    .pip_install("git+https://github.com/google-deepmind/mujoco_warp.git")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(_ROOT / "opensk"), remote_path="/root/src/opensk")
)

# The phone's aspect, 19.5:9, at a size a world model can actually consume.
# Rendering at capture resolution and downsampling would measure the wrong
# thing: what matters is the cost of the frames that get trained on.
# Reporting only. The size that actually applies is the camera's `resolution`
# in the MJCF, from SkateParams.render_height/render_width; these must match.
RENDER_H, RENDER_W = 128, 64


@app.function(image=image, gpu="A10G", timeout=3600, memory=32768)
def bench(worlds: list[int], frames: int = 68, segmentation: bool = False) -> dict:
    import time

    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx

    from opensk.sim.model.build import FLAT_PARK, build_scene
    from opensk.sim.params import SkateParams

    p = SkateParams()
    mjm = mujoco.MjModel.from_xml_string(build_scene(p, FLAT_PARK))
    # NOTE: offheight/offwidth do NOT size the batch renderer -- it reads the
    # CAMERA's resolution, set in the MJCF. Setting them here rendered 1x1
    # images and produced throughput numbers that meant nothing.
    import inspect
    print("put_model", inspect.signature(mjx.put_model), flush=True)
    print("make_data", inspect.signature(mjx.make_data), flush=True)
    try:
        import mujoco_warp
        print("mujoco_warp OK", getattr(mujoco_warp, "__version__", "?"), flush=True)
    except Exception as exc:
        print("mujoco_warp MISSING", exc, flush=True)
    # The Warp backend is what carries the batch renderer; the default JAX
    # implementation raises NotImplementedError from mjx.render.
    mx = None   # built per world count: the Warp model carries its batch size

    results = []
    for n in worlds:
        try:
            # batch_sizes is for batched MODEL fields (per-world geometry),
            # not for nworld -- passing nworld there is rejected outright.
            mx = mjx.put_model(mjm, impl="warp")
            # `render` takes the JAX-compatible handle, but the CONTEXT owns
            # the Warp buffers and deregisters them when it is collected --
            # dropping it leaves render looking up a key that no longer exists
            # (KeyError on the buffer registry). Both have to stay alive.
            rc = mjx.create_render_context(mjm, nworld=n,
                                           **({"render_seg": True}
                                              if segmentation else {}))
            ctx = rc.pytree() if hasattr(rc, "pytree") else rc
            # make_data must be told the Warp implementation too, or it hands
            # back a JAX Data the Warp model refuses. The batch axis comes from
            # vmapping make_data; render itself then consumes the batched Data
            # whole, which is the constraint that forces batch-major rollouts.
            d = jax.vmap(lambda _: mjx.make_data(mjm, impl="warp"))(jnp.arange(n))

            fn = mjx.render_with_segmentation if segmentation else mjx.render

            def once(dd):
                out = fn(mx, dd, ctx)
                return out[0], out[-1]

            jit = jax.jit(once)
            t = time.perf_counter()
            rgb, dd = jit(d)
            jax.block_until_ready(rgb)
            compile_s = time.perf_counter() - t

            t = time.perf_counter()
            for _ in range(10):
                rgb, dd = jit(dd)
            jax.block_until_ready(rgb)
            per_call = (time.perf_counter() - t) / 10.0

            fps = n / per_call
            # The number that decides the question: seconds to render every
            # frame of an n-wide episode batch, against the ~4 s its physics
            # takes at n=1024.
            row = {"worlds": n, "compile_s": compile_s, "s_per_call": per_call,
                   "frames_per_s": fps, "batch_render_s": n * frames / fps,
                   "segmentation": segmentation}
            print(f"worlds {n:>6}  compile {compile_s:6.1f}s  "
                  f"{per_call*1e3:8.2f} ms/call  {fps:12,.0f} frames/s  "
                  f"episode batch {row['batch_render_s']:8.2f}s", flush=True)
            results.append(row)
        except Exception as exc:
            import traceback
            print(f"worlds {n}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            results.append({"worlds": n, "error": f"{type(exc).__name__}: {exc}"})
    return {"resolution": [RENDER_H, RENDER_W], "results": results}


@app.local_entrypoint()
def main(worlds: str = "64,256,1024", segmentation: bool = False,
         out: str = "results/render_gpu.json"):
    payload = bench.remote([int(x) for x in worlds.split(",")],
                           segmentation=segmentation)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
