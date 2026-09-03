"""Render a real batch of episodes end to end, on GPU, and report honestly.

Everything up to here measured pieces: physics throughput on one path,
renderer throughput on a static scene. This runs the actual thing -- the
batch-major rollout with the fitted chase camera driving the model camera,
rendering every frame of a real gesture batch -- and reports the cost of a live
rollout rather than a still.

That difference matters: `results/THROUGHPUT.md`'s render figure excludes
whatever BVH refit a moving board costs per step, and the caveat is recorded
there precisely so this run can settle it.

    modal run bench/pixels_modal.py --batch 256
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("open-skate-pixels")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libglx-mesa0", "libosmesa6", "git")
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx[warp]", "numpy")
    .pip_install("git+https://github.com/google-deepmind/mujoco_warp.git")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(_ROOT / "opensk"), remote_path="/root/src/opensk")
)

# Reporting only. The size that actually applies is the camera's `resolution`
# in the MJCF, from SkateParams.render_height/render_width; these must match.
RENDER_H, RENDER_W = 128, 64


@app.function(image=image, gpu="A10G", timeout=3600, memory=32768)
def run(batch: int, save_frames: int = 4) -> dict:
    import time

    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx

    from opensk.mjx.outcomes import random_recipes
    from opensk.mjx.rollout import gesture_arrays
    from opensk.mjx.rollout_batched import rollout_batched
    from opensk.sim.core import SkateSim
    from opensk.sim.model.build import FLAT_PARK, build_scene
    from opensk.sim.params import SkateParams

    p = SkateParams()
    xml = build_scene(p, FLAT_PARK)
    mjm = mujoco.MjModel.from_xml_string(xml)
    # NOTE: offheight/offwidth do NOT size the batch renderer -- it reads the
    # CAMERA's resolution, set in the MJCF. Setting them here rendered 1x1
    # images and produced throughput numbers that meant nothing.
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "chase")
    cpu = SkateSim(p, FLAT_PARK)
    cpu.reset(seed=0)

    mx = mjx.put_model(mjm, impl="warp")
    # The Warp backend preallocates contact buffers and DROPS contacts past
    # them, printing "broadphase overflow" rather than failing. Left at the
    # default it silently simulates a board that is partly not touching the
    # ground. The counts are totals across all worlds, so they scale with the
    # batch; these are the observed requirement (~9 per world) with headroom.
    d0 = jax.vmap(lambda _: mjx.make_data(
        mjm, impl="warp", naconmax=64 * batch,
        njmax=64 * batch))(jnp.arange(batch))
    d0 = d0.replace(qpos=jnp.broadcast_to(jnp.array(cpu.data.qpos),
                                          (batch, cpu.model.nq)))

    # Settle the whole batch together, in the batch-major shape it will run in.
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    for _ in range(200):
        d0 = step(mx, d0)

    # The context owns Warp buffers and deregisters them when collected, so it
    # must outlive every use of its pytree handle.
    rc = mjx.create_render_context(mjm, nworld=batch)
    ctx = rc.pytree()

    def render(d):
        rgb, _, d = mjx.render(mx, d, ctx)
        return mjx.get_rgb(ctx, cam_id, rgb), d

    recipes = random_recipes(batch, seed=5)
    arrays = [gesture_arrays(r, 2) for r in recipes]
    P = jnp.asarray(np.stack([a[0] for a in arrays]))
    S = jnp.asarray(np.stack([a[1] for a in arrays]))
    T = jnp.asarray(np.stack([a[2] for a in arrays]))

    def go(P, S, T, d):
        return rollout_batched(mx, cpu.model, p, cpu.deck_bid,
                               sorted(cpu._deck_gids), d, P, S, T,
                               render=render, cam_id=cam_id)

    jit = jax.jit(go)
    t = time.perf_counter()
    out = jit(P, S, T, d0)
    jax.block_until_ready(out.pos)
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    out = jit(P, S, T, d0)
    jax.block_until_ready(out.pos)
    run_s = time.perf_counter() - t

    rgb = np.asarray(out.rgb)
    frames = rgb.shape[0]
    # Per-episode visibility, not one number over everything. A single
    # aggregate cannot tell "the renderer is broken" from "the board flew out
    # of frame", and those need completely different fixes.
    visible = (rgb.std(axis=-1) > 0.02).any(axis=(2, 3))     # (frames, batch)
    disp = np.linalg.norm(np.asarray(out.pos)[-1, :, :2]
                          - np.asarray(out.pos)[0, :, :2], axis=-1)
    res = {"batch": batch, "compile_s": compile_s, "run_s": run_s,
           "episodes_per_hour": batch / run_s * 3600.0,
           "frames": frames, "rgb_shape": list(rgb.shape),
           "mean_pixel": float(rgb.mean()),
           "board_visible_first_frame": float(visible[0].mean()),
           "board_visible_last_frame": float(visible[-1].mean()),
           "board_visible_all_frames": float(visible.all(axis=0).mean()),
           "median_displacement_m": float(np.median(disp)),
           "visible_and_settled": float(
               visible.all(axis=0)[disp < 2.0].mean() if (disp < 2.0).any() else -1.0),
           "frac_settled": float((disp < 2.0).mean())}
    print(json.dumps(res, indent=2), flush=True)
    # A few frames come back so the render can actually be LOOKED at. Two
    # "findings" in this project were bugs that statistics hid.
    # Returned as a compressed buffer, not JSON lists: a modal return value of
    # a few hundred thousand floats is what terminated the runner last time.
    import io

    # Frames from the CALMEST episode: a board that flew 20 m leaves an empty
    # frame, which looks identical to a broken renderer.
    calm = int(np.argmin(disp))
    buf = io.BytesIO()
    np.savez_compressed(
        buf, frames=rgb[::max(1, frames // save_frames)][:save_frames, calm],
        calm_displacement=disp[calm])
    res["sample_npz"] = buf.getvalue()
    return res


@app.local_entrypoint()
def main(batch: int = 256, out: str = "results/pixels_gpu.json"):
    res = run.remote(batch)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    npz = res.pop("sample_npz", None)
    Path(out).write_text(json.dumps(res, indent=2))
    if npz:
        Path(out).with_suffix(".npz").write_bytes(npz)
    print(f"wrote {out}")
