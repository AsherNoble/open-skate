"""Collect a real batch of pixel rollouts through the ENVIRONMENT, and store it.

Everything so far exercised the pieces from a benchmark script. This drives
`GestureEnv(pixels=True)` exactly as a training run would, writes a shard with
`rl.store`, reads it back, and checks the frames survive the trip -- which is
the last link between "the renderer works" and "a world model can be trained
on this".

    modal run bench/collect_modal.py --batch 64
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
app = modal.App("open-skate-collect")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libglx-mesa0", "libosmesa6", "git")
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx[warp]", "numpy")
    .pip_install("git+https://github.com/google-deepmind/mujoco_warp.git")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(_ROOT / "opensk"), remote_path="/root/src/opensk")
)


@app.function(image=image, gpu="A10G", timeout=3600, memory=32768)
def collect(batch: int) -> dict:
    import time

    import numpy as np

    from opensk.rl import store
    from opensk.rl.env import GestureEnv

    env = GestureEnv(pixels=True, batch=batch)
    actions = env.sample_actions(batch, seed=11)

    t = time.perf_counter()
    ep = env.step(actions)
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    ep = env.step(actions)
    run_s = time.perf_counter() - t

    path = store.save("/tmp/shard_00000.npz", actions, ep)
    back = store.load(path)
    kept = back.filtered()

    # A second, independent batch, so the world model is judged on episodes it
    # has never seen. Frames within one episode are nearly identical, so a
    # split across frames rather than episodes would leak the answer.
    held_actions = env.sample_actions(batch, seed=99)
    held = env.step(held_actions)
    held_path = store.save("/tmp/shard_00001.npz", held_actions, held)

    grey = np.asarray(ep.rgb).mean(axis=-1)
    bg = np.median(grey, axis=(2, 3), keepdims=True)
    visible = (np.abs(grey - bg) > 0.05).mean(axis=(2, 3)) > 0.005

    res = {
        "batch": batch, "compile_s": compile_s, "run_s": run_s,
        "episodes_per_hour": batch / run_s * 3600.0,
        "rgb_shape": list(np.asarray(ep.rgb).shape),
        "shard_bytes": Path(path).stat().st_size,
        "bytes_per_episode": Path(path).stat().st_size / batch,
        "round_trip_max_error": float(
            np.abs(back.frames_float() - np.asarray(ep.rgb)).max()),
        # nan-aware: an unstable episode carries NaN poses, and NaN != NaN
        # would report a storage failure where there is none.
        "pose_round_trip_max_error": float(
            np.nanmax(np.abs(np.asarray(back.pos) - np.asarray(ep.pos)))),
        "pose_nan_mismatch": int(np.sum(
            np.isnan(np.asarray(back.pos)) != np.isnan(np.asarray(ep.pos)))),
        "physical_episodes": int(np.asarray(ep.valid).sum()),
        "board_visible_all_frames": float(visible.all(axis=1).mean()),
        # The number that separates "the pixel path is broken" from "these
        # actions throw the board off the map": among episodes that stayed
        # physical at all, is the board framed?
        "board_visible_among_physical": float(
            visible.all(axis=1)[np.asarray(ep.valid)].mean()
            if np.asarray(ep.valid).any() else -1.0),
        "median_displacement_m": float(
            np.nanmedian(np.asarray(ep.displacement))),
        "median_displacement_physical_m": float(
            np.median(np.asarray(ep.displacement)[np.asarray(ep.valid)])
            if np.asarray(ep.valid).any() else -1.0),
        "kept_after_filter": len(kept),
    }
    from opensk.rl.worldmodel import evaluate

    scores = evaluate(back, store.load(held_path))
    res["worldmodel"] = {
        "model_mse": scores.model, "persistence_mse": scores.persistence,
        "mean_frame_mse": scores.mean_frame,
        "beats_persistence": scores.beats_persistence}
    print("world model:", scores, flush=True)
    print(json.dumps(res, indent=2), flush=True)

    # Frames and trajectory for a typical PHYSICAL episode, so "why is the
    # board not in frame" is answered by looking rather than by inference.
    import io

    idx = np.where(np.asarray(ep.valid))[0]
    pick = int(idx[np.argsort(np.asarray(ep.displacement)[idx])[len(idx) // 2]])
    rgb = np.asarray(ep.rgb)
    buf = io.BytesIO()
    np.savez_compressed(
        buf, frames=rgb[pick, ::max(1, rgb.shape[1] // 6)][:6],
        pos=np.asarray(ep.pos)[pick], quat=np.asarray(ep.quat)[pick],
        displacement=np.asarray(ep.displacement)[pick],
        peak=np.asarray(ep.peak_height)[pick])
    res["sample_npz"] = buf.getvalue()
    return res


@app.local_entrypoint()
def main(batch: int = 64, out: str = "results/collect_gpu.json"):
    res = collect.remote(batch)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    npz = res.pop("sample_npz", None)
    Path(out).write_text(json.dumps(res, indent=2))
    if npz:
        Path(out).with_suffix(".npz").write_bytes(npz)
    print(f"wrote {out}")
