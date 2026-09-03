"""Run `bench/throughput.py` on an NVIDIA GPU, via Modal.

This exists because the decisive measurement cannot be taken on the M2. Apple
silicon runs MJX through the CPU backend, where `vmap` is B times the work done
serially and batching buys exactly nothing (measured: 13.3K episodes/hour, flat
from B=1 to B=64, against the rig's 15K). The claim under test -- that thousands
of environments step at once -- is a claim about GPU parallelism specifically.

Mirrors the rig's proven Modal usage (`scripts/cloud/*_modal.py`, `gpu="A10G"`).
Physics only; rendering needs MJX-Warp and is measured separately.

    modal run bench/throughput_modal.py --batches 1,1024,4096,16384
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("open-skate-throughput")

# jax[cuda12] pulls the CUDA runtime as wheels, so the base image stays slim.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx", "numpy")
    .env({"PYTHONPATH": "/root/src", "MUJOCO_GL": "disable"})
    .add_local_dir(str(_ROOT / "opensk"), remote_path="/root/src/opensk")
    .add_local_file(str(_ROOT / "bench" / "throughput.py"),
                    remote_path="/root/src/bench_throughput.py")
)


@app.function(image=image, gpu="A10G", timeout=3600, memory=32768)
def run_sweep(batches: list[int], steps: int | None = None) -> dict:
    import importlib

    import jax

    bench = importlib.import_module("bench_throughput")
    info = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}
    print(info, flush=True)
    results = []
    for b in batches:
        try:
            r = bench.benchmark(b, n_steps=steps)
        except Exception as exc:                       # OOM at the top end is a result
            print(f"batch {b}: {type(exc).__name__}: {exc}", flush=True)
            results.append({"batch": b, "error": f"{type(exc).__name__}: {exc}"})
            continue
        d = r.__dict__ | {"x_rig": r.episodes_per_hour / bench.RIG_EPISODES_PER_HOUR}
        print(f"batch {b:>6}  compile {r.compile_s:7.1f}s  run {r.run_s:8.2f}s  "
              f"{r.episodes_per_hour:12,.0f} ep/h  {d['x_rig']:6.1f}x rig", flush=True)
        results.append(d)
    return {"info": info, "results": results}


@app.local_entrypoint()
def main(batches: str = "1,1024,4096,16384", steps: int = 0,
         out: str = "results/throughput_gpu.json"):
    bs = [int(x) for x in batches.split(",")]
    payload = run_sweep.remote(bs, steps or None)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
