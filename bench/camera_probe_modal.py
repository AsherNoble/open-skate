"""Does a mocap write actually move the rendered camera? Read it back and see.

The host-side check is unambiguous: the fitted chase camera holds the board at
screen centre (sx ~ 0.00, depth 2.2 m) for all 68 frames of an episode where
the board rolls 1.77 m along the ground. The render shows the board in frame 0
and then loses it. So the camera the RENDERER uses is not the camera the touch
model uses, and the question is only where the write is lost.

This probes the one link that cannot be checked without a GPU: set mocap, step,
and read `cam_xpos` back.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent
app = modal.App("open-skate-camera-probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libglx-mesa0", "libosmesa6", "git")
    .pip_install("jax[cuda12]", "mujoco", "mujoco-mjx[warp]", "numpy")
    .pip_install("git+https://github.com/google-deepmind/mujoco_warp.git")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(_ROOT / "opensk"), remote_path="/root/src/opensk")
)


@app.function(image=image, gpu="A10G", timeout=1800, memory=16384)
def probe() -> dict:
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    from mujoco import mjx

    from opensk.sim.model.build import FLAT_PARK, build_scene
    from opensk.sim.params import SkateParams

    p = SkateParams()
    mjm = mujoco.MjModel.from_xml_string(build_scene(p, FLAT_PARK))
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "chase")
    out = {"ncam": int(mjm.ncam), "nmocap": int(mjm.nmocap), "cam_id": cam_id,
           "cam_bodyid": int(mjm.cam_bodyid[cam_id]),
           "cam_mocapid_of_body": int(mjm.body_mocapid[mjm.cam_bodyid[cam_id]])}

    target = np.array([1.5, -0.7, 0.9])
    for impl in ("jax", "warp"):
        try:
            mx = mjx.put_model(mjm, impl=impl)
            d = (mjx.make_data(mjm, impl="warp", naconmax=256, njmax=256)
                 if impl == "warp" else mjx.make_data(mx))
            d = jax.tree.map(lambda x: jnp.broadcast_to(x, (2,) + x.shape)
                             if hasattr(x, "shape") else x, d)
            d = d.replace(mocap_pos=d.mocap_pos.at[:, 0].set(jnp.asarray(target)))
            before = np.asarray(d.cam_xpos)[0, cam_id].tolist()
            d = jax.vmap(mjx.step, in_axes=(None, 0))(mx, d)
            after = np.asarray(d.cam_xpos)[0, cam_id].tolist()
            out[impl] = {"mocap_set_to": target.tolist(),
                         "cam_xpos_before_step": before,
                         "cam_xpos_after_step": after,
                         "follows_mocap": bool(
                             np.allclose(after, target, atol=1e-4))}
        except Exception as exc:
            out[impl] = {"error": f"{type(exc).__name__}: {exc}"}
        print(impl, json.dumps(out[impl]), flush=True)
    print("MODEL", json.dumps({k: v for k, v in out.items()
                               if k not in ("jax", "warp")}), flush=True)
    return out


@app.local_entrypoint()
def main(out: str = "results/camera_probe.json"):
    res = probe.remote()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
