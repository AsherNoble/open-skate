"""Check that the MJX backend reproduces the CPU reference.

WHAT PARITY CAN AND CANNOT MEAN HERE. Measured on this model, MJX and MuJoCo C
agree to MACHINE PRECISION through smooth dynamics -- 4.8e-18 in position and
6.7e-16 in orientation at 50 steps -- and then diverge sharply once the deck
first touches the ground:

    step   |pos diff|   |quat diff|      note
      25     4.8e-18       6.7e-16       smooth flight
      50     1.0e-17       8.9e-16       smooth flight
     100     6.6e-03       6.4e-02       after first deck contact (step 47)
     250     3.0e-01       4.8e-01       chaotic amplification

The two implementations resolve simultaneous contacts differently, and a
flipping board is chaotic, so trajectory-level agreement past the first impact
is not achievable and is the wrong thing to demand.

So parity is tested in two parts:
  * `precontact_divergence` -- must be at machine precision. This is what
    actually proves the port: same model, same forces, same integration.
  * outcome statistics over many gestures -- what has to hold for the fitted
    parameters to mean the same thing on GPU. Trajectories may differ; the
    distribution of what the board DOES must not.

Note MJX runs in float32 unless `JAX_ENABLE_X64=1`. Training in float32 is
normal; parity checks need x64 or the comparison is dominated by precision.
"""
from __future__ import annotations

import numpy as np

from ..sim.core import SkateSim
from ..sim.params import SkateParams


def make_mjx(params: SkateParams | None = None, park: str | None = None):
    """(mjx_model, mjx_data, cpu_sim) all built from the same MJCF."""
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    from ..sim.model.build import FLAT_PARK, build_scene

    params = params or SkateParams()
    cpu = SkateSim(params, park or FLAT_PARK)
    cpu.reset(seed=0)
    model = mujoco.MjModel.from_xml_string(build_scene(params, park or FLAT_PARK))
    mx = mjx.put_model(model)
    d = mjx.make_data(mx).replace(qpos=jnp.array(cpu.data.qpos),
                                  qvel=jnp.array(cpu.data.qvel))
    return mx, d, cpu


def precontact_divergence(steps: int = 40, params: SkateParams | None = None
                          ) -> tuple[float, float]:
    """(max |pos diff|, max |quat diff|) over `steps` of free flight.

    Deliberately stops before the deck reaches the ground -- contact is where
    the two implementations legitimately part company. Kept short for the same
    reason: this measures the PORT, not the contact solver.
    """
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    params = params or SkateParams()
    mx, d, cpu = make_mjx(params)
    step = jax.jit(lambda dd: mjx.step(mx, dd))
    for _ in range(steps):
        mujoco.mj_step(cpu.model, cpu.data)
        d = step(d)
    cq = np.asarray(cpu.data.qpos)
    mq = np.asarray(d.qpos)
    return (float(np.abs(cq[:3] - mq[:3]).max()),
            float(np.abs(cq[3:7] - mq[3:7]).max()))
