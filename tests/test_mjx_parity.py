"""The MJX backend must reproduce the CPU reference before anything is built on it.

If the two disagree, every fitted parameter is void on GPU -- so this is the
gate for the whole batched-training effort.

It asserts agreement only BEFORE first contact. Measured, the two agree to
machine precision in smooth dynamics and then diverge at the first deck impact
(step 47 in the reference gesture), because they resolve simultaneous contacts
differently and a flipping board amplifies the difference chaotically. Demanding
trajectory agreement past that point would be demanding something false.
"""
import pytest

mjx = pytest.importorskip("mujoco.mjx", reason="mujoco-mjx not installed")


def test_mjx_matches_cpu_before_contact():
    from opensk.mjx.parity import precontact_divergence

    pos, quat = precontact_divergence(steps=40)
    # Loose enough for float32 (the JAX default); with JAX_ENABLE_X64=1 these
    # come out around 1e-17 and 1e-15.
    assert pos < 1e-4, f"position diverged before contact: {pos:.2e}"
    assert quat < 1e-4, f"orientation diverged before contact: {quat:.2e}"


def test_mjx_accepts_the_model_and_exposes_applied_force():
    """xfrc_applied is the entire touch pathway; without it nothing ports."""
    import dataclasses

    from opensk.mjx.parity import make_mjx

    _, d, _ = make_mjx()
    fields = {f.name for f in dataclasses.fields(d)}
    assert "xfrc_applied" in fields
    assert d.xfrc_applied.shape[1] == 6
