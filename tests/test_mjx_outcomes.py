"""The second half of the parity gate: outcome DISTRIBUTIONS must agree.

Individual trajectories legitimately diverge after the first contact (see
`test_mjx_parity.py`), so the thing that has to hold instead is that a
population of gestures produces the same population of outcomes. Without that,
every parameter fitted on the CPU is void on GPU.

Building this test found four port defects that trajectory parity had not:
a missed ray treated as a ground contact, a quaternion recovered through the
trace form's singularity, a missing camera lag, and force applied on the
touch-down substep. Each showed up here as boards launched metres into the air
where the reference did nothing.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

# 24 gestures, not fewer: at 16 this same seed happens to draw only mild
# flicks, and a population where nothing pops or flips agrees trivially.
N_GESTURES = 24


@pytest.fixture(scope="module")
def populations():
    from opensk.mjx.outcomes import cpu_outcomes, mjx_outcomes, random_recipes

    recipes = random_recipes(N_GESTURES, seed=1)
    return cpu_outcomes(recipes), mjx_outcomes(recipes)


def _by(rows, name):
    return next(r for r in rows if r.name == name)


def test_the_population_is_not_trivially_inert(populations):
    """A parity check on gestures that all do nothing would pass for free."""
    cpu, _ = populations
    assert max(o.peak_height for o in cpu) > 0.05, "no gesture pops the board"
    assert max(abs(o.roll_deg) for o in cpu) > 30.0, "no gesture rolls the board"


def test_translation_outcomes_agree(populations):
    """Height, air time and travel are not chaotic and must match closely.

    These are the outcomes an optimiser and a world model both lean on hardest,
    and unlike rotation they do not amplify a contact-solver difference into a
    different answer. Measured KS 0.04-0.08 with rank correlation 1.00.
    """
    from opensk.mjx.outcomes import compare

    rows = compare(*populations)
    for field in ("peak_height", "air_s", "displacement"):
        r = _by(rows, field)
        assert r.ks < 0.25, f"{field}: distributions differ, KS {r.ks:.2f}"
        assert r.rank_corr > 0.9, f"{field}: ranking differs, rho {r.rank_corr:.2f}"


def test_rotation_outcomes_agree_in_distribution(populations):
    """Rotation is where the two contact solvers genuinely part company.

    A board on the point of flipping goes either way, so the two backends will
    disagree about individual gestures. The distribution still has to line up,
    which is the property the fitted parameters actually depend on.
    """
    from opensk.mjx.outcomes import compare

    rows = compare(*populations)
    for field in ("roll_deg", "yaw_deg"):
        r = _by(rows, field)
        assert r.ks < 0.5, f"{field}: distributions differ, KS {r.ks:.2f}"
        assert abs(r.cpu_sd - r.mjx_sd) < 0.4 * max(r.cpu_sd, 1e-9), \
            f"{field}: spread differs, {r.cpu_sd:.1f} vs {r.mjx_sd:.1f}"
        assert r.rank_corr > 0.5, f"{field}: ranking differs, rho {r.rank_corr:.2f}"


def test_no_backend_produces_impossible_motion(populations):
    """The bug class that started this: a finger flinging the board kilometres.

    Both backends must stay inside what a finger on a 0.9 kg deck can do over
    2.3 s. Stated as an absolute bound rather than a comparison, because when
    this failed it failed by two orders of magnitude.
    """
    for pop in populations:
        for o in pop:
            assert np.isfinite(o.peak_height) and o.peak_height < 3.0
            assert np.isfinite(o.displacement) and o.displacement < 40.0
