"""Guard MJX (GPU) portability.

MJX-JAX implements a subset of MuJoCo. Every violation below still runs
perfectly on CPU and silently fails to port, which is the worst possible
failure mode: it would surface as wrong physics on the GPU run, months after
the model changed. So it is a test, not a comment.
"""
import mujoco
import pytest

from opensk.sim.model.build import FLAT_PARK, build_scene
from opensk.sim.params import SkateParams

# `int()` is load-bearing. `model.geom_type[i]` is a numpy int32, and
# `np.int32(5) in {mujoco.mjtGeom.mjGEOM_CYLINDER}` is **False** while
# `np.int32(5) == mujoco.mjtGeom.mjGEOM_CYLINDER` is True: set membership
# falls through pybind11's enum comparison and never matches. Written the
# obvious way, this guard sat green for the whole project over a model
# carrying four cylinders and two ellipsoids. A test that cannot fail is not
# a test -- standing rule 3, in the test suite this time.
BANNED = {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_scene(SkateParams(), FLAT_PARK))


def test_no_collidable_cylinder_or_ellipsoid_geoms(model):
    """MJX-JAX cannot collide these with a box or mesh, and parks are boxes.

    The constraint is on CONTACT, not on the shape existing: a cylinder with
    contype/conaffinity 0 never enters a collision pair, and the visual wheels
    are exactly that. So this asserts what is actually required, in the same
    form as the mesh guard below.
    """
    bad = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if int(model.geom_type[i]) in BANNED
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    assert not bad, f"MJX cannot collide these reliably: {bad}"


def test_the_banned_set_actually_matches_geom_type(model):
    """The guard above must be able to fail. It could not, for months."""
    types = {int(t) for t in model.geom_type}
    assert types & BANNED, (
        "no cylinder or ellipsoid in the model at all -- this canary can no "
        "longer detect the numpy/pybind membership bug it exists to catch")
    assert any(int(model.geom_type[i]) in BANNED for i in range(model.ngeom))


def test_no_collidable_meshes(model):
    """Visual meshes are fine, but they must not participate in contact."""
    bad = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    assert not bad, f"collidable meshes break MJX portability: {bad}"


def test_solver_and_integrator(model):
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL


def test_condim_supported(model):
    """MJX-JAX supports condim in {1, 3, 4, 6}; we standardise on 3."""
    assert set(model.geom_condim) <= {1, 3, 4, 6}
