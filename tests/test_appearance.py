"""Appearance may change pixels and absolutely nothing physical."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from opensk.sim.appearance import APPEARANCE_PRESETS, resolve_appearance
from opensk.sim.core import SkateSim
from opensk.sim.model.build import build_scene
from opensk.sim.model.parks import PARKS
from opensk.sim.params import SkateParams


VISUAL_PREFIXES = ("vis_", "boardfx_", "hw_", "parkvis_", "fx_")


def _is_visual(name: str) -> bool:
    return name.startswith(VISUAL_PREFIXES)


@pytest.mark.parametrize("appearance", APPEARANCE_PRESETS)
@pytest.mark.parametrize("park", PARKS)
def test_every_park_compiles_under_every_appearance(appearance, park):
    model = mujoco.MjModel.from_xml_string(
        build_scene(SkateParams(), PARKS[park], appearance))
    assert model.ngeom > 0


def test_unknown_appearance_is_refused_with_available_names():
    with pytest.raises(ValueError, match="day, indoor, overcast"):
        resolve_appearance("branded-megapark")


@pytest.mark.parametrize("appearance", APPEARANCE_PRESETS)
@pytest.mark.parametrize("park", PARKS)
def test_render_geometry_has_no_mass_or_collision(appearance, park):
    """Check source MJCF, because per-geom mass is folded into body inertia."""
    root = ET.fromstring(build_scene(SkateParams(), PARKS[park], appearance))
    seen = set()
    for geom in root.iter("geom"):
        name = geom.attrib.get("name", "")
        if not _is_visual(name):
            continue
        seen.add(name)
        assert geom.attrib.get("mass") == "0", name
        assert geom.attrib.get("contype") == "0", name
        assert geom.attrib.get("conaffinity") == "0", name
        assert geom.attrib.get("group") in {"1", "2"}, name
    assert {"vis_deck", "boardfx_grip", "parkvis_ground"} <= seen


@pytest.mark.parametrize("appearance", APPEARANCE_PRESETS)
@pytest.mark.parametrize("park", PARKS)
def test_authoritative_collision_geometry_is_hidden_from_rgb(appearance, park):
    root = ET.fromstring(build_scene(SkateParams(), PARKS[park], appearance))
    physical = []
    for geom in root.iter("geom"):
        name = geom.attrib.get("name", "")
        if name and not _is_visual(name):
            physical.append(name)
            assert geom.attrib.get("group") == "3", name
            assert geom.attrib.get("contype", "1") != "0", name
    assert {"ground", "deck_flat", "front_wheel_l"} <= set(physical)


def _physical_signature(sim: SkateSim) -> dict[str, bytes]:
    """All geometry fields which can affect contacts, keyed by stable name."""
    model = sim.model
    fields = []
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if _is_visual(name):
            continue
        numeric = np.concatenate((
            np.asarray([model.geom_type[gid], model.geom_contype[gid],
                        model.geom_conaffinity[gid], model.geom_condim[gid]],
                       dtype=np.float64),
            model.geom_size[gid], model.geom_pos[gid], model.geom_quat[gid],
            model.geom_friction[gid], model.geom_solref[gid],
            model.geom_solimp[gid],
        ))
        fields.append((name, numeric.tobytes()))
    return dict(fields)


def test_presets_have_identical_physical_models():
    sims = [SkateSim(park=PARKS["plaza"], appearance=name)
            for name in APPEARANCE_PRESETS]
    reference = sims[0]
    for sim in sims[1:]:
        assert _physical_signature(sim) == _physical_signature(reference)
        assert np.array_equal(sim.model.body_mass, reference.model.body_mass)
        assert np.array_equal(sim.model.body_inertia, reference.model.body_inertia)
        assert np.array_equal(sim.model.dof_armature, reference.model.dof_armature)


def _trajectory(appearance: str) -> np.ndarray:
    sim = SkateSim(appearance=appearance)
    sim.reset(seed=7, speed=2.0)
    rows = []
    for i in range(700):
        if i % 150 < 35:
            sim.apply_force([11.0, -17.0, -43.0],
                            sim.body_point([-0.28, 0.03, 0.006]))
        sim.step()
        rows.append(np.concatenate((sim.data.qpos.copy(), sim.data.qvel.copy())))
    return np.stack(rows)


def test_appearance_presets_produce_bit_identical_trajectories():
    trajectories = [_trajectory(name) for name in APPEARANCE_PRESETS]
    for candidate in trajectories[1:]:
        assert np.array_equal(candidate, trajectories[0])
    # Useful in failures and review: this pins the exact local trajectory, not
    # just a tolerance which cosmetic edits could quietly consume.
    assert hashlib.sha256(trajectories[0].tobytes()).hexdigest() == (
        "fc46fb33da962e57b1940d698c391d0c747173a7915035772176d546df00c853")


def test_training_camera_resolution_and_projection_stay_fitted():
    params = SkateParams()
    assert (params.render_width, params.render_height) == (64, 128)
    for name in APPEARANCE_PRESETS:
        sim = SkateSim(appearance=name)
        cam_id = sim.model.camera("chase").id
        assert tuple(sim.model.cam_resolution[cam_id]) == (64, 128)
        assert sim.model.cam_fovy[cam_id] == pytest.approx(params.cam_fov_deg)
