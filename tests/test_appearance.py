"""Appearance may change pixels and absolutely nothing physical."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from opensk.sim.appearance import APPEARANCE_PRESETS, resolve_appearance
from opensk.sim.core import SkateSim
from opensk.sim.model.build import build_scene
from opensk.sim.model.parks import PARKS
from opensk.sim.params import SkateParams
from opensk.sim.touch import TouchModel


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


def test_visual_scene_has_fine_concrete_contact_shadows_and_indoor_depth():
    root = ET.fromstring(build_scene(SkateParams(), PARKS["plaza"], "indoor"))
    texture = next(node for node in root.iter("texture")
                   if node.attrib.get("name") == "tex_ground")
    material = next(node for node in root.iter("material")
                    if node.attrib.get("name") == "mat_ground")
    lights = [node for node in root.iter("light")
              if node.attrib.get("name", "").startswith("key")]
    geom_names = {node.attrib.get("name", "") for node in root.iter("geom")}

    # The procedural aggregate is much finer than the old visible slab grid;
    # sparse joint geoms carry scale independently of the texture.
    assert texture.attrib["builtin"] == "checker"
    assert texture.attrib.get("mark") is None
    assert tuple(map(float, material.attrib["texrepeat"].split())) >= (64, 64)
    assert len(lights) == 1
    assert all(light.attrib.get("castshadow") == "true" for light in lights)
    assert {"fx_indoor_back_wall", "fx_indoor_window_l",
            "fx_indoor_column_l", "fx_indoor_wall_beam_high"} <= geom_names


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


def _baseline_trajectory(park: str, *, visuals: bool = True) -> np.ndarray:
    sim = SkateSim(park=PARKS[park], visuals=visuals)
    sim.reset(seed=17, speed=2.5)
    rows = []
    for i in range(1200):
        if i % 240 < 70:
            sim.apply_force([13.0, -21.0, -47.0],
                            sim.body_point([-0.29, 0.035, 0.006]))
        sim.step()
        rows.append(np.concatenate((sim.data.qpos.copy(), sim.data.qvel.copy())))
    return np.stack(rows)


@pytest.mark.parametrize("park, expected", [
    ("flat", "73a9c011bc2de3a986a1e8da4e0e8d755835a8a927969bae82b04af7e2ba507a"),
    ("sls", "2dca65616e8d65cf467468092be2b24a8f31e1dd7ff46c53b864468bb27ca8a9"),
])
def test_pre_visual_direction_trajectories_remain_bit_identical(park, expected):
    trajectory = _baseline_trajectory(park)
    assert hashlib.sha256(trajectory.tobytes()).hexdigest() == expected
    assert np.array_equal(trajectory, _baseline_trajectory(park, visuals=False))


def _review_collision_hash(park: str) -> str:
    sim = SkateSim(park=PARKS[park])
    physical = []
    for gid in range(sim.model.ngeom):
        name = sim.model.geom(gid).name
        if _is_visual(name):
            continue
        physical.append((
            name, int(sim.model.geom_type[gid]), sim.model.geom_size[gid].tolist(),
            sim.model.geom_pos[gid].tolist(), sim.model.geom_quat[gid].tolist(),
            sim.model.geom_friction[gid].tolist(),
            int(sim.model.geom_contype[gid]),
            int(sim.model.geom_conaffinity[gid]),
        ))
    payload = json.dumps(physical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize("park, expected", [
    ("flat", "3f6f9b6c2154e8133890411bb825c3aa72538388662d038540f92c60ac9ef4a9"),
    ("sls", "17ae94e27e2dbce4e3c300f26261024d8bf493b73b7b730ac33bd0681b172212"),
])
def test_collision_signatures_remain_unchanged(park, expected):
    assert _review_collision_hash(park) == expected


def _touch_ray_signature(*, visuals: bool) -> tuple[str, list]:
    sim = SkateSim(visuals=visuals)
    sim.reset(seed=0)
    touch = TouchModel(sim)
    rays = []
    for x in np.linspace(0.2, 0.8, 7):
        for y in np.linspace(0.25, 0.9, 14):
            kind, hit = touch.cast(float(x), float(y))
            rays.append((str(kind), None if hit is None else np.asarray(hit).tolist()))
    payload = json.dumps(rays, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), rays


def test_touch_rays_are_unchanged_and_ignore_visual_geometry():
    full_hash, full = _touch_ray_signature(visuals=True)
    slim_hash, slim = _touch_ray_signature(visuals=False)
    assert full_hash == "bdf3ca9498387dbf3e58e4a4717a9b5548dd949bcb0a1422835f2c79180a802f"
    assert slim_hash == full_hash
    assert slim == full
