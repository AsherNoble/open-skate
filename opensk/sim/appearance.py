"""Procedural, neutral appearance presets for Open Skate.

These values are deliberately outside :class:`SkateParams`: appearance is a
domain-randomisation input, not a fitted physical parameter.  A preset may
change materials, the sky and lights, but never collision geometry, camera
projection, mass or inertia.
"""
from __future__ import annotations

from dataclasses import dataclass


RGB = tuple[float, float, float]


@dataclass(frozen=True)
class AppearancePreset:
    """One coherent renderer look, expressed entirely as procedural values."""

    name: str
    sky_top: RGB
    sky_horizon: RGB
    ground_light: RGB
    ground_dark: RGB
    joint: RGB
    concrete_light: RGB
    concrete_dark: RGB
    ledge_light: RGB
    ledge_dark: RGB
    accent: RGB
    accent_secondary: RGB
    grip_light: RGB
    grip_dark: RGB
    underside: RGB
    plywood: RGB
    truck: RGB
    wheel: RGB
    ambient: RGB
    key_diffuse: RGB
    key_specular: RGB
    key_pos: tuple[float, float, float]
    key_dir: tuple[float, float, float]
    fill_diffuse: RGB
    fill_pos: tuple[float, float, float]
    fill_dir: tuple[float, float, float]
    environment_dark: RGB
    environment_mid: RGB
    environment_light: RGB
    environment: str = ""


_VISUAL = 'contype="0" conaffinity="0" mass="0" group="2"'


DAY = AppearancePreset(
    name="day",
    sky_top=(0.34, 0.43, 0.52),
    sky_horizon=(0.78, 0.82, 0.84),
    ground_light=(0.610, 0.604, 0.588),
    ground_dark=(0.598, 0.592, 0.576),
    joint=(0.46, 0.47, 0.47),
    concrete_light=(0.68, 0.68, 0.66),
    concrete_dark=(0.62, 0.62, 0.60),
    ledge_light=(0.73, 0.73, 0.70),
    ledge_dark=(0.67, 0.67, 0.64),
    accent=(0.76, 0.39, 0.30),
    accent_secondary=(0.28, 0.55, 0.61),
    grip_light=(0.105, 0.115, 0.125),
    grip_dark=(0.065, 0.072, 0.082),
    underside=(0.32, 0.34, 0.35),
    plywood=(0.70, 0.49, 0.27),
    truck=(0.50, 0.52, 0.54),
    wheel=(0.87, 0.86, 0.80),
    ambient=(0.23, 0.235, 0.24),
    key_diffuse=(0.72, 0.70, 0.65),
    key_specular=(0.19, 0.18, 0.16),
    key_pos=(3.0, -4.0, 7.0),
    key_dir=(-0.20, 0.24, -1.0),
    fill_diffuse=(0.18, 0.20, 0.23),
    fill_pos=(-5.0, 4.0, 5.0),
    fill_dir=(0.50, -0.40, -1.0),
    environment_dark=(0.28, 0.30, 0.31),
    environment_mid=(0.48, 0.50, 0.49),
    environment_light=(0.64, 0.65, 0.62),
    environment=f"""
    <!-- Abstract civic massing gives the plaza a horizon without depicting a
         named place.  It is deliberately distant and entirely visual-only. -->
    <geom name="fx_day_horizon" type="box" size="0.10 10 0.62"
          pos="10.5 0 0.62" {_VISUAL} material="mat_environment_mid"/>
    <geom name="fx_day_block_left" type="box" size="1.8 2.1 1.35"
          pos="12.1 4.5 1.35" {_VISUAL} material="mat_environment_dark"/>
    <geom name="fx_day_block_right" type="box" size="1.25 1.7 1.05"
          pos="11.7 -4.2 1.05" {_VISUAL} material="mat_environment_light"/>""",
)


INDOOR = AppearancePreset(
    name="indoor",
    sky_top=(0.055, 0.060, 0.070),
    sky_horizon=(0.13, 0.14, 0.16),
    ground_light=(0.610, 0.615, 0.620),
    ground_dark=(0.598, 0.603, 0.608),
    joint=(0.43, 0.44, 0.46),
    concrete_light=(0.70, 0.70, 0.69),
    concrete_dark=(0.64, 0.64, 0.63),
    ledge_light=(0.77, 0.77, 0.75),
    ledge_dark=(0.69, 0.69, 0.68),
    accent=(0.78, 0.36, 0.22),
    accent_secondary=(0.34, 0.55, 0.58),
    grip_light=(0.105, 0.112, 0.120),
    grip_dark=(0.060, 0.067, 0.076),
    underside=(0.30, 0.31, 0.33),
    plywood=(0.68, 0.46, 0.25),
    truck=(0.46, 0.48, 0.50),
    wheel=(0.89, 0.88, 0.83),
    ambient=(0.17, 0.18, 0.20),
    key_diffuse=(0.78, 0.77, 0.73),
    key_specular=(0.22, 0.22, 0.21),
    key_pos=(-1.5, -1.0, 6.0),
    key_dir=(0.10, 0.14, -1.0),
    fill_diffuse=(0.22, 0.24, 0.27),
    fill_pos=(6.0, 3.0, 5.0),
    fill_dir=(-0.35, -0.18, -1.0),
    environment_dark=(0.105, 0.115, 0.130),
    environment_mid=(0.245, 0.260, 0.275),
    environment_light=(0.590, 0.605, 0.610),
    environment=f"""
    <!-- A complete but sparse hall envelope.  The old back wall was only 3.4 m
         ahead and read as a black void; this keeps useful depth behind the park. -->
    <geom name="fx_indoor_back_wall" type="box" size="0.12 5.5 2.8"
          pos="6.8 0 2.8" {_VISUAL} material="mat_environment_mid"/>
    <geom name="fx_indoor_dado" type="box" size="0.015 5.35 0.58"
          pos="6.66 0 0.58" {_VISUAL} material="mat_environment_dark"/>
    <geom name="fx_indoor_window_l" type="box" size="0.015 1.55 0.72"
          pos="6.64 3.45 3.20" {_VISUAL} material="mat_environment_light"/>
    <geom name="fx_indoor_window_c" type="box" size="0.015 1.55 0.72"
          pos="6.64 0 3.20" {_VISUAL} material="mat_environment_light"/>
    <geom name="fx_indoor_window_r" type="box" size="0.015 1.55 0.72"
          pos="6.64 -3.45 3.20" {_VISUAL} material="mat_environment_light"/>
    <geom name="fx_indoor_column_l" type="box" size="0.015 0.10 2.55"
          pos="6.64 2.0 2.55" {_VISUAL} material="mat_environment_dark"/>
    <geom name="fx_indoor_column_r" type="box" size="0.015 0.10 2.55"
          pos="6.64 -2.0 2.55" {_VISUAL} material="mat_environment_dark"/>
    <geom name="fx_indoor_wall_beam_low" type="box" size="0.015 5.35 0.08"
          pos="6.64 0 1.32" {_VISUAL} material="mat_environment_dark"/>
    <geom name="fx_indoor_wall_beam_high" type="box" size="0.015 5.35 0.10"
          pos="6.64 0 4.42" {_VISUAL} material="mat_environment_dark"/>""",
)


OVERCAST = AppearancePreset(
    name="overcast",
    sky_top=(0.39, 0.45, 0.48),
    sky_horizon=(0.66, 0.70, 0.71),
    ground_light=(0.570, 0.595, 0.600),
    ground_dark=(0.558, 0.583, 0.588),
    joint=(0.39, 0.43, 0.44),
    concrete_light=(0.62, 0.66, 0.66),
    concrete_dark=(0.56, 0.60, 0.61),
    ledge_light=(0.67, 0.70, 0.70),
    ledge_dark=(0.60, 0.64, 0.64),
    accent=(0.48, 0.68, 0.55),
    accent_secondary=(0.68, 0.62, 0.24),
    grip_light=(0.095, 0.110, 0.118),
    grip_dark=(0.055, 0.066, 0.073),
    underside=(0.29, 0.33, 0.34),
    plywood=(0.61, 0.46, 0.29),
    truck=(0.43, 0.47, 0.48),
    wheel=(0.84, 0.85, 0.81),
    ambient=(0.27, 0.285, 0.29),
    key_diffuse=(0.42, 0.45, 0.46),
    key_specular=(0.10, 0.11, 0.11),
    key_pos=(2.0, -3.0, 7.0),
    key_dir=(-0.12, 0.15, -1.0),
    fill_diffuse=(0.25, 0.27, 0.28),
    fill_pos=(-5.0, 4.0, 6.0),
    fill_dir=(0.42, -0.34, -1.0),
    environment_dark=(0.31, 0.36, 0.37),
    environment_mid=(0.48, 0.53, 0.53),
    environment_light=(0.61, 0.65, 0.64),
    environment=f"""
    <!-- Layered, generic civic forms make haze and scale readable while
         remaining too abstract to evoke any distinctive real-world park. -->
    <geom name="fx_overcast_horizon" type="box" size="0.10 11 0.72"
          pos="11.2 0 0.72" {_VISUAL} material="mat_environment_mid"/>
    <geom name="fx_overcast_block_left" type="box" size="1.4 2.4 1.6"
          pos="12.4 4.8 1.6" {_VISUAL} material="mat_environment_dark"/>
    <geom name="fx_overcast_block_mid" type="box" size="1.0 1.5 1.12"
          pos="12.1 0.8 1.12" {_VISUAL} material="mat_environment_light"/>
    <geom name="fx_overcast_block_right" type="box" size="1.8 2.1 1.32"
          pos="12.5 -4.3 1.32" {_VISUAL} material="mat_environment_dark"/>""",
)


APPEARANCE_PRESETS = {p.name: p for p in (DAY, INDOOR, OVERCAST)}
DEFAULT_APPEARANCE = DAY.name


def resolve_appearance(value: str | AppearancePreset) -> AppearancePreset:
    """Resolve a public preset name while giving errors useful CLI text."""
    if isinstance(value, AppearancePreset):
        return value
    try:
        return APPEARANCE_PRESETS[value]
    except KeyError as exc:
        names = ", ".join(APPEARANCE_PRESETS)
        raise ValueError(f"unknown appearance {value!r}; choose one of: {names}") from exc


def rgb(value: RGB) -> str:
    """An MJCF-ready three-channel colour."""
    return " ".join(f"{channel:.3f}" for channel in value)


def xyz(value: tuple[float, float, float]) -> str:
    """An MJCF-ready vector."""
    return " ".join(f"{component:.3f}" for component in value)
