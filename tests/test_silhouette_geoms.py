"""The fitting silhouette is the DECK OUTLINE and nothing else.

This exists because it has already gone wrong once, silently. The appearance
pass added visual cylinder wheels named `vis_nose_wheel_l` and friends, and
`SkateSim._deck_visual_gids` selects by the PREFIX `vis_` -- so the wheels
walked straight into the mask the physics is fitted against, undoing a
decision the code comment records as measured: wheels sit at the axle
half-width plus their own radius and widen the silhouette ~26%, against real
frames whose length and framing match to within 1%.

Nothing in the test suite moved. What exposed it was a score that changed for
the STORED parameters (0.866 -> 0.839 on the same held-out half) while the
INERT reference stayed bit-identical at 0.650036 -- because an inert board
lies flat with its wheels hidden under the deck, and a flipping one does not.

A cosmetic change is not cosmetic if it renames a geom.
"""
import re

from opensk.sim.core import SkateSim
from opensk.sim.params import SkateParams

# The deck's visual shell is now a single generated mesh, swept along the
# profile measured off the game's own geometry. It replaced eleven
# constant-width boxes plus two ellipsoid tips.
OUTLINE = re.compile(r"^vis_deck$")


def test_only_the_deck_outline_is_in_the_silhouette():
    sim = SkateSim(SkateParams())
    names = [sim.model.geom(g).name for g in sorted(sim._deck_visual_gids)]
    assert names, "the silhouette cannot be empty"
    bad = [n for n in names if not OUTLINE.match(n)]
    assert not bad, f"non-outline geoms in the fitting silhouette: {bad}"


def test_no_hardware_is_named_into_the_silhouette():
    """The complement of the above: hardware must not use the `vis_` prefix."""
    sim = SkateSim(SkateParams())
    for g in range(sim.model.ngeom):
        n = sim.model.geom(g).name
        if any(k in n for k in ("wheel", "truck", "axle")):
            assert not n.startswith("vis_"), (
                f"{n} would be selected into the fitting silhouette by prefix")
