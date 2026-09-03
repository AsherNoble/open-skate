"""Storage contract: a sim rollout and a real demonstration stay the same object.

The world model is trained on simulated rollouts and fine-tuned on real ones,
so the moment the two stop sharing a shape the fine-tuning stage turns into a
translation exercise -- and translation is where a corpus convention gets
quietly mismatched, which has already cost this project once.
"""
import numpy as np
import pytest

from opensk.rl import store


def _episodes(b=5, f=68):
    from opensk.rl.env import Episodes
    rng = np.random.default_rng(0)
    return Episodes(pos=rng.normal(size=(b, f, 3)), quat=rng.normal(size=(b, f, 4)),
                    roll_deg=rng.normal(size=b), yaw_deg=rng.normal(size=b),
                    peak_height=rng.normal(size=b), air_s=rng.normal(size=b),
                    displacement=rng.normal(size=b),
                    valid=np.array([True, False, True, True, False]))


def test_round_trip_preserves_shapes_and_values(tmp_path):
    ep = _episodes()
    actions = np.random.default_rng(1).normal(size=(5, 17))
    got = store.load(store.save(tmp_path / "s.npz", actions, ep))
    assert len(got) == 5
    assert got.pos.shape == (5, 68, 3) and got.quat.shape == (5, 68, 4)
    assert np.allclose(got.actions, actions, atol=1e-6)
    assert np.array_equal(got.valid, np.asarray(ep.valid))


def test_filtering_keeps_only_physical_episodes(tmp_path):
    got = store.load(store.save(tmp_path / "s.npz",
                                np.zeros((5, 17)), _episodes())).filtered()
    assert len(got) == 3
    assert got.valid.all()


def test_a_shard_from_an_unknown_format_is_refused(tmp_path):
    """Silently misreading a layout is worse than failing to read it."""
    import json

    p = tmp_path / "future.npz"
    store.save(p, np.zeros((5, 17)), _episodes())
    with np.load(p) as z:
        data = {k: z[k] for k in z.files}
    data["meta"] = np.frombuffer(
        json.dumps({"version": store.FORMAT_VERSION + 1, "source": "sim"}).encode(),
        dtype=np.uint8)
    np.savez_compressed(p, **data)
    with pytest.raises(ValueError, match="format version"):
        store.load(p)


def test_shards_iterate_in_a_reproducible_order(tmp_path):
    for i in (2, 0, 1):
        store.save(tmp_path / f"shard_{i:05d}.npz",
                   np.full((5, 17), float(i)), _episodes())
    order = [float(s.actions[0, 0]) for s in store.iter_shards(tmp_path)]
    assert order == [0.0, 1.0, 2.0]
