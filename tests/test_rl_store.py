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


def test_frames_round_trip_through_uint8_without_visible_loss(tmp_path):
    """Frames are stored as uint8, 4x smaller. The renderer's output is already
    quantised to 8 bits, so the round trip must be exact to within one level."""
    from opensk.rl.env import Episodes

    rng = np.random.default_rng(3)
    frames = (rng.integers(0, 256, (4, 6, 8, 5, 3)) / 255.0).astype(np.float32)
    ep = Episodes(pos=np.zeros((4, 6, 3)), quat=np.zeros((4, 6, 4)),
                  roll_deg=np.zeros(4), yaw_deg=np.zeros(4),
                  peak_height=np.zeros(4), air_s=np.zeros(4),
                  displacement=np.zeros(4), valid=np.ones(4, bool), rgb=frames)
    got = store.load(store.save(tmp_path / "s.npz", np.zeros((4, 17)), ep))
    assert got.rgb is not None and got.rgb.dtype == np.uint8
    assert np.abs(got.frames_float() - frames).max() < 1.0 / 255.0


def test_filtering_keeps_frames_aligned_with_episodes(tmp_path):
    """A filter that drops episodes but not their frames silently pairs every
    remaining episode with the wrong images."""
    from opensk.rl.env import Episodes

    frames = np.zeros((5, 2, 3, 3, 3), dtype=np.float32)
    for i in range(5):
        frames[i] = i / 255.0            # each episode's frames tagged by index
    ep = Episodes(pos=np.zeros((5, 2, 3)), quat=np.zeros((5, 2, 4)),
                  roll_deg=np.zeros(5), yaw_deg=np.zeros(5),
                  peak_height=np.zeros(5), air_s=np.zeros(5),
                  displacement=np.zeros(5),
                  valid=np.array([True, False, True, True, False]), rgb=frames)
    got = store.load(store.save(tmp_path / "s.npz",
                                np.zeros((5, 17)), ep)).filtered()
    assert len(got) == 3
    assert [int(got.rgb[i].flat[0]) for i in range(3)] == [0, 2, 3]
