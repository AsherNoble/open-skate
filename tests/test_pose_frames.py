import json
from pathlib import Path

import numpy as np
from PIL import Image

from opensk.pose import frames


def test_explicit_waypoint_sample_preserves_absolute_spin_hold(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "meta.json").write_text(json.dumps({
        "waypoints": [[0.4, 0.7], [0.6, 0.3]],
        "duration": 0.3,
        "easing_power": 1.2,
        "frame_times": [0.0],
        "spin_active": True,
        "spin_hold_start_s": 0.04,
        "spin_hold_end_s": 0.19,
    }))
    Image.new("RGB", (2, 2)).save(sample_dir / "frame_000.png")
    sample = frames.load_sample(sample_dir)
    assert sample is not None
    assert sample.recipe()["spin"] == {
        "enabled": True,
        "spin_hold_start_s": 0.04,
        "spin_hold_end_s": 0.19,
    }


def test_gameplay_filter_receives_decoded_video_frames(monkeypatch, tmp_path):
    decoded_bgr = np.zeros((3, 4, 3), dtype=np.uint8)
    decoded_bgr[..., 2] = 255
    sample = frames.Sample(
        path=tmp_path, waypoints=np.array([[0.4, 0.7], [0.6, 0.3]]),
        duration=0.2, easing_power=1.0, frame_times=np.array([0.0]),
        frame_paths=[Path("does-not-need-to-exist.mp4")], park=None,
        spin_active=False, video=Path("packed.mp4"), _decoded=[decoded_bgr])

    class Filter:
        @staticmethod
        def is_gameplay_frame(image):
            assert isinstance(image, Image.Image)
            assert image.getpixel((0, 0)) == (255, 0, 0)
            return True

    monkeypatch.setattr(frames, "_gameplay_filter", lambda: Filter)
    assert frames.gameplay_flags(sample).tolist() == [True]
