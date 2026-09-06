"""Local preview tooling preserves exact output modes and matrix balance."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from opensk.pose.preview import write_contact_sheet


def test_contact_sheet_never_rescales_training_samples(tmp_path):
    paths = []
    for i in range(9):
        path = tmp_path / f"sample_{i}.png"
        image = np.full((128, 64, 3), i * 20, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        paths.append(path)

    output = write_contact_sheet(paths, tmp_path / "sheet.png")
    sheet = cv2.imread(str(output))
    assert output == Path(tmp_path / "sheet.png")
    assert sheet.shape == (3 * 128, 3 * 64, 3)
    assert np.array_equal(sheet[:128, :64], cv2.imread(str(paths[0])))
