from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.reliability_external import validate_external_manifest


def test_external_manifest_requires_binary_labels_and_existing_files(
    tmp_path: Path,
):
    normal = tmp_path / "normal.png"
    dfu = tmp_path / "dfu.png"
    Image.new("RGB", (16, 16)).save(normal)
    Image.new("RGB", (16, 16)).save(dfu)
    frame = validate_external_manifest(
        pd.DataFrame({"image_path": [normal, dfu], "label": [0, 1]})
    )
    assert set(frame.label) == {0, 1}
    assert "image_id" in frame


def test_external_manifest_rejects_single_class(tmp_path: Path):
    image = tmp_path / "one.png"
    Image.new("RGB", (16, 16)).save(image)
    with pytest.raises(ValueError):
        validate_external_manifest(
            pd.DataFrame({"image_path": [image], "label": [1]})
        )
