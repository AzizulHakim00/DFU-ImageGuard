from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import src  # noqa: F401 - installs the runtime I/O guards
from src.runtime_io import storage_write_probe


root = Path(tempfile.mkdtemp(prefix="dfu-imageguard-io-"))
try:
    result = storage_write_probe(root)
    assert result["status"] == "PASS"

    nested = root / "deleted_parent" / "deep"
    shutil.rmtree(nested.parent, ignore_errors=True)
    npy_path = nested / "array.npy"
    np.save(npy_path, np.arange(5, dtype=np.int64))
    assert np.array_equal(np.load(npy_path), np.arange(5, dtype=np.int64))

    shutil.rmtree(nested.parent, ignore_errors=True)
    csv_path = nested / "frame.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(csv_path, index=False)
    assert pd.read_csv(csv_path).x.tolist() == [1, 2]

    shutil.rmtree(nested.parent, ignore_errors=True)
    png_path = nested / "image.png"
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(png_path)
    with Image.open(png_path) as image:
        assert image.size == (4, 4)

    print("Runtime I/O missing-parent regression test: PASS")
finally:
    shutil.rmtree(root, ignore_errors=True)
