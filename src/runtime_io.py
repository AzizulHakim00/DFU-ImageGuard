from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image

_INSTALLED = False
_ORIGINAL_NP_SAVE = np.save
_ORIGINAL_NP_SAVEZ = np.savez
_ORIGINAL_NP_SAVEZ_COMPRESSED = np.savez_compressed
_ORIGINAL_TO_CSV = pd.DataFrame.to_csv
_ORIGINAL_IMAGE_SAVE = Image.Image.save


def _path_from_target(target: Any) -> Path | None:
    if isinstance(target, (str, os.PathLike)):
        return Path(target)
    return None


def ensure_parent(target: Any) -> Path | None:
    path = _path_from_target(target)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _retry(operation: Callable[[], Any], target: Any, attempts: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            ensure_parent(target)
            return operation()
        except (FileNotFoundError, OSError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(0.35 * (attempt + 1))
    assert last_error is not None
    raise last_error


def resilient_np_save(file: Any, arr: Any, allow_pickle: bool = True, fix_imports: bool = True) -> None:
    path = _path_from_target(file)
    if path is None:
        _ORIGINAL_NP_SAVE(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)
        return

    if path.suffix != ".npy":
        path = Path(str(path) + ".npy")

    def write_once() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                _ORIGINAL_NP_SAVE(handle, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    _retry(write_once, path)


def _resilient_npz(original: Callable[..., Any], file: Any, *args: Any, **kwargs: Any) -> None:
    path = _path_from_target(file)
    if path is None:
        original(file, *args, **kwargs)
        return
    if path.suffix != ".npz":
        path = Path(str(path) + ".npz")

    def write_once() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.npz"
        try:
            original(temporary, *args, **kwargs)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    _retry(write_once, path)


def resilient_np_savez(file: Any, *args: Any, **kwargs: Any) -> None:
    _resilient_npz(_ORIGINAL_NP_SAVEZ, file, *args, **kwargs)


def resilient_np_savez_compressed(file: Any, *args: Any, **kwargs: Any) -> None:
    _resilient_npz(_ORIGINAL_NP_SAVEZ_COMPRESSED, file, *args, **kwargs)


def resilient_to_csv(self: pd.DataFrame, path_or_buf: Any = None, *args: Any, **kwargs: Any) -> Any:
    path = _path_from_target(path_or_buf)
    if path is None:
        return _ORIGINAL_TO_CSV(self, path_or_buf, *args, **kwargs)

    def write_once() -> Any:
        path.parent.mkdir(parents=True, exist_ok=True)
        return _ORIGINAL_TO_CSV(self, path, *args, **kwargs)

    return _retry(write_once, path)


def resilient_image_save(self: Image.Image, fp: Any, *args: Any, **kwargs: Any) -> Any:
    path = _path_from_target(fp)
    if path is None:
        return _ORIGINAL_IMAGE_SAVE(self, fp, *args, **kwargs)

    def write_once() -> Any:
        path.parent.mkdir(parents=True, exist_ok=True)
        return _ORIGINAL_IMAGE_SAVE(self, path, *args, **kwargs)

    return _retry(write_once, path)


def storage_write_probe(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    probe_dir = root / ".dfu_imageguard_write_probe" / "nested"
    npy_path = probe_dir / "probe.npy"
    csv_path = probe_dir / "probe.csv"
    image_path = probe_dir / "probe.png"

    resilient_np_save(npy_path, np.asarray([1, 2, 3], dtype=np.int64))
    pd.DataFrame({"value": [1]}).to_csv(csv_path, index=False)
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(image_path)

    valid = (
        npy_path.exists()
        and csv_path.exists()
        and image_path.exists()
        and np.array_equal(np.load(npy_path), np.asarray([1, 2, 3], dtype=np.int64))
    )
    for path in [npy_path, csv_path, image_path]:
        path.unlink(missing_ok=True)
    try:
        probe_dir.rmdir()
        probe_dir.parent.rmdir()
    except OSError:
        pass
    if not valid:
        raise RuntimeError(f"Storage write probe failed under {root}")
    return {"root": str(root), "status": "PASS"}


def install_runtime_io_guards() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    np.save = resilient_np_save
    np.savez = resilient_np_savez
    np.savez_compressed = resilient_np_savez_compressed
    pd.DataFrame.to_csv = resilient_to_csv
    Image.Image.save = resilient_image_save
    _INSTALLED = True
