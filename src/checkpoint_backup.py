from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def chunk_file_for_git(source: str | Path, destination_dir: str | Path, chunk_bytes: int) -> dict[str, Any]:
    source = Path(source)
    destination_dir = Path(destination_dir)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint missing or empty: {source}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for old in destination_dir.glob("part-*.bin"):
        old.unlink()
    parts = []
    with source.open("rb") as handle:
        index = 0
        while True:
            block = handle.read(int(chunk_bytes))
            if not block:
                break
            part = destination_dir / f"part-{index:04d}.bin"
            part.write_bytes(block)
            parts.append({
                "path": part.name,
                "bytes": len(block),
                "sha256": hashlib.sha256(block).hexdigest(),
            })
            index += 1
    manifest = {
        "original_name": source.name,
        "original_bytes": int(source.stat().st_size),
        "original_sha256": sha256_file(source),
        "chunk_bytes": int(chunk_bytes),
        "parts": parts,
        "reconstruction": "Concatenate part-*.bin in lexical order to recover the exact file.",
    }
    _atomic_json(destination_dir / "chunks.json", manifest)
    return manifest


def reconstruct_chunked_checkpoint(chunk_dir: str | Path, output_path: str | Path) -> Path:
    chunk_dir = Path(chunk_dir)
    output_path = Path(output_path)
    manifest_path = chunk_dir / "chunks.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as out:
        for item in manifest["parts"]:
            part = chunk_dir / item["path"]
            if sha256_file(part) != item["sha256"]:
                raise RuntimeError(f"Corrupt checkpoint chunk: {part}")
            with part.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    out.write(block)
    os.replace(temporary, output_path)
    if output_path.stat().st_size != int(manifest["original_bytes"]):
        raise RuntimeError("Reconstructed checkpoint size mismatch")
    if sha256_file(output_path) != manifest["original_sha256"]:
        raise RuntimeError("Reconstructed checkpoint SHA-256 mismatch")
    return output_path
