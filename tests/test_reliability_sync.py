from pathlib import Path

from src.reliability_sync import chunk_file_for_git, reconstruct_chunked_file


def test_checkpoint_chunk_roundtrip(tmp_path: Path):
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(bytes(range(256)) * 1000)
    chunk_dir = tmp_path / "checkpoint.pt.chunks"
    manifest = chunk_file_for_git(source, chunk_dir, 32768)
    assert len(manifest["parts"]) > 1
    restored = tmp_path / "restored.pt"
    result = reconstruct_chunked_file(chunk_dir, restored)
    assert result["success"] is True
    assert restored.read_bytes() == source.read_bytes()
