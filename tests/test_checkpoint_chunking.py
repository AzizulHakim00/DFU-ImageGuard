def test_checkpoint_chunk_roundtrip(tmp_path):
    from src.checkpoint_backup import chunk_file_for_git, reconstruct_chunked_checkpoint
    source=tmp_path/'model.pt'
    source.write_bytes(bytes(range(256))*1000)
    chunk_dir=tmp_path/'model.pt.chunks'
    manifest=chunk_file_for_git(source,chunk_dir,32768)
    assert len(manifest['parts'])>1
    restored=tmp_path/'restored.pt'
    reconstruct_chunked_checkpoint(chunk_dir,restored)
    assert restored.read_bytes()==source.read_bytes()
