#!/usr/bin/env python3
from pathlib import Path
import base64, hashlib, io, zipfile, shutil
ROOT=Path(__file__).resolve().parent
parts=sorted((ROOT/'bootstrap_chunks').glob('chunk_*.txt'))
if not parts:
    raise RuntimeError('Bootstrap chunks are missing.')
data=base64.b64decode(''.join(p.read_text().strip() for p in parts))
expected='d8817a6de9f146ac8bd11b6489e470e4d937fc582c54bb467f3c2ec5724af313'
actual=hashlib.sha256(data).hexdigest()
if actual!=expected:
    raise RuntimeError(f'Bootstrap payload checksum mismatch: {actual} != {expected}')
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for member in archive.infolist():
        target=(ROOT/member.filename).resolve()
        if ROOT.resolve() not in target.parents and target!=ROOT.resolve():
            raise RuntimeError(f'Unsafe archive path: {member.filename}')
    archive.extractall(ROOT)
shutil.rmtree(ROOT/'bootstrap_chunks',ignore_errors=True)
for rel in ['bootstrap_payload.py','.github/workflows/bootstrap.yml']:
    path=ROOT/rel
    if path.exists(): path.unlink()
print('Extracted and validated the complete DFU-ImageGuard repository payload.')
# Triggered after all checksum-verified payload chunks were uploaded.
