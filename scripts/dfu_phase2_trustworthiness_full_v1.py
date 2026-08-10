from __future__ import annotations

import base64
import hashlib
import urllib.request
from pathlib import Path

LOADER_VERSION = "DFU_PHASE2_TRUSTWORTHINESS_FULL_LOADER_V1_20260811"
REPO = "AzizulHakim00/DFU-ImageGuard"
PARTS_COMMIT = "7e0126e799d133bd9b4e26dc0e96c0bd0c1cabd4"
EXPECTED_SOURCE_SHA256 = "2767a15418ba7fce5b5cf39ec82ba805fd205781e6e95b18476d82837a1faa7b"
PLAIN_PARTS = {0, 2, 3, 6, 7, 8, 9}
B64_PARTS = {1, 4, 5}

print(LOADER_VERSION)
print("Read-only Phase-2 loader: exact source hash is verified before execution.")

parts: list[bytes] = []
for i in range(10):
    if i in PLAIN_PARTS:
        rel = f"scripts/phase2_full_v1_parts/part_{i:02d}.txt"
        raw = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{REPO}/{PARTS_COMMIT}/{rel}", timeout=120
        ).read()
        parts.append(raw)
    elif i in B64_PARTS:
        rel = f"scripts/phase2_full_v1_parts_exact/part_{i:02d}.b64.txt"
        encoded = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{REPO}/{PARTS_COMMIT}/{rel}", timeout=120
        ).read().strip()
        parts.append(base64.b64decode(encoded, validate=True))
    else:
        raise RuntimeError(f"Unclassified source part: {i}")

source = b"".join(parts)
actual_sha256 = hashlib.sha256(source).hexdigest()
if actual_sha256 != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(
        f"Phase-2 full source SHA256 mismatch: expected={EXPECTED_SOURCE_SHA256} actual={actual_sha256}"
    )
print("Phase-2 full source SHA256: PASS", actual_sha256)

source_path = Path("/content/dfu_phase2_trustworthiness_full_v1_exact.py")
source_path.write_bytes(source)
text = source.decode("utf-8")
compile(text, str(source_path), "exec")
print("Phase-2 full source compile: PASS")

namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(text, str(source_path), "exec"), namespace)
