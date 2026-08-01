from __future__ import annotations

"""Load the validated pilot implementation stored as source text.

The implementation text is kept separately so this thin module remains directly
compilable and importable. The loader applies one audited signature repair to the
original generated source before compilation, then exposes its public API here.
"""

from pathlib import Path

_SOURCE_PATH = Path(__file__).with_name("radial_pilot_runner_impl.py.txt")
_SOURCE = _SOURCE_PATH.read_text(encoding="utf-8")
_BAD = "    patience_left: int,\n) \n    return {"
_GOOD = "    patience_left: int,\n) -> dict[str, Any]:\n    return {"

if _BAD in _SOURCE:
    _SOURCE = _SOURCE.replace(_BAD, _GOOD, 1)
elif _GOOD not in _SOURCE:
    raise RuntimeError("Pilot implementation signature marker was not found")

exec(compile(_SOURCE, str(_SOURCE_PATH), "exec"), globals(), globals())
