from pathlib import Path
import ast
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "requirements.txt", "configs/default_config.yaml",
    "src/dfu_imageguard_pipeline.py",
    "notebooks/DFU_ImageGuard_Conference_Complete.ipynb",
    "notebooks/DFU_ImageGuard_Load_Artifacts.ipynb",
    "notebooks/DFU_ImageGuard_Upload_Existing_Run.ipynb",
]
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")

for py_path in (ROOT / "src").glob("*.py"):
    ast.parse(py_path.read_text(encoding="utf-8"))
for nb_path in (ROOT / "notebooks").rglob("*.ipynb"):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, nb_path
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if nb_path.name == "DFU_ImageGuard_Conference_Complete.ipynb":
        assert len(code_cells) == 1, "Conference notebook must contain exactly one executable code cell"
        assert len(nb["cells"]) == 2 and nb["cells"][0]["cell_type"] == "markdown"
    for cell in code_cells:
        source = "".join(cell.get("source", []))
        if not source.lstrip().startswith("!"):
            try:
                ast.parse(source)
            except SyntaxError:
                pass
print("Repository structure and notebook JSON validation: PASS")
