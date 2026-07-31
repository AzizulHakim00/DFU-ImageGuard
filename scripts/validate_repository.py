from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "requirements.txt",
    "configs/default_config.yaml",
    "src/config_data.py",
    "src/models_training.py",
    "src/evaluation.py",
    "src/statistics_figures.py",
    "src/robustness_xai.py",
    "src/artifacts.py",
    "src/dfu_imageguard_pipeline.py",
    "notebooks/DFU_ImageGuard_Conference_Complete.ipynb",
    "notebooks/DFU_ImageGuard_Load_Artifacts.ipynb",
    "notebooks/DFU_ImageGuard_Upload_Existing_Run.ipynb",
]
missing = [path for path in REQUIRED if not (ROOT / path).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")

for py_path in sorted((ROOT / "src").glob("*.py")):
    ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))

for nb_path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4, nb_path
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if nb_path.name == "DFU_ImageGuard_Conference_Complete.ipynb":
        assert len(code_cells) == 1, "Conference notebook must contain exactly one executable code cell"
        assert len(notebook["cells"]) == 2, "Conference notebook must contain exactly two cells"
        assert notebook["cells"][0]["cell_type"] == "markdown"
        assert code_cells[0].get("outputs", []) == [], "Repository notebook must not contain fake outputs"
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"{nb_path}:code_cell_{index}")

print("Repository structure, Python syntax, notebook syntax, and one-cell contract: PASS")
