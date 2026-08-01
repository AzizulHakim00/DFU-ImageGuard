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
    "src/runtime_io.py",
    "src/dfu_imageguard_pipeline.py",
    "polarmorphnet/DFU_PolarMorphNet_AllInOne.py",
    "scripts/test_runtime_io.py",
    "notebooks/DFU_ImageGuard_Conference_Complete.ipynb",
    "notebooks/DFU_ImageGuard_Load_Artifacts.ipynb",
    "notebooks/DFU_ImageGuard_Upload_Existing_Run.ipynb",
    "notebooks/DFU_PolarMorphNet_Complete.ipynb",
    "notebooks/DFU_PolarMorphNet_Artifacts_Only.ipynb",
    "notebooks/DFU_PolarMorphNet_Upload_Existing_Run.ipynb",
    "docs/DFU_POLARMORPHNET_PROTOCOL.md",
]
missing = [path for path in REQUIRED if not (ROOT / path).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")

for folder in ["src", "scripts", "polarmorphnet"]:
    for py_path in sorted((ROOT / folder).glob("*.py")):
        ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))

single_cell_notebooks = {
    "DFU_ImageGuard_Conference_Complete.ipynb",
    "DFU_PolarMorphNet_Complete.ipynb",
    "DFU_PolarMorphNet_Artifacts_Only.ipynb",
    "DFU_PolarMorphNet_Upload_Existing_Run.ipynb",
}
for nb_path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4, nb_path
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if nb_path.name in single_cell_notebooks:
        assert len(code_cells) == 1, f"{nb_path.name} must contain exactly one executable code cell"
        assert len(notebook["cells"]) == 2, f"{nb_path.name} must contain exactly two cells"
        assert notebook["cells"][0]["cell_type"] == "markdown"
        assert code_cells[0].get("outputs", []) == [], f"{nb_path.name} must not contain fake outputs"
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"{nb_path}:code_cell_{index}")

print("Repository structure, Python syntax, notebook syntax, and one-cell contracts: PASS")
