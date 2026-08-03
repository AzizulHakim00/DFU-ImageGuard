import json
from pathlib import Path


def test_runner_uses_fresh_v2_run_id_and_retention_policy():
    source = Path("src/reliable_runner_v2.py").read_text(encoding="utf-8")
    assert 'run_id: str = "RELIABLE_DFU_CV_V2"' in source
    assert "retire_completed_full_checkpoints" in source
    assert "completed_trial_is_valid" in source
    assert "source_commit" in source


def test_v2_notebook_is_pinned_and_compiles():
    path = Path("notebooks/DFU_Reliable_Framework_V2_OneCell.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "TO_BE_PINNED" not in code
    assert "RELIABLE_DFU_CV_V2" in code
    compile(code, str(path), "exec")
