from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_thresholds_documented():
    text = (ROOT / "src" / "radial_pilot_runner_impl.py.txt").read_text(encoding="utf-8")
    assert "delta_balanced_accuracy >= -0.005" in text
    assert "delta_sensitivity >= -0.010" in text
    assert "FAIL_DO_NOT_RUN_FULL_CV" in text
    assert '"full_cv_started": False' in text
