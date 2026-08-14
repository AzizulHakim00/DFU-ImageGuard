from pathlib import Path


def test_runner_freezes_temperature_and_threshold_before_outer_test():
    text = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "reliability_training.py"
    ).read_text()
    calibration_position = text.index(
        "temperature = fit_temperature(calibration_logits"
    )
    threshold_position = text.index(
        "threshold, threshold_rule = select_threshold"
    )
    test_position = text.index("test_logits, test_y, _ = predict_logits")
    assert calibration_position < threshold_position < test_position
    assert (
        "outer test"
        not in text[calibration_position:threshold_position].lower()
    )
