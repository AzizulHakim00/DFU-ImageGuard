import numpy as np
import pandas as pd

from src.reliability_metrics import (
    paired_group_bootstrap,
    selective_prediction_table,
)


def _frame(probabilities):
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = (np.asarray(probabilities) >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(len(y))],
            "group_id": ["g0", "g0", "g1", "g2", "g3", "g4"],
            "label": y,
            "prob_calibrated": probabilities,
            "pred_calibrated": pred,
            "confidence": np.maximum(probabilities, 1 - np.asarray(probabilities)),
        }
    )


def test_selective_prediction_retains_requested_order():
    table = selective_prediction_table(
        _frame([0.01, 0.20, 0.95, 0.70, 0.45, 0.55])
    )
    assert list(table.requested_coverage) == [1.0, 0.95, 0.90, 0.80]
    assert table.retained_n.is_monotonic_decreasing
    assert (table.actual_coverage <= 1.0).all()


def test_group_bootstrap_is_reproducible():
    a = _frame([0.01, 0.20, 0.95, 0.70, 0.45, 0.55])
    b = _frame([0.10, 0.30, 0.70, 0.52, 0.55, 0.49])
    first = paired_group_bootstrap(
        a, b, metric="balanced_accuracy", reps=600, seed=42
    )
    second = paired_group_bootstrap(
        a, b, metric="balanced_accuracy", reps=600, seed=42
    )
    assert first == second
    assert first.valid_replicates >= 500
