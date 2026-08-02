from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .reliable_analysis import metric_dict


REQUIRED_COLUMNS={"image_path","label","image_id"}


def validate_external_manifest(path: str | Path) -> pd.DataFrame:
    frame=pd.read_csv(path)
    missing=REQUIRED_COLUMNS-set(frame.columns)
    if missing: raise ValueError(f"External manifest missing columns: {sorted(missing)}")
    if not set(frame.label.unique()).issubset({0,1}): raise ValueError("External labels must be binary 0/1 and task-compatible")
    if frame.image_id.duplicated().any(): raise ValueError("External image_id values must be unique")
    missing_files=[p for p in frame.image_path.astype(str) if not Path(p).is_file()]
    if missing_files: raise FileNotFoundError(f"External files missing; first={missing_files[0]}")
    return frame


def summarize_frozen_external_predictions(prediction_csv: str | Path, output_json: str | Path):
    frame=pd.read_csv(prediction_csv)
    required={"label","prob_calibrated","pred","model_key","seed","outer_fold"}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    rows=[]
    for keys,g in frame.groupby(["model_key","seed","outer_fold"]):
        rows.append({"model_key":keys[0],"seed":int(keys[1]),"outer_fold":int(keys[2]),**metric_dict(g.label,g.prob_calibrated,g.pred)})
    result={"policy":"No retraining, threshold tuning, temperature refitting or model selection on external data.","metrics":rows}
    Path(output_json).write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
