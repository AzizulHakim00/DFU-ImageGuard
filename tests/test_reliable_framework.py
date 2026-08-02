import numpy as np
import pandas as pd

from src.reliable_analysis import metric_dict, selective_prediction_table


def test_metric_dict_perfect():
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9]); pred=np.array([0,0,1,1])
    m=metric_dict(y,p,pred)
    assert m['balanced_accuracy']==1.0
    assert m['sensitivity']==1.0
    assert m['specificity']==1.0


def test_selective_prediction_reduces_coverage():
    frame=pd.DataFrame({
        'model_key':['convnextv2_tiny']*4,
        'seed':[2026]*4,
        'outer_fold':[1]*4,
        'label':[0,0,1,1],
        'prob_calibrated':[.05,.45,.55,.95],
        'pred':[0,0,1,1],
    })
    out=selective_prediction_table(frame,coverages=(1.0,.5))
    assert set(out.coverage)=={1.0,.5}
    assert int(out.loc[out.coverage==.5,'referred'].iloc[0])==2
