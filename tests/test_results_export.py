import pandas as pd, pytest
from evaluation.results_export import export_predictions_and_metrics

def test_export_requires_probabilities(tmp_path):
    df=pd.DataFrame({"model_name":["m"],"subject_id":["s"],"fold":["f"],"y_true":[1],"y_pred":[1]})
    with pytest.raises(ValueError): export_predictions_and_metrics(df,tmp_path,0)
