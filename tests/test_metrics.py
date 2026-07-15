from evaluation.metrics import compute_binary_metrics, generalization_gap

def test_known_metrics():
    m=compute_binary_metrics([0,0,1,1],[0,1,1,1],[0.1,0.6,0.8,0.9])
    assert m["accuracy"] == 0.75
    assert m["specificity"] == 0.5
    assert m["recall_sensitivity"] == 1.0
    assert m["auc_roc"] == 1.0
    assert generalization_gap(0.8,0.7) == 0.10000000000000009
