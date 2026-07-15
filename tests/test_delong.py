
from evaluation.delong import delong_roc_test

def test_delong_reference_identical_predictions():
    p=delong_roc_test([0,0,1,1],[0.1,0.2,0.8,0.9],[0.1,0.2,0.8,0.9])
    assert p == 1.0
