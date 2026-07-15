import json, pytest
from training.gates import evaluate_generalization_gate, assert_external_test_allowed
from training.loso_harness import generate_loso_folds, assert_loso_folds_match

def test_gap_exceeded():
    r=evaluate_generalization_gate(0.9,0.79); assert r["generalization_gap_exceeded"] is True

def test_fold_drift_detection(tmp_path):
    p=tmp_path/"loso.json"; saved=generate_loso_folds(); p.write_text(json.dumps(saved))
    drift=generate_loso_folds(); drift["folds"][0]["held_out_site"]="X"
    with pytest.raises(ValueError): assert_loso_folds_match(drift,p)

def test_external_lock(tmp_path):
    lock=tmp_path/"lock.json"; cfg={"a":1}
    with pytest.raises(PermissionError): assert_external_test_allowed(cfg,False,lock)
    assert_external_test_allowed(cfg,True,lock)
    with pytest.raises(RuntimeError): assert_external_test_allowed(cfg,True,lock)
