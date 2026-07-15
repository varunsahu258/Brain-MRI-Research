# Brain Tumor Radiogenomics Staged Pipeline

Part 0 scaffolds shared infrastructure for staged MGMT promoter methylation and IDH mutation prediction across UPENN-GBM, UCSF-PDGM, and TCGA-GBM. The controlling metric is `pooled_CV_MGMT_AUROC - LOSO_MGMT_AUROC`.

Run audit:

```bash
python -m data.audit --config configs/base.yaml
```

Run tests:

```bash
pytest
```
