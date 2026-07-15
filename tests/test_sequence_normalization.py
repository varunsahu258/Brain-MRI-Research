from data.sequence_normalization import normalize_sequence_name
from data.datasets import UPENNDataset,UCSFDataset,TCGADataset,SubjectRecord

def test_t1gd_maps_to_t1ce_across_loaders():
    for cls,site in [(UPENNDataset,"UPENN"),(UCSFDataset,"UCSF"),(TCGADataset,"TCGA")]:
        ds=cls([SubjectRecord("s1",site,{"T1":"a","T1GD":"b","T2":"c","FLAIR":"d"})])
        assert "T1CE" in ds[0]["sequences"]
        assert normalize_sequence_name("T1GD") == "T1CE"
