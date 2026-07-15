from data.datasets import *
def rec(site, bonus=None): return SubjectRecord("s",site,{"T1":"a","T1CE":"b","T2":"c","FLAIR":"d"},1,0,bonus)
def test_sequence_count_and_bonus_and_pool_site():
    u=UCSFDataset([rec("UCSF", {"DWI":"x"})]); assert len(u[0]["sequences"])==4; assert u[0]["bonus_modalities"]=={"DWI":"x"}
    p=PooledDataset([UPENNDataset([rec("UPENN")]), u, TCGADataset([rec("TCGA")])])
    assert [p[i]["site"] for i in range(3)] == ["UPENN","UCSF","TCGA"]
