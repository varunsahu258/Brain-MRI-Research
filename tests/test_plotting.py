from PIL import Image
import matplotlib.pyplot as plt
from utils.plotting import save_figure

def test_save_figure_both_and_dpi(tmp_path):
    fig,ax=plt.subplots(figsize=(3.5,2.5)); ax.plot([0,1],[0,1])
    paths=save_figure(fig,tmp_path/"plot",kind="both")
    assert (tmp_path/"plot.png") in paths and (tmp_path/"plot.svg") in paths
    im=Image.open(tmp_path/"plot.png"); assert im.info.get("dpi",(0,0))[0] >= 299
