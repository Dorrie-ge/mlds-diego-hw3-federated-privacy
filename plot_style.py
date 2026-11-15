from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager


def apply_style():
    fonts=Path('/System/Library/Fonts/Supplemental')
    if (fonts/'Times New Roman.ttf').exists():
        for file in fonts.glob('Times New Roman*.ttf'):
            font_manager.fontManager.addfont(str(file))
        plt.rcParams['font.family']='Times New Roman'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.15,'savefig.facecolor':'white'})
