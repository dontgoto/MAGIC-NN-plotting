from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_hexbin(mc, dl, extent=None, gridsize=30, norm=None, **kwargs):
    if norm is None:
        norm=LogNorm()
    if extent is None:
        plt.hexbin(mc, dl, norm=norm, gridsize=gridsize, **kwargs)
    else:
        plt.hexbin(mc, dl, extent=extent, norm=norm, gridsize=gridsize, **kwargs)
    plt.colorbar()
    plt.xlabel("true E /GeV")
    plt.ylabel("DL E / GeV")
