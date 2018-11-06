import copy
from os import path
from cycler import cycler
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from uncertainties import unumpy as unp
from sklearn import metrics

# c = plt.get_cmap('tab2b').colors
CBLIND = cycler('color', ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79', 'CFCFCF'])
plt.style.use('tableau-colorblind10')
mpl.rcParams['figure.dpi']= 100
mpl.rcParams.update({'font.size': 12})
mpl.rcParams['figure.figsize'] = 5.0, 4.0
SINGLE_FIGSIZE = 5.5, 4.0
DOUBLE_FIGSIZE = 5.0, 4.0
DOUBLE_GRDS = (30, 23)
SINGLE_GRDS = (30, 30)
mpl.rcParams['figure.autolayout'] = True
SAVEDIR = "/net/nfshome/home/phoffmann/masterthesis/thesis/plots/"
DPI = 300

def set_params(single=False):
    if single is False:
        mpl.rcParams['figure.figsize'] = DOUBLE_FIGSIZE
        return DOUBLE_GRDS
    if single is True:
        mpl.rcParams['figure.figsize'] = SINGLE_FIGSIZE
        return SINGLE_GRDS

def sname(fname):
    return path.join(SAVEDIR, fname)

def get_metrics(mc, reco):
    print("\n std of standardized residuals:", np.std((mc-reco)/mc))
    print("mean of standardized residuals:", np.mean((mc-reco)/mc))
    print("correlation:", np.corrcoef(mc, reco))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(mc, reco)))
    print("MAE:", metrics.mean_absolute_error(mc, reco))

def draw_diagonal(poly):
    xlims = list(poly.axes.get_xlim())
    ylims = list(poly.axes.get_ylim())
    linex = [0,0]
    liney = [0,0]
    linex[1] = xlims[1] * 10
    liney[1] = ylims[1] * 10
    plt.plot(linex, liney, color="k", linewidth=0.35)
    poly.axes.set_ylim(ylims)
    poly.axes.set_xlim(xlims)

def wavg(group, attr_name, weight_name):
    d = group[attr_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()

def calc_mean_for_bins(poly, bins=10):
    b = poly.get_paths()
    counts = poly.get_array()
    xy = []
    for x in range(len(b)):
        xav = np.mean(b[x].vertices[0:6,0]) #center in x (RA)
        yav = np.mean(b[x].vertices[0:6,1]) #center in y (DEC)
        xy.append([xav,yav, counts[x]])

    xy = np.array(xy)
    hist = plt.hist(np.log10(xy[::,0]))
    binedges = hist[1]
    # plt.close()
    binmids = get_binmids(binedges)
    binid = np.digitize(xy[::,0], bins=10**hist[1])
    tef = pd.DataFrame(xy, columns=["x", "y", "w"], index=binid)
    ys = tef.groupby(tef.index).apply(wavg, "y", "w").values
    ys = tef.groupby(tef.index).apply(wavg, "y", "w").values
    for x, y in zip(10**binmids, ys):
        plt.plot(x,y,'k.',zorder=100)

def plot_hexbin(mc, dl, extent=None, gridsize=DOUBLE_GRDS, norm=None,
                xlabel=None, ylabel=None, diagonal=True, mean=True, mincnt=1, single=False, fname=None, **kwargs):
    gridsize = set_params(single)
    if norm is None:
        norm=LogNorm()
    if isinstance(dl, pd.core.series.Series):
        ylabel = dl.name
    elif ylabel is None:
        ylabel = "DL Energy / GeV"
    if isinstance(mc, pd.core.series.Series):
        xlabel = mc.name
    if xlabel is None:
        xlabel = "MC Energy / GeV"
    if extent is None:
        hexes = plt.hexbin(mc, dl, norm=norm, gridsize=gridsize,
                           linewidth=0.03, mincnt=mincnt, **kwargs)
    else:
        hexes = plt.hexbin(mc, dl, extent=extent, norm=norm,
                          linewidth=0.03, gridsize=gridsize, mincnt=mincnt, **kwargs)

    if diagonal is True:
        draw_diagonal(hexes)
    if mean is True:
        pass
        # calc_mean_for_bins(hexes)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if fname is not None:
        plt.savefig(sname(fname), dpi=DPI)
    return hexes

def replacenan(lut):
    arr = copy.deepcopy(lut)
    arr[arr == -1.0] = 11.
    return arr

def dropnan(array):
    array =  array[np.isnan(array) == False]
    array[np.isinf(array) == True] = 1.0
    return array

def get_hexdiff(mc, a, b, func=lambda a,b : a-b, **kwargs):
    ahex = plt.hexbin(a, mc, mincnt=0, **kwargs)
    bhex = plt.hexbin(b, mc, mincnt=0, **kwargs)
    diff = ahex.get_array() - bhex.get_array()
    # diff = func(a,b)
    plt.close()
    return diff

def plot_diffhex(mc, a, diff, fname=None, **kwargs):
    maxval = np.max(np.abs(diff.min()), np.abs(diff.max))
    vmax = maxval
    vmin = -maxval
    diffhex = plt.hexbin(a, mc, vmax=vmax, vmin=vmin, **kwargs)
    # print(func(a,b))
    diffhex.set_array(diff)
    plt.colorbar()
    if fname is not None:
        plt.savefig(sname(fname))

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
        b = np.asarray(b)
    # shuffled_a = np.empty(a.shape, dtype=a.dtype)
    # shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    # for old_index, new_index in enumerate(permutation):
        # shuffled_a[new_index] = a[old_index]
        # shuffled_b[new_index] = b[old_index]
    return a[permutation], b[permutation]

def shuffle_and_partition(a, b, n=5):
    a, b = shuffle_in_unison(a, b)
    a = a[:len(a)-(len(a)%n)]
    b = b[:len(b)-(len(b)%n)]
    return a.reshape(n, -1), b.reshape(n, -1)

def get_bias_res_wstd(mc, reco, binedges, nfolds=5):
    mc, reco = shuffle_and_partition(mc, reco, n=nfolds)
    biasreso = np.array([get_bias_res(m, r, binedges) for m, r in zip(mc, reco)])
    biases = biasreso[::,0,::]
    resolutions = biasreso[::,1,::]
    biasmean = biases.mean(axis=0)
    resolutionmean = resolutions.mean(axis=0)
    biasstd = biases.std(axis=0)
    resolutionstd = resolutions.std(axis=0)
    return unp.uarray(list(biasmean), list(biasstd)), \
           unp.uarray(resolutionmean, resolutionstd)

def get_percentiles(binnedvals, per):
    percentiles = []
    for vals in binnedvals:
        if len(vals) == 0:
            percentiles.append(0.)
        else:
            percentiles.append(np.percentile(vals, per))
    return percentiles

def _calc_bias_res(mc, reco):
    """needs binned mc and reco as list of list"""
    #from IPython import embed; embed()
    rawbias = (reco-mc)/mc
    bias = get_percentiles(rawbias, 50.)
    a = get_percentiles(rawbias, 15.9)
    b = get_percentiles(rawbias, 84.1)
    resolution = [(bx-ax)/2 for ax, bx in zip(a,b)]
    return bias, resolution

def bin_arrs(mc, reco, binedges):
    """mc and reco must have every event in the same order"""
    mc = np.array(mc)
    reco = np.array(reco)
    binnedmc = []
    binnedreco = []
    for i in range(len(binedges)-1):
        if i < len(binedges)-2:
            mask = (mc >= binedges[i])  & (mc < binedges[i+1])
        else:
            mask = (mc >= binedges[i])  & (mc <= binedges[i+1])
        binnedmc.append(mc[mask])
        binnedreco.append(reco[mask])
    return np.array(binnedmc), np.array(binnedreco)

def get_bias_res_2dzd(mcE, recoE, zeniths, binedges):
    """mcE, recoE and zeniths need to be in the eventwise same order
       returns bias and res in 2d shape (zenithbinning,energybinning)"""
    zenithedges = np.linspace(np.min(zeniths), np.max(zeniths), len(binedges))
    binnumbers = np.digitize(zeniths, zenithedges) - 1
    binmasks = [binnumbers == i for i in range(len(zenithedges)-1)]
    biasres = np.array([get_bias_res(mcE[mask], recoE[mask], binedges) for mask in binmasks])
    bias = biasres[::,0]
    res = biasres[::,1]

    return bias, res

def get_bias_res(mc, reco, binedges):
    mcbinned, recobinned = bin_arrs(mc, reco, binedges)
    bias, resolution = _calc_bias_res(mcbinned, recobinned)
    return bias, resolution

def get_binmids(binedges, log=False):
    if log is True:
        binedges = np.log10(binedges)
    middle = binedges[:-1] + np.diff(binedges)/2
    if log is True:
        middle = 10**middle
    return middle

def get_logxerr(binmiddles, binedges):
    def logmiddle(l, r):
        return np.logspace(np.log10(l), np.log10(r), num=3)[1]

    xerr = [[logmiddle(binedges[i],   binmid),
             logmiddle(binedges[i+1], binmid)]
            for i, binmid in enumerate(binmiddles)]

    return np.array(xerr).T/len(binmiddles)*3.0

def plot_errorbar(uarray, binmiddle, ylabel="", xlabel="MC Energy / GeV", xscale=None, test1=None, fname=None, **kwargs):
    size = len(uarray)
    if test1 is None:
        valrange = np.max(binmiddle) - np.min(binmiddle)
        xerr = valrange/size * 0.43 * np.ones((2, size))
    else:
        xerr = test1
        # print(xerr)
    plt.errorbar(binmiddle[:size], unp.nominal_values(uarray),
                 xerr=xerr, yerr=unp.std_devs(uarray),
                 linestyle="none", **kwargs)
    if xscale:
        plt.xscale(xscale)
    # plt.rcParams['axes.prop_cycle'] = CBLIND
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if fname is not None:
        plt.savefig(sname(fname))

def imshow(X, xedges, yedges=None, xlabel=None, ylabel=None, name=None, **kwargs):
    if yedges is None:
        yedges = [5,35]
    cmap = kwargs.pop("cmap", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if cmap is None:
        if np.any(X < 0.0):
            cmap = "RdBu_r"
            if vmax is None:
                vmax = np.max(np.abs([np.min(X), np.max(X)]))
            if vmin is None:
                vmin = -vmax
        else:
            cmap = "viridis"
            if vmin is None:
                vmin = 0.
    plt.imshow(X, extent=(xedges[0], xedges[-1],
                          yedges[0], yedges[-1]),
               interpolation="none", cmap=cmap,
               vmin=vmin, vmax=vmax,
               **kwargs)
    ax = plt.gca()
    ax.set_aspect("auto")
    if xlabel is None:
        xlabel = r"$\log _{10}(E_\mathrm{true}\, /\, \mathrm{GeV}$)"
    if ylabel is None:
        ylabel = "Zenith Angle / °"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if name is not None:
        plt.savefig(mp.sname(name), dpi=300, bbox_inches="tight")
