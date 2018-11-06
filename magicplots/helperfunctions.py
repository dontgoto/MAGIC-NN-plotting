from itertools import zip_longest
from itertools import cycle
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
from functools import partial
from magicplots import plotting as mp



CUTVALS = np.array([[5.540000e+00, 7.540000e+00, 1.024000e+01, 1.392000e+01,
        1.893000e+01, 2.573000e+01, 3.498000e+01, 4.755000e+01,
        6.463000e+01, 8.786000e+01, 1.194300e+02, 1.623500e+02,
        2.206900e+02, 3.000000e+02, 4.078100e+02, 5.543500e+02,
        7.535700e+02, 1.024360e+03, 1.392480e+03, 1.892870e+03,
        2.573090e+03, 3.497740e+03, 4.754680e+03, 6.463300e+03,
        8.785930e+03, 1.194322e+04, 1.623509e+04, 2.206927e+04,
        3.000000e+04, 4.078069e+04],
       [7.540000e+00, 1.024000e+01, 1.392000e+01, 1.893000e+01,
        2.573000e+01, 3.498000e+01, 4.755000e+01, 6.463000e+01,
        8.786000e+01, 1.194300e+02, 1.623500e+02, 2.206900e+02,
        3.000000e+02, 4.078100e+02, 5.543500e+02, 7.535700e+02,
        1.024360e+03, 1.392480e+03, 1.892870e+03, 2.573090e+03,
        3.497740e+03, 4.754680e+03, 6.463300e+03, 8.785930e+03,
        1.194322e+04, 1.623509e+04, 2.206927e+04, 3.000000e+04,
        4.078069e+04, 5.543549e+04],
       [2.000000e-01, 2.000000e-01, 5.000000e-02, 6.000000e-02,
        3.000000e-02, 4.000000e-02, 6.000000e-02, 5.000000e-02,
        4.000000e-02, 3.000000e-02, 3.000000e-02, 2.000000e-02,
        2.000000e-02, 1.000000e-02, 1.000000e-02, 1.000000e-02,
        1.000000e-02, 1.000000e-02, 1.000000e-02, 1.000000e-02,
        1.000000e-02, 1.000000e-02, 1.000000e-02, 1.000000e-02,
        1.000000e-02, 1.000000e-02, 1.000000e-02, 1.000000e-02,
        1.000000e-02, 1.000000e-02],
       [9.500000e-01, 9.500000e-01, 7.425000e-01, 9.207500e-01,
        9.500000e-01, 9.500000e-01, 8.822500e-01, 5.917500e-01,
        3.650000e-01, 3.055000e-01, 3.255000e-01, 3.300000e-01,
        3.025000e-01, 2.867500e-01, 2.670000e-01, 2.352500e-01,
        2.210000e-01, 1.880000e-01, 1.605000e-01, 1.500000e-01,
        1.500000e-01, 1.500000e-01, 1.500000e-01, 1.500000e-01,
        1.725000e-01, 2.272500e-01, 3.570000e-01, 4.997500e-01,
        3.385000e-01, 3.385000e-01]])
binmins = CUTVALS[0]
binmaxs = CUTVALS[1]
thetas = CUTVALS[2]
hads = CUTVALS[3]
bins = np.ones(len(binmins)*2)
bins[::2] = binmins
bins[1::2] = binmaxs
BINS = np.unique(bins)
binmids = mp.get_binmids(bins)


def reshaper(counts, gridsize=30):
    widths = cycle([gridsize+1, gridsize])
    length = len(counts)
    totlen = 0
    newcounts = []
    for width in widths:
        partialcounts = (counts[totlen:totlen+width])
        newcounts.append(partialcounts)
        totlen += width
        if totlen > length+1:
            break
    return np.array(newcounts)

def get_doublebins(mc, gridsize=30, valrange=(1,4), log=True):
    if log is True:
        mc = np.log10(mc)
    shortbins, _ = np.histogram(mc, range=valrange,
                                bins=gridsize)
    longbins, _ = np.histogram(mc, range=valrange,
                               bins=gridsize+1)
    return shortbins, longbins

# for some reason it doesnt work in the notebook, but when you do ipython embed it works wtf

def x_norm_hexbin(reshapedreco, binshort, binlong):
    bins = cycle([binlong, binshort])
    normedreco = []
    for i, normbins in enumerate(bins):
        normedreco.append(reshapedreco[i]/normbins)
    return normedreco

# this variant always works but produces clipping

def x_xnorm_hexbin(reshaped, norm, binlong, binshort):
    longre = np.array([re.tolist() for re in reshaped[::2]])
    shortre = np.array([re.tolist() for re in reshaped[1::2]][:-1])
    normedlong = longre/binlong
    normedshort = shortre/binshort

    normed = []
    for longb,short in zip_longest(normedlong,normedshort):
        normed.append(longb)
        if short is not None:
            normed.append(short)
    normed = np.array([i for norm in normed for i in norm])
    return normed

def makefull(length, binlong, binshort):
    widths = cycle([binlong, binshort])
    totlen = 0
    filled = []
    for width in widths:
        totlen += len(width)

        if totlen > length:
            break
        filled.append(width)
    filled = [i for fill in filled for i in fill]
    return np.array(filled)

def norm_hexbin(counts, norm, gridsize=30, valrange=(1,4), log=True):
    shortbins, longbins = get_doublebins(norm, gridsize, valrange, log)
    fullbins = makefull(len(counts), longbins, shortbins)
    return fullbins

def make_fnames_equal(fnames, qfnames):
    fnames = np.asarray(sorted(fnames))
    qfnames = np.asarray(sorted(qfnames))
    #  './melibea/lut/mergecols/GA_za05to35_8_1737992_Q_wr_mergecols.csv',
    ids = [get_id(f) for f in fnames]
    qids = [get_id(f) for f in qfnames]
    mask = np.isin(ids, qids)
    qmask = np.isin(qids, ids)
    return fnames[mask], qfnames[qmask]

def get_id(fname):
    if "._I_" in fname:
        splitter = "._I_"
    elif "_Q_" in fname:
        splitter = "_Q_"
    elif "_Y_" in fname:
        splitter = "_Y_"
    elif "_I_" in fname:
        splitter = "_I_"
    else:
        raise NotImplementedError(f"{fname}")
    if "adjusted_validate" not in fname:
        fid = fname.split(splitter)[0].split("_")[-1]
    else:
        fid = fname.split(splitter)[0]
        fid = fid.split("_")[-1]
    return fid

def geneffcut(energy, array, cutvals=hads, bins=BINS):
    """array assumes hadronnessvalues from the qframes as default, cutvals are the hadron cuts that get applied"""
    binning = np.digitize(energy, bins) - 1
    binning[binning < 0] = 0.
    binning[binning >= len(bins)-1] = 0.
    hadeffcut = np.zeros(len(energy), dtype=bool)
    for i, cutval in enumerate(cutvals):
        binmask = binning == i
        hadeffcut[binmask] = array[binmask] < cutval
    binning = np.digitize(energy, bins) - 1
    binning[binning < 0] = -1
    binning[binning >= len(bins)-1] = -1
    hadeffcut[binning == -1] = 0

    return hadeffcut

def get_fixedcuts(qframe):
    cuts = ((qframe["energy"] != -1) &
            (qframe["dispvalid"] != 0) &
            (qframe["cherdens"] != -1) &
            (qframe["size1"] > 50.) &
            (qframe["size2"] > 50.))
    return cuts

def get_mc_livetime(n_gen, γ=-1.6, f_crab=3.39*10**-11, r_max=35000, e_min=0.01, e_max=30):
    """ See S. Eineckes dissertation p. 134, all energies in TeV, lengths in cm
    """
    return n_gen*(γ+1) / (f_crab * r_max**2 * np.pi * (e_max**(γ+1) - e_min**(γ+1)))

def get_mc_weights(E, γ=-1.6, a_crab=-2.51, b_crab=-0.21):
    """ See S. Eineckes dissertation p. 134, all energies in TeV
    """
    return E**(-γ + a_crab + (b_crab*np.log10(E)))

def w_m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def w_cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - w_m(x, w)) * (y - w_m(y, w))) / np.sum(w)

def weighted_corr(x, y, w):
    """Weighted Correlation"""
    return w_cov(x, y, w) / np.sqrt(w_cov(x, x, w) * w_cov(y, y, w))

def get_mismatching_files(frame1, frame2):
    index = [val[0] for val in frame1.index.values]
    index = list(set(index))
    notequals = []
    for ind in index:
        shape1 = frame1.query(f"Filenumber == '{ind}'").shape[0]
        shape2 = frame2.query(f"Filenumber == '{ind}'").shape[0]
        if shape1 != shape2:
            print(shape1)
            print(shape2)
            print(f"not equal file {ind}")
            notequals.append(ind)
    return notequals
