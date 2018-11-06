import ROOT
ROOT.gSystem.Load('libmars')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

SED = ['SED', 'TGraphErrors', 'SED']
LIGHT_CURVE = ['LightCurve', 'TGraphErrors', 'Light Curve']
LIGHT_CURVE_RETRIEVER = ['TLine', 'TLine', 'Light Curve']
CRAB_1409 = np.array([3.39*10**-11, -2.51, -0.21])
CRAB_1406= [3.23*10**-11, -2.47, -0.24]
CRAB_1406_ERR = [0.03*10**-11, 0.01, 0.01]
CRAB_1409_ERR = np.array([0.09*10**-11, 0.02, 0.03])

def get_disp(filename):
    f = ROOT.TFile(filename)
    return f.Get('MStatusDisplay')

def get_xy(disp, retriever=None, retlen=18):
    if retriever is None:
        retriever = SED
    messpunkte = disp.FindObjectInCanvas(*retriever)
    x = messpunkte.GetX()
    x = [x[i] for i in range(retlen)]
    y = messpunkte.GetY()
    y = [y[i] for i in range(retlen)]
    yerr = [messpunkte.GetErrorY(i) for i in range(retlen)]
    xerr = [messpunkte.GetErrorX(i) for i in range(retlen)]
    return (np.array(x), np.array(xerr),
            np.array(y), np.array(yerr))

def getref(disp, refretriever=None):
    if refretriever is None:
        refretriever = LIGHT_CURVE_RETRIEVER
    referenz = disp.FindObjectInCanvas(*refretriever)
    referenz = referenz.GetY1()
    print(referenz)
    return referenz

def get_crabref(E, f0=3.39*10**-11, a=-2.51, b=-0.21):
    """E in TeV, defaults to arxiv 1409.5594 parameters"""
    return f0*(E)**(a+(b*np.log10(E)))

def get_mean_up_low(E, params, errors, func=get_crabref):
    params = np.array(params)
    errors = np.array(errors)
    mean = func(E, *params)
    low = func(E, *list(params-errors))
    high = func(E, *list(params+errors))
    return mean, low, high

def line_and_fill(x, y, y_lower, y_upper, alpha=0.30, **kwargs):
    plot = plt.plot(x, y, **kwargs)
    color = plot[-1].get_color()
    plt.fill_between(x, y_lower, y_upper, color=color, alpha=alpha)

def ref_with_errors_timesE(E, params, errors, reffunc=get_crabref, efactor=2., **kwargs):
    """E in GeV, reffunc in TeV"""
    mean, low, high = get_mean_up_low(E, params,
                                      errors, reffunc)*np.array(E)**efactor
    line_and_fill(E*1000., mean, low, high, **kwargs)
