import os
from os import path
import h5py as h5
import click
import numpy as np
from tqdm import tqdm
from dataprepro import csv2hdf5 as cth


def get_new_filename(filename, deltazd):
    olddir, oldbasename = path.split(filename)
    newdir = path.join(olddir, "zddiff")
    os.makedirs(newdir, exist_ok=True)
    newbasename = f"zddiff_{deltazd}_"+oldbasename
    return path.join(newdir, newbasename)

def add_zenith(deltazd, filename):
    """file is the path to the normed file, the deltazd is in degrees and then normed to the norm from dataprepro"""
    deltazdnormed = deltazd/26.
    data = h5.File(filename, "r")["data"].value
    data[::,-2] += deltazdnormed
    newfilename = get_new_filename(filename, deltazd)
    cth.create_hdf5_from_dataset(data, newfilename)
    del data
    return newfilename

def add_zeniths(infileglob, sub=-35, add=35):
    infiles = cth.glob_and_check(infileglob)
    zdrange = range(sub, add+1)
    for infile in infiles:
        for diff in tqdm(zdrange):
            print(diff)
            add_zenith(diff, infile)

def classify_and_adjust(infile, model, normfile):
    pass

def get_new_energies(adjustedglob, sub, add):
    infiles = cth.glob_and_check(adjustedglob)
    zdrange = range(sub, add+1)
    new_energies = pd.DataFrame(get_new_energies(zdrange), columns=zdrange)
