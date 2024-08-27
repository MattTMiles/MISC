import numpy as np
import matplotlib.pyplot as plt
import os, glob
import libstempo as T2
from libstempo import toasim as LT
from libstempo import plot as LP
import enterprise
from enterprise.pulsar import Pulsar
from simFuncs import *
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Fake data maker.")
parser.add_argument("-noise", type = str.lower, nargs="+",dest="noise", help="Noise values to induce", \
    choices={"red", "dm"}, required=False)
parser.add_argument('--rand', default=False, help="If this is turned on, randomise a spectral index and amplitude", action=argparse.BooleanOptionalAction)
parser.add_argument('--out', type = str.lower, dest="out", help="full path of savefile", required=True)
args = parser.parse_args()

noise = args.noise
out = str(args.out)

# get parfiles containing pulsar params
datadir = '/fred/oz002/users/mmiles/MPTA_GW/partim_DJR_copy/'
parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))

# create pulsar objects from parfiles
# default TOA errors are set to 0.4 microseconds
psrs = []
psrs = create_ideal_psrs(parfiles, timfiles=timfiles)

#Give pulsars noise
for n in noise:
    if not args.rand:
        if n == "red":
            add_red(psrs)
        if n == "dm":
            add_dm(psrs)
    else:
        if n == "red":
            add_red(psrs,rand=True)
        if n == "dm":
            add_dm(psrs,rand=True)


#Save the pulsars as MJD, frequencies residuals

#toa_all, freq_all, res_all = [], [], []
data_all = []
for psr in psrs:
    psrname = psr.name
    toas = psr.toas()
    freq = psr.freqs.astype(np.float128)
    res = psr.residuals()
    psrname_list = [psrname]*len(toas)
    data = np.array([psrname_list, toas,freq,res])
    data_all.append(data.T)

df = pd.DataFrame(np.row_stack(data_all),columns=["psr","toas","freqs","res"])
df.to_pickle(out)