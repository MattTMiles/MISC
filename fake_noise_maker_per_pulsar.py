import numpy as np
import os, glob
import libstempo as T2
from libstempo import toasim as LT
from libstempo import plot as LP
from simFuncs import *
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Fake data maker.")
parser.add_argument("-noise", type = str.lower, nargs="+",dest="noise", help="Noise values to induce", \
    choices={"red", "dm"}, required=False)
parser.add_argument("-freqave", default=False, dest="freqave", help="If this is turned on, it returns a frequency averaged version instead.", action=argparse.BooleanOptionalAction)
parser.add_argument('-rand', default=False, help="If this is turned on, randomise a spectral index and amplitude", action=argparse.BooleanOptionalAction)
parser.add_argument('-out', type = str.lower, dest="out", help="full path of savefile", required=False)
parser.add_argument('-pulsar',dest="pulsar", help="Pulsar to run on", required=True)
parser.add_argument('-int', dest="int", help="integer of array to save", required=False)
args = parser.parse_args()

noise = args.noise
out = str(args.out)
pulsar = str(args.pulsar)
index = args.int


# get parfiles containing pulsar params
datadir = '/fred/oz002/users/mmiles/MPTA_GW/partim_DJR_copy/'
parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))

parfiles = [ p for p in parfiles if pulsar in p ]
timfiles = [ t for t in timfiles if pulsar in t ]

# create pulsar objects from parfiles
# default TOA errors are set to 0.4 microseconds
psrs = []
psrs = create_ideal_psrs(parfiles, timfiles=timfiles)


def weighted_average(dataframe, value, weight):
    val = dataframe[value]
    wt = dataframe[weight]

    return np.average(val, weights = 1/(np.array(wt)**2))

def quad_sum(dataframe, errs):

    errors = dataframe[errs]

    return np.sqrt(np.sum(errors**2))

if not args.freqave:
    for psr in psrs:
        data_all = []
        truth_all = []
        i=0
        print("Creating sims for: {}".format(psr.name))
        while i < 1000:
            print("Running sim number: {}".format(i+1))

            #Give pulsars noise
            for n in noise:
                if not args.rand:
                    if n == "red":
                        add_red(psr)
                    if n == "dm":
                        add_dm(psr)
                else:
                    if n == "red":
                        psrnew, amp_temp, gam_temp = add_red_singular(psr,rand=True)
                    if n == "dm":
                        add_dm(psr,rand=True)
                    
            data_temp = np.array([psrnew.toas(), psrnew.freqs.astype(np.float128), psrnew.residuals(), psrnew.toaerrs])
            truth_temp = np.array([amp_temp, gam_temp])


            data_all.append(data_temp)
            truth_all.append(truth_temp)
            i=i+1

        print("Combining and saving data for: {}".format(psr.name))
        data_all_array = np.array(data_all)
        truth_all_array = np.array(truth_all)

        if index:

            np.save("/fred/oz002/users/mmiles/ML_noise/data_new/"+psr.name+"_data_{}.npy".format(int(index)), data_all_array)
            np.save("/fred/oz002/users/mmiles/ML_noise/data_new/"+psr.name+"_truth_{}.npy".format(int(index)), truth_all_array)

        else:

            np.save("/fred/oz002/users/mmiles/ML_noise/data_new/"+psr.name+"_data.npy", data_all_array)
            np.save("/fred/oz002/users/mmiles/ML_noise/data_new/"+psr.name+"_truth.npy", truth_all_array)

elif args.freqave:
    for psr in psrs:
        data_all = []
        truth_all = []
        i=0
        print("Creating sims for: {}".format(psr.name))
        while i < 1000:
            print("Running sim number: {}".format(i+1))

            #Give pulsars noise
            for n in noise:
                if not args.rand:
                    if n == "red":
                        add_red(psr)
                    if n == "dm":
                        add_dm(psr)
                else:
                    if n == "red":
                        psrnew, amp_temp, gam_temp = add_red_singular(psr,rand=True)
                    if n == "dm":
                        add_dm(psr,rand=True)

            df = pd.DataFrame(np.array([psrnew.toas(), psrnew.residuals(), psrnew.toaerrs]).T, columns=["mjd", "res", "err"])
            df["rounded mjd"] = df.mjd.round().values

            res_ave = df.groupby("rounded mjd").apply(weighted_average, "res", "err")
            err_ave = df.groupby("rounded mjd").apply(quad_sum,"err")

            data_temp = np.array([res_ave.index, res_ave.values, err_ave.values])

            #data_temp = np.array([psrnew.toas(), psrnew.freqs.astype(np.float128), psrnew.residuals(), psrnew.toaerrs])
            truth_temp = np.array([amp_temp, gam_temp])


            data_all.append(data_temp)
            truth_all.append(truth_temp)
            i=i+1

        print("Combining and saving data for: {}".format(psr.name))
        data_all_array = np.array(data_all)
        truth_all_array = np.array(truth_all)

        if index:

            np.save("/fred/oz002/users/mmiles/ML_noise/data_F/"+psr.name+"_data_{}.npy".format(int(index)), data_all_array)
            np.save("/fred/oz002/users/mmiles/ML_noise/data_F/"+psr.name+"_truth_{}.npy".format(int(index)), truth_all_array)

        else:

            np.save("/fred/oz002/users/mmiles/ML_noise/data_F/"+psr.name+"_data.npy", data_all_array)
            np.save("/fred/oz002/users/mmiles/ML_noise/data_F/"+psr.name+"_truth.npy", truth_all_array)

