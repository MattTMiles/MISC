import numpy as np
import sys
import importlib.util
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.table import QTable, Table, Column
import pandas as pd
import astropy.time as time

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
import enterprise.constants as econst
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm
import enterprise_extensions
from enterprise_extensions import models, model_utils, hypermodel, blocks
from enterprise_extensions import timing

import gc
from scipy import stats
from scipy.stats import anderson

import argparse
import os

#sys.path.insert(0,"/home/mmiles/soft/PINT/src")
import pint
from pint.models import *

import pint.fitter
from pint.residuals import Residuals
from pint.toa import get_TOAs
import pint.logging
import pint.config

import astropy.constants as const

plt.rcParams.update({'axes.facecolor':'white'})

parser = argparse.ArgumentParser(description="Single pulsar gaussianity check.")
parser.add_argument("-pulsar", dest="pulsar", help="Pulsar to run noise analysis on", required = False)
args = parser.parse_args()
pulsar = str(args.pulsar)

def _get_ssb_lsec(toas, obs_planet):
    """Get the planet to SSB vector in lightseconds from Pint table"""
    if obs_planet not in toas.table.colnames:
        err_msg = f"{obs_planet} is not in toas.table.colnames. Either "
        err_msg += "`planet` flag is not True  in `toas` or further Pint "
        err_msg += "development to add additional planets is needed."
        raise ValueError(err_msg)
    vec = toas.table[obs_planet] + toas.table["ssb_obs_pos"]
    return (vec / const.c).to("s").value

def _get_planetssb(toas):
    planetssb = None
    """
    Currently Pint only has position vectors for:
    [Earth, Jupiter, Saturn, Uranus, Neptune]
    No velocity vectors available
    [Mercury, Venus, Mars, Pluto] unavailable pending Pint enhancements.
    """
    #if self.planets:
    planetssb = np.empty((len(toas), 9, 6))
    planetssb[:] = np.nan
    planetssb[:, 2, :3] = _get_ssb_lsec(toas, "obs_earth_pos")
    planetssb[:, 4, :3] = _get_ssb_lsec(toas, "obs_jupiter_pos")
    planetssb[:, 5, :3] = _get_ssb_lsec(toas, "obs_saturn_pos")
    planetssb[:, 6, :3] = _get_ssb_lsec(toas, "obs_uranus_pos")
    planetssb[:, 7, :3] = _get_ssb_lsec(toas, "obs_neptune_pos")

        # if hasattr(model, "ELAT") and hasattr(model, "ELONG"):
        #     for ii in range(9):
        #         planetssb[:, ii, :3] = utils.ecl2eq_vec(planetssb[:, ii, :3])
        #         # planetssb[:, ii, 3:] = utils.ecl2eq_vec(planetssb[:, ii, 3:])
    return planetssb


def theta_impact(planetssb, sunssb, pos_t):
    """
    Use the attributes of an enterprise Pulsar object to calculate the
    solar impact angle.

    ::param :planetssb Solar system barycenter time series supplied with
        enterprise.Pulsar objects.
    ::param :sunssb Solar system sun-to-barycenter timeseries supplied with
        enterprise.Pulsar objects.
    ::param :pos_t Unit vector to pulsar position over time in ecliptic
        coordinates. Supplied with enterprise.Pulsar objects.

    returns: Solar impact angle (rad), Distance to Earth (R_earth),
            impact distance (b), perpendicular distance (z_earth)
    """
    earth = planetssb[:, 2, :3]
    sun = sunssb[:, :3]
    earthsun = earth - sun
    R_earth = np.sqrt(np.einsum('ij,ij->i', earthsun, earthsun))
    Re_cos_theta_impact = np.einsum('ij,ij->i', earthsun, pos_t)

    theta_impact = np.arccos(-Re_cos_theta_impact / R_earth)
    b = np.sqrt(R_earth**2 - Re_cos_theta_impact**2)

    return theta_impact, R_earth, b, -Re_cos_theta_impact

def theta_grabber(parfile, timfile):

    psr = Pulsar(parfile, timfile, ephem="DE440")
    m, t_all = get_model_and_toas(parfile, timfile, allow_name_mixing=True, planets=True)

    glsfit = pint.fitter.GLSFitter(toas=t_all, model=m)
    glsfit.fit_toas(maxiter=10)
    mjds = glsfit.toas.get_mjds()
    which_astrometry = (
        "AstrometryEquatorial" if "AstrometryEquatorial" in m._parent.components else "AstrometryEcliptic"
    )
    pos_t = m._parent.components[which_astrometry].ssb_to_psb_xyz_ICRS(m._parent.get_barycentric_toas(t_all)).value
    
    planetssb = m._get_planetssb(t_all)
    sunssb = m._get_sunssb(t_all)

    theta, R_earth, _, _ = theta_impact(planetssb, sunssb, pos_t)
    
    
    return m, t_all, theta, R_earth


parfile = pulsar+"_tdb_noise.par"
timfile = pulsar+".tim"

m, t_all, theta, R_earth = theta_grabber(parfile,timfile)
