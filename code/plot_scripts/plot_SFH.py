import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
import pickle

import sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
sys.path.insert(0, "..")
from colors import tableau20
import util
import os

# Define Cosmology
cosmo = util.get_cosmology()

# Only used to ensure continuity in plots. No physical meaning
def log10_with_dummy_numbers(prop, dummy=-99.):
    prop = np.log10(prop)
    prop[~np.isfinite(prop)] = dummy
    
    return prop


######################################################################
#######					  LOAD EELG1002 RESULTS			       #######
######################################################################

cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
cigale_mass = cigale["bayes.stellar.m_star"]
cigale_mass_err = cigale["bayes.stellar.m_star_err"]/(np.log(10.)*cigale_mass)
cigale_mass = np.log10(cigale_mass)
cigale_SFR10Myrs = cigale["bayes.sfh.sfr10Myrs"]
cigale_SFR10Myrs_err = cigale["bayes.sfh.sfr10Myrs_err"]
cigale_sfh = fits.open("../../data/SED_results/cigale_results/1002_SFH.fits")[1].data

bagpipes_sfh = fits.open("../../data/SED_results/bagpipes_results/best_fit_SFH_sfh_continuity_spec_BPASS.fits")[1].data
bagpipes_results = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")

# Stellar Mass
bagpipes_mass = bagpipes_results["median"][10]
bagpipes_mass_err_low,bagpipes_mass_err_up = (bagpipes_mass - bagpipes_mass), (bagpipes_results["conf_int"][1][10] - bagpipes_mass)

# Gas-Phase Metallicity
pyneb_stat = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1].data
metal_med = pyneb_stat["12+log10(O/H)_med"]-8.69

# Age at z = 0.8275
time_EELG1002 = np.array(cosmo.lookback_time(0.8275).value)


######################################################################
#######				    	Prepare the Figure			       #######
######################################################################

# Initialize the Figure
fig = plt.figure()
fig.set_size_inches(3.6,3.6)
ax = fig.add_subplot(111)

# Define Scale
ax.set_xscale("log")

# Define Limits
ax.set_xlim(6.5,13.6)

######################################################################
#######					SFH PLOT of EELG1002 				   #######
######################################################################

ax.plot(cosmo.age(0.).value - bagpipes_sfh["Lookback Time"], bagpipes_sfh["SFH_median"],color=tableau20("Blue"),ls="--")
ax.fill_between(cosmo.age(0.).value - bagpipes_sfh["Lookback Time"], bagpipes_sfh["SFH_1sig_low"], bagpipes_sfh["SFH_1sig_upp"],color=tableau20("Blue"),alpha=0.5)


ax.plot(cosmo.age(0.).value - (cosmo.age(0.8275).value - (np.max(cigale_sfh["time"]) - cigale_sfh["time"])/1e3), cigale_sfh["SFR"],color=tableau20("Red"),ls=":")

id_analog_2 = "TNG300-2_125608"
analog = pickle.load(open(f"../../data/Illustris_Analogs/{id_analog_2}_analogs_with_histories.pkl","rb"))

# Star Formation History
ax.plot(cosmo.lookback_time(analog["redshift"]).value,analog["sfr"],color=tableau20("Red"))


plt.show()