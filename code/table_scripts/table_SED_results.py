from astropy.io import fits
import numpy as np
import h5py

import sys
sys.path.insert(0, '..')
import util


# Load in the Data
cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
bagpipes = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")
emlines = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data

# Stellar Mass Measurements
cigale_mass = cigale["bayes.stellar.m_star"]
cigale_mass_err = cigale["bayes.stellar.m_star_err"]
bagpipes_mass,bagpipes_mass_err_low,bagpipes_mass_err_up = util.stats(pow(10,np.transpose(bagpipes["samples2d"])[-5]))
print(r"Stellar Mass (M$_\odot$) & $(%0.2f\pm%0.2f) \times 10^7$ & $(%0.2f^{+%0.2f}_{-%0.2f}) \times 10^7$ & -- \\" % (cigale_mass/1e7,cigale_mass_err/1e7, bagpipes_mass/1e7, bagpipes_mass_err_up/1e7, bagpipes_mass_err_low/1e7))

# Star Formation Rates
cigale_SFR = cigale["bayes.sfr.sfr"]; cigale_SFR_err = cigale["bayes.sfr.sfr_err"]
cigale_SFR10Myrs = cigale["bayes.sfr.sfr10Myrs"]; cigale_SFR10Myrs_err = cigale["bayes.sfr.sfr10Myrs_err"]
cigale_SFR100Myrs = cigale["bayes.sfr.sfr100Myrs"]; cigale_SFR10Myrs_err = cigale["bayes.sfr.sfr100Myrs_err"]

bagpipes_SFR_pdf = util.stats(pow(10,np.transpose(bagpipes["samples2d"])[9]))