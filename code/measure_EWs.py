import numpy as np 
import pickle
from astropy.io import fits
from scipy.interpolate import interp1d

def Calzetti(lam):
	return 2.659*( -2.156 +1.509/lam - 0.198/lam**2. + 0.011/lam**3. ) + 4.05

# Load in the data
data = fits.open("../data/emline_fits/43158747673038238/1002_lineprops_new.fits")[1].data
cigale = fits.open("../data/final_SED_results/cigale_results/1002_best_model.fits")[1].data
bagpipes = fits.open("../data/final_SED_results/best_fit_SED_sfh_continuity_spec_BPASS_nocalib.fits")[1].data
#bagpipes = fits.open("../data/final_SED_results/bagpipes_results/best_fit_SED_sfh_continuity_spec_BC03_final_w_calib_w_dust_delta_Zconst.fits")[1].data
#bagpipes = fits.open("../data/final_SED_results/bagpipes_results/best_fit_SED_sfh_continuity_spec_BC03_final_w_calib_w_dust.fits")[1].data


# Convert Wavelengths
cigale["wavelength"] *= 10. # nm to Angstrom
bagpipes["log_wave"] = pow(10,bagpipes["log_wave"]) # log-scale to linear-scale wavelength in Angstrom

# Lines
lines = np.array(["Hb_na_1","Hg_na_1","Hd_na_1","Hep_na_1","H9_1","H8_1",\
		"OIII5007c_1","OIII4959c_1","OIII4363_1","[OII]","OII3726_1","OII3728_1",\
		"NeIII3869_1","NeIII3968_1"])

# Masks
masks = [ [6790., 6833.], [6921., 7145.], [7212., 7291.] , [7422., 7548.], [7840., 7996.], [8826., 8936.], [9029., 9096.], [9120., 9184.] ]

# Initialize the mask with all True
mask_cigale = np.ones(cigale["wavelength"].shape, dtype=bool)
mask_bagpipes = np.ones(bagpipes["log_wave"].shape, dtype=bool)

# Apply the masks
for start, end in masks:
    mask_cigale &= ~((cigale["wavelength"] >= start) & (cigale["wavelength"] <= end))
    mask_bagpipes &= ~((bagpipes["log_wave"] >= start) & (bagpipes["log_wave"] <= end))

# Apply the mask to the data
cigale_wave = cigale["wavelength"][mask_cigale]
cigale_flam = cigale["Fnu"][mask_cigale]*1e-26*3e18/cigale_wave**2.

bagpipes_wave = bagpipes["log_wave"][mask_bagpipes]
bagpipes_flam = bagpipes["SED_median"][mask_bagpipes]*1e-18

# Interpolate the SED
interp_cigale = interp1d(cigale_wave,cigale_flam)
interp_bagpipes = interp1d(bagpipes_wave,bagpipes_flam)

# Now Let's Calculate the EWs

EW_cigale = (data["lineflux_med"][data["line_ID"] == "Hb_na_1"]/interp_cigale(4861.*1.8275) + data["lineflux_med"][data["line_ID"] == "OIII4959c_1"]/interp_cigale(4959.*1.8275) + data["lineflux_med"][data["line_ID"] == "OIII5007c_1"]/interp_cigale(5007.*1.8275))*1e-17/1.8275
EW_bagpipes = (data["lineflux_med"][data["line_ID"] == "Hb_na_1"]/interp_bagpipes(4861.*1.8275) + data["lineflux_med"][data["line_ID"] == "OIII4959c_1"]/interp_bagpipes(4959.*1.8275) + data["lineflux_med"][data["line_ID"] == "OIII5007c_1"]/interp_bagpipes(5007.*1.8275))*1e-17/1.8275 


#Hbeta = data["lineflux_med"][data["line_ID"] == "Hb_na_1"]*1e-17/interp_bagpipes(4861.*1.8275)*pow(10,0.4*Calzetti(0.4861)*(EBV_nebular - EBV_stellar))
#O3_4959 = data["lineflux_med"][data["line_ID"] == "OIII4959c_1"]*1e-17/interp_bagpipes(4959.*1.8275)*pow(10,0.4*Calzetti(0.4959)*(EBV_nebular - EBV_stellar))
#O3_5007 = data["lineflux_med"][data["line_ID"] == "OIII5007c_1"]*1e-17/interp_bagpipes(5007.*1.8275)*pow(10,0.4*Calzetti(0.5007)*(EBV_nebular - EBV_stellar))
#O2 = data["lineflux_med"][data["line_ID"] == "[OII]"]*pow(10,0.4*Calzetti(0.3727)*0.08)
#
#
#
#EW_bagpipes = (Hbeta + O3_4959 + O3_5007)/1.8275 

print(EW_cigale)
print(EW_bagpipes)
interp_cigale(5007.*1.8275)
interp_bagpipes(5007.*1.8275)
import pdb; pdb.set_trace()

