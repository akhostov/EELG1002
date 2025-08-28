import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.integrate import simps
import os
from astropy.convolution import Gaussian2DKernel,convolve
import util


def get_fluxcorr():

	FWHM_ground = 0.9/0.03 #0.9'' seeing with a 0.03''/pix scale
	FWHM_F814W = 0.09/0.03 # arcsec
	FWHM = np.sqrt(FWHM_ground**2. - FWHM_F814W**2.)
	gaussian_2D_kernel = Gaussian2DKernel(FWHM/2.634) #0.9'' seeing with a 0.03''/pix scale

	# Load the image
	data = fits.open("../data/cutouts/ACS/1002_HST_ACS_F814W_10arcsec_unrot_sci.fits")

	# Now let's do Aperture Photometry with Sextractor
	os.system("cd ../data/cutouts/ACS/; sex 1002_HST_ACS_F814W_10arcsec_unrot_sci.fits -c original_HST.sex")

	# Get the original flux
	original_flux = util.ab_mag_to_fnu(fits.open("../data/cutouts/ACS/original_HST.fits")[1].data["MAG_APER"],unit="uJy")

	# Range of the GMOS Slit
	dx = (168-round(0.25/0.03),168+round(0.25/0.03))
	dy = (168-round(2.5/0.03),168+round(2.5/0.03))

	# Smooth out the images
	data[0].data = convolve(data[0].data, gaussian_2D_kernel)
	data.writeto("../data/cutouts/ACS/1002_HST_ACS_F814W_10arcsec_unrot_sci_ground_PSF.fits",overwrite=True)

	# Calculate Original Flux
	mag_new = util.fnu_to_ab_mag(np.sum(data[0].data[dx[0]:dx[1],dy[0]:dy[1]]),unit="custom",ZP=25.936)
	flux_new = util.ab_mag_to_fnu(mag_new,unit="uJy")

	# Calculate the Correction Factor
	corr_factor = 1. - flux_new/original_flux

	print(f"Original Flux: {original_flux} uJy")
	print(f"Smoothed Flux: {flux_new} uJy")
	print(f"Correction: {corr_factor}")

	return corr_factor


def apply_correction():

	# Load in the Spectra
	spec1d = fits.open("../data/Science_coadd/spec1d_43158747673038238.fits")
	hdr = spec1d[5].header
	spec1d = spec1d[5].data

	# Get the Slitloss Correction Factor
	corr_factor = get_fluxcorr()

	spec1d["OPT_FLAM"] = spec1d["OPT_FLAM"]*(1.+corr_factor)
	spec1d["OPT_FLAM_SIG"] = spec1d["OPT_FLAM_SIG"]*(1.+corr_factor)

	# Write this out
	table_hdu = fits.BinTableHDU(spec1d)
	table_hdu.header = hdr

	new_hdulist = fits.HDUList([fits.PrimaryHDU(), table_hdu])
	new_hdulist.writeto("../data/flux_corr_spectra/EELG1002_1dspec_slitloss_corrected.fits", overwrite=True)

def main():
	get_fluxcorr()
	apply_correction()

if __name__ == "__main__":
	main()