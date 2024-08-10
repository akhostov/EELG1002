import warnings
import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models
from astropy import units as u
from astropy.io import fits

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines,fit_generic_continuum
from specutils.fitting import estimate_line_parameters
import pdb



mask_name = ["43158747673038238","43128094491436101","41099882085246065"]
for maskname in mask_name:

	print(f"Running {maskname}")

	# Load in the 1D Spectra File
	data = fits.open(f"../data/Science_coadd/spec1d_{maskname}.fits")

	# Get the IDs
	source_id = [data[ii+1].header["MASKDEF_OBJNAME"] for ii in range(0,data[0].header["NSPEC"])]

	# Load Redshift File
	_,slitid,ra,dec,gal_id,zspec,Qf,_,_ = np.loadtxt(f'../data/specpro/{maskname}/{maskname}_zinfo.dat',unpack=True,dtype=str)

	count_serendip = 0
	# Now Let's go Through Each Source
	for jj in range(len(source_id)):
	
		jj = 4
		print(f"Running for source {source_id[jj]}")
		# Extract Spectra
		spec1d = data[jj+1].data
		lam, flux, sig = spec1d["OPT_WAVE"],spec1d["OPT_FLAM"],spec1d["OPT_FLAM_SIG"]

		# Indices of 0s towards the end of the array
		index = np.flatnonzero(lam)

		lam = lam[index]
		flux = flux[index]
		sig = sig[index]

		# Find the redshift in the specpro file
		keep = gal_id == source_id[jj]

		if source_id[jj] == "SERENDIP":
			zsource = np.double(zspec[keep][count_serendip])
			count_serendip+=1

		else:
			zsource = np.double(zspec[keep])

		# Go to the next one if zspec is -99.
		#import pdb; pdb.set_trace()
		try:
			if zsource < 0.:
				continue
		except:
			pdb.set_trace()


		lam = lam/(1.+zsource)
		flux = flux*(1.+zsource)
		sig = sig*(1.+zsource)

		keep = (lam > 3840.) & (lam < 3875.)
		lam = lam[keep]
		flux = flux[keep]
		sig = sig[keep]

		# Create a simple spectrum with a Gaussian.
		yy = flux*1e-17*(lam**2/3e18)*1e23
		spectrum = Spectrum1D(flux=yy*u.Jy, spectral_axis=lam*u.Angstrom)

		with warnings.catch_warnings():  # Ignore warnings
			warnings.simplefilter('ignore')
			g1_fit = fit_generic_continuum(spectrum)

		y_continuum_fitted = g1_fit(lam*u.Angstrom)


		# Fit the spectrum and calculate the fitted flux values (``y_fit``)
		g1_init = models.Gaussian1D(amplitude=1, mean=3726.*u.Angstrom, stddev=1.*u.Angstrom)
		g2_init = models.Gaussian1D(amplitude=3, mean=3869.*u.Angstrom, stddev=1.*u.Angstrom)

		g_fit = fit_lines(spectrum, g2_init)
		y_fit = g_fit(lam*u.Angstrom)

		# Plot the original spectrum and the fitted.
		plt.plot(lam, yy, label="Original spectrum")
		plt.plot(lam, y_fit, label="Fit result")
		plt.title('Single fit peak')
		plt.grid(True)
		plt.legend()
		plt.show()

		import pdb; pdb.set_trace()