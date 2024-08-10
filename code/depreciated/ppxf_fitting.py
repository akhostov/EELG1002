from os import path

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
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
		#index = np.flatnonzero(lam)

		#lam = lam[index]
		#flux = flux[index]
		#sig = sig[index]

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


		R = 631/2.
		FWHM_gal = 3.884
		print( f"FWHM_gal: {FWHM_gal:.1f} Å")   # 8.5 Angstrom  

		c = 299792.458                      # speed of light in km/s
		sigma_inst = c/(R*2.355)
		print( f"sigma_inst: {sigma_inst:.0f} km/s")   # 47 km/s

		z = zsource                      # Initial estimate of the galaxy redshift
		lam /= (1 + z)               # Compute approximate restframe wavelength
		FWHM_gal /= (1 + z)     # Adjust resolution in Angstrom
		print(f"de-redshifted resolution FWHM in Å: {FWHM_gal:.1f}")

		# Only the well fit area
		keep = (lam > 3000.) & (lam < 5300.)
		lam = lam[keep]
		flux = flux[keep]
		sig = sig[keep]

		galaxy = flux/np.median(flux)       # Normalize spectrum to avoid numerical issues
		noise = np.full_like(flux, 0.05)      # Assume constant noise per pixel here. I adopt a noise that gives chi2/DOF~1

		velscale = c*np.log(lam[1]/lam[0])  # eq.(8) of Cappellari (2017)
		print(f"Velocity scale per pixel: {velscale:.2f} km/s")

		FWHM_temp = 2.51   # Resolution of E-MILES templates in the fitted range

		ppxf_dir = path.dirname(path.realpath(lib.__file__))
		pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
		miles = lib.miles(pathname, velscale, norm_range=[5070, 5300], age_range=[0, 2.2])

		reg_dim = miles.templates.shape[1:]
		stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

		lam_range_gal = [np.min(lam), np.max(lam)]
		gas_templates, gas_names, line_wave = util.emission_lines(miles.ln_lam_temp, lam_range_gal, FWHM_gal)

		templates = np.column_stack([stars_templates, gas_templates])


		### SETUP PPXF PARAMS
		c = 299792.458
		start = [1200, 200.]     # (km/s), starting guess for [V, sigma]


		n_stars = stars_templates.shape[1]
		n_gas = len(gas_names)
		component = [0]*n_stars + [1]*n_gas
		gas_component = np.array(component) > 0  # gas_component=True for gas templates

		moments = [2, 2]

		start = [start, start]

		pp = ppxf(templates, galaxy, noise, velscale, start,
					moments=moments, degree=-1, mdegree=-1, lam=lam, #lam_temp=miles.lam_temp,
					reg_dim=reg_dim, component=component, gas_component=gas_component,
					reddening=0, gas_reddening=0, gas_names=gas_names)
		#pdb.set_trace()

		### START FIT
		#pp = ppxf(templates, galaxy, noise, velscale, start,
		#		moments=moments, degree=-1, mdegree=-1, lam=lam, lam_temp=miles.lam_temp,
		#		reg_dim=reg_dim, component=component, gas_component=gas_component,
		#		reddening=0, gas_reddening=0, gas_names=gas_names)
		plt.figure(figsize=(15, 5))
		pp.plot()


		#plt.figure(figsize=(15, 5))
		#pp.plot(gas_clip=1)
		#plt.xlim([0.42, 0.52]);
		plt.show()
