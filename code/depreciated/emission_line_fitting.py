import glob, os,sys,timeit
import matplotlib
import numpy as np
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

QSOFit.set_mpl_style()

# Show the versions so we know what works
import astropy
import lmfit
import pyqsofit
print(astropy.__version__)
print(lmfit.__version__)
print(pyqsofit.__version__)

import emcee # optional, for MCMC
print(emcee.__version__)

import pdb


"""
Create parameter file
lambda    complexname  minwav maxwav linename ngauss inisca minsca maxsca inisig minsig maxsig voff vindex windex findex fvalue vary
"""

newdata = np.rec.array([

#(6564.61, r'H$\alpha$', 6400, 6800, 'Ha_br',   3, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.015, 0, 0, 0, 0.05 , 1),

#(6564.61, r'H$\alpha$', 6400, 6800, 'Ha_na',      1, 0.1, 0.0, 1e10, 1e-3, 5e-4,   0.0017, 0.01,  1, 1, 0, 0.002, 1),
#(6549.85, r'H$\alpha$', 6400, 6800, 'NII6549',    1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 1, 0.001, 1),
#(6585.28, r'H$\alpha$', 6400, 6800, 'NII6585',    1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 1, 0.003, 1),
#(6718.29, r'H$\alpha$', 6400, 6800, 'SII6718',    1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 2, 0.001, 1),
#(6732.67, r'H$\alpha$', 6400, 6800, 'SII6732',    1, 0.1, 0.0, 1e10, 1e-3, 2.3e-4, 0.0017, 5e-3,  1, 1, 2, 0.001, 1),

#(4861.333, r'H$\beta$', 4640, 5100, 'Hb_br',     2, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.01 , 1),
#(4960.30, r'H$\beta$', 4640, 5100, 'OIII4959w',   1, 0.1, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.001, 1),
#(5008.24, r'H$\beta$', 4640, 5100, 'OIII5007w',   1, 0.1, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.002, 1),
#(4687.02, r'H$\beta$', 4640, 5100, 'HeII4687_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),

#(4340.471, r'H$\gamma$', 4200, 4400, 'Hg_br',     2, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.01 , 1),

#(4101.742, r'H$\delta$', 4000, 4200, 'Hd_br',     2, 0.1, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.01, 0, 0, 0, 0.01 , 1),


#(3728.815, 'OII', 3700, 4200, 'OII3728'  , 1, 1.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 1, 1, 0, 0.001, 1),
#(3726.032, 'OII', 3700, 4200, 'OII3726'  , 1, 1.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 1, 1, 0, 0.001, 1),
#(3868.760, 'OII', 3700, 4200, 'NeIII3869', 1, 3.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 2, 2, 0, 0.001, 1),
#(3889.064, 'OII', 3700, 4200, 'H8', 	   1, 0.8, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
#(3835.391, 'OII', 3700, 4200, 'H9',        1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
#
#(3970.079, r'OII', 3700, 4200, 'Hep_na',     1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
#(3967.470, r'OII', 3700, 4200, 'NeIII3968',  1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 2, 2, 0, 0.001, 1),
#(4101.742, r'OII', 3700, 4200, 'Hd_na',      1, 0.4, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
##
##
#(4363.210, r'H$\beta$', 4200, 5100, 'OIII4363',   1, 0.2, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 1, 1, 0, 0.001, 1),
#(4340.471, r'H$\beta$', 4200, 5100, 'Hg_na',      1, 1.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
##
#(5006.843, r'H$\beta$', 4200, 5100, 'OIII5007c',   1, 6.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0,  1,  1, 0, 0.001, 1),
#(4958.911, r'H$\beta$', 4200, 5100, 'OIII4959c',   1, 3.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0,  1,  1, 0, 0.001, 1),
#(4861.333, r'H$\beta$', 4200, 5100, 'Hb_na',       1, 2.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0,  3,  3, 0, 0.001, 1)],

(3728.815, 'ALL', 3700, 5100, 'OII3728'  , 1, 1.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 1, 1, 0, 0.001, 1),
(3726.032, 'ALL', 3700, 5100, 'OII3726'  , 1, 1.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 1, 1, 0, 0.001, 1),
(3868.760, 'ALL', 3700, 5100, 'NeIII3869', 1, 3.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 2, 2, 0, 0.001, 1),
(3889.064, 'ALL', 3700, 5100, 'H8', 	   1, 0.8, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
(3835.391, 'ALL', 3700, 5100, 'H9',        1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),

(3970.079, r'ALL', 3700, 5100, 'Hep_na',     1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
(3967.470, r'ALL', 3700, 5100, 'NeIII3968',  1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 2, 2, 0, 0.001, 1),
(4101.742, r'ALL', 3700, 5100, 'Hd_na',      1, 0.4, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
#
#
(4363.210, r'ALL', 3700, 5100, 'OIII4363',   1, 0.2, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 1, 1, 0, 0.001, 1),
(4340.471, r'ALL', 3700, 5100, 'Hg_na',      1, 1.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0, 3, 3, 0, 0.001, 1),
(5006.843, r'ALL', 3700, 5100, 'OIII5007c',   1, 6.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0,  1,  1, 0, 0.001, 1),
(4958.911, r'ALL', 3700, 5100, 'OIII4959c',   1, 3.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0,  1,  1, 0, 0.001, 1),
(4861.333, r'ALL', 3700, 5100, 'Hb_na',       1, 2.0, 0.0, 1e10, 0.0005, 1e-7, 0.001, 1.0,  3,  3, 0, 0.001, 1)],



formats = 'float32,      a20,  float32, float32,      a20,  int32, float32, float32, float32, float32, float32, float32, float32,   int32,  int32,  int32,   float32, int32',
names  =  ' lambda, compname,   minwav,  maxwav, linename, ngauss,  inisca,  minsca,  maxsca,  inisig,  minsig,  maxsig,  voff,     vindex, windex,  findex,  fvalue,  vary')

# Header
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'

# Can be set to negative for absorption lines if you want
hdr['inisca'] = 'Initial guess of line scale [flux]'
hdr['minsca'] = 'Lower range of line scale [flux]'
hdr['maxsca'] = 'Upper range of line scale [flux]'

hdr['inisig'] = 'Initial guess of linesigma [lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'  
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'

hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'

hdr['vary'] = 'Whether or not to vary the line parameters (set to 0 to fix the line parameters to initial values)'

mask_na = [True if not '_br' in str(row['linename']) else False for row in newdata]

print(Table(newdata))

# Save line info
hdu = fits.BinTableHDU(data=newdata[mask_na], header=hdr, name='data')
hdu.writeto(os.path.join("../data/", 'qsopar.fits'), overwrite=True)




# Load in the 1D Spectra File
data = fits.open(f"../data/flux_corr_spectra/43158747673038238/1002.fits")

# Get the IDs
source_id = '1002'

# Load Redshift File
_,slitid,ra,dec,gal_id,zspec,Qf,_,_ = np.loadtxt(f'../data/specpro/43158747673038238/43158747673038238_zinfo.dat',unpack=True,dtype=str)

count_serendip = 0
# Now Let's go Through Each Source
for jj in range(len(source_id)):

	#jj = 4
	#print(f"Running for source {source_id[jj]}")
	# Extract Spectra
	spec1d = data[1].data
	lam, flux, sig = spec1d["OPT_WAVE"],spec1d["OPT_FLAM"],spec1d["OPT_FLAM_SIG"]

	# Indices of 0s towards the end of the array
	index = np.flatnonzero(lam)

	lam = lam[index]
	flux = flux[index]
	sig = sig[index]

	# Find the redshift in the specpro file
	keep = gal_id == source_id
	zsource = np.double(zspec[keep])


	start = timeit.default_timer()		
	q_mcmc = QSOFit(lam, flux, sig, zsource, path="../data",plateid=0.,mjd=0.,fiberid=0.)#,ra=float(ra[jj]),dec=float(dec[jj]))

	q_mcmc.Fit(name=str(source_id), nsmooth=1, deredden=True, reject_badpix=False, wave_range=(3000,5100), \
				wave_mask=None, decompose_host=False, BC03=False, Fe_uv_op=False, poly=False,\
				MCMC=True, MC=False, epsilon_jitter=0,nburn=200, nsamp=1000, nthin=1, \
				plot_fig=True,plot_corner = True, BC=False, rej_abs_conti=False, rej_abs_line=False, linefit=True, save_result=True, save_fig=True, param_file_name="../data/qsopar.fits",\
				kwargs_plot={'save_fig_path':f"../data/emline_fits/43158747673038238", 'plot_line_name':True}, save_fits_path=f"../data/emline_fits/43158747673038238",verbose=True)

	# Save PDF in a Fits File
	t = Table(q_mcmc.gauss_result_all,names=q_mcmc.gauss_result_name[::2].T)
	t.write(f"../data/emline_fits/43158747673038238/1002_pdf.fits",format="fits",overwrite=True)


	fwhm_5007, sigma_5007, ew_5007, peak_5007, area_5007, snr = q_mcmc.line_prop_from_name('OIII5007c','narrow')
	fwhm_4959, sigma_4959, ew_4959, peak_4959, area_4959, snr = q_mcmc.line_prop_from_name('OIII4959c','narrow')
	fwhm_4861, sigma_4861, ew_4861, peak_4861, area_4861, snr = q_mcmc.line_prop_from_name('Hb_na','narrow')
	fwhm_4343, sigma_4343, ew_4343, peak_4343, area_4343, snr = q_mcmc.line_prop_from_name('Hg_na','narrow')
	fwhm_4101, sigma_4101, ew_4101, peak_4101, area_4101, snr = q_mcmc.line_prop_from_name('Hd_na','narrow')
	fwhm_4363, sigma_4363, ew_4363, peak_4363, area_4363, snr = q_mcmc.line_prop_from_name('OIII4363','narrow')
	fwhm_3869, sigma_3869, ew_3869, peak_3869, area_3869, snr = q_mcmc.line_prop_from_name('NeIII3869','narrow')

	xx = np.log10(area_4363/(area_4959+area_5007))

	logTe = (3.5363 + 7.2939*xx)/(1.0000 + 1.6298*xx -0.1221*xx**2 - 0.0074*xx**3)

	print("Electron Temperature:", pow(10,logTe))
		

	fwhm_3728, sigma_3728, ew_3728, peak_3728, area_3728, snr = q_mcmc.line_prop_from_name('OII3728','narrow')
	fwhm_3726, sigma_3726, ew_3726, peak_3726, area_3726, snr = q_mcmc.line_prop_from_name('OII3726','narrow')

	print("[OII] ratio:",area_3728/area_3726)






	end = timeit.default_timer()

	print(f'Fitting finished in {np.round(end - start, 1)}s')

	pdb.set_trace()
	exit()































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


		start = timeit.default_timer()		
		q_mcmc = QSOFit(lam, flux, sig, zsource, path="../data",plateid=0.,mjd=0.,fiberid=0.)#,ra=float(ra[jj]),dec=float(dec[jj]))

		q_mcmc.Fit(name=str(source_id[jj]), nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, \
					wave_mask=None, decompose_host=False, BC03=False, Fe_uv_op=False, poly=False,\
					MCMC=True, MC=False, epsilon_jitter=0.,nburn=200, nsamp=10000, nthin=10, \
					plot_fig=True,plot_corner = True, BC=False, rej_abs_conti=False, rej_abs_line=False, linefit=True, save_result=True, save_fig=True, param_file_name="../data/qsopar.fits",\
					kwargs_plot={'save_fig_path':f"../data/emline_fits/{maskname}", 'plot_line_name':True}, save_fits_path=f"../data/emline_fits/{maskname}",verbose=True)

		# Save PDF in a Fits File
		t = Table(q_mcmc.gauss_result_all,names=q_mcmc.gauss_result_name[::2].T)
		t.write(f"../data/emline_fits/{maskname}/{source_id[jj]}_pdf.fits",format="fits",overwrite=True)





		pdb.set_trace()



		fwhm, sigma, ew, peak, area_5007, snr = q_mcmc.line_prop_from_name('OIII5007c','narrow')
		fwhm, sigma, ew, peak, area_4959, snr = q_mcmc.line_prop_from_name('OIII4959c','narrow')
		fwhm, sigma, ew, peak, area_4861, snr = q_mcmc.line_prop_from_name('Hb_na','narrow')
		fwhm, sigma, ew, peak, area_4343, snr = q_mcmc.line_prop_from_name('Hg_na','narrow')
		fwhm, sigma, ew, peak, area_4101, snr = q_mcmc.line_prop_from_name('Hd_na','narrow')
		fwhm, sigma, ew, peak, area_4363, snr = q_mcmc.line_prop_from_name('OIII4363','narrow')
		fwhm, sigma, ew, peak, area_3869, snr = q_mcmc.line_prop_from_name('NeIII3869','narrow')

		xx = np.log10(area_4363/(area_4959+area_5007))

		logTe = (3.5363 + 7.2939*xx)/(1.0000 + 1.6298*xx -0.1221*xx**2 - 0.0074*xx**3)

		print("Electron Temperature:", pow(10,logTe))
			

		fwhm, sigma, ew, peak, area_3728, snr = q_mcmc.line_prop_from_name('OII3728','narrow')
		fwhm, sigma, ew, peak, area_3726, snr = q_mcmc.line_prop_from_name('OII3726','narrow')

		print("[OII] ratio:",area_3726/area_3728)






		end = timeit.default_timer()

		print(f'Fitting finished in {np.round(end - start, 1)}s')

		pdb.set_trace()



