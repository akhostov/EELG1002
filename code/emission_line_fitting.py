import os
import timeit
import numpy as np
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
from astropy.table import Table
import warnings
warnings.filterwarnings("ignore")

QSOFit.set_mpl_style()

import astropy
import lmfit
import pyqsofit
import emcee 

from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
import pickle
import h5py

import util

from tqdm import trange
import time
from functools import cached_property

class physical_PyQSOfit():

	def __init__(self, scale , cwave , sigma , redshift , ref_line , scale_err=None, cwave_err=None , sigma_err=None):
		self.scale = np.double(scale)
		self.cwave = np.double(cwave)
		self.sigma = np.double(sigma)
		self.scale_err = np.double(scale_err)
		self.cwave_err = np.double(cwave_err)
		self.sigma_err = np.double(sigma_err)
		self.redshift = np.double(redshift)
		self.ref_line = np.double(ref_line)

		self.wavelength()

	# Define the Gaussian Function for Line Profile
	@cached_property
	def gaussian(self):
		return np.sum(self.scale*np.exp(-0.5*((self.xx[:,np.newaxis] - self.cwave)/self.sigma)**2.),axis=1)

	def wavelength(self):
		left  = np.min(self.cwave - 100.*self.sigma)
		right = np.max(self.cwave + 100.*self.sigma)
		disp = 1e-4*np.log(10)
		npix = int((right - left)/disp)
		self.xx = np.linspace(left, right, npix)
		self.exp_xx = np.exp(self.xx)


	# Measure the Line Flux
	def line_flux(self):
		yy = self.gaussian
		self.lflux = simps(yy,x=self.exp_xx)
		return self.lflux

	# Measure the Dispersion
	def line_sigma(self,kms=True):
		
		lambda1 = simps(self.gaussian*self.exp_xx ,x=self.exp_xx )
		lambda2 = simps(self.gaussian*pow(self.exp_xx,2.),x=self.exp_xx)

		if kms:
			return np.sqrt(lambda2/self.lflux - (lambda1/self.lflux)**2)/np.exp(self.cwave)*3e5
		else:
			return np.sqrt(lambda2/self.lflux - (lambda1/self.lflux)**2)

	# Measure the FWHM
	def line_FWHM(self,kms=True):
		spline = UnivariateSpline(self.xx, self.gaussian - np.max(self.gaussian)/2, s=0)
		fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()

		if kms:
			return abs(np.exp(fwhm_left) - np.exp(fwhm_right))/np.exp(self.cwave)*3e5

		else:
			return abs(np.exp(fwhm_left) - np.exp(fwhm_right))


	# Measure the Redshifts for each line
	def line_redshift(self):
		#return np.exp(self.cwave)*(1.+self.redshift)/(self.ref_line) -1.
		return np.exp(self.cwave)*(1.+self.redshift)/self.ref_line*(1.+self.redshift)-1.
	
	# Measure the Errors
	def error_estimation(self,nsamples=2000):

		# Store the Best-Fit Parameters
		lflux = self.line_flux()
		lsigma = self.line_sigma()
		lfwhm = self.line_FWHM()
		lzred = self.line_redshift()

		rand_scale = np.random.normal(loc=self.scale,scale=self.scale_err,size=nsamples)
		rand_sigma = np.random.normal(loc=self.sigma,scale=self.sigma_err,size=nsamples)
		rand_cwave = np.random.normal(loc=self.cwave,scale=self.cwave_err,size=nsamples)

		pdf_lflux, pdf_lsigma, pdf_lfwhm, pdf_lzred = [],[],[],[]
		for ii in range(nsamples):
			self.scale = rand_scale[ii]
			self.sigma = rand_sigma[ii]
			self.cwave = rand_cwave[ii]

			pdf_lflux.append(self.line_flux())
			pdf_lsigma.append(self.line_sigma())
			pdf_lfwhm.append(self.line_FWHM())
			pdf_lzred.append(self.line_redshift())


		# Calculate the Percentiles
		lflux_elow,lflux_eupp = (lflux - np.percentile(pdf_lflux,16.) , np.percentile(pdf_lflux,84.) - lflux)
		lsigma_elow,lsigma_eupp = (lsigma - np.percentile(pdf_lsigma,16.) , np.percentile(pdf_lsigma,84.) - lsigma)
		lfwhm_elow,lfwhm_eupp = (lfwhm - np.percentile(pdf_lfwhm,16.) , np.percentile(pdf_lfwhm,84.) - lfwhm)
		lzred_elow,lzred_eupp = (lzred - np.percentile(pdf_lzred,16.) , np.percentile(pdf_lzred,84.) - lzred)

		return (lflux,lflux_elow,lflux_eupp,
				lsigma,lsigma_elow,lsigma_eupp,
				lfwhm,lfwhm_elow,lfwhm_eupp,
				lzred,lzred_elow,lzred_eupp)







import numpy as np
from scipy.integrate import simps

import numpy as np
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline

def vectorized_line_model_general(
    lam,                         # linear wavelength grid (1D)
    scale1, cwave1, sigma1,      # first Gaussian (log-lambda centers!)
    scale2=None, cwave2=None, sigma2=None,  # optional second Gaussian
    redshift_guess=0.0,
    ref_wave_rest=None,          # float or tuple of two floats
    return_model=False):
    """
    Generate a single or double Gaussian line model on a linear wavelength grid.
    
    Parameters:
    -----------
    lam : 1D array
        Linear wavelength array.
    scale1, cwave1, sigma1 : 1D arrays
        Amplitude, central wavelength, and sigma (in angstroms) for the first Gaussian.
    scale2, cwave2, sigma2 : optional 1D arrays
        Parameters for second Gaussian (used for doublets).
    redshift_guess : float
        Assumed redshift for computing fine-tuned redshift.
    ref_wave_rest : float or tuple
        Rest-frame reference wavelength(s).
    return_model : bool
        Whether to return the full model array.

    Returns:
    --------
    dict with keys:
        - model : (N, len(lam)) array if return_model=True, else None
        - flux : total flux
        - sigma_kms : velocity dispersion
        - FWHM_kms : full width at half max
        - redshift : fine-tuned redshift per line
    """

    # Ensure inputs are arrays
    scale1 = np.atleast_1d(scale1)
    cwave1 = np.atleast_1d(cwave1)
    sigma1 = np.atleast_1d(sigma1)

    N = len(scale1)
    lam = np.atleast_1d(lam)

    # First Gaussian
    g1 = scale1[:, None] * np.exp(-0.5 * ((lam[None, :] - cwave1[:, None]) / sigma1[:, None]) ** 2)

    if scale2 is not None and cwave2 is not None and sigma2 is not None:
        # Double Gaussian
        scale2 = np.atleast_1d(scale2)
        cwave2 = np.atleast_1d(cwave2)
        sigma2 = np.atleast_1d(sigma2)
        g2 = scale2[:, None] * np.exp(-0.5 * ((lam[None, :] - cwave2[:, None]) / sigma2[:, None]) ** 2)
        model = g1 + g2
    else:
        g2 = None
        model = g1
    
    # Total flux
    flux = simps(model, x=np.exp(lam), axis=1)

    # Compute sigma (std dev) in wavelength space
    lambda1 = simps(model * np.exp(lam[None, :]), x=np.exp(lam), axis=1)
    lambda2 = simps(model * np.exp(lam[None, :]) ** 2, x=np.exp(lam), axis=1)
    sigma_lam = np.sqrt(lambda2 / flux - (lambda1 / flux) ** 2)

    # Velocity dispersion in km/s
    mu = lambda1 / flux
    sigma_kms = sigma_lam / mu * 3e5

    # FWHM
    fwhm_kms = 2. * np.sqrt(2. * np.log(2.)) * sigma_kms

    # Fine-tuned redshift
    if g2 is not None:
        if isinstance(ref_wave_rest, (tuple, list, np.ndarray)) and len(ref_wave_rest) == 2:
            ref1, ref2 = ref_wave_rest
            mu1 = lambda1 / flux
            mu2 = simps(g2 * lam[None, :], x=lam, axis=1) / simps(g2, x=lam, axis=1)
            z1 = mu1 / ref1 * (1 + redshift_guess) - 1
            z2 = mu2 / ref2 * (1 + redshift_guess) - 1
            redshift = (z1 + z2) / 2
        else:
            raise ValueError("Double Gaussian requires two reference rest-frame wavelengths.")
    else:
        if isinstance(ref_wave_rest, (float, int)):
            mu1 = lambda1 / flux
            redshift = mu1 / ref_wave_rest * (1 + redshift_guess) - 1
        else:
            raise ValueError("Single Gaussian requires one reference rest-frame wavelength.")

    return {
        "model": model if return_model else None,
        "flux": flux,
        "sigma_kms": sigma_kms,
        "FWHM_kms": fwhm_kms,
        "redshift": redshift
    }



def extrap_cont(sed_wave,sed_flam,central_line,left_window,right_window):
	keep_cont_blue = (sed_wave > central_line + left_window[0]) & (sed_wave < central_line + left_window[1])
	keep_cont_red  = (sed_wave > central_line + right_window[0]) & (sed_wave < central_line + right_window[1])

	cont_flux = sed_flam[keep_cont_blue+keep_cont_red]
	cont_wave = sed_wave[keep_cont_blue+keep_cont_red]

	return UnivariateSpline(cont_wave,cont_flux)


def post_process_emline_fits(line_meas,redshift,obs_line_wave,cont_cigale,cont_bagpipes,line_disp_thermal,
							line_results,EW_results,stats_redshift,stats_lflux,stats_ew_cigale,stats_ew_bagpipes,stats_lsigma_obs,stats_lsigma_int,
							line_name):

	line_results[f"{line_name}_lflux_pdf"] = np.asarray(line_meas['flux'])
	stats_redshift.append( util.stats(line_meas['redshift']) )
	stats_lflux.append( util.stats(line_meas['flux']*1e-17) )
	stats_lsigma_obs.append( util.stats(line_meas['sigma_kms']) )

	line_disp_inst = (2.5/2.355) * 3.88/obs_line_wave * 3e5 # km/s
	stats_lsigma_int.append( util.stats(np.sqrt( line_meas['sigma_kms']**2. - line_disp_inst**2. - line_disp_thermal**2. )) )
	
	pdf_ew_cigale = line_meas['flux']*1e-17/((1.+redshift)*cont_cigale)
	pdf_ew_bagpipes = line_meas['flux']*1e-17/((1.+redshift)*cont_bagpipes)

	stats_ew_cigale.append( util.stats(pdf_ew_cigale) )
	stats_ew_bagpipes.append( util.stats(pdf_ew_bagpipes) )

	EW_results[f"{line_name}_EW_cigale_pdf"] = np.asarray(pdf_ew_cigale)
	EW_results[f"{line_name}_EW_bagpipes_pdf"] = np.asarray(pdf_ew_bagpipes)

	return (line_results,EW_results,stats_redshift,stats_lflux,stats_ew_cigale,stats_ew_bagpipes,stats_lsigma_obs,stats_lsigma_int)







def run_pyqsofit():

	# Initialize the Fits Setup File for PyQSOFit

	"""
	Create parameter file
	lambda    complexname  minwav maxwav linename ngauss inisca minsca maxsca inisig minsig maxsig voff vindex windex findex fvalue vary
	"""

	newdata = np.rec.array([
			(5006.843, 'ALL', 3700, 5100, 'OIII5007c',   1, 6.0, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 3, 3, 0, 0.001, 1),
			(4958.911, 'ALL', 3700, 5100, 'OIII4959c',   1, 3.0, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 3, 3, 0, 0.001, 1),
			(4861.333, 'ALL', 3700, 5100, 'Hb_na',       1, 2.0, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 4, 4, 0, 0.001, 1),
			(3728.815, 'ALL', 3700, 5100, 'OII3728'  ,   1, 5.0, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 3, 5, 0, 0.001, 1),
			(3726.032, 'ALL', 3700, 5100, 'OII3726'  ,   1, 5.0, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 3, 5, 0, 0.001, 1),
			(3868.760, 'ALL', 3700, 5100, 'NeIII3869',   1, 3.0, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 2, 2, 1, 1.000, 1),
			(3889.064, 'ALL', 3700, 5100, 'H8', 	     1, 0.8, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 4, 4, 0, 0.001, 1),
			(3835.391, 'ALL', 3700, 5100, 'H9',          1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 4, 4, 0, 0.001, 1),
			(3970.079, 'ALL', 3700, 5100, 'Hep_na',      1, 0.5, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 4, 4, 0, 0.001, 1),
			(3967.470, 'ALL', 3700, 5100, 'NeIII3968',   1, 0.9, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 2, 2, 1, 0.301, 1),
			(4101.742, 'ALL', 3700, 5100, 'Hd_na',       1, 0.4, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 4, 4, 0, 0.001, 1),
			(4363.210, 'ALL', 3700, 5100, 'OIII4363',    1, 0.2, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 3, 3, 0, 0.001, 1),
			(4340.471, 'ALL', 3700, 5100, 'Hg_na',       1, 1.0, 0.0, 1e10, 0.0005, 1e-7, 1e-3, 1.0, 4, 4, 0, 0.001, 1)],

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






	######################################################################################
	##########						FITTING PROCESS HERE						##########	
	######################################################################################
	# Load in the 1D Spectra File
	data = fits.open(f"../data/flux_corr_spectra/EELG1002_1dspec_slitloss_corrected.fits")

	# Get the IDs
	source_id = '1002'

	# Load Redshift File
	_,slitid,ra,dec,gal_id,zspec,Qf,_,_ = np.loadtxt(f'../data/specpro/43158747673038238/43158747673038238_zinfo.dat',unpack=True,dtype=str)


	# Extract Spectra
	spec1d = data[1].data
	mask = (spec1d["OPT_WAVE"] > 6300.) & (spec1d["OPT_WAVE"] < 9400.)
	spec1d = spec1d[mask]
	lam, flux, sig = spec1d["OPT_WAVE"],spec1d["OPT_FLAM"],spec1d["OPT_FLAM_SIG"]
	print(f"Limited Spectrum to {lam.min():0.2f} - {lam.max():0.2f} A")

	# Indices of 0s towards the end of the array
	index = np.flatnonzero(lam)
	lam = lam[index]
	flux = flux[index]
	sig = sig[index]

	# Find the redshift in the specpro file
	keep = gal_id == source_id
	zsource = np.double(zspec[keep])

	# Fit via MCMC
	start = timeit.default_timer()		
	q_mcmc = QSOFit(lam, flux, sig, zsource, path="../data",plateid=0.,mjd=0.,fiberid=0.)
	q_mcmc.Fit(name=str(source_id), nsmooth=1, deredden=True, reject_badpix=False, wave_range=(3000,5100), \
				wave_mask=None, decompose_host=False, BC03=False, Fe_uv_op=False, poly=False,\
				MCMC=True, MC=False, epsilon_jitter=0,nburn=3000, nsamp=15000, nthin=1, \
				plot_fig=True, plot_corner = True, BC=False, rej_abs_conti=False, rej_abs_line=False, linefit=True, save_result=True, save_fig=True, param_file_name="../data/qsopar.fits",\
				kwargs_plot={'save_fig_path':f"../data/emline_fits/", 'plot_line_name':True}, save_fits_path=f"../data/emline_fits/",verbose=True,kwargs_line_emcee={'nwalkers':100})

	# Save PDF in a Fits File
	t = Table(q_mcmc.gauss_result_all,names=q_mcmc.gauss_result_name[::2].T)
	t.write(f"../data/emline_fits/1002_pdf.fits",format="fits",overwrite=True)

	# Calculate Total Time
	end = timeit.default_timer()
	print(f'Fitting finished in {np.round(end - start, 1)}s')




def line_props(sed_module="cigale"):

	# Initialize Main Dictionary
	line_results = {}
	EW_results = {}

	### READ IN THE DATA
	path = "../data/emline_fits/"
	fname = "1002_pdf.fits"
	data = fits.open(f"{path}/{fname}")[1].data

	### GET THE LINES I USED IN PYQSOFIT
	qsopar = fits.open("../data/qsopar.fits")[1].data
	line_id = qsopar["linename"]
	line_wave = qsopar["lambda"]



	### TEMP: ASSIGNING REDSHIFT
	redshift = 0.8275
	
	# Define Thermal Broadening
	# Note the 19500 K is made after actually measuring the Electron Temperature
	line_disp_thermal = np.sqrt(8.247e-3 * 19500.)

	# This is where we will store all the line fit information
	stats_lflux,stats_ew_cigale,stats_ew_bagpipes,stats_lsigma_obs,stats_lsigma_int,stats_redshift = [],[],[],[],[],[]

	# Now Let's get the SEDs so we can use the continuum flux density to make measurements of the the emission line equivalent width	
	cigale = fits.open("../data/SED_results/cigale_results/1002_best_model.fits")[1].data
	cigale_flam = cigale["fnu"]*1e-26*3e18/pow(cigale["wavelength"]*10,2.)
	cigale_wave = cigale["wavelength"]*10.
	
	bagpipes = fits.open("../data/SED_results/bagpipes_results/best_fit_SED_sfh_continuity_spec_BPASS.fits")[1].data
	bagpipes_wave = pow(10,bagpipes["log_wave"])
	bagpipes_flam = bagpipes["SED_median"]*1e-18
	
	
	# O2 Doublet will be done first
	these_lines = [ll for ll in line_id if "OII3" in ll]

	scales = np.array([data[f"{ll}_1_scale"] for ll in these_lines]).T
	cwaves = np.array([data[f"{ll}_1_centerwave"] for ll in these_lines]).T
	sigmas = np.array([data[f"{ll}_1_sigma"] for ll in these_lines]).T

	left_window = (-20.*(1.+redshift),-10.*(1.+redshift))
	right_window = (10.*(1.+redshift),20.*(1.+redshift))	
	fx_cigale = extrap_cont(sed_wave=cigale_wave,sed_flam=cigale_flam,central_line=3727.*(1.+redshift),left_window=left_window,right_window=right_window)
	fx_bagpipes = extrap_cont(sed_wave=bagpipes_wave,sed_flam=bagpipes_flam,central_line=3727.*(1.+redshift),left_window=left_window,right_window=right_window)

	#pdf_lflux, pdf_ew_cigale, pdf_ew_bagpipes, pdf_lsigma_int, pdf_lsigma_obs, pdf_redshift = [],[],[],[],[],[]

	############### [OII] Doublet
	start = time.time()
	print("Running for line: [OII] Doublet")
	oii_doublet = vectorized_line_model_general(np.log(np.linspace(3700,3750,100)),scale1=scales.T[0],cwave1=cwaves.T[0],sigma1=sigmas.T[0],scale2=scales.T[1],cwave2=cwaves.T[1],sigma2=sigmas.T[1],redshift_guess=redshift,ref_wave_rest=(3727.0,3729.0),return_model=True)
	obs_line_wave = 3727.*(1.+redshift)
	cont_cigale = fx_cigale(obs_line_wave)
	cont_bagpipes = fx_bagpipes(obs_line_wave)	
	
	line_results,EW_results,stats_redshift,stats_lflux,\
		stats_ew_cigale,stats_ew_bagpipes,\
		stats_lsigma_obs,stats_lsigma_int = post_process_emline_fits(oii_doublet,redshift,obs_line_wave,cont_cigale,cont_bagpipes,line_disp_thermal*np.sqrt(16.),\
						  												line_results,EW_results,stats_redshift,stats_lflux,stats_ew_cigale,stats_ew_bagpipes,stats_lsigma_obs,stats_lsigma_int,
																		line_name="OII") # note the sqrt(16.) is because of Oxygen
	print(f"Time taken for [OII] Doublet: {time.time()-start:.4f} seconds")

	# This defines the Windows that will be used for the EW measurement
	window_definitions = {
		"OII3728":     ((-20, -10), (10, 20)),
		"OII3726":     ((-20, -10), (10, 20)),
		"Hb_na":       ((-20, -10), (10, 20)),
		"HeII4687_na":   ((-20, -10), (10, 20)),
		"Hd_na":       ((-20, -10), (10, 20)),
		"HeI4143":       ((-20, -10), (10, 20)),
		"NeIII3968":   ((-20, -10), (10, 20)),
		"Hep_na":      ((-20, -10), (10, 20)),
		"HeI4026":       ((-20, -10), (10, 20)),
		"H9":          ((-20, -10), (10, 20)),
		"OIII4959c":   ((-35, -15), (15, 35)),
		"OIII5007c":   ((-35, -15), (15, 35)),
		"Hg_na":       ((-20, -10), (35, 45)),
		"NeIII3869":   ((-20, -10), (35, 45)),
		"OIII4363":    ((-15, -6), (10, 20)),
		"H8":          ((-45, -35), (10, 20)),
		"HeI3889":       ((-45, -35), (10, 20))
	}


	### NOW RUN THROUGH EACH LINE
	for lid,lwave in zip(line_id,line_wave):
		start = time.time()
		print("Running for line: ", lid)

		# Extract the PyQSOFit results
		scales = data[f"{lid}_1_scale"]
		cwaves = data[f"{lid}_1_centerwave"]
		sigmas = data[f"{lid}_1_sigma"]

		obs_line_wave = lwave*(1.+redshift)

		# Define the windows
		if lid in window_definitions:
			blue_range, red_range = window_definitions[lid]
			
			# Convert Windows to Observer Frame
			blue_range = tuple([bb*(1.+redshift) for bb in blue_range])
			red_range = tuple([bb*(1.+redshift) for bb in red_range])

			fx_cigale = extrap_cont(sed_wave=cigale_wave,sed_flam=cigale_flam,central_line=obs_line_wave,left_window=blue_range,right_window=red_range)
			fx_bagpipes = extrap_cont(sed_wave=bagpipes_wave,sed_flam=bagpipes_flam,central_line=obs_line_wave,left_window=blue_range,right_window=red_range)
		else:
			raise ValueError(f"Unknown line identifier: {lid}")

		# Get the continuum flux density about the emission line
		cont_cigale = fx_cigale(obs_line_wave)
		cont_bagpipes = fx_bagpipes(obs_line_wave)

		line_of_interest = vectorized_line_model_general(np.log(np.linspace(lwave-25,lwave+25,100)),scale1=scales.T,cwave1=cwaves,sigma1=sigmas.T,redshift_guess=redshift,ref_wave_rest=float(lwave))

		# Get the mass factor for the line to augment the thermal broadening
		if "OI" in lid:
			mass_factor = np.sqrt(16.)
		if "NeI" in lid:
			mass_factor = np.sqrt(20.18)
		else:
			mass_factor = 1.
		
		line_results,EW_results,stats_redshift,stats_lflux,\
			stats_ew_cigale,stats_ew_bagpipes,\
			stats_lsigma_obs,stats_lsigma_int = post_process_emline_fits(line_of_interest,redshift,obs_line_wave,cont_cigale,cont_bagpipes,line_disp_thermal*mass_factor,\
																			line_results,EW_results,stats_redshift,stats_lflux,stats_ew_cigale,stats_ew_bagpipes,stats_lsigma_obs,stats_lsigma_int,
																			line_name=lid) 

		print(f"Time taken for {lid}: {time.time()-start:.4f} seconds")

	# WRITE EVERYTHING OUT
	line_id = line_id.tolist()
	line_id.insert(0,"[OII]")
	line_id = np.asarray(line_id)

	final = np.column_stack((line_id,stats_redshift,stats_lflux,stats_ew_cigale,stats_ew_bagpipes,stats_lsigma_obs,stats_lsigma_int))
	names = np.array(["line_ID",
						"linez_med","linez_elow","linez_eupp",
						"lineflux_med","lineflux_elow","lineflux_eupp",
						"lineEW_Cigale_med","lineEW_Cigale_elow","lineEW_Cigale_eupp",
						"lineEW_Bagpipes_med","lineEW_Bagpipes_elow","lineEW_Bagpipes_eupp",
						"linesigma_obs_med","linesigma_obs_elow","linesigma_obs_eupp",
						"linesigma_int_med","linesigma_int_elow","linesigma_int_eupp"])
	dtypes = np.array(["str",
						"float","float","float",
						"float","float","float",
						"float","float","float",
						"float","float","float",
						"float","float","float",
						"float","float","float"])

	t = Table(final,names=names,dtype=dtypes)
	t.write("../data/emline_fits/1002_lineprops.fits",format="fits",overwrite=True)

	# Output Dictionary
	with open("../data/emline_fits/1002_lineflux_pdfs.pkl","wb") as outfile:
		pickle.dump(line_results,outfile)

	with open("../data/emline_fits/1002_EW_pdfs.pkl","wb") as outfile:
		pickle.dump(EW_results,outfile)

	outfile.close()


def line_ratios():

	# Load the PDFs
	with open("../data/emline_fits/1002_lineflux_pdfs.pkl","rb") as outfile:
		line_results = pickle.load(outfile)
	
	outfile.close()

	
	##### START MAKING THE RATIOS HERE
	
	# Initialize the Dictionary
	ratio_measurements = {
		"name": [],
		"low_1sigma": [],
		"median": [],
		"upp_1sigma": []
	}

	# Define EBV which we will base on the Hb/Hg ratio
	pdf_EBV = 2.5/(util.calzetti(4343.) - util.calzetti(4861.)) * np.log10( (line_results["Hb_na_lflux_pdf"]/line_results["Hg_na_lflux_pdf"])/2.11 )
	pdf_EBV[pdf_EBV<0.] = 0. # ignore negative EBVs

	# Balmer Decrement
	ratio_measurements["name"].append("Hb/Hg")
	median,low_1sigma,upp_1sigma  = util.stats(line_results["Hb_na_lflux_pdf"]/line_results["Hg_na_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	ratio_measurements["name"].append("Hb/Hd")
	median,low_1sigma,upp_1sigma  = util.stats(line_results["Hb_na_lflux_pdf"]/line_results["Hd_na_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	ratio_measurements["name"].append("Hb/Hep")
	median,low_1sigma,upp_1sigma = util.stats(line_results["Hb_na_lflux_pdf"]/line_results["Hep_na_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	ratio_measurements["name"].append("Hb/H8")
	median,low_1sigma,upp_1sigma = util.stats(line_results["Hb_na_lflux_pdf"]/line_results["H8_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	ratio_measurements["name"].append("Hb/H9")
	median,low_1sigma,upp_1sigma = util.stats(line_results["Hb_na_lflux_pdf"]/line_results["H9_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# [OIII]5007/Hbeta
	ratio_measurements["name"].append("O3HB")
	median,low_1sigma,upp_1sigma  = util.stats(line_results["OIII5007c_lflux_pdf"]/line_results["Hb_na_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# [OIII]5007/[OII]3726,3729
	ratio_measurements["name"].append("O32")
	median,low_1sigma,upp_1sigma  = util.stats(line_results["OIII5007c_lflux_pdf"]/line_results["OII_lflux_pdf"]*pow(10,0.4*pdf_EBV*(util.calzetti(5007.) - util.calzetti(3727.) )))
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# [OII]/Hbeta
	ratio_measurements["name"].append("R2")
	R2_pdf_dustcorr = line_results["OII_lflux_pdf"]/line_results["Hb_na_lflux_pdf"]*pow(10,0.4*pdf_EBV*(util.calzetti(3727.) - util.calzetti(4861.) ))
	median,low_1sigma,upp_1sigma = util.stats(R2_pdf_dustcorr)
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# ([OIII]5007 + [OIII]4959)/Hbeta
	ratio_measurements["name"].append("R3")
	upper = line_results["OIII5007c_lflux_pdf"] * pow(10,0.4*pdf_EBV*util.calzetti(5007.)) + line_results["OIII4959c_lflux_pdf"] * pow(10,0.4*pdf_EBV*util.calzetti(4959.)) 
	lower = line_results["Hb_na_lflux_pdf"]*pow(10,0.4*pdf_EBV*util.calzetti(4861.))
	R3_pdf_dustcorr = upper/lower
	median,low_1sigma,upp_1sigma = util.stats(R3_pdf_dustcorr)
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# R23
	ratio_measurements["name"].append("R23")
	R23_pdf_dustcorr = R2_pdf_dustcorr + R3_pdf_dustcorr
	median,low_1sigma,upp_1sigma = util.stats(R23_pdf_dustcorr)
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# O3d
	ratio_measurements["name"].append("O3d")
	median,low_1sigma,upp_1sigma  = util.stats(line_results["OIII5007c_lflux_pdf"]/line_results["OIII4959c_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# [NeIII]3869/[OII]
	ratio_measurements["name"].append("Ne3O2")
	median,low_1sigma,upp_1sigma  = util.stats(line_results["NeIII3869_lflux_pdf"]/line_results["OII_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# [NeIII]3869/[OIII]5007
	ratio_measurements["name"].append("Ne3O3")
	median,low_1sigma,upp_1sigma = util.stats(line_results["NeIII3869_lflux_pdf"]/line_results["OIII5007c_lflux_pdf"] * pow(10,0.4*pdf_EBV*(util.calzetti(3869.) - util.calzetti(5007.))))
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)

	# [OIII]4363/[OIII]5007
	ratio_measurements["name"].append("Auroral")
	upper = line_results["OIII4363_lflux_pdf"]*pow(10,0.4*pdf_EBV*util.calzetti(4363.))
	lower = line_results["OIII5007c_lflux_pdf"]*pow(10,0.4*pdf_EBV*util.calzetti(5007.))	
	median,low_1sigma,upp_1sigma = util.stats(upper/lower)
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)


	# [OII]d
	ratio_measurements["name"].append("OII3728/OII3726")
	median,low_1sigma,upp_1sigma = util.stats(line_results["OII3728_lflux_pdf"]/line_results["OII3726_lflux_pdf"])
	ratio_measurements["low_1sigma"].append(low_1sigma); ratio_measurements["median"].append(median); ratio_measurements["upp_1sigma"].append(upp_1sigma)


	with open("../data/emline_fits/1002_line_ratios.pkl", "wb") as file:
		pickle.dump(ratio_measurements, file)
	
	print("And Done :D")

def get_1500A_luminosity(wave, flam, redshift=0.8275):
    
    mask = (wave > 1450.) & (wave < 1550.)
    wave = wave[mask]
    flam = flam[mask]
    
    fnu = (1./3e18) * simps(flam*wave, x=wave)/simps(1./wave, x=wave)

    return util.lineFlux_to_Luminosity(fnu,redshift)

def get_xi_ion(uv_lum, ha_lum):
    return ha_lum/(1.36e-12 * uv_lum)

def get_uv_props_bagpipes(gmos, bagpipes_SED, bagpipes_params):

	# Calculate the 1500A Luminosity
	wave = pow(10,bagpipes_SED["log_wave"])/1.8275
	flam = bagpipes_SED["SED_median"]*1.8275*1e-18
	flam_elow = bagpipes_SED["SED_1sig_low"]*1.8275*1e-18
	flam_eupp = bagpipes_SED["SED_1sig_upp"]*1.8275*1e-18

	Lnu_1500, Lnu_1500_low, Lnu_1500_upp =  (get_1500A_luminosity(wave,flam=flam),  get_1500A_luminosity(wave,flam=flam - flam_elow),  get_1500A_luminosity(wave,flam=flam + flam_eupp))
	Lnu_1500_pdf = util.sampling(Lnu_1500,(Lnu_1500 - Lnu_1500_low, Lnu_1500_upp - Lnu_1500))

	# Extract the Hbeta Luminosity
	HB_pdf = util.sampling(gmos["lineflux_med"],(gmos["lineflux_elow"],gmos["lineflux_eupp"]))
	lum_HB_pdf = util.lineFlux_to_Luminosity(HB_pdf,redshift=0.8275)

	# Xi Ion Measurement
	xi_ion_pdf = get_xi_ion(uv_lum = Lnu_1500_pdf, ha_lum = lum_HB_pdf*2.86)

	# UV Luminosity Measurement
	M_UV_pdf = -2.5*np.log10(Lnu_1500_pdf/(4.*np.pi*(10*3.08e18)**2.)) - 48.6
	return (xi_ion_pdf, M_UV_pdf)

def get_uv_props_cigale(gmos, cigale_SED, cigale_params):

	# 1500A Luminosity
	#flam = cigale_SED["Fnu"]*1e-26*3e18/(cigale_SED["wavelength"]*10.)**2.*1.8275
	#wave = cigale_SED["wavelength"]*10./1.8275

	HB_pdf = util.sampling(gmos["lineflux_med"],(gmos["lineflux_elow"],gmos["lineflux_eupp"]))

	Lnu_1500_pdf = util.sampling(central = cigale_params["bayes.param.restframe_Lnu(TopHat_1500_100)"]*1e7,
									sigma = cigale_params["bayes.param.restframe_Lnu(TopHat_1500_100)_err"]*1e7)
	lum_HB_pdf = util.lineFlux_to_Luminosity(HB_pdf,redshift=0.8275)

	# Xi Ion Measurement
	xi_ion_pdf = get_xi_ion(uv_lum = Lnu_1500_pdf, ha_lum = lum_HB_pdf*2.86)

	# UV Luminosity Measurement
	M_UV_pdf = -2.5*np.log10(Lnu_1500_pdf/(4.*np.pi*(10*3.08e18)**2.)) - 48.6

	return (xi_ion_pdf, M_UV_pdf)

def measure_xi_ion(gmos,cigale_SED,cigale_params,bagpipes_SED,bagpipes_params,lineprops_pdf):

    output = {}

    # Measure Xi_Ion with Bagpipes SED
    bagpipes_xi_ion_pdf,bagpipes_M_UV_pdf_corr = get_uv_props_bagpipes(gmos=gmos,bagpipes_SED=bagpipes_SED,bagpipes_params=bagpipes_params)

    # Measure [OIII]+Hb EW with Bagpipes SED
    EW_O3HB_bagpipes_pdf = lineprops_pdf["OIII5007c_EW_bagpipes_pdf"] + lineprops_pdf["OIII4959c_EW_bagpipes_pdf" ] + lineprops_pdf["Hb_na_EW_bagpipes_pdf"]

    output["bagpipes_xi_ion"] = util.stats(bagpipes_xi_ion_pdf)
    output["bagpipes_M_UV"] = util.stats(bagpipes_M_UV_pdf_corr)
    output["bagpipes_xi_ion_pdf"] = bagpipes_xi_ion_pdf
    output["bagpipes_M_UV_pdf"] = bagpipes_M_UV_pdf_corr
    output["bagpipes_O3HB_EW_pdf"] = EW_O3HB_bagpipes_pdf
    output["bagpipes_O3HB_EW"] = util.stats(EW_O3HB_bagpipes_pdf)
    

    # Cigale Xi_ion
    xi_ion_pdf, M_UV_pdf_corr = get_uv_props_cigale(gmos, cigale_SED, cigale_params)

    # Measure [OIII]+Hb EW with Cigale SED
    EW_O3HB_cigale_pdf = lineprops_pdf["OIII5007c_EW_cigale_pdf"] + lineprops_pdf["OIII4959c_EW_cigale_pdf" ] + lineprops_pdf["Hb_na_EW_cigale_pdf"]

    output["cigale_xi_ion"] = util.stats(xi_ion_pdf)
    output["cigale_M_UV"] = util.stats(M_UV_pdf_corr)
    output["cigale_xi_ion_pdf"] = xi_ion_pdf
    output["cigale_M_UV_pdf"] = M_UV_pdf_corr
    output["cigale_O3HB_EW_pdf"] = EW_O3HB_cigale_pdf
    output["cigale_O3HB_EW"] = util.stats(EW_O3HB_cigale_pdf)
    
    with open("../data/xi_ion_measurements.pkl","wb") as f:
        pickle.dump(output,f)


def main():
	#run_pyqsofit()
	line_props()
	line_ratios()

	# Load in the GMOS Line Properties
	gmos = fits.open("../data/emline_fits/1002_lineprops.fits")[1].data
	gmos = gmos[ gmos["line_ID"] == "Hb_na" ]

	# Load in the cigale sed fit and results
	cigale_SED = fits.open("../data/SED_results/cigale_results/1002_best_model.fits")[1].data
	cigale_params = fits.open("../data/SED_results/cigale_results/results.fits")[1].data

	# Load in the bagpipes sed fit and results
	bagpipes_SED = fits.open("../data/SED_results/bagpipes_results/best_fit_SED_sfh_continuity_spec_BPASS.fits")[1].data
	bagpipes_params = h5py.File("../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")

	# Load in the Line Prop PDF
	with open("../data/emline_fits/1002_EW_pdfs.pkl","rb") as f:
		lineprops_pdf = pickle.load(f)

	# Measure Xi_Ion
	measure_xi_ion(gmos,cigale_SED,cigale_params,bagpipes_SED,bagpipes_params,lineprops_pdf)


if __name__ == "__main__":
	main()







