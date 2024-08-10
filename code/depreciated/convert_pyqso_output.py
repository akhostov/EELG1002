import numpy as np
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
from astropy.io import fits
from astropy.table import Table
import pickle

from tqdm import trange

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
	def gaussian(self):
		return np.sum(self.scale*np.exp(-0.5*((self.xx[:,np.newaxis] - self.cwave)/self.sigma)**2.),axis=1)

	def wavelength(self):
		left  = np.min(self.cwave - 100.*self.sigma)
		right = np.max(self.cwave + 100.*self.sigma)
		disp = 1e-4*np.log(10)
		npix = int((right - left)/disp)
		self.xx = np.linspace(left, right, npix)


	# Measure the Line Flux
	def line_flux(self):
		yy = self.gaussian()
		self.lflux = simps(yy,x=np.exp(self.xx))
		return (self.lflux)

	def line_flux_EW(self,cont_fx):
		yy = self.gaussian()
		self.lflux = simps(yy,x=np.exp(self.xx))
		ew = simps(yy/cont_fx(np.exp(self.xx)),x=np.exp(self.xx))
		return (self.lflux,ew)

	# Measure the Dispersion
	def line_sigma(self,kms=True):
		lambda1 = simps(self.gaussian()*np.exp(self.xx),x=np.exp(self.xx))
		lambda2 = simps(self.gaussian()*pow(np.exp(self.xx),2.),x=np.exp(self.xx))

		if kms:
			return np.sqrt(lambda2/self.lflux - (lambda1/self.lflux)**2)/np.exp(self.cwave)*3e5
		else:
			return np.sqrt(lambda2/self.lflux - (lambda1/self.lflux)**2)

	# Measure the FWHM
	def line_FWHM(self,kms=True):
		spline = UnivariateSpline(self.xx, self.gaussian() - np.max(self.gaussian())/2, s=0)
		fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()

		if kms:
			return abs(np.exp(fwhm_left) - np.exp(fwhm_right))/np.exp(self.cwave)*3e5

		else:
			return abs(np.exp(fwhm_left) - np.exp(fwhm_right))


	# Measure the Redshifts for each line
	def line_redshift(self):
		return np.exp(self.cwave)/self.ref_line*(1.+self.redshift) -1.

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

def lib_lines():
	lines = np.array([
			(4861.333, 'Hb_na'),
			(4958.911, 'OIII4959c'),
			(5006.843, 'OIII5007c'),
			(4687.02,  'HeII4687_na'),
			(4340.471, 'Hg_na'),
			(4363.210, 'OIII4363'),
			(4101.742, 'Hd_na'),
			(4143.761, 'HeI4143'),
			(3967.470, 'NeIII3968'),
			(3970.079, 'Hep_na'),
			(4026.190, 'HeI4026'),
			(3889.064, 'H8'),
			(3888.647, 'HeI3889'),
			(3868.760, 'NeIII3869'),
			(3835.391, 'H9'),  
			(3726.032, 'OII3726'),
			(3728.815, 'OII3728')])

	return lines


def line_props():

	### READ IN THE DATA
	path = "../data/emline_fits/43158747673038238/"
	fname = "1002_pdf.fits"
	data = fits.open(f"{path}/{fname}")[1].data

	### MATCH LINES IN LIBRARY TO WHAT PYQSOFIT HAS
	line_id = [cc.split("_scale")[0] for cc in data.columns.names if "_scale" in cc and "_scale_err" not in cc]
	lines = lib_lines()

	### TEMP: ASSIGNING REDSHIFT
	redshift = 0.8275

	# This is where we will store all the line fit information
	stats_lflux,stats_lew,stats_lsigma,stats_lfwhm,stats_redshift = [],[],[],[],[]


	# Let's now do cigale
	cigale = fits.open("../data/cigale_results/sfh_delayed_nodust/1002_best_model.fits")[1].data
	cigale_flam = cigale["fnu"]*1e-26*3e18/pow(cigale["wavelength"]*10,2.)*(1.+redshift)
	cigale_wave = cigale["wavelength"]*10./(1.+redshift)


	### I'll do the doublet first
	these_lines = [ll for ll in line_id if "OII3" in ll]

	scales = np.array([data[f"{ll}_scale"] for ll in these_lines]).T
	cwaves = np.array([data[f"{ll}_centerwave"] for ll in these_lines]).T
	sigmas = np.array([data[f"{ll}_sigma"] for ll in these_lines]).T

	keep_cont_blue = (cigale_wave < 3727.-10.) & (cigale_wave > 3727.-20.)
	keep_cont_red  = (cigale_wave > 3727.+10.) & (cigale_wave < 3727.+20.)

	cont_flux = cigale_flam[keep_cont_blue+keep_cont_red]
	cont_wave = cigale_wave[keep_cont_blue+keep_cont_red]

	fx = UnivariateSpline(cont_wave,cont_flux*1e17)

	pdf_lflux, pdf_ew, pdf_lsigma,pdf_lfwhm,pdf_redshift = [],[],[],[],[]
	for jj in trange(scales.shape[0]):
		emission_line = physical_PyQSOfit(scale=scales[jj],cwave=cwaves[jj],sigma=sigmas[jj],redshift=redshift, ref_line = 3727.)

		lflux,ew = emission_line.line_flux_EW(fx)

		pdf_redshift.append(emission_line.line_redshift())
		pdf_lflux.append(lflux)
		pdf_ew.append(ew)
		pdf_lsigma.append(emission_line.line_sigma())
		pdf_lfwhm.append(emission_line.line_FWHM())

	med_redshift = np.median(pdf_redshift)
	med_lflux = np.median(pdf_lflux)
	med_ew = np.median(pdf_ew)
	med_lsigma = np.median(pdf_lsigma)
	med_lfwhm = np.median(pdf_lfwhm)

	stats_redshift.append((med_redshift, med_redshift - np.percentile(pdf_redshift,16.), np.percentile(pdf_redshift,84.) - med_redshift))
	stats_lflux.append((med_lflux, med_lflux - np.percentile(pdf_lflux,16.), np.percentile(pdf_lflux,84.) - med_lflux))
	stats_lew.append((med_ew, med_ew - np.percentile(pdf_ew,16.), np.percentile(pdf_ew,84.) - med_ew))
	stats_lsigma.append((med_lsigma, med_lsigma - np.percentile(pdf_lsigma,16.), np.percentile(pdf_lsigma,84.) - med_lsigma))
	stats_lfwhm.append((med_lfwhm, med_lfwhm - np.percentile(pdf_lfwhm,16.), np.percentile(pdf_lfwhm,84.) - med_lfwhm))



	### NOW RUN THROUGH EACH LINE
	ind = 0
	for ll in line_id:
		print("Running for line: ", ll)
		# Find Associated Reference Wavelength
		this = ll[:-2] == lines.T[1]

		scales = data[f"{ll}_scale"]
		cwaves = data[f"{ll}_centerwave"]
		sigmas = data[f"{ll}_sigma"]

		# Define the windows
		if ll in ("Hb_na_1","HeII4687_na","Hd_na_1","HeI4143","NeIII3968_1","Hep_na_1","HeI4026","H9_1"):
			keep_cont_blue = (cigale_wave < np.double(lines.T[0][this])-10.) & (cigale_wave > np.double(lines.T[0][this])-20.)
			keep_cont_red  = (cigale_wave > np.double(lines.T[0][this])+10.) & (cigale_wave < np.double(lines.T[0][this])+20.)

		elif ll in ("OIII4959c_1","OIII5007c_1"):
			keep_cont_blue = (cigale_wave < np.double(lines.T[0][this])-15.) & (cigale_wave > np.double(lines.T[0][this])-35.)
			keep_cont_red  = (cigale_wave > np.double(lines.T[0][this])+15.) & (cigale_wave < np.double(lines.T[0][this])+35.)

		elif ll in ("Hg_na_1","NeIII3869_1"):
			keep_cont_blue = (cigale_wave < np.double(lines.T[0][this])-10.) & (cigale_wave > np.double(lines.T[0][this])-20.)
			keep_cont_red  = (cigale_wave > np.double(lines.T[0][this])+35.) & (cigale_wave < np.double(lines.T[0][this])+45.)

		elif ll in ("OIII4363_1","H8_1","HeI3889"):
			keep_cont_blue = (cigale_wave < np.double(lines.T[0][this])-35.) & (cigale_wave > np.double(lines.T[0][this])-45.)
			keep_cont_red  = (cigale_wave > np.double(lines.T[0][this])+10.) & (cigale_wave < np.double(lines.T[0][this])+20.)

		elif ll in ("H8"):
			keep_cont_blue = (cigale_wave < np.double(lines.T[0][this])-35.) & (cigale_wave > np.double(lines.T[0][this])-45.)
			keep_cont_red  = (cigale_wave > np.double(lines.T[0][this])+10.) & (cigale_wave < np.double(lines.T[0][this])+20.)

		cont_flux = cigale_flam[keep_cont_blue+keep_cont_red]
		cont_wave = cigale_wave[keep_cont_blue+keep_cont_red]

		fx = UnivariateSpline(cont_wave,cont_flux*1e17)

		pdf_lflux,pdf_ew,pdf_lsigma,pdf_lfwhm,pdf_redshift = [],[],[],[],[]
		for jj in trange(len(scales)):
			emission_line = physical_PyQSOfit(scale=scales[jj],cwave=cwaves[jj],sigma=sigmas[jj],redshift=redshift, ref_line = lines.T[0][this])

			lflux,ew = emission_line.line_flux_EW(fx)

			pdf_redshift.append(emission_line.line_redshift())
			pdf_lflux.append(lflux)
			pdf_ew.append(ew)			
			pdf_lsigma.append(emission_line.line_sigma())
			pdf_lfwhm.append(emission_line.line_FWHM())

		
		med_redshift = np.median(pdf_redshift)
		med_lflux = np.median(pdf_lflux)
		med_ew = np.median(pdf_ew)
		med_lsigma = np.median(pdf_lsigma)
		med_lfwhm = np.median(pdf_lfwhm)

		stats_redshift.append((med_redshift, med_redshift - np.percentile(pdf_redshift,16.), np.percentile(pdf_redshift,84.) - med_redshift))
		stats_lflux.append((med_lflux, med_lflux - np.percentile(pdf_lflux,16.), np.percentile(pdf_lflux,84.) - med_lflux))
		stats_lew.append((med_ew, med_ew - np.percentile(pdf_ew,16.), np.percentile(pdf_ew,84.) - med_ew))	
		stats_lsigma.append((med_lsigma, med_lsigma - np.percentile(pdf_lsigma,16.), np.percentile(pdf_lsigma,84.) - med_lsigma))
		stats_lfwhm.append((med_lfwhm, med_lfwhm - np.percentile(pdf_lfwhm,16.), np.percentile(pdf_lfwhm,84.) - med_lfwhm))

		ind = ind + 1

	# WRITE EVERYTHING OUT
	line_id.insert(0,"[OII]")

	final = np.column_stack((line_id,stats_redshift,stats_lflux,stats_lew,stats_lsigma,stats_lfwhm))
	names = np.array(["line_ID",
						"linez_med","linez_elow","linez_eupp",
						"lineflux_med","lineflux_elow","lineflux_eupp",
						"lineEW_med","lineEW_elow","lineEW_eupp",
						"linesigma_med","linesigma_elow","linesigma_eupp",
						"linefwhm_med","linefwhm_elow","linefwhm_eupp"])
	dtypes = np.array(["str",
						"float","float","float",
						"float","float","float",
						"float","float","float",
						"float","float","float",
						"float","float","float"])

	t = Table(final,names=names,dtype=dtypes)
	t.write("../data/emline_fits/43158747673038238/1002_lineprops_new.fits",format="fits",overwrite=True)




def line_ratios():

	### READ IN THE DATA
	path = "../data/emline_fits/43158747673038238/"
	fname = "1002_pdf.fits"

	data = fits.open(f"{path}/{fname}")[1].data

	redshift = 0.8275

	o2_3726,o2_3729,o2_doublet = [],[],[]
	o3_5007,o3_4959,o3_4363 = [],[],[]
	hbeta,hgamma,hdelta,hepsilon,h8,h9 = [],[],[],[],[],[]
	ne3_3869,ne3_3968 = [],[]
	#heI_3889,heI_4026,heI_4143,heII_4687 = [],[],[],[]
	for jj in trange(len(data["OII3726_1_scale"])):

		# [OII]3729,3726
		id_line = "OII3726_1"
		o2_3726.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "OII3728_1"
		o2_3729.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		# [OII] Double combined
		o2_doublet.append(physical_PyQSOfit(scale=np.array([data["OII3726_1_scale"][jj],data["OII3728_1_scale"][jj]]),\
							cwave=np.array([data["OII3726_1_centerwave"][jj],data["OII3728_1_centerwave"][jj]]),\
							sigma=np.array([data["OII3726_1_sigma"][jj],data["OII3728_1_sigma"][jj]]),redshift=redshift, ref_line = None).line_flux())

		# Other Lines
		id_line = "OIII5007c_1"
		o3_5007.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "OIII4959c_1"
		o3_4959.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "Hb_na_1"
		hbeta.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		#id_line = "HeII4687_na_1"
		#heII_4687.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "Hg_na_1"
		hgamma.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "OIII4363_1"
		o3_4363.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		#id_line = "HeI4143_1"
		#heI_4143.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "Hd_na_1"
		hdelta.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		#id_line = "HeI4026_1"
		#heI_4026.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "NeIII3968_1"
		ne3_3968.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "Hep_na_1"
		hepsilon.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		#id_line = "HeI3889_1"
		#heI_3889.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "H8_1"
		h8.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "H9_1"
		h9.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())

		id_line = "NeIII3869_1"
		ne3_3869.append(physical_PyQSOfit(scale=data[f"{id_line}_scale"][jj],cwave=data[f"{id_line}_centerwave"][jj],sigma=data[f"{id_line}_sigma"][jj],redshift=redshift, ref_line = None).line_flux())




	#########################################################
	#########		 Get Physical Measurements		#########
	#########################################################

	def stats(pdf,upper_limit=False): 

		if upper_limit:
			return np.percentile(pdf,upper_limit)
		else:
			return (np.median(pdf),np.median(pdf) - np.percentile(pdf,16.),np.percentile(pdf,84.) - np.median(pdf))

	def calzetti(lam):
		lam = lam/1e4
		return (2.659*(-2.156 + 1.509/lam - 0.198/(lam**2.) + 0.011/(lam**3.)) + 4.05)

	# Initialize Pickle
	data_pickle = {}


	# Save the Lines
	data_pickle["pdf_OII_combined"] = np.asarray(o2_doublet)
	data_pickle["pdf_OII3726"] = np.asarray(o2_3726)
	data_pickle["pdf_OII3728"] = np.asarray(o2_3729)
	data_pickle["pdf_OIII5007"] = np.asarray(o3_5007)
	data_pickle["pdf_OIII4959"] = np.asarray(o3_4959)
	data_pickle["pdf_OIII4363"] = np.asarray(o3_4363)
	data_pickle["pdf_NeIII3968"] = np.asarray(ne3_3968)
	data_pickle["pdf_NeIII3869"] = np.asarray(ne3_3869)
	data_pickle["pdf_Hb"] = np.asarray(hbeta)
	data_pickle["pdf_Hg"] = np.asarray(hgamma)
	data_pickle["pdf_Hd"] = np.asarray(hdelta)
	data_pickle["pdf_Hep"] = np.asarray(hepsilon)
	data_pickle["pdf_H8"] = np.asarray(h8)
	data_pickle["pdf_H9"] = np.asarray(h9)



	# Balmer Decrement
	pdf_Hb_Hg = np.asarray(hbeta)/np.asarray(hgamma)
	pdf_Hb_Hd = np.asarray(hbeta)/np.asarray(hdelta)
	pdf_Hb_Heps = np.asarray(hbeta)/np.asarray(hepsilon)
	pdf_Hb_H8 = np.asarray(hbeta)/np.asarray(h8)
	pdf_Hb_H9 = np.asarray(hbeta)/np.asarray(h9)

	pdf_A_Hb_Hg = 3.33*2.5/(calzetti(4343.) - calzetti(4861.))*np.log10(pdf_Hb_Hg/2.11)
	pdf_A_Hb_Hd = 3.33*2.5/(calzetti(4101.) - calzetti(4861.))*np.log10(pdf_Hb_Hd/3.87)
	pdf_A_Hb_Heps = 3.33*2.5/(calzetti(3970.) - calzetti(4861.))*np.log10(pdf_Hb_Heps/6.135)
	pdf_A_Hb_H8 = 3.33*2.5/(calzetti(3889.) - calzetti(4861.))*np.log10(pdf_Hb_H8/9.345)
	pdf_A_Hb_H9 = 3.33*2.5/(calzetti(3835.) - calzetti(4861.))*np.log10(pdf_Hb_H9/13.405)

	pdf_A_Hb_Hg[pdf_A_Hb_Hg<0.] = 0.
	pdf_A_Hb_Hd[pdf_A_Hb_Hd<0.] = 0.
	pdf_A_Hb_Heps[pdf_A_Hb_Heps<0.] = 0.
	pdf_A_Hb_H8[pdf_A_Hb_H8 < 0.] = 0.
	pdf_A_Hb_H9[pdf_A_Hb_H9 < 0.] = 0.

	final_A_Hb_Hg = stats(pdf_A_Hb_Hg)
	final_A_Hb_Hd = stats(pdf_A_Hb_Hd)

	data_pickle["pdf_Hb_Hg"] = pdf_Hb_Hg
	data_pickle["pdf_Hb_Hd"] = pdf_Hb_Hd
	data_pickle["pdf_A_Hb_Hg"] = pdf_A_Hb_Hg
	data_pickle["pdf_A_Hb_Hd"] = pdf_A_Hb_Hd
	data_pickle["final_A_Hb_Hg"] = stats(pdf_A_Hb_Hg)
	data_pickle["final_A_Hb_Hd"] = stats(pdf_A_Hb_Hd)







	# Electron Temperature -- Nicholls et al. (2020)
	pdf_o3_4363_div_o3_5007_4959 = np.log10(np.asarray(o3_4363)/(np.asarray(o3_4959) + np.asarray(o3_5007)))
	pdf_te = pow(10,(3.5363+7.2939*pdf_o3_4363_div_o3_5007_4959)/(1 + 1.6298*pdf_o3_4363_div_o3_5007_4959-0.1221*pdf_o3_4363_div_o3_5007_4959**2. - 0.0074*pdf_o3_4363_div_o3_5007_4959**3.))

	final_pdf_o3_4363_div_o3_5007_4959 = stats(pdf_o3_4363_div_o3_5007_4959)
	final_te = stats(pdf_te)

	data_pickle["pdf_o3_4363_div_o3_5007_4959"] = pdf_o3_4363_div_o3_5007_4959
	data_pickle["pdf_te"] = pdf_te
	data_pickle["final_pdf_o3_4363_div_o3_5007_4959"] = final_pdf_o3_4363_div_o3_5007_4959
	data_pickle["final_te"] = final_te








	# Direct Metallicity
	pdf_12OH = 9.72 - 1.70*pdf_te/1e4 + 0.32*(pdf_te/1e4)**2.
	pdf_12OH_solar = pow(10,pdf_12OH - np.random.normal(8.69,0.04))
	pdf_12OH_solar_cigale = pdf_12OH_solar*0.014

	final_12OH = stats(pdf_12OH)
	final_12OH_solar = stats(pdf_12OH_solar)


	data_pickle["pdf_12OH"] = pdf_12OH
	data_pickle["pdf_12OH_solar"] = pdf_12OH_solar
	data_pickle["pdf_12OH_solar_cigale"] = pdf_12OH_solar_cigale
	data_pickle["final_12OH"] = final_12OH
	data_pickle["final_12OH_solar"] = final_12OH_solar






	# Electron Density & ISM Pressure
	# OII doublet ratio
	pdf_o2_3729_3726 = np.asarray(o2_3729)/np.asarray(o2_3726)
	rmin = 0.3839
	rmax = 1.4558
	keep = (pdf_o2_3729_3726 >= rmin) & (pdf_o2_3729_3726 <= rmax)
	pdf_o2_3729_3726 = pdf_o2_3729_3726[keep]
	pdf_ne = (638.4*pdf_o2_3729_3726 - 0.3771*2468)/(0.3771-pdf_o2_3729_3726)*np.sqrt(1e4/pdf_te[keep])
	pdf_ISM = pdf_te[keep]*pdf_ne/1e7

	final_pdf_o2_3729_3726 = stats(pdf_o2_3729_3726)
	final_ne = stats(pdf_ne)
	final_ISM = stats(pdf_ISM)

	data_pickle["pdf_o2_3729_3726"] = pdf_o2_3729_3726
	data_pickle["pdf_ne"] = pdf_ne
	data_pickle["pdf_ISM"] = pdf_ISM
	data_pickle["final_pdf_o2_3729_3726"] = final_pdf_o2_3729_3726
	data_pickle["final_ne"] = final_ne
	data_pickle["final_ISM"] = final_ISM





	# O32 -- Ionization logU (using Lisa Kewley's review paper)
	pdf_o32 = np.log10(np.asarray(o3_5007)/np.asarray(o2_doublet))
	pdf_o32_tot = np.log10((np.asarray(o3_5007)+np.asarray(o3_4959))/np.asarray(o2_doublet))
	#pdf_logU_Pk_5 =  13.768 + 9.494*pdf_o32 -  4.3223*pdf_12OH - 2.3531*pdf_o32*pdf_12OH - 0.5768*pdf_o32**2. + 0.2794*pdf_12OH**2. + 0.1574*pdf_o32*pdf_12OH**2. + 0.0890*pdf_12OH*pdf_o32**2. + 0.0311*pdf_o32**3.
	pdf_logU_Pk_7 = -48.953 + 6.076*pdf_o32 + 18.1390*pdf_12OH - 1.4759*pdf_o32*pdf_12OH - 0.4753*pdf_o32**2. - 2.3925*pdf_12OH**2. + 0.1010*pdf_o32*pdf_12OH**2. + 0.0758*pdf_12OH*pdf_o32**2. + 0.0332*pdf_o32**3. + 0.1055*pdf_12OH**3.

	final_pdf_o32 = stats(pdf_o32)
	final_pdf_o32_tot = stats(pdf_o32_tot)
	final_pdf_logU_Pk_7 = stats(pdf_logU_Pk_7)

	data_pickle["pdf_o32"] = pdf_o32
	data_pickle["pdf_o32_tot"] = pdf_o32_tot
	data_pickle["pdf_logU_Pk_7"] = pdf_logU_Pk_7
	data_pickle["final_pdf_o32"] = final_pdf_o32
	data_pickle["final_pdf_o32_tot"] = final_pdf_o32_tot
	data_pickle["final_pdf_logU_Pk_7"] = final_pdf_logU_Pk_7



	# R23
	pdf_r23 = np.log10((np.asarray(o3_4959) + np.asarray(o3_5007) + np.asarray(o2_doublet))/np.asarray(hbeta))
	final_pdf_r23 = stats(pdf_r23)


	data_pickle["pdf_r23"] = pdf_r23
	data_pickle["final_pdf_r23"] = final_pdf_r23



	# R2
	pdf_r2 = np.log10(np.asarray(o2_doublet)/np.asarray(hbeta))
	final_pdf_r2 = stats(pdf_r2)

	data_pickle["pdf_r2"] = pdf_r2
	data_pickle["final_pdf_r2"] = final_pdf_r2




	# R3
	pdf_r3 = np.log10(np.asarray(o3_5007)/np.asarray(hbeta))
	final_pdf_r3 = stats(pdf_r3)

	data_pickle["pdf_r3"] = pdf_r3
	data_pickle["final_pdf_r3"] = final_pdf_r3



	# O3
	pdf_o3 = np.log10(np.asarray(o3_5007)/np.asarray(o3_4959))
	final_pdf_o3 = stats(pdf_o3)

	data_pickle["pdf_o3"] = pdf_o3
	data_pickle["final_pdf_o3"] = final_pdf_o3



	# Ne3O3
	pdf_Ne3O3 = np.log10(np.asarray(ne3_3869)/np.asarray(o3_5007))
	final_pdf_Ne3O3 = stats(pdf_Ne3O3)

	data_pickle["pdf_Ne3O3"] = pdf_Ne3O3
	data_pickle["final_pdf_Ne3O3"] = final_pdf_Ne3O3



	# Ne3O2
	pdf_NO = np.log10(np.asarray(ne3_3869)/np.asarray(o2_doublet))
	final_pdf_Ne3O2 = stats(pdf_NO)

	data_pickle["pdf_NO"] = pdf_NO
	data_pickle["final_pdf_Ne3O2"] = final_pdf_Ne3O2


	def levesque(qq):
		#return -129.649+37.6573*qq-3.61621*qq**2.+0.116192*qq**3.
		return 56.3416 - 23.1202*qq + 3.00640*qq**2. - 0.124519*qq**3. # 0.001
		#return 53.5278 - 22.2764*qq + 2.93137*qq**2. - 0.122954*qq**3. # 0.004

	qq = np.arange(np.log10(1e7),np.log10(4e8),0.01)
	rr = levesque(qq)
	fx = UnivariateSpline(rr,qq)

	pdf_logU_NO = fx(pdf_NO)
	final_logU_NO = stats(pdf_logU_NO)

	data_pickle["pdf_logU_NO"] = pdf_logU_NO
	data_pickle["final_logU_NO"] = final_logU_NO


	with open("../data/emline_fits/43158747673038238/ratios_and_ISM_props_1002_new.pkl", "wb") as file:
		 pickle.dump(data_pickle, file, protocol=pickle.HIGHEST_PROTOCOL)





"""
def main():

	path = "../data/emline_fits/43158747673038238/"
	fname = "1002.fits"

	data = fits.open(f"{path}/{fname}")[1].data

	line_id = [cc.split("_scale")[0] for cc in data.columns.names if "_scale" in cc and "_scale_err" not in cc]
	lines = lib_lines()

	final = []
	for ll in line_id:

		print("Running for line: ", ll)
		# Find Associated Reference Wavelength
		this = ll[:-2] == lines.T[1]


		emission_line = physical_PyQSOfit(scale=data[f"{ll}_scale"],cwave=data[f"{ll}_centerwave"],sigma=data[f"{ll}_sigma"],\
									scale_err=data[f"{ll}_scale_err"],cwave_err=data[f"{ll}_centerwave_err"],sigma_err=data[f"{ll}_sigma_err"],redshift=data["redshift"], ref_line = lines.T[0][this])

		final.append(emission_line.error_estimation())

	import pdb; pdb.set_trace()
	source_id = fname.split(".fits")[0]
	np.savetxt(f"{path}/{source_id}_line_props.txt",\
				np.column_stack((line_id,np.transpose(final))),\
				header="lineID lineflux lineflux_elow lineflux_eupp sigma sigma_elow sigma_eupp fwhm fwhm_elow fwhm_eupp redshift redshift_elow redshift_eupp")
"""

if __name__ == "__main__":
	line_props() 
	line_ratios()
