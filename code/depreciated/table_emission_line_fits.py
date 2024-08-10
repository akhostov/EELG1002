from astropy.io import fits
import numpy as np
import pickle

def measure_EW(cEW,cEW_err,clflux,clflux_err,lflux,lflux_err):

	fc = clflux/cEW
	fc_err = fc*np.sqrt((clflux_err/clflux)**2. + (cEW_err/cEW)**2.)

	EW = lflux/fc
	EW_err = EW*np.sqrt((lflux_err/lflux)**2. + (fc_err/fc)**2.)

	lflux_rand = np.random.normal(lflux,lflux_err,100000)
	clflux_rand = np.random.normal(clflux,clflux_err,100000)
	cEW_rand = np.random.normal(cEW,cEW_err,100000)

	pdf_EW = lflux_rand/clflux_rand*cEW_rand

	return (EW,EW-np.percentile(pdf_EW,16.),np.percentile(pdf_EW,84.)-EW)

def table_emission_line_fits():
	# Load in the data
	data = fits.open("../data/emline_fits/43158747673038238/1002_lineprops_new.fits")[1].data

	cigale = fits.open("../data/final_SED_results/cigale_results/results.fits")[1].data

	# Sort to Table
	sort = np.array(["Hb_na_1","Hg_na_1","Hd_na_1","Hep_na_1","H9_1","H8_1",\
			"OIII5007c_1","OIII4959c_1","OIII4363_1","[OII]","OII3726_1","OII3728_1",\
			"NeIII3869_1","NeIII3968_1"])

	names = np.array([r"\hbeta",r"H$\gamma$",r"H$\delta",r"H$\epsilon",r"H$\zeta+$He{\sc i}3889\AA",r"H$\eta$",\
						r"\oiii5007\AA",r"\oiii4959\AA",r"\oiii4363\AA",r"\oii3726,3729\AA",r"\oii3726\AA",r"\oii3729\AA",\
						r"\neiii3869\AA",r"\neiii3968\AA"])

	keep = np.hstack([np.argwhere(tt == data["line_id"])[0] for tt in sort])
	data = data[keep]

	for ff in range(len(data["line_id"])):

		str_redshift = "$%0.4f^{+%0.4f}_{-%0.4f}$" % (data['linez_med'][ff],data['linez_eupp'][ff],data['linez_elow'][ff])
		str_lflux = "$%0.2f^{+%0.2f}_{-%0.2f}$" % (data['lineflux_med'][ff],data['lineflux_eupp'][ff],data['lineflux_elow'][ff])
		str_lEW   = "$%0.2f^{+%0.2f}_{-%0.2f}$" % (data['lineEW_med'][ff],data['lineEW_eupp'][ff],data['lineEW_elow'][ff])	
		str_lfwhm = "$%0.0f^{+%0.0f}_{-%0.0f}$" % (data['linefwhm_med'][ff],data['linefwhm_eupp'][ff],data['linefwhm_elow'][ff])
		#str_lsigma = "$%0.0f^{+%0.0f}_{-%0.0f}$" % (data['linesigma_med'][ff],data['linesigma_eupp'][ff],data['linesigma_elow'][ff])

		if data["line_id"][ff] == "OII3728_1":
			str_cigale_lflux = "$%0.2f\pm%0.2f$" % (cigale["bayes.line.OII-372.9"]*1e20,cigale["bayes.line.OII-372.9_err"]*1e20)
			str_cigale_EW = " -- "
		elif data["line_id"][ff] == "OII3726_1":
			str_cigale_lflux = "$%0.2f\pm%0.2f$" % (cigale["bayes.line.OII-372.6"]*1e20,cigale["bayes.line.OII-372.6_err"]*1e20)
			str_cigale_EW = " -- "

		elif data["line_id"][ff] == "Hd_na_1":
			str_cigale_lflux = "$%0.2f\pm%0.2f$" % (cigale["bayes.line.H-delta"]*1e20,cigale["bayes.line.H-delta_err"]*1e20)
			str_cigale_EW = " -- "

		elif data["line_id"][ff] == "Hg_na_1":
			str_cigale_lflux = "$%0.2f\pm%0.2f$" % (cigale["bayes.line.H-gamma"]*1e20,cigale["bayes.line.H-gamma_err"]*1e20)
			str_cigale_EW = " -- "

		elif data["line_id"][ff] == "Hb_na_1":
			str_cigale_lflux = "$%0.2f\pm%0.2f$" % (cigale["bayes.line.H-beta"]*1e20,cigale["bayes.line.H-beta_err"]*1e20)
			str_cigale_EW = "$%0.2f\pm%0.2f$" % (cigale["bayes.param.EW(486.1/1.0)"]*10.,cigale["bayes.param.EW(486.1/1.0)_err"]*10.)
			EW,EW_elow,EW_eupp = measure_EW(cigale["bayes.param.EW(486.1/1.0)"]*10.,cigale["bayes.param.EW(486.1/1.0)_err"]*10.,\
									cigale["bayes.line.H-beta"]*1e3,cigale["bayes.line.H-beta_err"]*1e3,\
									data['lineflux_med'][ff]*1e-17,0.5*(data['lineflux_eupp'][ff]+data['lineflux_elow'][ff])*1e-17)

			print(EW,EW_elow,EW_eupp)

		elif data["line_id"][ff] == "OIII4959c_1":
			str_cigale_lflux = "$%0.2f\pm%0.2f$" % (cigale["bayes.line.OIII-495.9"]*1e20,cigale["bayes.line.OIII-495.9_err"]*1e20)
			str_cigale_EW = "$%0.2f\pm%0.2f$" % (cigale["bayes.param.EW(495.9/1.0)"]*10.,cigale["bayes.param.EW(495.9/1.0)_err"]*10.)
			EW,EW_elow,EW_eupp = measure_EW(cigale["bayes.param.EW(495.9/1.0)"]*10.,cigale["bayes.param.EW(495.9/1.0)_err"]*10.,\
									cigale["bayes.line.OIII-495.9"]*1e3,cigale["bayes.line.OIII-495.9_err"]*1e3,\
									data['lineflux_med'][ff]*1e-17,0.5*(data['lineflux_eupp'][ff]+data['lineflux_elow'][ff])*1e-17)

			print(EW,EW_elow,EW_eupp)

		elif data["line_id"][ff] == "OIII5007c_1":
			str_cigale_lflux = "$%0.2f\pm%0.2f$" % (cigale["bayes.line.OIII-500.7"]*1e20,cigale["bayes.line.OIII-500.7_err"]*1e20)
			str_cigale_EW = "$%0.2f\pm%0.2f$" % (cigale["bayes.param.EW(500.7/1.0)"]*10.,cigale["bayes.param.EW(500.7/1.0)_err"]*10.)
			EW,EW_elow,EW_eupp = measure_EW(cigale["bayes.param.EW(500.7/1.0)"]*10.,cigale["bayes.param.EW(500.7/1.0)_err"]*10.,\
									cigale["bayes.line.OIII-500.7"]*1e3,cigale["bayes.line.OIII-500.7_err"]*1e3,\
									data['lineflux_med'][ff]*1e-17,0.5*(data['lineflux_eupp'][ff]+data['lineflux_elow'][ff])*1e-17)

			print(EW,EW_elow,EW_eupp)
			
		else:
			str_cigale_lflux = " -- "
			str_cigale_EW = " -- "
		print(f"{names[ff]} & {str_redshift} & {str_lflux} & {str_cigale_lflux} & {str_lEW} & {str_cigale_EW} & {str_lfwhm} \\\\") #" & {str_lsigma} \\\\")


def table_emission_line_ratios():

	# Load in the pickle file
	with open("../data/emline_fits/43158747673038238/ratios_and_ISM_props_1002.pkl","rb") as file:
		data = pickle.load(file)


	# Let's Through Each
	str_final_r23 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_r23"]
	str_final_r2 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_r2"]
	str_final_r3 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_r3"]

	str_final_Ne3O3 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_Ne3O3"]
	str_final_O32 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_O32"]
	str_final_O3 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_o3"]
	str_final_O2 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_o2_3729_3726"]
	str_final_O3_4363_5007_4959 = "%0.3f^{+%0.3f}_{-%0.3f}" % data["final_pdf_o3_4363_div_o3_5007_4959"]

	import pdb; pdb.set_trace()
	#print(f"R23 \& {}")



#table_emission_line_ratios()
table_emission_line_fits()
#def table_SED_fits():

