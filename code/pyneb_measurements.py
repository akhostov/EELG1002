import numpy as np
import pyneb as pn
from astropy.io import fits
import pickle
import util
import multiprocessing
import os

# Load in the Emission Line Fluxe
def load_data():
	"""
	Loads the emission line fluxes and their associated uncertainties from a .pkl file

	Returns:
		data (dict): A dictionary of line fluxes and their associated uncertainties
	"""
	file = open("../data/emline_fits/1002_lineflux_pdfs.pkl","rb")
	data = pickle.load(file)
	file.close()
	return data

def correct_fluxes(data, pdf_EBV):
	"""
	Corrects the emission line fluxes for dust extinction.

	Parameters:
		data (dict): A dictionary of line fluxes and their associated uncertainties
		pdf_EBV (array): An array of E(B-V) values from the PDF

	Returns:
		o3_5007 (array): The dust-corrected OIII 5007 line fluxes
		o3_4959 (array): The dust-corrected OIII 4959 line fluxes
		o3_4363 (array): The dust-corrected OIII 4363 line fluxes
		o2_all (array): The dust-corrected OII 3727 line fluxes
		ne3_3869 (array): The dust-corrected NeIII 3869 line fluxes
		hbeta (array): The dust-corrected Hbeta line fluxes
	"""
	o3_5007 = data["OIII5007c_lflux_pdf"] * pow(10, 0.4 * pdf_EBV * util.calzetti(5007.))
	o3_4959 = data["OIII4959c_lflux_pdf"] * pow(10, 0.4 * pdf_EBV * util.calzetti(4959.))
	o3_4363 = data["OIII4363_lflux_pdf"] * pow(10, 0.4 * pdf_EBV * util.calzetti(4363.))
	o2_all = data["OII_lflux_pdf"] * pow(10, 0.4 * pdf_EBV * util.calzetti(3727.))
	ne3_3869 = data["NeIII3869_lflux_pdf"] * pow(10, 0.4 * pdf_EBV * util.calzetti(3869.))
	hbeta = data["Hb_na_lflux_pdf"] * pow(10, 0.4 * pdf_EBV * util.calzetti(4861.))

	return o3_5007, o3_4959, o3_4363, o2_all, ne3_3869, hbeta

def calc_Te(o3_4363, o3_5007):
	"""
	Calculate the electron temperatures from auroral [OIII] line ratios.

	The function uses the given [OIII] 4363 and 5007 fluxes to compute the electron 
	temperature using a line ratio method. It also applies a correction based on 
	Izotov et al. (2006) for lower electron temperatures.

	Parameters:
		o3_4363 (array): Flux values of the [OIII] 4363 line.
		o3_5007 (array): Flux values of the [OIII] 5007 line.

	Returns:
		tuple: A tuple containing:
			- pdf_te_high (array): High electron temperature derived from the [OIII] 4363/5007 ratio.
			- pdf_te_low (array): Low electron temperature correction based on Izotov et al. (2006).
	"""
	# Initialize the PyNeb object for [OIII]
	O3 = pn.Atom('O', 3)

	# Calculate the auroral [OIII] line ratio
	pdf_auroral_O3_ratio = o3_4363 / o3_5007

	# Calculate the high electron temperature using PyNeb
	pdf_te_high = O3.getTemDen(int_ratio=pdf_auroral_O3_ratio, den=1000., to_eval='L(4363) / (L(5007))')

	# Apply Izotov et al. (2006) correction for low electron temperature
	pdf_te_low = (-0.577 + 1e-4 * pdf_te_high * (2.065 - 0.498 * 1e-4 * pdf_te_high)) * 1e4

	return pdf_te_high, pdf_te_low

def calc_ne(pdf_density_O2_ratio, pdf_te_low):
	"""
	Calculates the electron density based on the given [OII] line ratios.

	Parameters:
		pdf_density_O2_ratio (array): Flux values of the [OII] 3729/3726 line ratios.
		pdf_te_low (array): Low electron temperature derived from the [OIII] 4363/5007 ratio with Izotov et al. (2006) calibration.

	Returns:
		pdf_ne (array): Electron density derived from the [OII] 3729/3726 line ratio.
	"""

	# Initialize the PyNeb object for [OII]
	O2 = pn.Atom('O', 2)

	# Use PyNeb's getTemDen method to calculate electron density
	pdf_ne = O2.getTemDen(int_ratio=pdf_density_O2_ratio, tem=pdf_te_low, to_eval='L(3729) / L(3726)')

	# Return the calculated electron density
	return pdf_ne

def calc_ionic_abundance(o3_5007, o3_4959, hbeta, o2_all, ne3_3869, pdf_te_high, pdf_te_low, pdf_ne):
	"""
	Calculates the ionic abundances of Oxygen and Neon based on the
	observed line fluxes and derived electron temperatures and densities.

	Parameters:
		o3_5007 (array): Flux values of the [OIII] 5007 line.
		o3_4959 (array): Flux values of the [OIII] 4959 line.
		hbeta (array): Flux values of the Hbeta line.
		o2_all (array): Flux values of the [OII] 3727 line.
		ne3_3869 (array): Flux values of the [NeIII] 3869 line.
		pdf_te_high (array): High electron temperature derived from the [OIII] 4363/5007 ratio.
		pdf_te_low (array): Low electron temperature correction based on Izotov et al. (2006).
		pdf_ne (array): Electron density derived from the [OII] 3729/3726 line ratio.

	Returns:
		tuple: A tuple containing:
			- o3_h (array): Ionic abundance of O++ derived from the [OIII] 4959+5007/Hbeta ratio.
			- o2_h (array): Ionic abundance of O+ derived from the [OII] 3726+3729/Hbeta ratio.
			- o_h (array): Total ionic abundance of Oxygen derived from o3_h and o2_h.
			- ne_h (array): Ionic abundance of Ne++ derived from the [NeIII] 3869/Hbeta ratio and the ICF correction.
	"""

	# Initialize the PyNeb object for [OIII], [OII], and [NeIII]
	O3 = pn.Atom('O', 3)
	O2 = pn.Atom('O', 2)
	Ne3 = pn.Atom('Ne', 3)	

	# Calculate the R3 and R2 line ratios
	pdf_R3 = (o3_5007 + o3_4959) / hbeta
	pdf_R2 = o2_all / hbeta

	# Calculate the ionic abundance of O++
	o3_h = O3.getIonAbundance(int_ratio=pdf_R3, tem=pdf_te_high, den=pdf_ne, to_eval='L(4959)+L(5007)', Hbeta=1.)

	# Calculate the ionic abundance of O+
	o2_h = O2.getIonAbundance(int_ratio=pdf_R2, tem=pdf_te_low, den=pdf_ne, to_eval='L(3726)+L(3729)', Hbeta=1.)

	# Calculate the total ionic abundance of Oxygen
	o_h = o3_h + o2_h

	# Calculate the ionic abundance of Ne++
	pdf_Ne3Hb = ne3_3869 / hbeta
	ne3_h = Ne3.getIonAbundance(int_ratio=pdf_Ne3Hb, tem=pdf_te_high, den=pdf_ne, to_eval='L(3869)', Hbeta=1.)

	# Apply the ICF correction to the Ne++ abundance
	ICF = (-0.385 * o3_h / (o2_h + o3_h) + 1.365 + 0.022 * (o2_h + o3_h) / o3_h)
	ne_h = ne3_h * ICF # Second term is the low metallicity ICF correction from Izotov et al. (2006)

	return o3_h, o2_h, o_h, ne3_h, ne_h, ICF

def save_results(data_pickle):
	"""
	Saves the results of the PyNeb fit to a FITS file.

	Parameters:
		data_pickle (dict): A dictionary containing the results of the fit.

	Writes a FITS file to '../data/emline_fits/1002_pyneb_stats.fits' with the
	median and 2-sigma limits of the parameters in the dictionary.

	Returns:
		None
	"""
	columns = []
	names = ["te_O++", "te_O+", "ne", "12+log10(O++/H)", "12+log10(O+/H)", "12+log10(O/H)", "12+log10(Ne++/H)", "12+log10(Ne/H)", "ICF_Ne", "log10(Ne/O)"]
	for name in names:
		if ("EBV" in name) or ("A_HA" in name):
			balmer = data_pickle[name]
			balmer[balmer < 0.] = 0.
			med, err_low, err_up = util.stats(balmer)
			columns.append(fits.Column(name=name + "_2sig_limit", format="E", array=np.array([util.stats(balmer, upper_limit=98.)])))
		else:
			med, err_low, err_up = util.stats(data_pickle[name])
		columns.append(fits.Column(name=name + '_med', format='E', array=np.array([med])))
		columns.append(fits.Column(name=name + '_err_low', format='E', array=np.array([err_low])))
		columns.append(fits.Column(name=name + '_err_up', format='E', array=np.array([err_up])))

	# Create a FITS table HDU
	hdu = fits.BinTableHDU.from_columns(columns)

	# Write the HDU to a FITS file
	hdu.writeto('../data/emline_fits/1002_pyneb_stats.fits', overwrite=True)

	print("All Done!")
    

def main():
	import time
	t0 = time.time()
	# Define Number of CPUs to Use
	num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

	# Initialize Atom Objects that we need
	H1 = pn.RecAtom('H', 1)
	O3 = pn.Atom('O', 3)
	O2 = pn.Atom('O', 2)
	Ne3 = pn.Atom('Ne', 3)

	# Load the line PDFs
	data = load_data()
	#data = {key: val[:10000] for key, val in data_orig.items()}

	# Correct the fluxes for dust extinction
	pdf_EBV = 2.5 / (util.calzetti(4343.) - util.calzetti(4861.)) * np.log10((data["Hb_na_lflux_pdf"] / data["Hg_na_lflux_pdf"]) / 2.11)
	pdf_EBV[pdf_EBV < 0.] = 0.
	o3_5007, o3_4959, o3_4363, o2_all, ne3_3869, hbeta = correct_fluxes(data, pdf_EBV)

	# Calculate the electron temperatures
	print(f"Doing Electron Temperature Measurements...")
	with multiprocessing.Pool(processes=num_cpus) as pool:
		results = pool.starmap(calc_Te, zip(o3_4363, o3_5007))
	pdf_te_high, pdf_te_low = zip(*results)

	# Calculate the electron density
	print("Doing Electron Density Measurements...")
	pdf_density_O2_ratio = data["OII3728_lflux_pdf"]/data["OII3726_lflux_pdf"]
	with multiprocessing.Pool(processes=num_cpus) as pool:
		pdf_ne = pool.starmap(calc_ne, zip(pdf_density_O2_ratio, pdf_te_low))

	# Calculate the ionic abundances
	print("Doing Abundance Measurements...")
	with multiprocessing.Pool(processes=num_cpus) as pool:
		results = pool.starmap(calc_ionic_abundance, zip(o3_5007, o3_4959, hbeta, o2_all, ne3_3869, pdf_te_high, pdf_te_low, pdf_ne))

	o3_h, o2_h, o_h, ne3_h, ne_h, ICF = zip(*results)
	elapsed = time.time() - t0
	print(f"All Done with Abundance Measurements... {elapsed:.2f} seconds")




	########## BALMER DECREMENT
	print("Doing Balmer Decrement Measurements...")

	# Now let's get Balmer Decrement
	hbeta = H1.getEmissivity(tem=np.nanmedian(pdf_te_high), den=np.nanmedian(pdf_ne), wave=4861)
	hgamma = H1.getEmissivity(tem=np.nanmedian(pdf_te_high), den=np.nanmedian(pdf_ne), wave=4341)
	hdelta = H1.getEmissivity(tem=np.nanmedian(pdf_te_high), den=np.nanmedian(pdf_ne), wave=4101)
	hepsilon = H1.getEmissivity(tem=np.nanmedian(pdf_te_high), den=np.nanmedian(pdf_ne), wave=3970)
	h8 = H1.getEmissivity(tem=np.nanmedian(pdf_te_high), den=np.nanmedian(pdf_ne), wave=3889)

	hghb_int = hgamma/hbeta
	hdhb_int = hdelta/hbeta
	hehb_int = hepsilon/hbeta
	h8hb_int = h8/hbeta

	hghb_obs = data["Hg_na_lflux_pdf"]/data["Hb_na_lflux_pdf"]
	hdhb_obs = data["Hd_na_lflux_pdf"]/data["Hb_na_lflux_pdf"]
	hehb_obs = data["Hep_na_lflux_pdf"]/data["Hb_na_lflux_pdf"]
	h8hb_obs = data["H8_lflux_pdf"]/data["Hb_na_lflux_pdf"]

	pdf_EBV_hghb = 2.5/(util.calzetti(4861.) - util.calzetti(4341.))*np.log10(hghb_obs/hghb_int)
	pdf_A_HA_hghb = util.calzetti(6563.)*pdf_EBV_hghb

	pdf_EBV_hdhb = 2.5/(util.calzetti(4861.) - util.calzetti(4101.))*np.log10(hdhb_obs/hdhb_int)
	pdf_A_HA_hdhb = util.calzetti(6563.)*pdf_EBV_hdhb

	pdf_EBV_hehb = 2.5/(util.calzetti(4861.) - util.calzetti(3970.))*np.log10(hehb_obs/hehb_int)
	pdf_A_HA_hehb = util.calzetti(6563.)*pdf_EBV_hehb

	pdf_EBV_h8hb = 2.5/(util.calzetti(4861.) - util.calzetti(3889.))*np.log10(h8hb_obs/h8hb_int)
	pdf_A_HA_h8hb = util.calzetti(6563.)*pdf_EBV_h8hb



	#### Now Pickle It All Out
	data_pickle = {} # initialize the pickle

	data_pickle["te_O++"] = pdf_te_high
	data_pickle["te_O+"] = pdf_te_low
	data_pickle["ne"] = pdf_ne
	data_pickle["12+log10(O++/H)"] = 12.+np.log10(o3_h)
	data_pickle["12+log10(O+/H)"] = 12.+np.log10(o2_h)
	data_pickle["12+log10(O/H)"] = 12.+np.log10(o_h)
	data_pickle["12+log10(Ne++/H)"] = 12.+np.log10(ne3_h)
	data_pickle["ICF_Ne"] = ICF
	data_pickle["12+log10(Ne/H)"] = 12.+np.log10(ne_h)
	data_pickle["log10(Ne/O)"] = np.log10(np.asarray(ne_h)/np.asarray(o_h))
	data_pickle["EBV_hghb"] = pdf_EBV_hghb
	data_pickle["EBV_hdhb"] = pdf_EBV_hdhb
	data_pickle["A_HA_hghb"] = pdf_A_HA_hghb
	data_pickle["A_HA_hdhb"] = pdf_A_HA_hdhb


	# Pickle out all the PDFs
	with open("../data/emline_fits/1002_pyneb_pdf.pkl", "wb") as file:
		pickle.dump(data_pickle, file, protocol=pickle.HIGHEST_PROTOCOL)


	# Now let's make the stat measurements

	columns = []
	names = ["te_O++","te_O+","ne","12+log10(O++/H)","12+log10(O+/H)","12+log10(O/H)","12+log10(Ne++/H)","12+log10(Ne/H)","ICF_Ne","log10(Ne/O)","EBV_hghb","EBV_hdhb","A_HA_hghb","A_HA_hdhb"]
	for name in names:
		if ("EBV" in name) or ("A_HA" in name):
			balmer = data_pickle[name]
			balmer[balmer < 0.] = 0.
			med, err_low, err_up = util.stats(balmer)
			columns.append(fits.Column(name=name + "_2sig_limit",format="E", array=np.array([util.stats(balmer,upper_limit=98.)])))
		else:
			med, err_low, err_up = util.stats(data_pickle[name])
		
		columns.append(fits.Column(name=name + '_med', format='E', array=np.array([med])))
		columns.append(fits.Column(name=name + '_err_low', format='E', array=np.array([err_low])))
		columns.append(fits.Column(name=name + '_err_up', format='E', array=np.array([err_up])))	

	# Create a FITS table HDU
	hdu = fits.BinTableHDU.from_columns(columns)

	# Write the HDU to a FITS file
	hdu.writeto('../data/emline_fits/1002_pyneb_stats.fits', overwrite=True)

	print("All Done!")




	

if __name__ == '__main__':
    main()

