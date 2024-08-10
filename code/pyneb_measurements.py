import numpy as np
import pyneb as pn
from astropy.io import fits
import pickle
import util

# Load in the Emission Line Fluxes
file = open("../data/emline_fits/1002_lineflux_pdfs.pkl","rb")
data = pickle.load(file)
file.close()

# Recombination line Atom class
H1 = pn.RecAtom('H', 1)

# Collisionally-excited line Atom class
O3 = pn.Atom('O', 3)  # = [OIII]
O2 = pn.Atom('O', 2)  # = [OII]
Ne3 = pn.Atom('Ne', 3)  # = [NeIII]

# Define initial EBV which we will base on the Hb/Hg ratio
pdf_EBV = 2.5/(util.calzetti(4343.) - util.calzetti(4861.)) * np.log10( (data["Hb_na_lflux_pdf"]/data["Hg_na_lflux_pdf"])/2.11 )
pdf_EBV[pdf_EBV<0.] = 0. # ignore negative EBVs



##################################################
# Dust Correct Lines of Interest for Next Section
o3_5007 = data["OIII5007c_lflux_pdf"] * pow(10,0.4*pdf_EBV*util.calzetti(5007.))
o3_4959 = data["OIII4959c_lflux_pdf"] * pow(10,0.4*pdf_EBV*util.calzetti(4959.))
o3_4363 = data["OIII4363_lflux_pdf"]*pow(10,0.4*pdf_EBV*util.calzetti(4363.))
o2_all = data["OII_lflux_pdf"]*pow(10,0.4*pdf_EBV*util.calzetti(3727.))
ne3_3869 = data["NeIII3869_lflux_pdf"]*pow(10,0.4*pdf_EBV*util.calzetti(3869.))
hbeta =  data["Hb_na_lflux_pdf"] * pow(10,0.4*pdf_EBV*util.calzetti(4861.))

### ELECTRON TEMPERATURE
print("Doing Electron Temperature Measurements...")
pdf_auroral_O3_ratio = o3_4363/o3_5007
pdf_density_O2_ratio = data["OII3728_lflux_pdf"]/data["OII3726_lflux_pdf"]

pdf_te_high = O3.getTemDen(int_ratio=pdf_auroral_O3_ratio,den=1000., to_eval = 'L(4363) / (L(5007))')
pdf_te_low = (-0.577 + 1e-4*pdf_te_high*( 2.065 - 0.498 * 1e-4 * pdf_te_high))*1e4 # Izotov et al. (2006)


### ELECTRON DENSITY
print("Doing Electron Density Measurements...")
pdf_ne = O2.getTemDen(int_ratio=pdf_density_O2_ratio,tem=pdf_te_low,to_eval = 'L(3729) / L(3726)')


##### IONIC ABUNDANCE
print("Doing Abundance Measurements...")
# Oxygen Abundance                                   
pdf_R3 = (o3_5007 + o3_4959)/hbeta
pdf_R2 = data["OII_lflux_pdf"]/data["Hb_na_lflux_pdf"]*pow(10,0.4*pdf_EBV*(util.calzetti(3727.) - util.calzetti(4861.) ))

o3_h = O3.getIonAbundance(int_ratio=pdf_R3, tem=pdf_te_high, den=pdf_ne, to_eval='L(4959)+L(5007)',Hbeta=1.)
o2_h = O2.getIonAbundance(int_ratio=pdf_R2, tem=pdf_te_low, den=pdf_ne, to_eval='L(3726)+L(3729)', Hbeta=1.)
o_h = o3_h+o2_h

# Neon Abundance
pdf_Ne3Hb = ne3_3869/hbeta
ne3_h = Ne3.getIonAbundance(int_ratio=pdf_Ne3Hb, tem=pdf_te_high, den=pdf_ne, to_eval='L(3869)', Hbeta=1.)
ICF = (-0.385*o3_h/(o2_h+o3_h) + 1.365 + 0.022*(o2_h+o3_h)/o3_h)
ne_h = ne3_h*ICF # Second term is the low metallicity ICF correction from Izotov et al. (2006)


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
data_pickle["12+log10(O/H)"] = 12.+np.log10(o3_h+o2_h)
data_pickle["12+log10(Ne++/H)"] = 12.+np.log10(ne3_h)
data_pickle["ICF_Ne"] = ICF
data_pickle["12+log10(Ne/H)"] = 12.+np.log10(ne_h)
data_pickle["log10(Ne/O)"] = np.log10(ne_h/o_h)
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