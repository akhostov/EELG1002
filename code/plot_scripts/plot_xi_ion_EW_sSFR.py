import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
import numpy as np
from astropy.io import fits
import h5py
import pickle
from astropy.cosmology import FlatLambdaCDM
import sys


sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20
sys.path.insert(0,"..")
import util


def convert_str_to_double(arr):
	"""
	Converts a string or tuple of strings to a double or tuple of doubles.

	Parameters:
		arr (str or tuple of str): The string or tuple of strings to be converted.

	Returns:
		double or tuple of double: The converted double or tuple of doubles.
	"""
	if isinstance(arr, tuple):
		return tuple(np.double(aa) for aa in arr)
	else:
		return np.double(arr)

# Sun et al. (2023) -- # https://iopscience.iop.org/article/10.3847/1538-4357/acd53c/pdf
def Sun_2023(ax, EW=False, sSFR=False):

	# Load the Data
	Sun_EW_OIII5007,Sun_EW_OIII5007_err,Sun_EW_OIII4959,Sun_EW_OIII4959_err,\
		Sun_EW_HB,Sun_EW_HB_err,Sun_SFR, Sun_SFR_err,Sun_mass, Sun_mass_err,\
			Sun_xi_ion, Sun_xi_ion_err = np.loadtxt("../../data/literature_measurements/Sun_et_al_2023.txt",unpack=True,usecols=np.arange(5,17))
	Sun_EW_HB[Sun_EW_HB<1] = Sun_EW_HB_err[Sun_EW_HB<1]

	if sSFR == True:
		Sun_mass_err = Sun_mass_err*np.log(10.)*pow(10,Sun_mass)
		Sun_mass = pow(10,Sun_mass)

		Sun_sSFR_med, Sun_sSFR_elow, Sun_sSFR_eupp = np.array([ util.stats(util.sampling(sfr,sfr_err)/util.sampling(mass,mass_err)) for sfr,sfr_err,mass,mass_err in zip(Sun_SFR,Sun_SFR_err,Sun_mass,Sun_mass_err) ]).T
		Sun_sSFR_elow = Sun_sSFR_elow/(np.log(10.)*Sun_sSFR_med)
		Sun_sSFR_eupp = Sun_sSFR_eupp/(np.log(10.)*Sun_sSFR_med)
		Sun_sSFR_med = np.log10(Sun_sSFR_med)

		ax.errorbar(Sun_sSFR_med,Sun_xi_ion,xerr=(Sun_sSFR_elow,Sun_sSFR_eupp),yerr=Sun_xi_ion_err,ls="none",marker="^",ms=6,mfc=tableau20("Light Brown"),mec=tableau20("Brown"),\
									ecolor=tableau20("Brown"),\
									mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

	if EW == True:
		ax.errorbar(Sun_EW_OIII5007+Sun_EW_OIII4959+Sun_EW_HB,
			  		Sun_xi_ion,yerr=(Sun_xi_ion_err),
					marker="d",ls="none",ms=6,\
					mec=tableau20("Red"),mfc=tableau20("Light Red"),ecolor=tableau20("Grey"),\
					mew=0.5,capsize=2,capthick=1,elinewidth=0.5)

	return ax

def Tang_2019(ax):

	# Load in the Data
	Ta19_EW,Ta19_xi = np.loadtxt("../../data/literature_measurements/Tang_et_al_2019.txt",unpack=True)
	ax.plot(Ta19_EW,Ta19_xi,marker="s",ls="none",ms=3,mec=tableau20("Purple"),mfc="none",mew=0.2)


	# Also Plot the Model assuming Calzetti Attenuation
	# This model is only valid for 225A < EW_0([OIII]) < 2500A
	func = lambda x, a, b: a*x + b
	params = np.array([0.76,23.27])
	perr = np.array([0.05,0.15])
	xx = np.linspace(225,2500,100)
	
	ax.plot(xx,func(np.log10(xx),*params),ls="--",color=tableau20("Purple"),lw=1.5)
	bound_upper = func(np.log10(xx), *(params +perr))
	bound_lower = func(np.log10(xx), *(params -perr))

	ax.fill_between(xx,bound_lower,bound_upper,color=tableau20("Purple"),alpha=0.2)

	return ax

def Tang_2023(ax):

	# Initialize lists to store data
	Ta23_EW,Ta23_xi_ion = [],[]

	# Read the file line by line
	with open("../../data/literature_measurements/Tang_et_al_2023.txt", "r") as file:
		for line in file:
			# Skip comment lines or empty lines
			if line.startswith("#") or not line.strip():
				continue
			# Stop reading when reaching the second table
			if "Next Table" in line:
				break

			# Store the data
			columns = line.split()
			Ta23_EW.append(float(columns[5]))
			Ta23_xi_ion.append(float(columns[8]))

	# close the file
	file.close()

	# Convert the lists to numpy arrays
	Ta23_EW = np.array(Ta23_EW)
	Ta23_xi_ion = np.array(Ta23_xi_ion)

	ax.plot(Ta23_EW,Ta23_xi_ion,marker="^",ls="none",ms=3,
							mfc="none",mec=tableau20("Green"),mew=0.2)

	return ax

def Endsley_2021(ax):

	# Load in the Data
	En21_ew, En21_xi = np.loadtxt("../../data/literature_measurements/Endsley_et_al_2021.txt",unpack=True,usecols=(13,16))
	
	ax.plot(En21_ew,En21_xi,marker="v",ls="none",ms=3,
							mfc="none",mec=tableau20("Brown"),
							mew=0.2)	

	return ax

def Stark_2017(ax):

	# Load in the Data
	St17_ew,St17_xi = np.loadtxt("../../data/literature_measurements/Stark_et_al_2017.txt",unpack=True,usecols=(2,5))

	ax.plot(St17_ew,St17_xi,marker=">",ls="none",ms=3,
						mfc="none",mec=tableau20("Sky Blue"),
						mew=0.2)

	return ax

def Matthee_2023(ax,EW=False, sSFR=False):

	# Load in the Data
	# log10_Mass    log10_Mass_err      EW_0_O3HB       EW_0_O3HB_err       SFR_HB      SFR_HB_err      log10_xi_ion        log10_xi_ion_err    12+log10(O/H)       12+log10(O/H)_err
	eiger_mass, eiger_mass_err, \
		eiger_EW, eiger_EW_err, \
		eiger_sfr, eiger_sfr_err, \
		eiger_xi_ion, eiger_xi_ion_err = np.loadtxt("../../data/literature_measurements/Matthee_et_al_2023.txt",unpack=True,usecols=(0,1,2,3,4,5,6,7))

	if EW == True:
		ax.errorbar([eiger_EW],[eiger_xi_ion],xerr=[eiger_EW_err],yerr=[eiger_xi_ion_err],ls="none",marker="o",ms=6,mfc=tableau20("Light Red"),mec=tableau20("Red"),\
									ecolor=tableau20("Red"),\
									mew=1.0,capsize=2,capthick=1,elinewidth=0.5)


	if sSFR == True:
		eiger_mass_err = eiger_mass_err*np.log(10.)*pow(10,eiger_mass)
		eiger_mass = pow(10,eiger_mass)

		eiger_ssfr_med, eiger_ssfr_elow, eiger_ssfr_eupp = util.stats(util.sampling(eiger_sfr,eiger_sfr_err)/util.sampling(eiger_mass,eiger_mass_err))
		eiger_ssfr_elow = eiger_ssfr_elow/(np.log(10.)*eiger_ssfr_med)
		eiger_ssfr_eupp = eiger_ssfr_eupp/(np.log(10.)*eiger_ssfr_med)
		eiger_ssfr_med = np.log10(eiger_ssfr_med)

		ax.errorbar([eiger_ssfr_med],[eiger_xi_ion],xerr=([eiger_ssfr_elow],[eiger_ssfr_eupp]),yerr=[eiger_xi_ion_err],ls="none",marker="o",ms=6,mfc=tableau20("Light Red"),mec=tableau20("Red"),\
									ecolor=tableau20("Red"),\
									mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

	return ax

def Chen_2024(ax):

	# Load in the Data
	Ch24_EW, Ch24_EW_elow, Ch24_EW_eupp, Ch24_xi_ion, Ch24_xi_ion_elow, Ch24_xi_ion_eupp = np.loadtxt("../../data/literature_measurements/Chen_et_al_2024.txt",unpack=True,usecols=(0,1,2,3,4,5))
	
	ax.errorbar(Ch24_EW,Ch24_xi_ion,xerr=(Ch24_EW_elow,Ch24_EW_eupp),yerr=(Ch24_xi_ion_elow,Ch24_xi_ion_eupp),ls="none",marker="p",ms=6,mec=tableau20("Green"),mfc=tableau20("Light Green"),\
								ecolor=tableau20("Green"),\
								mew=0.5,capsize=2,capthick=1,elinewidth=0.5)

	return ax

def Rinaldi_2024(ax):

	# Load in the Data
	Ri24_sSFR,Ri24_xi_ion = np.loadtxt("../../data/literature_measurements/Rinaldi_et_al_2024.txt",unpack=True,usecols=(0,1),skiprows=26)

	ax.plot(Ri24_sSFR, Ri24_xi_ion,ls="none",marker="p",ms=6,mfc=tableau20("Light Green"),mec=tableau20("Green"),mew=1.0)

	return ax


def Whitler_2023(ax):
	# Load in the Data
	# ID 	RA				DEC				F200W	F200W_elow	F200_eupp	Beta	Beta_elow Beta_eupp	Redshift	Redshift_elow	Redshift_eupp	Mass	Mass_elow	Mass_eupp	sSFR	sSFR_elow	sSFR_eupp	xi_ion	xi_ion_elow	xi_ion_eupp	
	Wh23_sSFR,Wh23_xi_ion = np.loadtxt("../../data/literature_measurements/Whitler_et_al_2023.txt",unpack=True,usecols=(-6,-3))

	Wh23_sSFR = np.log10(Wh23_sSFR)-9.
	ax.plot(Wh23_sSFR, Wh23_xi_ion,ls="none",marker="d",ms=6,mfc=tableau20("Light Purple"),mec=tableau20("Purple"),mew=1.0)

	return ax

def Castellano_2023(ax):

	# Load in the Data
	Ca23_sSFR,Ca23_sSFR_elow,Ca23_sSFR_eupp,Ca23_xi_ion,Ca23_xi_ion_elow,Ca23_xi_ion_eupp = np.loadtxt("../../data/literature_measurements/Castellano_et_al_2023.txt",unpack=True)

	ax.errorbar(Ca23_sSFR,Ca23_xi_ion,xerr=(Ca23_sSFR_elow,Ca23_sSFR_eupp),yerr=(Ca23_xi_ion_elow,Ca23_xi_ion_eupp),
			 		ls="none",marker="s",ms=6,mfc=tableau20("Light Orange"),mec=tableau20("Orange"),\
					ecolor=tableau20("Orange"),\
					mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

	return ax

def Schaerer2016(ax,EW=False,sSFR=False):

	# Load in the data
	xi_ion_0, err_xi_ion_0, SFR, Mass, EW_Hbeta, EW_O3_4959, EW_O3_5007 = np.loadtxt("../../data/literature_measurements/Schaerer_et_al_2016.txt",unpack=True,usecols=(2,3,4,5,6,7,8))

	if EW:
		ax.errorbar(EW_Hbeta+EW_O3_4959+EW_O3_5007, xi_ion_0, yerr=err_xi_ion_0, ls="none",marker="H",ms=6,mfc=tableau20("Light Grey"),mec=tableau20("Grey"),mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

	if sSFR:
		ax.errorbar(np.log10(SFR) - Mass - 0.20, xi_ion_0, yerr=err_xi_ion_0, ls="none",marker="H",ms=6,mfc=tableau20("Light Grey"),mec=tableau20("Grey"),mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

	return ax


def Onodera_2020(ax,EW=False,sSFR=False):

	# Load in the data
	F_HB, E_HB, F_O3, EW_O3, err_EW_O3, xi_ion_0, err_xi_ion_0, log10_mass, err_log10_mass, log10_SFR_UV , err_log10_SFR_UV = np.loadtxt("../../data/literature_measurements/Onodera_et_al_2020.txt",unpack=True,usecols=(2,3,4,6,7,12,13,14,15,18,19))

	# Calculate how much to increase the EW_O3 to convert to [OIII]+Hbeta
	frac_HB  = F_HB/(F_HB+F_O3)
	EW_O3_HB = (1.+frac_HB)*EW_O3
	sSFR_UV = log10_SFR_UV - log10_mass

	if EW:
		det = (np.abs(E_HB) < 99.) & (np.abs(err_xi_ion_0) < 99.)
		ax.errorbar(EW_O3_HB[det], xi_ion_0[det], yerr=err_xi_ion_0[det], ls="none",marker="<",ms=3,mfc="none",mec=tableau20("Light Pink"),ecolor=tableau20("Light Pink"),mew=0.5,capsize=2,capthick=1,elinewidth=0.5)

		uplims = (E_HB > 100.) & (err_xi_ion_0 > 100.)
		ax.errorbar(EW_O3_HB[uplims], xi_ion_0[uplims], xerr=EW_O3_HB[uplims]*0.1, yerr=xi_ion_0[uplims]*0.0025, xuplims=True, uplims=True, ls="none",marker="<",ms=3,mfc="none",mec=tableau20("Light Pink"),ecolor=tableau20("Light Pink"),mew=0.5,capsize=1.5,capthick=1,elinewidth=0.5)

	if sSFR:
		det = (err_xi_ion_0 < 99.)
		ax.errorbar(sSFR_UV[det], xi_ion_0[det], yerr=err_xi_ion_0[det], ls="none",marker="<",ms=6,mfc=tableau20("Light Pink"),mec=tableau20("Pink"),mew=1.0,capsize=2,capthick=1,elinewidth=0.5,zorder=1)

		uplims = (err_xi_ion_0 > 100.)
		ax.errorbar(sSFR_UV[uplims], xi_ion_0[uplims], yerr=0.1, uplims=True, ls="none",marker="<",ms=6,mfc=tableau20("Light Pink"),mec=tableau20("Pink"),ecolor=tableau20("Pink"),mew=1.0,capsize=2,capthick=1,elinewidth=0.5,zorder=1)

	return ax


cosmo = FlatLambdaCDM(H0=70.,Om0=0.3)

# Let's first define the figure parameters
fig = plt.figure()
fig.set_size_inches(8,4)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.rcParams["xtick.labelsize"] = "7"
plt.rcParams["ytick.labelsize"] = "7"



##################################################################################################################################
#############						Ionizing Photon Production Efficiency vs [OIII] + Hbeta EW						 #############
##################################################################################################################################



ax1 = fig.add_subplot(121)


##############################
##### Literature Measurements

# Plot Sun et al. (2023)
ax1 = Sun_2023(ax1,EW=True)

# Plot Tang et al. (2023)
ax1 = Tang_2023(ax1)

# Plot Tang et al. (2019)
ax1 = Tang_2019(ax1)

# Plot Endsley et al. (2021)
ax1 = Endsley_2021(ax1)

# Plot Stark et al. (2017)
ax1 = Stark_2017(ax1)

# EIGER (Matthee+23)
ax1 = Matthee_2023(ax1,EW=True)

# CEERS (Chen et al. 2024)
ax1 = Chen_2024(ax1)

# Schaerer et al. (2016)
ax1 = Schaerer2016(ax1,EW=True)

# Onodera et al. (2020)
ax1 = Onodera_2020(ax1,EW=True)

##############################
##### EELG1002

# Load in EELG1002 Results
with open("../../data/xi_ion_measurements.pkl","rb") as f: uv_measurements = pickle.load(f)

cigale_xi_ion_elow = uv_measurements["cigale_xi_ion"][1]/(np.log(10.)*uv_measurements["cigale_xi_ion"][0])
cigale_xi_ion_eupp = uv_measurements["cigale_xi_ion"][2]/(np.log(10.)*uv_measurements["cigale_xi_ion"][0])
cigale_xi_ion = np.log10(uv_measurements["cigale_xi_ion"][0])

ax1.errorbar([uv_measurements["cigale_O3HB_EW"][0]],[cigale_xi_ion],
            	xerr=([uv_measurements["cigale_O3HB_EW"][2]],[uv_measurements["cigale_O3HB_EW"][1]]),
                yerr=([cigale_xi_ion_elow],[cigale_xi_ion_eupp]),
                ls="none",marker="*",ms=15,
				mfc=util.color_scheme("Cigale",mfc=True),
				mec=util.color_scheme("Cigale",mec=True),
				ecolor=util.color_scheme("Cigale",mfc=True),
				mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

bagpipes_xi_ion_elow = uv_measurements["bagpipes_xi_ion"][1]/(np.log(10.)*uv_measurements["bagpipes_xi_ion"][0])
bagpipes_xi_ion_eupp = uv_measurements["bagpipes_xi_ion"][2]/(np.log(10.)*uv_measurements["bagpipes_xi_ion"][0])
bagpipes_xi_ion = np.log10(uv_measurements["bagpipes_xi_ion"][0])

ax1.errorbar([uv_measurements["bagpipes_O3HB_EW"][0]],[bagpipes_xi_ion],
            	xerr=([uv_measurements["bagpipes_O3HB_EW"][2]],[uv_measurements["bagpipes_O3HB_EW"][1]]),
                yerr=([bagpipes_xi_ion_elow],[bagpipes_xi_ion_eupp]),
				mfc=util.color_scheme("Bagpipes",mfc=True),
				mec=util.color_scheme("Bagpipes",mec=True),
				ecolor=util.color_scheme("Bagpipes",mfc=True),
                marker="*",ms=15,ls="none",\
				mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

handles = [	Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),label=r"\textbf{\textit{EELG1002 (Cigale)}}"),
			Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),label=r"\textbf{\textit{EELG1002 (Bagpipes)}}")]

leg = ax1.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg)	

handles = [ Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"EIGER (Ma+23; $z \sim 5 - 7$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"Su+23 ($z \sim 6.2$)"),			
			Line2D([],[],ls="none",marker="p",ms=4,mec=tableau20("Green"),mfc=tableau20("Light Green"),label=r"CEERS (Ch+23; $z \sim 5 - 8$)"),
			Line2D([],[],ls="--",color=tableau20("Purple"),label=r"Ta+19 ($z \sim 1.3 - 2.4$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Purple"),mfc="none",label=r"EELGs (Ta+19; $z \sim 1.3 - 2.4$)"),
			Line2D([],[],ls="none",marker="^",ms=4,mec=tableau20("Green"),mfc="none",label=r"CEERS (Ta+23; $z \sim 7 - 9$)"),
			Line2D([],[],ls="none",marker="v",ms=4,mec=tableau20("Brown"),mfc="none",label=r"En+21 ($z \sim 7$)"),
			Line2D([],[],ls="none",marker=">",ms=4,mec=tableau20("Sky Blue"),mfc="none",label=r"St+17 ($z \sim 7 - 9$)"),
			Line2D([],[],ls="none",marker="<",ms=4,mec=tableau20("Light Pink"),mfc="none",label=r"On+20 ($z \sim 3 - 3.7$)"),
			Line2D([],[],ls="none",marker="H",ms=4,mec=tableau20("Grey"),mfc=tableau20("Light Grey"),label=r"LyC Leakers (Sc+16; $z \sim 0.3$)")]

leg2 = ax1.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg2)	

ax1.set_yticks([23.5,24.0,24.5,25.0,25.5,26.0,26.5])
ax1.set_yticks(np.arange(23.5,27.,0.1),minor=True)
ax1.set_yticklabels([23.5,24.0,24.5,25.0,25.5,26.0,26.5])


ax1.set_ylim(23.6,26.3)
ax1.set_xlim(100.,6000.)
ax1.set_xscale("log")

ax1.set_xticks([100.,300.,600.,1000.,3000.,6000.])
ax1.set_xticks([100.,200.,300.,450.,600.,800.,1000.,2000.,3000.,4500.,6000.],minor=True)
ax1.set_xticklabels(["100","300","600","1000","3000","6000"])

ax1.set_xlabel(r"$\log_{10}$ EW$_0$([O{\sc iii}]$+$H$\beta$) (\AA)",fontsize=8)#, **hfont)
ax1.set_ylabel(r"$\log_{10} \xi_\textrm{ion}^\textrm{H{\sc ii}}$ (erg$^{-1}$ Hz)",fontsize=8)#, **hfont)












##################################################################################################################################
#############					Ionizing Photon Production Efficiency vs Specific Star Formation Rate				 #############
##################################################################################################################################


ax2 = fig.add_subplot(122)

##############################
##### Literature Measurements

# EIGER 
ax2 = Matthee_2023(ax2,sSFR=True)

# MIDIS (Rinaldi et al. 2024) 
ax2 = Rinaldi_2024(ax2)

# CEERS (Whitler et al. 2023) -- # https://ui.adsabs.harvard.edu/abs/2024MNRAS.529..855W/abstract
ax2 = Whitler_2023(ax2)

# Castellano et al. (2023) -- # https://www.aanda.org/articles/aa/pdf/2023/07/aa46069-23.pdf
ax2 = Castellano_2023(ax2)

# Sun et al. (2023) -- # https://iopscience.iop.org/article/10.3847/1538-4357/acd53c/pdf
ax2 = Sun_2023(ax2,sSFR=True)

# Schaerer et al. (2016)
ax2 = Schaerer2016(ax2,sSFR=True)

# Onodera et al. (2020)
ax2 = Onodera_2020(ax2,sSFR=True)

##############################
##### EELG1002

# Load in EELG1002 Results
with open("../../data/xi_ion_measurements.pkl","rb") as f: uv_measurements = pickle.load(f)
cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
bagpipes = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")

cigale_xi_ion_elow = uv_measurements["cigale_xi_ion"][1]/(np.log(10.)*uv_measurements["cigale_xi_ion"][0])
cigale_xi_ion_eupp = uv_measurements["cigale_xi_ion"][2]/(np.log(10.)*uv_measurements["cigale_xi_ion"][0])
cigale_xi_ion = np.log10(uv_measurements["cigale_xi_ion"][0])

# Get the Hbeta Emission Line SFR from the GMOS Spectra
emlines = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data
mask = emlines["line_ID"] == "Hb_na"
emlines = emlines[mask]

SFR_Hbeta      = (4.4e-42*2.86)*util.lineFlux_to_Luminosity(lineFlux=emlines["lineflux_med"], redshift=0.8275)
SFR_Hbeta_elow = (4.4e-42*2.86)*util.lineFlux_to_Luminosity(lineFlux=emlines["lineflux_elow"],redshift=0.8275)
SFR_Hbeta_eupp = (4.4e-42*2.86)*util.lineFlux_to_Luminosity(lineFlux=emlines["lineflux_eupp"],redshift=0.8275)


# Cigale
cigale_mass = cigale["bayes.stellar.m_star"]
cigale_mass_err = cigale["bayes.stellar.m_star_err"]

# Bagpipes
bagpipes_mass = pow(10,bagpipes["median"][10])
bagpipes_mass_err_low,bagpipes_mass_err_up = bagpipes_mass - pow(10,bagpipes["conf_int"][0][10]), pow(10,bagpipes["conf_int"][1][10]) - bagpipes_mass

# GMOS
GMOS_bagpipes = SFR_Hbeta/bagpipes_mass
GMOS_bagpipes_err_low = GMOS_bagpipes*np.sqrt( (SFR_Hbeta_elow/SFR_Hbeta)**2 + (bagpipes_mass_err_low/bagpipes_mass)**2 )
GMOS_bagpipes_err_up = GMOS_bagpipes*np.sqrt( (SFR_Hbeta_eupp/SFR_Hbeta)**2 + (bagpipes_mass_err_up/bagpipes_mass)**2 )

GMOS_bagpipes_err_low = GMOS_bagpipes_err_low/(np.log(10.)*GMOS_bagpipes)
GMOS_bagpipes_err_up = GMOS_bagpipes_err_up/(np.log(10.)*GMOS_bagpipes)
GMOS_bagpipes = np.log10(GMOS_bagpipes)

GMOS_cigale = SFR_Hbeta/cigale_mass
GMOS_cigale_err_low = GMOS_cigale * np.sqrt( (SFR_Hbeta_elow/SFR_Hbeta)**2 + (cigale_mass_err/cigale_mass)**2 )
GMOS_cigale_err_up = GMOS_cigale * np.sqrt( (SFR_Hbeta_eupp/SFR_Hbeta)**2 + (cigale_mass_err/cigale_mass)**2 )

GMOS_cigale_err_low = GMOS_cigale_err_low/(np.log(10.)*GMOS_cigale)
GMOS_cigale_err_up = GMOS_cigale_err_up/(np.log(10.)*GMOS_cigale)
GMOS_cigale = np.log10(GMOS_cigale)

ax2.errorbar(GMOS_cigale,[cigale_xi_ion],
				xerr=(GMOS_cigale_err_low,GMOS_cigale_err_up),
				yerr=([cigale_xi_ion_elow],[cigale_xi_ion_eupp]),
				ls="none",marker="*",ms=15,
				mfc=util.color_scheme("Cigale",mfc=True),
				mec=util.color_scheme("Cigale",mec=True),
				ecolor=util.color_scheme("Cigale",mfc=True),
				mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

ax2.errorbar(GMOS_bagpipes,[bagpipes_xi_ion],
				xerr=(GMOS_bagpipes_err_low,GMOS_bagpipes_err_up),
				yerr=([bagpipes_xi_ion_elow],[bagpipes_xi_ion_eupp]),
				ls="none",marker="*",ms=15,
				mfc=util.color_scheme("Bagpipes",mfc=True),
				mec=util.color_scheme("Bagpipes",mec=True),
				ecolor=util.color_scheme("Bagpipes",mfc=True),
				mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

handles = [	Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),label=r"\textbf{\textit{EELG1002 (Cigale; sSFR(H$\beta$))}}"),
			Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),label=r"\textbf{\textit{EELG1002 (Bagpipes; sSFR(H$\beta$))}}")]

leg = ax2.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg)	


handles = [ Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"EIGER (Ma+23; $z \sim 5 - 7$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Brown"),mfc=tableau20("Light Brown"),label=r"Su+23 ($z \sim 6.2$)"),			
			Line2D([],[],ls="none",marker="p",ms=4,mec=tableau20("Green"),mfc=tableau20("Light Green"),label=r"MIDIS (Ri+23; $z \sim 7 - 8$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),label=r"CEERS (Wh+23; $z \sim 7 - 9$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"VANDELS (Ca+23; $z \sim 2 - 5$)"),
			Line2D([],[],ls="none",marker="<",ms=4,mec=tableau20("Pink"),mfc=tableau20("Light Pink"),label=r"On+20 ($z \sim 3 - 3.7$)"),
			Line2D([],[],ls="none",marker="H",ms=4,mec=tableau20("Grey"),mfc=tableau20("Light Grey"),label=r"LyC Leakers (Sc+16; $z \sim 0.3$)")]

		
leg2 = ax2.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg2)	

ax2.set_yticks([23.5,24.0,24.5,25.0,25.5,26.0,26.5])
ax2.set_yticks(np.arange(23.5,27.,0.1),minor=True)
ax2.set_yticklabels([])


ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(-10.2,-6.)

ax2.set_xlabel(r"$\log_{10}$ sSFR (yr$^{-1}$)",fontsize=8)#, **hfont)



fig.savefig("../../plots/xi_ion_O3HB_EW_sSFR.png",bbox_inches="tight",dpi=300,format="png")


