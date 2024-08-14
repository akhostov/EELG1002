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


#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Helvetica']
#hfont = {'fontname':'Helvetica'}

cosmo = FlatLambdaCDM(H0=70.,Om0=0.3)

# Load in Rinaldi et al. (2023)
_,_,_,Ri23_zphot,_,_,_,\
	Ri23_mass,Ri23_mass_elow,Ri23_mass_eupp,_,_,_,_,_,_,\
		Ri23_MUV,Ri23_MUV_elow,Ri23_MUV_eupp,_,_,_,\
		Ri23_xi_ion,Ri23_xi_ion_elow,Ri23_xi_ion_eupp = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Rinaldi2023_xi_ion.txt" ,unpack=True,dtype=str)        

Ri23_mass, Ri23_mass_elow, Ri23_mass_eupp = convert_str_to_double((Ri23_mass, Ri23_mass_elow, Ri23_mass_eupp))

Ri23_mass = np.double(Ri23_mass) 
Ri23_mass_elow = np.double(Ri23_mass_elow)
Ri23_mass_eupp = np.double(Ri23_mass_eupp)
Ri23_MUV = np.double(Ri23_MUV)
Ri23_MUV_elow = np.double(Ri23_MUV_elow)
Ri23_MUV_eupp = np.double(Ri23_MUV_eupp)
Ri23_xi_ion = np.double(Ri23_xi_ion)
Ri23_xi_ion_elow = np.double(Ri23_xi_ion_elow)
Ri23_xi_ion_eupp = np.double(Ri23_xi_ion_eupp)


# Load in Simmonds et al. (2023)
_,_,_,Si23_zspec,_,_,\
	Si23_xi_ion,Si23_xi_ion_elow,Si23_xi_ion_eupp,\
	Si23_mass,Si23_mass_elow,Si23_mass_eupp = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Simmonds2023_xi_ion.txt",unpack=True,dtype=str)

Si23_xi_ion = np.double(Si23_xi_ion)
Si23_xi_ion_elow = np.double(Si23_xi_ion_elow)
Si23_xi_ion_eupp = np.double(Si23_xi_ion_eupp)
Si23_mass = np.double(Si23_mass)
Si23_mass_elow = np.double(Si23_mass_elow)
Si23_mass_eupp = np.double(Si23_mass_eupp)

# Sun et al. (2023)
_,_,_,Sun23_redshift,Sun23_redshift_err,\
	Sun23_EW_OIII5007,Sun23_EW_OIII5007_err,\
	Sun23_EW_OIII4959,Sun23_EW_OIII4959_err,\
	Sun23_EW_HB,Sun23_EW_HB_err,\
	Sun23_Mass,Sun23_Mass_err,\
	Sun23_xi_ion,Sun23_xi_ion_err = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Sun2023_xi_ion_EW.txt",unpack=True,dtype=str)

Sun23_redshift = np.double(Sun23_redshift)
Sun23_redshift_err = np.double(Sun23_redshift_err)
Sun23_EW_OIII5007 = np.double(Sun23_EW_OIII5007)
Sun23_EW_OIII5007_err = np.double(Sun23_EW_OIII5007_err)
Sun23_EW_OIII4959 = np.double(Sun23_EW_OIII4959)
Sun23_EW_OIII4959_err = np.double(Sun23_EW_OIII4959_err)
Sun23_EW_HB = np.double(Sun23_EW_HB)
Sun23_EW_HB_err = np.double(Sun23_EW_HB_err)
Sun23_Mass = np.double(Sun23_Mass)
Sun23_Mass_err = np.double(Sun23_Mass_err)
Sun23_xi_ion = np.double(Sun23_xi_ion)
Sun23_xi_ion_err = np.double(Sun23_xi_ion_err)

Sun23_EW_HB[Sun23_EW_HB<1] = Sun23_EW_HB_err[Sun23_EW_HB<1]

# Tang et al. (2023)
_,Ta23_zspec,Ta23_mass,Ta23_mass_elow,Ta23_mass_eupp,\
	Ta23_EW_O3_HB,Ta23_EW_O3_HB_elow,Ta23_EW_O3_HB_eupp,\
	Ta23_xi_ion,Ta23_xi_ion_elow,Ta23_xi_ion_eupp = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Tang2023_xi_ion_EW.txt",unpack=True,dtype=str)

Ta23_zspec = np.double(Ta23_zspec)
Ta23_mass = np.double(Ta23_mass)
Ta23_mass_elow = np.double(Ta23_mass_elow)
Ta23_mass_eupp = np.double(Ta23_mass_eupp)
Ta23_EW_O3_HB = np.double(Ta23_EW_O3_HB)
Ta23_EW_O3_HB_elow = np.double(Ta23_EW_O3_HB_elow)
Ta23_EW_O3_HB_eupp = np.double(Ta23_EW_O3_HB_eupp)
Ta23_xi_ion = np.double(Ta23_xi_ion)
Ta23_xi_ion_elow = np.double(Ta23_xi_ion_elow)
Ta23_xi_ion_eupp = np.double(Ta23_xi_ion_eupp)

# Whitler et al. (2023)
_,_,_,_,_,_,_,_,_,Wh23_redshift,Wh23_redshift_elow,Wh23_redshift_eupp,\
	Wh23_mass,Wh23_mass_elow,Wh23_mass_eupp,_,_,_,\
	Wh23_xi_ion,Wh23_xi_ion_elow,Wh23_xi_ion_eupp = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Whitler2023_xi_ion_Mass_ssfr.txt",unpack=True,dtype=str)

Wh23_redshift = np.double(Wh23_redshift)
Wh23_redshift_elow = np.double(Wh23_redshift_elow)
Wh23_redshift_eupp = np.double(Wh23_redshift_eupp)
Wh23_mass = np.double(Wh23_mass)
Wh23_mass_elow = np.double(Wh23_mass_elow)
Wh23_mass_eupp = np.double(Wh23_mass_eupp)
Wh23_xi_ion = np.double(Wh23_xi_ion)
Wh23_xi_ion_elow = np.double(Wh23_xi_ion_elow)
Wh23_xi_ion_eupp = np.double(Wh23_xi_ion_eupp)

# Endsley et al. (2021) z ~ 7
_,En21_z,En21_zlow,En2021_zupp,_,_,_,_,_,_,En21_mass,En21_mass_elow,En21_mass_eupp,\
	En21_ew,En21_ew_elow,En21_ew_eupp,\
	En21_xi,En21_xi_elow,En21_xi_eupp = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Endsley2021.txt",unpack=True,dtype=str)

En21_ew = np.double(En21_ew)
En21_ew_elow = np.double(En21_ew_elow)
En21_ew_eupp = np.double(En21_ew_eupp)
En21_xi = np.double(En21_xi)
En21_xi_elow = np.double(En21_xi_elow)
En21_xi_eupp = np.double(En21_xi_eupp)



# Stark et al. (2017) z ~ 7 - 9
St17_z,St17_ew,St17_ew_elow,St17_ew_eupp,St17_xi,St17_xi_elow,St17_eupp = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Stark2017.txt",unpack=True)



# Tang et al. (2019) z ~ 2
Ta19_EW,Ta19_xi = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Tang2019_prelim.csv",delimiter=",",unpack=True)











# Let's first define the figure parameters
fig = plt.figure()
fig.set_size_inches(8,4)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

#plt.rcParams["font.family"] = "sans-serif"
#plt.rcParams["font.sans-serif"] = "DejaVu Sans"
#plt.rcParams["text.usetex"] = True
plt.rcParams["xtick.labelsize"] = "7"
plt.rcParams["ytick.labelsize"] = "7"

ax1 = fig.add_subplot(121)


# Plot Sun et al. (2023)
ax1.errorbar(Sun23_EW_OIII5007+Sun23_EW_OIII4959+Sun23_EW_HB,Sun23_xi_ion,yerr=(Sun23_xi_ion_err),marker="d",ls="none",ms=6,\
			mec=tableau20("Red"),mfc=tableau20("Light Red"),ecolor=tableau20("Grey"),\
							mew=0.5,capsize=2,capthick=1,elinewidth=0.5)

# Plot Tang et al. (2023)
#ax.errorbar(Ta23_EW_O3_HB,Ta23_xi_ion,xerr=(Ta23_EW_O3_HB_elow,Ta23_EW_O3_HB_eupp),yerr=(Ta23_xi_ion_elow,Ta23_xi_ion_eupp),marker="^",ls="none",ms=3,
#						mfc="none",mec=tableau20("Red"),ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)
ax1.plot(Ta23_EW_O3_HB,Ta23_xi_ion,marker="^",ls="none",ms=3,
						mfc="none",mec=tableau20("Green"),mew=0.2)

# Plot Tang et al. (2019)
ax1.plot(Ta19_EW,Ta19_xi,marker="s",ls="none",ms=3,mec=tableau20("Purple"),mfc="none",mew=0.2)

# Plot Endsley et al. (2021)
#ax.errorbar(En21_ew,En21_xi,xerr=(En21_ew_elow,En21_ew_eupp),yerr=(En21_xi_elow,En21_xi_eupp),marker="v",ls="none",ms=3,
#						mfc="none",mec=tableau20("Green"),ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)
ax1.plot(En21_ew,En21_xi,marker="v",ls="none",ms=3,
						mfc="none",mec=tableau20("Brown"),
							mew=0.2)


# Plot Stark et al. (2017)
#ax.errorbar(St17_ew,St17_xi,xerr=(St17_ew_elow,St17_ew_eupp),yerr=(St17_xi_elow,St17_eupp),marker=">",ls="none",ms=3,
#						mfc="none",mec=tableau20("Sky Blue"),ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)
ax1.plot(St17_ew,St17_xi,marker=">",ls="none",ms=3,
						mfc="none",mec=tableau20("Sky Blue"),
							mew=0.2)

# EIGER (Matthee+23)
ax1.errorbar([845.],[25.3],xerr=[70.],yerr=[0.2],ls="none",marker="o",ms=6,mfc=tableau20("Light Red"),mec=tableau20("Red"),\
							ecolor=tableau20("Red"),\
							mew=1.0,capsize=2,capthick=1,elinewidth=0.5)


# CEERS (Chen et al. (2024)) -- # https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.7052C/abstract
Ch24_EW = np.array([1982.,682.,835.,4300.,2266.,1094.,1475.,3865.,1416.,2289.])
Ch24_EW_elow = np.array([333.,315.,181.,1006,552,308,315,1329,348,832])
Ch24_EW_eupp = np.array([462,344,253,693,1063,550,565,883,1298,1674])

Ch24_xi_ion = np.array([25.76,25.86,25.53,25.83,25.79,25.60,25.73,25.82,25.65,25.80])
Ch24_xi_ion_elow = np.array([0.14,0.23,0.07,0.12,0.10,0.13,0.16,0.16,0.12,0.16])
Ch24_xi_ion_eupp = np.array([0.13,0.04,0.10,0.06,0.11,0.19,0.08,0.08,0.16,0.15])

ax1.errorbar(Ch24_EW,Ch24_xi_ion,xerr=(Ch24_EW_elow,Ch24_EW_eupp),yerr=(Ch24_xi_ion_elow,Ch24_xi_ion_eupp),ls="none",marker="p",ms=6,mec=tableau20("Green"),mfc=tableau20("Light Green"),\
							ecolor=tableau20("Green"),\
							mew=0.5,capsize=2,capthick=1,elinewidth=0.5)


# How many in sweet spot?
#ax.fill_between([800.,800.,1e4,1e4],26.,28.,color=tableau20("Red"),alpha=0.15)

#alpha=0.8
# Plot Our Measurement
#ax.plot(EW[data["EELG"]],xi_ion[data["EELG"]],marker="o",ls="none",ms=4,alpha=alpha,mec="black",mfc=tableau20("Red"),mew=0.5,zorder=1)
#ax.plot(EW[data["SELG"]],xi_ion[data["SELG"]],marker="o",ls="none",ms=4,alpha=alpha,mec="black",mfc=tableau20("Light Blue"),mew=0.5,zorder=1)

# Tang ?
xdata = np.arange(1.,4.,0.1)
ydata = 0.76*xdata+23.27
ax1.plot(pow(10,xdata),ydata,ls="--",color=tableau20("Purple"),zorder=1)



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
			Line2D([],[],ls="none",marker=">",ms=4,mec=tableau20("Sky Blue"),mfc="none",label=r"St+17 ($z \sim 7 - 9$)")]

leg2 = ax1.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg2)	

ax1.set_yticks([23.5,24.0,24.5,25.0,25.5,26.0,26.5])
ax1.set_yticks(np.arange(23.5,27.,0.1),minor=True)
ax1.set_yticklabels([23.5,24.0,24.5,25.0,25.5,26.0,26.5])


ax1.set_ylim(23.8,26.3)
ax1.set_xlim(100.,6000.)
ax1.set_xscale("log")

ax1.set_xticks([100.,300.,600.,1000.,3000.,6000.])
ax1.set_xticks([100.,200.,300.,450.,600.,800.,1000.,2000.,3000.,4500.,6000.],minor=True)
ax1.set_xticklabels(["100","300","600","1000","3000","6000"])

ax1.set_xlabel(r"$\log_{10}$ EW$_0$([O{\sc iii}]$+$H$\beta$) (\AA)",fontsize=8)#, **hfont)
ax1.set_ylabel(r"$\log_{10} \xi_\textrm{ion}^\textrm{H{\sc ii}}$ (erg$^{-1}$ Hz)",fontsize=8)#, **hfont)









ax2 = fig.add_subplot(122)

# EIGER 
EIGER_sSFR = 4./pow(10,8.38)
EIGER_sSFR_err = EIGER_sSFR * np.sqrt( (1/np.log10(4.))**2. + (0.07*np.log(10.))**2. ); EIGER_sSFR_err = EIGER_sSFR_err/(np.log(10.)*EIGER_sSFR)
EIGER_sSFR = np.log10(EIGER_sSFR)
ax2.errorbar([EIGER_sSFR],[25.3],xerr=[EIGER_sSFR_err],yerr=[0.2],ls="none",marker="o",ms=6,mfc=tableau20("Light Red"),mec=tableau20("Red"),\
							ecolor=tableau20("Red"),\
							mew=1.0,capsize=2,capthick=1,elinewidth=0.5)



# MIDIS (Rinaldi et al. 2023) 
MIDIS_sSFR = np.array([-7.515,-7.261,-7.796,-7.800,-7.692,-7.871,-7.895,-7.484,-8.037,-8.183,-8.022,-8.267])
MIDIS_xi_ion = np.array([25.794,25.722,25.656,25.599,25.558,25.572,25.437,25.209,25.244,25.273,25.179,24.932])

ax2.plot(MIDIS_sSFR, MIDIS_xi_ion,ls="none",marker="p",ms=6,mfc=tableau20("Light Green"),mec=tableau20("Green"),mew=1.0)



# CEERS (Whitler et al. 2023) -- # https://ui.adsabs.harvard.edu/abs/2024MNRAS.529..855W/abstract
Wh23_sSFR = np.array([-6.670,-6.573,-6.648,-6.666,-6.780,-6.834,-6.762,-7.083,-7.075,-7.108,-7.379,-7.432,-7.457,-7.543,-7.514,-7.539,-7.557,-7.436,-7.421,-7.514,-7.592,-7.660,-7.703,-8.302])
Wh23_xi_ion = np.array([25.928,25.837,25.827,25.807,25.847,25.837,25.739,25.747,25.660,25.650,25.668,25.680,25.646,25.663,25.634,25.617,25.619,25.619,25.577,25.589,25.568,25.607,25.600,25.470])
ax2.plot(Wh23_sSFR, Wh23_xi_ion,ls="none",marker="d",ms=6,mfc=tableau20("Light Purple"),mec=tableau20("Purple"),mew=1.0)


# Castellano et al. (2023) -- # https://www.aanda.org/articles/aa/pdf/2023/07/aa46069-23.pdf
Ca23_sSFR = np.array([-9.63,-9.09,-8.71,-8.34,-7.90,-7.44])
Ca23_sSFR_elow = np.array([-10.,-9.5,-9.0,-8.5,-8.0,-7.5])
Ca23_sSFR_eupp = np.array([-9.5,-9.0,-8.5,-8.0,-7.5,-7.0])
Ca23_sSFR_elow = Ca23_sSFR - Ca23_sSFR_elow
Ca23_sSFR_eupp = Ca23_sSFR_eupp - Ca23_sSFR

Ca23_xi_ion = np.array([24.57,24.70,24.78,25.05,25.35,25.45])
Ca23_xi_ion_elow = np.array([0.61,0.65,0.35,0.37,0.12,0.10])
Ca23_xi_ion_eupp = np.array([0.24,0.25,0.19,0.20,0.10,0.08])

ax2.errorbar(Ca23_sSFR,Ca23_xi_ion,xerr=(Ca23_sSFR_elow,Ca23_sSFR_eupp),yerr=(Ca23_xi_ion_elow,Ca23_xi_ion_eupp),ls="none",marker="s",ms=6,mfc=tableau20("Light Orange"),mec=tableau20("Orange"),\
							ecolor=tableau20("Orange"),\
							mew=1.0,capsize=2,capthick=1,elinewidth=0.5)

# Sun et al. (2023) -- # https://iopscience.iop.org/article/10.3847/1538-4357/acd53c/pdf
Sun_SFR = np.array([32.,38.,18.,64.])
Sun_SFR_err = np.array([5.,7.,5.,9.])

Sun_mass = np.array([9.1,8.9,9.2,9.5])
Sun_mass_err = np.array([0.2,0.4,0.3,0.3]); Sun_mass_err = Sun_mass_err*np.log(10.)*pow(10,Sun_mass)
Sun_mass = pow(10,Sun_mass)

Sun_sSFR = Sun_SFR/Sun_mass
Sun_sSFR_err = Sun_sSFR * np.sqrt( (Sun_mass_err/Sun_mass)**2. + (Sun_SFR_err/Sun_SFR)**2. )
Sun_sSFR_err = Sun_sSFR_err/(np.log(10.)*Sun_sSFR)
Sun_sSFR = np.log10(Sun_sSFR)

Sun_xi_ion = np.array([25.2,25.8,24.6,25.5])
Sun_xi_ion_err = np.array([0.1,0.3,0.2,0.2])

ax2.errorbar(Sun_sSFR,Sun_xi_ion,xerr=Sun_sSFR_err,yerr=Sun_xi_ion_err,ls="none",marker="^",ms=6,mfc=tableau20("Light Brown"),mec=tableau20("Brown"),\
							ecolor=tableau20("Brown"),\
							mew=1.0,capsize=2,capthick=1,elinewidth=0.5)


#### EELG1002
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
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"VANDELS (Ca+23; $z \sim 2 - 5$)")]

		
leg2 = ax2.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg2)	

ax2.set_yticks([23.5,24.0,24.5,25.0,25.5,26.0,26.5])
ax2.set_yticks(np.arange(23.5,27.,0.1),minor=True)
ax2.set_yticklabels([])


ax2.set_ylim(23.8,26.3)
ax2.set_xlim(-10.2,-6.)
#ax.set_xscale("log")

#ax.set_xticks([100.,300.,600.,1000.,3000.,6000.])
#ax.set_xticks([100.,200.,300.,450.,600.,800.,1000.,2000.,3000.,4500.,6000.],minor=True)
#ax.set_xticklabels(["100","300","600","1000","3000","6000"])

ax2.set_xlabel(r"$\log_{10}$ sSFR (yr$^{-1}$)",fontsize=8)#, **hfont)
#ax2.set_ylabel(r"$\log_{10} \xi_\textrm{ion}^\textrm{H{\sc ii}}$ (erg$^{-1}$ Hz)",fontsize=8)#, **hfont)



fig.savefig("../../plots/xi_ion_O3HB_EW_sSFR.png",bbox_inches="tight",dpi=300,format="png")




#fig.savefig("../../plots/xi_ion_O3HB_EW.png",bbox_inches="tight",dpi=300,format="png")