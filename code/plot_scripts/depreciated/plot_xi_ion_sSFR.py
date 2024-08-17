import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import sys
import pdb

sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20

#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Helvetica']
#hfont = {'fontname':'Helvetica'}

cosmo = FlatLambdaCDM(H0=70.,Om0=0.3)




# Let's first define the figure parameters
fig = plt.figure()
fig.set_size_inches(4,4)

#plt.rcParams["font.family"] = "sans-serif"
#plt.rcParams["font.sans-serif"] = "DejaVu Sans"
#plt.rcParams["text.usetex"] = True
plt.rcParams["xtick.labelsize"] = "7"
plt.rcParams["ytick.labelsize"] = "7"

ax = fig.add_subplot(111)



# EIGER 
EIGER_sSFR = 4./pow(10,8.38)
EIGER_sSFR_err = EIGER_sSFR * np.sqrt( (1/np.log10(4.))**2. + (0.07*np.log(10.))**2. ); EIGER_sSFR_err = EIGER_sSFR_err/(np.log(10.)*EIGER_sSFR)
EIGER_sSFR = np.log10(EIGER_sSFR)
ax.errorbar([EIGER_sSFR],[25.3],xerr=[EIGER_sSFR_err],yerr=[0.2],ls="none",marker="o",ms=6,mfc=tableau20("Light Red"),mec=tableau20("Red"),\
							ecolor=tableau20("Red"),\
							mew=1.0,capsize=2,capthick=1,elinewidth=0.5)



# MIDIS (Rinaldi et al. 2023) 
MIDIS_sSFR = np.array([-7.515,-7.261,-7.796,-7.800,-7.692,-7.871,-7.895,-7.484,-8.037,-8.183,-8.022,-8.267])
MIDIS_xi_ion = np.array([25.794,25.722,25.656,25.599,25.558,25.572,25.437,25.209,25.244,25.273,25.179,24.932])

ax.plot(MIDIS_sSFR, MIDIS_xi_ion,ls="none",marker="p",ms=6,mfc=tableau20("Light Green"),mec=tableau20("Green"),mew=1.0)



# CEERS (Whitler et al. 2023) -- # https://ui.adsabs.harvard.edu/abs/2024MNRAS.529..855W/abstract
Wh23_sSFR = np.array([-6.670,-6.573,-6.648,-6.666,-6.780,-6.834,-6.762,-7.083,-7.075,-7.108,-7.379,-7.432,-7.457,-7.543,-7.514,-7.539,-7.557,-7.436,-7.421,-7.514,-7.592,-7.660,-7.703,-8.302])
Wh23_xi_ion = np.array([25.928,25.837,25.827,25.807,25.847,25.837,25.739,25.747,25.660,25.650,25.668,25.680,25.646,25.663,25.634,25.617,25.619,25.619,25.577,25.589,25.568,25.607,25.600,25.470])
ax.plot(Wh23_sSFR, Wh23_xi_ion,ls="none",marker="d",ms=6,mfc=tableau20("Light Purple"),mec=tableau20("Purple"),mew=1.0)


# Castellano et al. (2023) -- # https://www.aanda.org/articles/aa/pdf/2023/07/aa46069-23.pdf
Ca23_sSFR = np.array([-9.63,-9.09,-8.71,-8.34,-7.90,-7.44])
Ca23_sSFR_elow = np.array([-10.,-9.5,-9.0,-8.5,-8.0,-7.5])
Ca23_sSFR_eupp = np.array([-9.5,-9.0,-8.5,-8.0,-7.5,-7.0])
Ca23_sSFR_elow = Ca23_sSFR - Ca23_sSFR_elow
Ca23_sSFR_eupp = Ca23_sSFR_eupp - Ca23_sSFR

Ca23_xi_ion = np.array([24.57,24.70,24.78,25.05,25.35,25.45])
Ca23_xi_ion_elow = np.array([0.61,0.65,0.35,0.37,0.12,0.10])
Ca23_xi_ion_eupp = np.array([0.24,0.25,0.19,0.20,0.10,0.08])

ax.errorbar(Ca23_sSFR,Ca23_xi_ion,xerr=(Ca23_sSFR_elow,Ca23_sSFR_eupp),yerr=(Ca23_xi_ion_elow,Ca23_xi_ion_eupp),ls="none",marker="s",ms=6,mfc=tableau20("Light Orange"),mec=tableau20("Orange"),\
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

ax.errorbar(Sun_sSFR,Sun_xi_ion,xerr=Sun_sSFR_err,yerr=Sun_xi_ion_err,ls="none",marker="^",ms=6,mfc=tableau20("Light Brown"),mec=tableau20("Brown"),\
							ecolor=tableau20("Brown"),\
							mew=1.0,capsize=2,capthick=1,elinewidth=0.5)



# Load in Cigale
sed = fits.open("../../../EELG_OIII_GMOS/data/cigale_results/sfh_delayed_nodust/results.fits")[1].data
sed_model = fits.open("../../../EELG_OIII_GMOS/data/cigale_results/sfh_delayed_nodust/1002_best_model.fits")[1].data
sed_model["fnu"] = sed_model["fnu"]*(1.8275)*1e-26
sed_model["wavelength"] = sed_model["wavelength"]/1.8275
keep = (sed_model["wavelength"] > 1499.) & (sed_model["wavelength"] < 1501.)

data = fits.open("../../data/emline_fits/43158747673038238/1002_lineprops_new.fits")[1].data

ha_lum = 2.86*data["lineflux_med"][-1]*1e-17*4*np.pi*(cosmo.luminosity_distance(0.8272).value*3.08e24)**2.
ha_lum_err = 2.86*data["lineflux_eupp"][-1]*1e-17*4*np.pi*(cosmo.luminosity_distance(0.8272).value*3.08e24)**2.

SFR = (ha_lum*7.9e-42/1.64)
SFR_err = ha_lum_err*7.9e-42/1.64

sSFR = SFR/sed["bayes.stellar.m_star"]
sSFR_err = sSFR*np.sqrt( (SFR_err/SFR)**2. + (sed["bayes.stellar.m_star_err"]/sed["bayes.stellar.m_star"])**2. )
sSFR_err = sSFR_err/(np.log(10.) * sSFR); sSFR = np.log10(sSFR)

uv_lum = sed_model["fnu"][keep]*4*np.pi*(cosmo.luminosity_distance(0.8272).value*3.08e24)**2.
uv_lum_err = sed["bayes.param.restframe_Lnu(FUV)_err"]*1e7


xi_ion = ha_lum/(1.36e-12*uv_lum)
xi_ion_err = xi_ion*np.sqrt( (ha_lum_err/ha_lum)**2. + (uv_lum_err/uv_lum)**2. )
xi_ion_err = xi_ion_err/(np.log(10.)*xi_ion)
#xi_ion_lower = data['bayes.stellar.n_ly']/(data['bayes.param.restframe_Lnu(FUV)']*1e7*10**(0.4*2.31*(data["bayes.param.beta_calz94"] - data["bayes.param.beta0_calz94"])))

ax.errorbar([sSFR],np.log10(xi_ion),xerr=([sSFR_err]),yerr=(xi_ion_err),marker="*",ms=15,mfc=tableau20("Light Blue"),mec=tableau20("Blue"),ls="none",ecolor=tableau20("Blue"),\
							mew=0.5,capsize=2,capthick=1,elinewidth=0.5)


#xi_ion_err = xi_ion * np.sqrt((data['bayes.stellar.n_ly_err']/data['bayes.stellar.n_ly'])**2. + (data['bayes.param.restframe_Lnu(FUV)_err']/data['bayes.param.restframe_Lnu(FUV)'])**2.)
#xi_ion_err = xi_ion_err/(np.log(10.)*xi_ion)

#ax.errorbar([3154.],np.log10(xi_ion),xerr=(133.),yerr=(xi_ion_err),marker="*",ms=10,mfc=tableau20("Orange"),mec="black",ls="none",ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)



handles = [	Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}") ]

leg = ax.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg)	


handles = [ Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"EIGER (Ma+23; $z \sim 5 - 7$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Brown"),mfc=tableau20("Light Brown"),label=r"Su+23 ($z \sim 6.2$)"),			
			Line2D([],[],ls="none",marker="p",ms=4,mec=tableau20("Green"),mfc=tableau20("Light Green"),label=r"MIDIS (Ri+23; $z \sim 7 - 8$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),label=r"CEERS (Wh+23; $z \sim 7 - 9$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"VANDELS (Ca+23; $z \sim 2 - 5$)")]

		
leg2 = ax.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg2)	

ax.set_yticks([23.5,24.0,24.5,25.0,25.5,26.0,26.5])
ax.set_yticks(np.arange(23.5,27.,0.1),minor=True)
ax.set_yticklabels([23.5,24.0,24.5,25.0,25.5,26.0,26.5])


ax.set_ylim(23.8,26.2)
ax.set_xlim(-10.2,-6.)
#ax.set_xscale("log")

#ax.set_xticks([100.,300.,600.,1000.,3000.,6000.])
#ax.set_xticks([100.,200.,300.,450.,600.,800.,1000.,2000.,3000.,4500.,6000.],minor=True)
#ax.set_xticklabels(["100","300","600","1000","3000","6000"])

ax.set_xlabel(r"$\log_{10}$ sSFR (yr$^{-1}$)",fontsize=8)#, **hfont)
ax.set_ylabel(r"$\log_{10} \xi_\textrm{ion}^\textrm{H{\sc ii}}$ (erg$^{-1}$ Hz)",fontsize=8)#, **hfont)



fig.savefig("../../plots/xi_ion_sSFR.png",bbox_inches="tight",dpi=300,format="png")