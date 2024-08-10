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

# Load in Rinaldi et al. (2023)
_,_,_,Ri23_zphot,_,_,_,\
	Ri23_mass,Ri23_mass_elow,Ri23_mass_eupp,_,_,_,_,_,_,\
		Ri23_MUV,Ri23_MUV_elow,Ri23_MUV_eupp,_,_,_,\
		Ri23_xi_ion,Ri23_xi_ion_elow,Ri23_xi_ion_eupp = np.loadtxt("../../../Proposal/JWST/JWST_Cycle3/COSMOS_NIRSpec/data/Rinaldi2023_xi_ion.txt" ,unpack=True,dtype=str)        

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
fig.set_size_inches(4,4)

#plt.rcParams["font.family"] = "sans-serif"
#plt.rcParams["font.sans-serif"] = "DejaVu Sans"
#plt.rcParams["text.usetex"] = True
plt.rcParams["xtick.labelsize"] = "7"
plt.rcParams["ytick.labelsize"] = "7"

ax = fig.add_subplot(111)


# Plot Sun et al. (2023)
ax.errorbar(Sun23_EW_OIII5007+Sun23_EW_OIII4959+Sun23_EW_HB,Sun23_xi_ion,yerr=(Sun23_xi_ion_err),marker="d",ls="none",ms=6,\
			mec=tableau20("Red"),mfc=tableau20("Light Red"),ecolor=tableau20("Grey"),\
							mew=0.5,capsize=2,capthick=1,elinewidth=0.5)

# Plot Tang et al. (2023)
#ax.errorbar(Ta23_EW_O3_HB,Ta23_xi_ion,xerr=(Ta23_EW_O3_HB_elow,Ta23_EW_O3_HB_eupp),yerr=(Ta23_xi_ion_elow,Ta23_xi_ion_eupp),marker="^",ls="none",ms=3,
#						mfc="none",mec=tableau20("Red"),ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)
ax.plot(Ta23_EW_O3_HB,Ta23_xi_ion,marker="^",ls="none",ms=3,
						mfc="none",mec=tableau20("Green"),mew=0.2)

# Plot Tang et al. (2019)
ax.plot(Ta19_EW,Ta19_xi,marker="s",ls="none",ms=3,mec=tableau20("Purple"),mfc="none",mew=0.2)

# Plot Endsley et al. (2021)
#ax.errorbar(En21_ew,En21_xi,xerr=(En21_ew_elow,En21_ew_eupp),yerr=(En21_xi_elow,En21_xi_eupp),marker="v",ls="none",ms=3,
#						mfc="none",mec=tableau20("Green"),ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)
ax.plot(En21_ew,En21_xi,marker="v",ls="none",ms=3,
						mfc="none",mec=tableau20("Brown"),
							mew=0.2)


# Plot Stark et al. (2017)
#ax.errorbar(St17_ew,St17_xi,xerr=(St17_ew_elow,St17_ew_eupp),yerr=(St17_xi_elow,St17_eupp),marker=">",ls="none",ms=3,
#						mfc="none",mec=tableau20("Sky Blue"),ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)
ax.plot(St17_ew,St17_xi,marker=">",ls="none",ms=3,
						mfc="none",mec=tableau20("Sky Blue"),
							mew=0.2)

# EIGER (Matthee+23)
ax.errorbar([845.],[25.3],xerr=[70.],yerr=[0.2],ls="none",marker="o",ms=6,mfc=tableau20("Light Red"),mec=tableau20("Red"),\
							ecolor=tableau20("Red"),\
							mew=1.0,capsize=2,capthick=1,elinewidth=0.5)


# CEERS (Chen et al. (2024)) -- # https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.7052C/abstract
Ch24_EW = np.array([1982.,682.,835.,4300.,2266.,1094.,1475.,3865.,1416.,2289.])
Ch24_EW_elow = np.array([333.,315.,181.,1006,552,308,315,1329,348,832])
Ch24_EW_eupp = np.array([462,344,253,693,1063,550,565,883,1298,1674])

Ch24_xi_ion = np.array([25.76,25.86,25.53,25.83,25.79,25.60,25.73,25.82,25.65,25.80])
Ch24_xi_ion_elow = np.array([0.14,0.23,0.07,0.12,0.10,0.13,0.16,0.16,0.12,0.16])
Ch24_xi_ion_eupp = np.array([0.13,0.04,0.10,0.06,0.11,0.19,0.08,0.08,0.16,0.15])

ax.errorbar(Ch24_EW,Ch24_xi_ion,xerr=(Ch24_EW_elow,Ch24_EW_eupp),yerr=(Ch24_xi_ion_elow,Ch24_xi_ion_eupp),ls="none",marker="p",ms=6,mec=tableau20("Green"),mfc=tableau20("Light Green"),\
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
ax.plot(pow(10,xdata),ydata,ls="--",color=tableau20("Purple"),zorder=1)




# Load in Cigale
sed = fits.open("../../../EELG_OIII_GMOS/data/cigale_results/sfh_delayed_nodust/results.fits")[1].data
sed_model = fits.open("../../../EELG_OIII_GMOS/data/cigale_results/sfh_delayed_nodust/1002_best_model.fits")[1].data
sed_model["fnu"] = sed_model["fnu"]*(1.8275)*1e-26
sed_model["wavelength"] = sed_model["wavelength"]/1.8275
keep = (sed_model["wavelength"] > 1499.) & (sed_model["wavelength"] < 1501.)

data = fits.open("../../data/emline_fits/43158747673038238/1002_lineprops_new.fits")[1].data

ha_lum = 2.86*data["lineflux_med"][-1]*1e-17*4*np.pi*(cosmo.luminosity_distance(0.8272).value*3.08e24)**2.
ha_lum_err = 2.86*data["lineflux_eupp"][-1]*1e-17*4*np.pi*(cosmo.luminosity_distance(0.8272).value*3.08e24)**2.

uv_lum = sed_model["fnu"][keep]*4*np.pi*(cosmo.luminosity_distance(0.8272).value*3.08e24)**2.
uv_lum_err = sed["bayes.param.restframe_Lnu(FUV)_err"]*1e7

EW_source = data["lineEW_med"][-1] + data["lineEW_med"][-2] + data["lineEW_med"][-3] 
EW_source_elow = np.sqrt( data["lineEW_elow"][-1]**2. + data["lineEW_elow"][-2]**2. + data["lineEW_elow"][-3]**2. )
EW_source_eupp = np.sqrt( data["lineEW_eupp"][-1]**2. + data["lineEW_eupp"][-2]**2. + data["lineEW_eupp"][-3]**2. )

xi_ion = ha_lum/(1.36e-12*uv_lum)
xi_ion_err = xi_ion*np.sqrt( (ha_lum_err/ha_lum)**2. + (uv_lum_err/uv_lum)**2. )
xi_ion_err = xi_ion_err/(np.log(10.)*xi_ion)
#xi_ion_lower = data['bayes.stellar.n_ly']/(data['bayes.param.restframe_Lnu(FUV)']*1e7*10**(0.4*2.31*(data["bayes.param.beta_calz94"] - data["bayes.param.beta0_calz94"])))

ax.errorbar([EW_source],np.log10(xi_ion),xerr=([EW_source_elow],[EW_source_eupp]),yerr=(xi_ion_err),marker="*",ms=15,mfc=tableau20("Light Blue"),mec=tableau20("Blue"),ls="none",ecolor=tableau20("Blue"),\
							mew=0.5,capsize=2,capthick=1,elinewidth=0.5)


#xi_ion_err = xi_ion * np.sqrt((data['bayes.stellar.n_ly_err']/data['bayes.stellar.n_ly'])**2. + (data['bayes.param.restframe_Lnu(FUV)_err']/data['bayes.param.restframe_Lnu(FUV)'])**2.)
#xi_ion_err = xi_ion_err/(np.log(10.)*xi_ion)

#ax.errorbar([3154.],np.log10(xi_ion),xerr=(133.),yerr=(xi_ion_err),marker="*",ms=10,mfc=tableau20("Orange"),mec="black",ls="none",ecolor=tableau20("Grey"),\
#							mew=0.2,capsize=2,capthick=1,elinewidth=0.5)



handles = [	Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}"),
			Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"EIGER (Ma+23; $z \sim 5 - 7$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"Su+23 ($z \sim 6.2$)"),			
			Line2D([],[],ls="none",marker="p",ms=4,mec=tableau20("Green"),mfc=tableau20("Light Green"),label=r"CEERS (Ch+23; $z \sim 5 - 8$)"),
			Line2D([],[],ls="--",color=tableau20("Purple"),label=r"Ta+19 ($z \sim 1.3 - 2.4$)")]

leg = ax.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg)	

handles = [ Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Purple"),mfc="none",label=r"EELGs (Ta+19; $z \sim 1.3 - 2.4$)"),
			Line2D([],[],ls="none",marker="^",ms=4,mec=tableau20("Green"),mfc="none",label=r"CEERS (Ta+23; $z \sim 7 - 9$)"),
			Line2D([],[],ls="none",marker="v",ms=4,mec=tableau20("Brown"),mfc="none",label=r"En+21 ($z \sim 7$)"),
			Line2D([],[],ls="none",marker=">",ms=4,mec=tableau20("Sky Blue"),mfc="none",label=r"St+17 ($z \sim 7 - 9$)")]

leg2 = ax.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=6,columnspacing=0.075)
plt.gca().add_artist(leg2)	

ax.set_yticks([24.5,25.0,25.5,26.0,26.5])
ax.set_yticks(np.arange(24.5,27.,0.1),minor=True)
ax.set_yticklabels([24.5,25.0,25.5,26.0,26.5])


ax.set_ylim(24.5,26.3)
ax.set_xlim(100.,6000.)
ax.set_xscale("log")

ax.set_xticks([100.,300.,600.,1000.,3000.,6000.])
ax.set_xticks([100.,200.,300.,450.,600.,800.,1000.,2000.,3000.,4500.,6000.],minor=True)
ax.set_xticklabels(["100","300","600","1000","3000","6000"])

ax.set_xlabel(r"$\log_{10}$ EW$_0$([O{\sc iii}]$+$H$\beta$) (\AA)",fontsize=8)#, **hfont)
ax.set_ylabel(r"$\log_{10} \xi_\textrm{ion}^\textrm{H{\sc ii}}$ (erg$^{-1}$ Hz)",fontsize=8)#, **hfont)



fig.savefig("../../plots/xi_ion_O3HB_EW.png",bbox_inches="tight",dpi=300,format="png")