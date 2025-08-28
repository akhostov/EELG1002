import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np 
from astropy.io import fits
import pickle
import h5py
import sys

sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20

sys.path.insert(0, '..')
import util


fig = plt.figure()
fig.set_size_inches(4, 4)
fig.subplots_adjust(hspace=0.03, wspace=0.03)

ax = fig.add_subplot(111)

# set label
ax.set_xlabel(r"$\log_{10}$ Stellar Mass (M$_\odot$)",fontsize=10)
ax.set_ylabel(r"Rest-Frame [O{\sc iii}] + H$\beta$ EW (\AA)",fontsize=10)
ax.tick_params(which="both",labelsize=10)


#Cardamone2009 GP Catalog
_,_,_,_,C09_EW,C09_Mass = np.loadtxt("../../../Proposal/HST/Cycle32/COS_EELG_GMOS/data/EELG_lit/Cardamone2009_GP_sources.txt",unpack=True)
Ca09, = ax.plot(C09_Mass-0.21,C09_EW,marker="s",mec=tableau20("Green"),mfc="none",ms=3,mew=0.5,alpha=0.5,ls="none",label=r"$z \sim 0$ GP (Ca+09)")


# Blueberries (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18470038Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_Blueberries/Yang2017_Blueberries.fits")[1].data
ax.plot(Ya17["mass"],Ya17["EW(OIII)"] + Ya17["EW(Hbeta)"],				
				ls="none",mec="#464196",mfc="none",\
                marker="s",ms=3,mew=0.2)

# JWST (EIGER)
EIGER_logM = np.array([7.5,8.2,8.9,9.5])
EIGER_EW = np.array([1870,980,690,410])
EIGER_EW_elow = np.array([590,240,300,100])
EIGER_EW_eupp = np.array([1200,670,340,200])
EIGER = ax.errorbar(EIGER_logM,EIGER_EW,yerr=[EIGER_EW_elow,EIGER_EW_eupp],\
					ls="none",mec="darkred",mfc="darkred",ecolor=tableau20("Grey"),\
					marker="p",ms=5,mew=0.2,capsize=2,capthick=1,elinewidth=0.5,label=r"$z \sim 5 - 7$ (EIGER; stacks)")

## Izotov et al. 2021 -- LyC
Iz21_mass,Iz21_hb,Iz21_o3_4959,Iz21_o3_5007 = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2021.txt",unpack=True)
Iz21_EW = Iz21_hb + Iz21_o3_4959 + Iz21_o3_5007
Iz21, = ax.plot(Iz21_mass-0.21,Iz21_EW,marker="d",ms=5,mec=tableau20("Purple"),mfc="none",alpha=0.8,ls="none",label=r"$z = 0.3 - 0.45$ (Iz+21)",zorder=99)

Iz16_mass,Iz16_hb,Iz16_o3_4959,Iz16_o3_5007 = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2016.txt",unpack=True,usecols=(-2,10,13,16))
Iz16_EW = Iz16_hb + Iz16_o3_4959 + Iz16_o3_5007
_, = ax.plot(Iz16_mass-0.21,Iz16_EW,marker="d",ms=5,mec=tableau20("Purple"),mfc="none",alpha=0.8,ls="none",zorder=99)

Iz18_mass,Iz18_hb,Iz18_o3_4959,Iz18_o3_5007 = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2018.txt",unpack=True,usecols=(-2,10,13,16))
Iz18_EW = Iz18_hb + Iz18_o3_4959 + Iz18_o3_5007
_, = ax.plot(Iz18_mass-0.21,Iz18_EW,marker="d",ms=5,mec=tableau20("Purple"),mfc="none",alpha=0.8,ls="none",zorder=99)


# JADES -- Boyett et al. (2024) 3 < z < 7
JADES_mass,JADES_mass_elow,JADES_mass_eupp,\
	JADES_EW, JADES_EW_err	= np.loadtxt("../../data/literature_measurements/Boyett_et_al_2024.txt",unpack=True,usecols=(4,6,5,2,3))

JADES = ax.plot(JADES_mass,JADES_EW,\
					ls="none",mec=tableau20("Pink"),mfc="none",mew=0.5,alpha=0.8,\
					marker="H",ms=5,label=r"$z \sim 3 - 9$ (JADES)")

# BOSS-EUVLG1 (Marques-Chaves et al. 2020)
BOSS, = ax.plot([10.0-0.21],[1125.],marker="^",ms=5,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),alpha=0.8,ls="none",label=r"$z = 2.469$ (BOSS-EUVLG1)")

# Ion 2 (de Barros et al. 2016)
Ion2, = ax.plot([9.2-0.21],[1103.],marker="v",ms=5,mec=tableau20("Red"),mfc=tableau20("Light Red"),alpha=0.8,ls="none",label=r"$z = 3.216$ (Ion2)")




# Plot the EW -- Stellar Mass Correlations
mass = np.arange(7.,11.,0.01)
EW_z084 = pow(10,4.72-0.33*mass)
EW_z142 = pow(10,5.33-0.33*mass)
EW_z223 = pow(10,6.20-0.38*mass)
EW_z324 = pow(10,6.66-0.43*mass)

ax.plot(mass,EW_z084,ls="--",color=tableau20("Blue"))
ax.plot(mass,EW_z142,ls="--",color=tableau20("Green"))
ax.plot(mass,EW_z223,ls="--",color=tableau20("Orange"))
ax.plot(mass,EW_z324,ls="--",color=tableau20("Red"))

# Bring in HiZELS
hizels_mass_z0p84 = np.array([8.23,8.48,8.74,8.99,9.24,9.49,9.74])
hizels_EW_z0p84 = np.array([1.978,1.893,1.813,1.707,1.675,1.590,1.462])
hizels_EW_z0p84_elow = (hizels_EW_z0p84 - np.array([1.760,1.664,1.574,1.489,1.446,1.414,1.340]))*np.log(10.)*pow(10,hizels_EW_z0p84)
hizels_EW_z0p84_eupp = (np.array([2.223,2.138,2.058,1.941,1.914,1.797,1.601] - hizels_EW_z0p84))*np.log(10.)*pow(10,hizels_EW_z0p84)
hizels_EW_z0p84 = pow(10,hizels_EW_z0p84)

ax.errorbar(hizels_mass_z0p84,hizels_EW_z0p84,yerr=(hizels_EW_z0p84_elow,hizels_EW_z0p84_eupp),\
			ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Grey"),\
                marker="o",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)

hizels_mass_z1p42 = np.array([8.986,9.299,9.596,9.901,10.189,10.502,10.791,11.096])
hizels_EW_z1p42 = np.array([2.462,2.281,2.090,2.026,1.914,1.914,1.723,1.707])
hizels_EW_z1p42_elow = (hizels_EW_z1p42 - np.array([2.207,2.127,1.920,1.856,1.749,1.734,1.563,1.547]))*np.log(10.)*pow(10,hizels_EW_z1p42)
hizels_EW_z1p42_eupp = (np.array([2.739,2.446,2.260,2.212,2.101,2.122,1.904,1.819]) - hizels_EW_z1p42)*np.log(10.)*pow(10,hizels_EW_z1p42)
hizels_EW_z1p42 = pow(10,hizels_EW_z1p42)

ax.errorbar(hizels_mass_z1p42,hizels_EW_z1p42,yerr=(hizels_EW_z1p42_elow,hizels_EW_z1p42_eupp),\
			ls="none",mec=tableau20("Green"),mfc=tableau20("Light Green"),ecolor=tableau20("Grey"),\
                marker="o",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)

hizels_mass_z2p23 = np.array([9.090,9.596,10.101,10.606,11.096])
hizels_EW_z2p23 = np.array([2.734,2.515,2.340,2.015,2.042])
hizels_EW_z2p23_elow = (hizels_EW_z2p23 - np.array([2.494,2.287,2.063,1.819,1.808]))*np.log(10.)*pow(10,hizels_EW_z2p23)
hizels_EW_z2p23_eupp = (np.array([3.010,2.776,2.638,2.271,2.292]) - hizels_EW_z2p23)*np.log(10.)*pow(10,hizels_EW_z2p23)
hizels_EW_z2p23 = pow(10,hizels_EW_z2p23)

ax.errorbar(hizels_mass_z2p23,hizels_EW_z2p23,yerr=(hizels_EW_z2p23_elow,hizels_EW_z2p23_eupp),\
			ls="none",mec=tableau20("Orange"),mfc=tableau20("Light Orange"),ecolor=tableau20("Grey"),\
                marker="o",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)


hizels_mass_z3p24 = np.array([9.596,10.093,10.590,11.096])
hizels_EW_z3p24 = np.array([2.510,2.260,2.117,1.840])
hizels_EW_z3p24_elow = (hizels_EW_z3p24 - np.array([2.303,2.090,1.962,1.691]))*np.log(10.)*pow(10,hizels_EW_z3p24)
hizels_EW_z3p24_eupp = (np.array([2.675,2.404,2.228,1.962]) - hizels_EW_z3p24)*np.log(10.)*pow(10,hizels_EW_z3p24)
hizels_EW_z3p24 = pow(10,hizels_EW_z3p24)

ax.errorbar(hizels_mass_z3p24,hizels_EW_z3p24,yerr=(hizels_EW_z3p24_elow,hizels_EW_z3p24_eupp),\
			ls="none",mec=tableau20("Red"),mfc=tableau20("Light Red"),ecolor=tableau20("Grey"),\
                marker="o",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)

##############################################################################
#######							 PLOT EELG1002						   #######	
##############################################################################

# Get the Stellar Mass
sed = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
cigale_mass = sed["bayes.stellar.m_star"]
cigale_mass_err = sed["bayes.stellar.m_star_err"]/(np.log(10.)*cigale_mass)
cigale_mass = np.log10(cigale_mass)

bagpipes = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")
bagpipes_mass = bagpipes["median"][10]
bagpipes_mass_err_low,bagpipes_mass_err_up = bagpipes_mass - bagpipes["conf_int"][0][10], bagpipes["conf_int"][1][10] - bagpipes_mass


# Get the EW
with open("../../data/xi_ion_measurements.pkl","rb") as f: EW_meas = pickle.load(f)


ax.errorbar(cigale_mass,[EW_meas["cigale_O3HB_EW"][0]],
				xerr=(cigale_mass_err,cigale_mass_err),
            	yerr=([EW_meas["cigale_O3HB_EW"][2]],[EW_meas["cigale_O3HB_EW"][1]]),ls="none",
                mec=util.color_scheme("Cigale",mec=True),
                mfc=util.color_scheme("Cigale",mfc=True),
                ecolor=util.color_scheme("Cigale",mec=True),\
	            marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,label=r"$EELG1002$",zorder=99)

ax.errorbar([bagpipes_mass],[EW_meas["bagpipes_O3HB_EW"][0]],
				xerr=([bagpipes_mass_err_low],[bagpipes_mass_err_up]),
            	yerr=([EW_meas["bagpipes_O3HB_EW"][2]],[EW_meas["bagpipes_O3HB_EW"][1]]),ls="none",
                mec=util.color_scheme("Bagpipes",mec=True),
                mfc=util.color_scheme("Bagpipes",mfc=True),
                ecolor=util.color_scheme("Bagpipes",mec=True),\
	            marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,label=r"$EELG1002$",zorder=99)

print("BAGPIPES:",EW_meas["bagpipes_O3HB_EW"][0]/pow(10,4.72-0.33*bagpipes_mass))
print("CIGALE:",EW_meas["cigale_O3HB_EW"][0]/pow(10,4.72-0.33*cigale_mass))


handles = [	Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),label=r"\textbf{\textit{EELG1002 (Cigale)}}"),
           	Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),label=r"\textbf{\textit{EELG1002 (Bagpipes)}}"),
            Line2D([],[],ls="none",marker="s",ms=4,mec="#464196",mfc="none",label=r"Blueberries (Ya+17; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Purple"),mfc="none",label=r"LyC Leakers (Iz+16,18,21; $z < 0.5$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Green"),mfc="none",label=r"GPs (Ca+08; $z \sim 0.2 - 0.3$)"),
			]		

leg = ax.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=1.0,columnspacing=0.075)
plt.gca().add_artist(leg)	

handles = [	
			Line2D([],[],ls="none",marker="p",ms=4,mec="darkred",mfc="darkred",label=r"\texttt{EIGER} (Ma+23; $z \sim 5 - 7$)"),
            Line2D([],[],ls="none",marker="H",ms=4,mec=tableau20("Pink"),mfc="none",label=r"\texttt{JADES} (Bo+24; $z \sim 3 - 9$)"),
			Line2D([],[],ls="none",marker="^",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"\textrm{\texttt{BOSS-EUVLG1}} (MC+20; $z \sim 2.5$)"),
			Line2D([],[],ls="none",marker="v",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"\textrm{\textit{Ion2}} (deB+16; $z \sim 3.2$)")
			]		

leg = ax.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=1.0,handletextpad=0.1)
plt.gca().add_artist(leg)	

handles = [	Line2D([],[],ls="--",marker="none",color=tableau20("Blue"),label =r"$z \sim 0.84$"),
			Line2D([],[],ls="--",marker="none",color=tableau20("Green"),label =r"$z \sim 1.42$"),
			Line2D([],[],ls="--",marker="none",color=tableau20("Orange"),label =r"$z \sim 2.23$"),
			Line2D([],[],ls="--",marker="none",color=tableau20("Red"),label =r"$z \sim 3.24$")
			]		

leg2 = ax.legend(handles=handles,loc="lower left",title=r"Khostovan+16",frameon=False,ncol=1,numpoints=1,fontsize=5,columnspacing=0.075)
plt.gca().add_artist(leg2)	
leg2.get_title().set_fontsize('5')

plt.xlim(7,11.)
plt.ylim(10.,5e4)
plt.yscale("log")
plt.savefig("../../plots/EW_OIII_distrib.png",format="png",dpi=300,bbox_inches="tight")

