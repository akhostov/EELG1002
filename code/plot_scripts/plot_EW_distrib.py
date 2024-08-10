import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits
import os,sys
import pdb

sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20

# setup LaTeX path
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'

# Define Key Parameters for Font
#mpl.rc('text',usetex=True)
#mpl.rcParams['text.latex.preamble']=[r'\usepackage{amsmath,amssymb,newtxtt,newtxmath}']
#mpl.rc('font',family='serif')

"""
fig = plt.figure()
fig.set_size_inches(3.75, 3.75)
fig.subplots_adjust(hspace=0.03, wspace=0.03)

ax = fig.add_subplot(111)

# set label
ax.set_xlabel(r"Rest-Frame EW (\AA)")
ax.set_ylabel(r"Number of Emitters")

# Load in the catalog
cat = fits.open("../data/NBH_HiZELS_HA_O3_sample.fits")
cat = cat[1].data

ewbins = pow(10,np.arange(1.,5.,0.2))

plt.hist(cat["EW_0"][cat["EL"] == "HA"],bins=ewbins,color=tableau20("Blue"),edgecolor="black",alpha=0.5,ls="--",lw=1,histtype="stepfilled",hatch="\\",label=r"H$\alpha$")
plt.hist(cat["EW_0"][cat["EL"] == "OIII"],bins=ewbins,color=tableau20("Red"),edgecolor="black",alpha=0.5,ls="--",lw=1,histtype="stepfilled",hatch="/",label=r"[O{\sc iii}]")

plt.legend(loc="upper right",ncol=1,fontsize=10,frameon=False)

plt.xscale("log")
plt.yscale("log")
plt.ylim(0.9,100.)
plt.xlim(10.,20000.)
plt.savefig("../plots/EW_distributions.png",format="png",dpi=300,bbox_inches="tight")
"""

"""
fig = plt.figure()
fig.set_size_inches(3.71,2.36)
fig.subplots_adjust(hspace=0.03, wspace=0.03)

ax = fig.add_subplot(111)

# set label
ax.set_xlabel(r"$\log_{10}$ Stellar Mass (M$_\odot$)")
ax.set_ylabel(r"Number of Galaxies")

# Load in the catalog
cat = fits.open("../data/NBH_HiZELS_HA_O3_sample.fits")
cat = cat[1].data

mass_bins = np.arange(7.5,12.,0.3)

plt.hist(cat["MASS"][cat["EL"] == "HA"],bins=mass_bins,color=tableau20("Blue"),edgecolor="black",alpha=0.5,ls="--",lw=1,histtype="stepfilled",hatch="\\",label=r"H$\alpha$")
plt.hist(cat["MASS"][cat["EL"] == "OIII"],bins=mass_bins,color=tableau20("Red"),edgecolor="black",alpha=0.5,ls="--",lw=1,histtype="stepfilled",hatch="/",label=r"[O{\sc iii}]")

plt.legend(loc="upper right",ncol=1,fontsize=10,frameon=False)

#plt.xscale("log")
#plt.yscale("log")
#plt.ylim(2e-17,5e-16.)
#plt.xlim(10.,20000.)
plt.savefig("../plots/Mass_distributions.png",format="png",dpi=300,bbox_inches="tight")
#plt.show()

"""


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
Iz21_mass,Iz21_hb,Iz21_o3_4959,Iz21_o3_5007 = np.loadtxt("../../../Proposal/HST/Cycle32/COS_EELG_GMOS/data/EELG_lit/Izotov_2021.txt",unpack=True)
Iz21_EW = Iz21_hb + Iz21_o3_4959 + Iz21_o3_5007
Iz21, = ax.plot(Iz21_mass-0.21,Iz21_EW,marker="d",ms=5,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),alpha=0.8,ls="none",label=r"$z = 0.3 - 0.45$ (Iz+21)",zorder=99)

# J0925+1403 (Izotov et al. 2016)
#J0925, = ax.plot([np.log10(8.2e8)-0.21],[1174.],marker="s",ms=5,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),alpha=0.8,ls="none",label=r"$z = 0.301$ (J0925+1403)",zorder=99)

# BOSS-EUVLG1 (Marques-Chaves et al. 2020)
BOSS, = ax.plot([10.0-0.21],[1125.],marker="^",ms=5,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),alpha=0.8,ls="none",label=r"$z = 2.469$ (BOSS-EUVLG1)")

# AUDFs01 (Saha et al. 2020)
#AUDF, = ax.plot([np.log10(1.45e9)-0.21],[680.],marker="s",ms=5,mec=tableau20("Green"),mfc=tableau20("Light Green"),alpha=0.8,ls="none",label=r"$z = 1.42$ (AUDFs01)")

# Ion 2 (de Barros et al. 2016)
Ion2, = ax.plot([9.2-0.21],[1103.],marker="v",ms=5,mec=tableau20("Red"),mfc=tableau20("Light Red"),alpha=0.8,ls="none",label=r"$z = 3.216$ (Ion2)")

#lgd2 = ax.legend(handles=[Ma14,AUDF,Ta21,BOSS,Tr20,Ion2,EIGER,Tang],loc="upper right",ncol=1,fontsize=7.,frameon=False,handletextpad=0.025,columnspacing=0.05)
#plt.gca().add_artist(lgd2)
#
#lgd3 = ax.legend(handles=[Ca09,JPLUS,Iz21],loc="upper left",ncol=1,fontsize=7.,frameon=False,handletextpad=0.025,columnspacing=0.05,bbox_to_anchor=(0.0,0.85))
#plt.gca().add_artist(lgd3)


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

# Plot in my Source
sed = fits.open("../../data/cigale_results/sfh_delayed_nodust/results.fits")[1].data
mass = sed["bayes.stellar.m_star"]
mass_err = sed["bayes.stellar.m_star_err"]/(np.log(10.)*mass)
mass = np.log10(mass)
pdb.set_trace()
data = fits.open("../../data/emline_fits/43158747673038238/1002_lineprops_new.fits")[1].data

EW_source = data["lineEW_med"][-1] + data["lineEW_med"][-2] + data["lineEW_med"][-3] 
EW_source_elow = np.sqrt( data["lineEW_elow"][-1]**2. + data["lineEW_elow"][-2]**2. + data["lineEW_elow"][-3]**2. )
EW_source_eupp = np.sqrt( data["lineEW_eupp"][-1]**2. + data["lineEW_eupp"][-2]**2. + data["lineEW_eupp"][-3]**2. )

ax.errorbar([mass],[EW_source],xerr=[mass_err],yerr=([EW_source_elow],[EW_source_eupp]),\
						ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
                		marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,label=r"$EELG1002$",zorder=99)

#high_ew_prop, = ax.plot(cat["MASS"][cat["EL"] == "OIII"][highEW], cat["EW_0"][cat["EL"] == "OIII"][highEW],color=tableau20("Green"),marker="o",ms=5,mec="black",ls="none",alpha=0.5,label=r"High EW")

#handles = [mpl.lines.Line2D([],[],mfc=tableau20("Light Blue"),marker="*",ms=10,mew=0.5,mec=tableau20("Blue"),ls="none",alpha=0.5,label=r"$EELG1002$"),
#		   mpl.lines.Line2D([],[],color="black",marker="s",ms=5,mew=0.5,mec="black",ls="none",alpha=0.5,label=r"Confirmed LyC")]

#lgd1 = ax.legend(handles=handles,loc="upper left",ncol=1,fontsize=7.,frameon=True,handletextpad=0.1,columnspacing=0.05)
#plt.gca().add_artist(lgd1)
#lgd1.get_title().set_fontsize('7')

handles = [	Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}"),
			Line2D([],[],ls="none",marker="s",ms=4,mec="#464196",mfc="none",label=r"Blueberries (Ya+17; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),label=r"LyC Leakers (Iz+21; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Green"),mfc="none",label=r"GPs (Ca+08; $z \sim 0.2 - 0.3$)"),
			Line2D([],[],ls="none",marker="p",ms=4,mec="darkred",mfc="darkred",label=r"EIGER (Ma+23; $z \sim 5 - 7$)"),
			Line2D([],[],ls="none",marker="^",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"BOSS-EUVLG1 (MC+20; $z = 2.469$)"),
			Line2D([],[],ls="none",marker="v",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"\textrm{Ion2} (deB+16; $z = 3.216$)")
			]		

leg = ax.legend(handles=handles,loc="upper right",frameon=False,ncol=2,numpoints=1,fontsize=5,columnspacing=0.075)
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
plt.ylim(10.,3e4)
plt.yscale("log")
#plt.savefig("../plots/EW_OIII_distrib.png",format="png",dpi=300,bbox_inches="tight")
plt.savefig("../../plots/EW_OIII_distrib_STScI_talk.png",format="png",dpi=300,bbox_inches="tight")

