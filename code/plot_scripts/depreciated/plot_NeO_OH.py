import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from astropy.io import fits
import numpy as np 
import pickle
import pdb

import sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20

sys.path.insert(0, "..")
import util


def LyC_Leakers(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    # Load in Izotov et al. (2016, 2018), Jaskot & Oey (2013)
    Iz16_OH,Iz16_OH_err,\
        Iz16_NeO,Iz16_NeO_err = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2016.txt",unpack=True,skiprows=5,usecols=(-7,-6,-5,-4))

    Iz18_OH,Iz18_OH_err,\
        Iz18_NeO,Iz18_NeO_err = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2018.txt",unpack=True,skiprows=5,usecols=(-7,-6,-5,-4))

    Gu20_OH,Gu20_OH_err,\
        Gu20_NeO,Gu20_NeO_err = np.loadtxt("../../data/literature_measurements/Guseva_et_al_2020.txt",unpack=True,skiprows=5,usecols=(2,3,4,5))


    # Combine the data
    OH = np.concatenate((Iz16_OH,Iz18_OH,Gu20_OH))
    OH_err = np.concatenate((Iz16_OH_err,Iz18_OH_err,Gu20_OH_err))
    NeO = np.concatenate((Iz16_NeO,Iz18_NeO,Gu20_NeO))
    NeO_err = np.concatenate((Iz16_NeO_err,Iz18_NeO_err,Gu20_NeO_err))

    # Plot
    ax.errorbar(OH,NeO,xerr=OH_err,yerr=NeO_err,ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=2,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)
    
    return ax

def Izotov2006(ax,mfc,mec,marker,ms,zorder=98,alpha=1):
    # Load in Izotov et al. (2006)
    data = fits.open("../../data/literature_measurements/Izotov_et_al_2006.fits")[1].data

    # Plot
    ax.plot(data["_12_logO_H"],data["logNe_O"],ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,
                    mew=0.5,zorder=zorder,alpha=alpha)

    return ax

def Guseva(ax,mfc,mec,marker,ms,zorder=98,alpha=1):
    # Load in Guseva et al. (2011)
    OH,OH_err,NeO,NeO_err = np.loadtxt("../../data/literature_measurements/Guseva_et_al_2011.txt",unpack=True,usecols=(1,2,3,4))

    # Plot
    ax.plot(OH,NeO,ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,
                    mew=0.5,zorder=zorder,alpha=alpha)
    """
    ax.errorbar(OH,NeO,xerr=OH_err,yerr=NeO_err,ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=1,capthick=0.5,elinewidth=0.5,zorder=zorder,alpha=alpha)
    """

    return ax
def EMPGs(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    # Load in Kojima et al. (2021)
    Ko21_OH,Ko21_OH_elow,Ko21_OH_eupp,\
        Ko21_NeO,Ko21_NeO_elow,Ko21_NeO_eupp = np.loadtxt("../../data/literature_measurements/Kojima_et_al_2021.txt",unpack=True,usecols=(2,3,4,5,6,7))

    # Watanabe et al. (2024) -- Ne/OUnits are in Asplund et al. 2021 solar abundances units (-0.75 for Ne/O)
    Wa24_OH = np.array([7.23,7.56]); Wa24_OH_err = np.array([0.01,0.01])
    Wa24_NeO = np.array([-0.27,-0.01])-0.75; Wa24_NeO_err = np.array([0.00,0.00])

    # Isobe et al. (2022)
    Is22_OH,Is22_OH_elow,Is22_OH_eupp,\
        Is22_NeO,Is22_NeO_elow,Is22_NeO_eupp = np.loadtxt("../../data/literature_measurements/Isobe_et_al_2022.txt",unpack=True,usecols=(2,3,4,5,6,7))

    # Berg et al. (2021)
    Be21_OH = np.array([7.44,7.62]); Be21_OH_err = np.array([0.01,0.01])
    Be21_NeO = np.array([-0.79,-0.75]); Be21_NeO_err = np.array([0.03,0.03])

    # Combine the Data
    OH = np.concatenate((Ko21_OH,Is22_OH,Wa24_OH,Be21_OH))
    OH_elow = np.concatenate((Ko21_OH_elow,Is22_OH_elow,Wa24_OH_err,Be21_OH_err))
    OH_eupp = np.concatenate((Ko21_OH_eupp,Is22_OH_eupp,Wa24_OH_err,Be21_OH_err))

    NeO = np.concatenate((Ko21_NeO,Is22_NeO,Wa24_NeO,Be21_NeO))
    NeO_elow = np.concatenate((Ko21_NeO_elow,Is22_NeO_elow,Wa24_NeO_err,Be21_NeO_err))
    NeO_eupp = np.concatenate((Ko21_NeO_eupp,Is22_NeO_eupp,Wa24_NeO_err,Be21_NeO_err))

    ax.errorbar(OH,NeO,xerr=(OH_elow,OH_eupp),yerr=(NeO_elow,NeO_eupp),ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=2,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)
    
    return ax

# Marques-Chaves et al. (2024)
def CEERS_1019(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    OH = np.array([7.70]); OH_err = np.array([0.18])
    NeO = np.array([-0.63]); NeO_err = np.array([0.07])

    ax.errorbar(OH,NeO,xerr=OH_err,yerr=NeO_err,ls="none",
                    mfc=mfc,mec=mec,ecolor=mec,
                    marker=marker,ms=ms,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)
    
    return ax


# Arellano-Cordova et al. (2022)  
def Arellano(ax,mfc,mec,marker,ms,zorder=98,alpha=1):
    
    # Using only o008
    OH = np.array([7.12,8.17,7.73]); OH_err = np.array([0.12,0.12,0.08])
    NeO = np.array([-0.52,-0.64,-0.58]); NeO_err = np.array([0.15,0.17,0.09])

    ax.errorbar(OH,NeO,xerr=OH_err,yerr=NeO_err,ls="none",
                    mfc=mfc,mec=mec,ecolor=mec,
                    marker=marker,ms=ms,mew=0.5,capsize=1,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)

    return ax

# Green Peas
def GPs(ax,mfc,mec,marker,ms,zorder=98,alpha=1):
    
    # Load Amorin et al. (2012)
    OH,OH_err,NeO,NeO_err = np.loadtxt("../../data/literature_measurements/Amorin_et_al_2012.txt",unpack=True,usecols=(1,2,3,4))

    ax.errorbar(OH,NeO,xerr=OH_err,yerr=NeO_err,ls="none",
                    mfc=mfc,mec=mec,ecolor=mec,
                    marker=marker,ms=ms,mew=0.5,capsize=1,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)

    return ax


######################################################################
#######						PREPARE THE FIGURE				   #######
######################################################################

fig = plt.figure()
fig.set_size_inches(3,3)
ax = fig.add_subplot(111)

# Define Labels
ax.set_ylabel(r"$\log_{10}$ (Ne/O)",fontsize=8)
ax.set_xlabel(r"$12+\log_{10}$(O/H)",fontsize=8)


# Define Limits
ax.set_xlim(6.8,8.5)
ax.set_ylim(-1.5,-0.3)


#### OUR SOURCE
data = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1].data


ax.errorbar(data["12+log10(O/H)_med"],data["log10(Ne/O)_med"],
			xerr=(data["12+log10(O/H)_err_low"],data["12+log10(O/H)_err_up"]),\
				yerr=(data["log10(Ne/O)_err_low"],data["log10(Ne/O)_err_up"]),\
				ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
                marker="*",ms=12,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

# Izotov et al. (2006)
ax = Izotov2006(ax,mec=tableau20("Grey"),mfc=tableau20("Light Grey"),marker="o",ms=2, alpha=0.5)

# Guseva et al. (2011) 
ax = Guseva(ax,mec=tableau20("Grey"),mfc=tableau20("Light Grey"),marker="o",ms=2,zorder=1, alpha=0.8)

# Local LyC Leakers
ax = LyC_Leakers(ax,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),marker="s",ms=3, alpha=0.8)

# EMPGs
ax = EMPGs(ax,mec=tableau20("Red"),mfc=tableau20("Light Red"),marker="d",ms=3, alpha=0.8)

# GPs
ax = GPs(ax,mec=tableau20("Green"),mfc=tableau20("Light Green"),marker="<",ms=3, alpha=0.8)

# Arellano-Cordova et al. (2022)  
ax = Arellano(ax,mec=tableau20("Brown"),mfc=tableau20("Light Brown"),marker="^",ms=4, alpha=0.8)

# CEERS-1019
ax = CEERS_1019(ax,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),marker="o",ms=4, alpha=0.8)


# Plot the Solar Metallicity (Asplund et al. 2021)
plt.plot([6.,10.],[-0.63,-0.63],ls="--",color="royalblue",lw=1.)
plt.fill_between([6.,10.],-0.63-0.06,-0.63+0.06,facecolor="royalblue",alpha=0.2)

handles = [	Line2D([],[],ls="none",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),marker="*",label=r"\textbf{\textit{EELG1002 (This Work)}}"),
			Line2D([],[],ls="none",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),marker="p",label=r"$z \sim 0$ EMPGs (Be+21,Ko+21,Is+22,Wa+24)"),
			Line2D([],[],ls="none",ms=4,mec=tableau20("Grey"),mfc=tableau20("Light Grey"),marker="o",label=r"$z \sim 0$ low-metallicity ELGs (Iz+06,Gu+11)"),
			Line2D([],[],ls="none",ms=4,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),marker="s",label=r"$z \sim 0 - 0.3$ LyC Leakers (Ja+13,Iz+16,18)"),
			Line2D([],[],ls="none",ms=4,mec=tableau20("Green"),mfc=tableau20("Light Green"),marker="<",label=r"$z \sim 0.3$ GPs (Am+12)"),
            Line2D([],[],ls="none",ms=4,mec=tableau20("Brown"),mfc=tableau20("Light Brown"),marker="^",label=r"$z \sim 7.7 - 8.5$ SFGs (Ar+22)"),
            Line2D([],[],ls="none",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),marker="o",label=r"$z = 10.6$ CEERS-1019 (Ma+24)")
			]		

leg = ax.legend(handles=handles,loc="lower left",frameon=False,ncol=1,numpoints=1,fontsize=4,columnspacing=0.075,labelspacing=0.8)
plt.gca().add_artist(leg)	


plt.savefig("../../plots/NeO_OH.png",format="png",dpi=300,bbox_inches="tight")

