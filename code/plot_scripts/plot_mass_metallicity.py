import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from astropy.io import fits
import numpy as np 
import pickle
import pdb

import sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20

######################################################################
#######						PREPARE THE FIGURE				   #######
######################################################################

fig = plt.figure()
fig.set_size_inches(3,3)
ax = fig.add_subplot(111)

# Define Labels
ax.set_xlabel(r"$\log_{10}$ Stellar Mass (M$_\odot$)",fontsize=8)
ax.set_ylabel(r"$12+\log_{10}($O/H$)$",fontsize=8)


# Define Limits
ax.set_ylim(6.7,9.)
ax.set_xlim(6.5,11)

#### OUR SOURCE
# Load in the Line properties
file = open("../../data/emline_fits/43158747673038238/ratios_and_ISM_props_1002_new.pkl","rb")
data = pickle.load(file)
file.close()

# Load in the SED properties (use CIGALE for now)
sed = fits.open("../../data/cigale_results/sfh_delayed_nodust/results.fits")[1].data
mass = sed["bayes.stellar.m_star"]
err_mass = sed["bayes.stellar.m_star_err"]/(np.log(10.)*mass)
mass = np.log10(mass)

# Get Metallicity for this Object
zgas,zgas_low,zgas_upp = np.median(data["pdf_12OH"]),np.percentile(data["pdf_12OH"],16.),np.percentile(data["pdf_12OH"],84.)
zgas_low = zgas - zgas_low
zgas_upp = zgas_upp - zgas

ax.errorbar([mass],[zgas],xerr=([err_mass]),yerr=([zgas_low],[zgas_upp]),\
				ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
                marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5)

#### SPECIAL SOURCES
# BOSS-EUVLG1 (Marques-Chaves et al. 2020)
mass = [10.0]
err_mass = [0.1]
BOSS_zgas = [8.13]
BOSS_zgas_err = [0.19]
ax.errorbar(mass,BOSS_zgas,xerr=(err_mass),yerr=(BOSS_zgas_err),\
				ls="none",mec=tableau20("Green"),mfc=tableau20("Light Green"),ecolor=tableau20("Green"),\
                marker="d",ms=8,mew=1,capsize=2,capthick=1,elinewidth=0.5)


# Ion2 (de Barros et al. 2016)
mass = [9.5]
err_mass = [0.2]
Ion2_zgas = [7.79]
Ion2_zgas_err = [0.35]
ax.errorbar(mass,Ion2_zgas,xerr=(err_mass),yerr=(Ion2_zgas_err),			
			ls="none",mec=tableau20("Orange"),mfc=tableau20("Light Orange"),ecolor=tableau20("Orange"),\
            marker="p",ms=8,mew=1,capsize=2,capthick=1,elinewidth=0.5)

#### Low-z Analogs

# Blueberries (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18470038Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_Blueberries/Yang2017_Blueberries.fits")[1].data
Ya17 = Ya17[ Ya17["12+log(O/H)"] > 0.]
ax.plot(Ya17["mass"],Ya17["12+log(O/H)"],				
				ls="none",mec="navy",mfc="none",\
                marker="s",ms=3,mew=0.2)

# Green Peas (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18440171Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_GreenPeas/Yang2017_GreenPeas_table1.fits")[1].data
Ya17 = Ya17[ Ya17["12+log(O/H)"] > 0.]
ax.plot(Ya17["logM"],Ya17["12+log(O/H)"],\
				ls="none",mec=tableau20("Green"),mfc="none",\
                marker="o",ms=3,mew=0.2)

"""
# EELG (Amorin et al. 2015) -- # https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.105A/abstract
Am15 = fits.open("../../../Main Catalogs/Amorin2015_EELGs/Amorin2015_EELGs.fits")[1].data
Am15 = Am15[ Am15["Ab_O_"] > 0.]
ax.plot(Am15["logMs"],Am15["Ab_O_"],\
				ls="none",mec=tableau20("Sky Blue"),mfc="none",\
                marker="o",ms=3,mew=0.2)
"""



handles = [Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}"),
			Line2D([],[],ls="none",marker="d",ms=5,mec=tableau20("Green"),mfc=tableau20("Light Green"),label=r"BOSS-EUVLG1 (MC+20; $z = 2.47$)"),
			Line2D([],[],ls="none",marker="p",ms=5,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"\textit{Ion2} (deBa+16; $z = 3.2$)"),
			Line2D([],[],ls="none",marker="s",ms=3,mec="navy",mfc="none",label=r"Blueberries (Ya+17; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="o",ms=3,mec=tableau20("Green"),mfc="none",label=r"GPs (Ya+17; $z \sim 0.2$)")]			

leg1 = ax.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg1)












# Tremonti et al. (2004) #https://ui.adsabs.harvard.edu/abs/2004ApJ...613..898T/abstract
#conversion = 1.5/1.64 #Kroupa to Chabrier
#mass = np.arange(8.5,11.6,0.1)+np.log10(conversion)

#Tr04_zgas = -1.492+1.847*mass-0.08026*mass**2.

#ax.plot(mass,Tr04_zgas,ls="--")


############## FITS GO HERE
### Andrews & Martini et al. (2013) #https://ui.adsabs.harvard.edu/abs/2013ApJ...765..140A/abstract
mass = np.arange(7.4,10.5,0.1)
AM13_zgas = 8.798 - np.log10(1+pow(10,8.901 - mass)**0.64)

ax.plot(mass,AM13_zgas,ls="--",color=tableau20("Purple"))

### Ly et al. (2016) # https://iopscience.iop.org/article/10.3847/0004-637X/828/2/67/pdf
# This is only the 0.5 < z < 1.0 data with [OIII]4363 detection
mass = np.arange(7.25,10.26,0.1)
Ly16_zgas = 8.53 - np.log10(1.+ pow(10,8.901-mass)**0.57)

ax.plot(mass,Ly16_zgas,ls="--",color=tableau20("Blue"))


### Sanders et al. (2021) # https://ui.adsabs.harvard.edu/abs/2021ApJ...914...19S/abstract
mass = np.arange(9.0,11.1,0.1)

Sa21_z2p3_zgas = 8.51 + (mass - 10.)*0.30
Sa21_z3p3_zgas = 8.41 + (mass - 10.)*0.29

ax.plot(mass,Sa21_z2p3_zgas,ls="--",color=tableau20("Olive"))
ax.plot(mass,Sa21_z3p3_zgas,ls="--",color=tableau20("Orange"))


### Langeroodi et al. (2023) # https://iopscience.iop.org/article/10.3847/1538-4357/acdbc1/pdf
mass = np.arange(7.5,10.6,0.1)
La23_zgas = 9.0 - 0.98 + 0.3*(mass - 10.)
ax.plot(mass,La23_zgas,ls="--",color=tableau20("Red"))


handles = [Line2D([],[],ls="--",color=tableau20("Purple"),label=r"A\&M+13 ($z \sim 0$)"),
			Line2D([],[],ls="--",color=tableau20("Blue"),label=r"Ly+16 ($z \sim 0.5 - 1$)"),
			Line2D([],[],ls="--",color=tableau20("Olive"),label=r"Sa+21 ($z \sim 2.2$)"),
			Line2D([],[],ls="--",color=tableau20("Orange"),label=r"Sa+21 ($z \sim 3.3$)"),
			Line2D([],[],ls="--",color=tableau20("Red"),label=r"La+23 ($z \sim 8$)")
			]

leg2 = ax.legend(handles=handles,loc="lower right",frameon=False,ncol=2,numpoints=1,fontsize=5)
plt.gca().add_artist(leg2)







# Ly et al. (2016) # https://iopscience.iop.org/article/10.3847/0004-637X/828/2/67/pdf
# This is only the 0.5 < z < 1.0 data with [OIII]4363 detection
mass = np.array([7.50,8.00,8.50,9.00,9.50,10.00])
err_mass = np.array([0.25,0.25,0.25,0.25,0.25,0.25])
Ly16_zgas = np.array([7.63,7.92,8.10,8.25,8.41,8.37])
Ly16_zgas_elow = np.array([0.14,0.07,0.05,0.04,0.08,0.16])
Ly16_zgas_eupp = np.array([0.11,0.07,0.04,0.04,0.09,0.23])

ax.errorbar(mass,Ly16_zgas,xerr=(err_mass),yerr=(Ly16_zgas_elow,Ly16_zgas_eupp),\
			ls="none",mec="black",mfc=tableau20("Blue"),ecolor=tableau20("Grey"),\
            marker="^",ms=3,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)


# MUSE (Revalski et al. 2024) # https://arxiv.org/pdf/2403.17047.pdf
# 1 < z < 2
mass = np.array([6.5,7.8,8.3,8.8,9.3,9.8,11.0])
err_mass = np.diff(mass)
mass = 0.5*(mass[1:] + mass[:-1])
MUSE_zgas = np.array([7.84,8.10,8.22,8.30,8.34,8.42])
MUSE_zgas_elow = np.array([0.16,0.16,0.08,0.03,0.04,0.08])
MUSE_zgas_eupp = np.array([0.16,0.12,0.08,0.04,0.06,0.06])

ax.errorbar(mass,MUSE_zgas,xerr=(err_mass),yerr=(MUSE_zgas_elow,MUSE_zgas_eupp),\
				ls="none",mec="black",mfc=tableau20("Green"),ecolor=tableau20("Grey"),\
                marker="v",ms=3,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

# Matthee et al. (2023) 5 < z < 7 #https://ui.adsabs.harvard.edu/abs/2023ApJ...950...67M/abstract
mass = np.array([7.5,8.2,8.9,9.5])
mass_elow = np.array([7.5-6.8,0.7/2.,0.7/2.,0.6/2.])
mass_eupp = np.array([0.7/2.,0.7/2.,0.6/2.,10.2-9.5])
Ma23_zgas = np.array([7.29,7.77,8.05,8.19])
Ma23_zgas_elow = np.array([0.14,0.20,0.35,0.06])
Ma23_zgas_eupp = np.array([0.15,0.22,0.12,0.06])

ax.errorbar(mass,Ma23_zgas,xerr=(mass_elow,mass_eupp),yerr=(Ma23_zgas_elow,Ma23_zgas_eupp),\
			ls="none",mec="black",mfc=tableau20("Red"),ecolor=tableau20("Grey"),\
            marker="o",ms=3,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)


handles = [Line2D([],[],ls="none",marker="^",ms=5,mec="black",mew=0.2,mfc=tableau20("Blue"),label=r"MACT (Ly+16; $z \sim 0.5 - 1$)"),
			Line2D([],[],ls="none",marker="v",ms=5,mec="black",mew=0.2,mfc=tableau20("Green"),label=r"MUSE (Re+24; $z \sim 1 - 2$)"),
			Line2D([],[],ls="none",marker="o",ms=5,mec="black",mew=0.2,mfc=tableau20("Red"),label=r"EIGER (Ma+23; $5 - 7$)"),
			]

leg3 = ax.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=5,bbox_to_anchor=(0.99,0.15),handletextpad=0.02)
plt.gca().add_artist(leg3)



"""
# Curti et al. (2023) JADES 6 < z < 10 # https://arxiv.org/pdf/2304.08516.pdf
mass = np.array([6.,7.75,8.5,10.])
err_mass = np.diff(mass)/2.
mass = 0.5*(mass[1:] + mass[:-1])
Cu23_zgas = np.array([7.64,7.67,7.73])
Cu23_zgas_err = np.array([0.07,0.06,0.08])

plt.errorbar(mass,Cu23_zgas,xerr=(err_mass),yerr=(Cu23_zgas_err),marker="s",ls="none")
"""




## CLEAR #https://iopscience.iop.org/article/10.3847/1538-4357/ac8058/pdf
## 1.1 < z < 2.3
#mass = np.array([9.0,9.5,10.,10.5])
#err_mass = np.array([0.25,0.25,0.25,0.25])
#uplims = np.array([False,False,False,True])
#CLEAR_zgas = np.array([8.42,8.55,8.65,8.83])
#CLEAR_zgas_err = np.array([0.03,0.02,0.03,0.05])
#
#ax.errorbar(mass,CLEAR_zgas,xerr=(err_mass,err_mass),yerr=(CLEAR_zgas_err),xuplims=np.repeat(False,len(mass)),xlolims=uplims,\
#			ls="none",mec="black",mfc=tableau20("Green"),ecolor=tableau20("Grey"),\
#            marker="o",ms=3,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)








plt.savefig("../../plots/MZR_EELG1002.png",format="png",dpi=300,bbox_inches="tight")


pdb.set_trace()