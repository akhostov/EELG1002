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
ax.set_xlabel(r"$\log_{10}$ [N{\sc ii}]/H$\alpha$",fontsize=8)
ax.set_ylabel(r"$\log_{10}$ [O{\sc iii}]5007\AA/H$\beta$",fontsize=8)


# Define Limits
ax.set_xlim(-3.,0.5)
ax.set_ylim(-0.5,1.25)
#ax.set_yscale("log")

#ax.set_yticks([1,2,3,4,6,10])
#ax.set_yticks([1,1.5,2,2.5,3,3.5,4,5,6,7,8,9,10],minor=True)
#ax.set_yticklabels(["1","2","3","4","6","10"])





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
o3hb,o3hb_elow,o3hb_eupp = np.median(data["pdf_r3"]),np.percentile(data["pdf_r3"],16.),np.percentile(data["pdf_r3"],84.)
o3hb_elow = o3hb - o3hb_elow
o3hb_eupp = o3hb_eupp - o3hb

# Infer NII/HA from this calibration (Equation 4 of https://www.aanda.org/articles/aa/pdf/2013/11/aa21956-13.pdf)
#pdf_n2ha = (data["pdf_12OH"] - 8.743)/0.462
pdf_n2ha = np.concatenate(((data["pdf_12OH"] - 9.07)/0.79,(data["pdf_12OH"] - 8.743)/0.462,(data["pdf_12OH"] - 8.90)/0.57))
n2ha,n2ha_elow,n2ha_eupp = np.median(pdf_n2ha),np.percentile(pdf_n2ha,16.),np.percentile(pdf_n2ha,84.)
#n2ha = np.log10(sed["bayes.line.NII-658.3"]/sed["bayes.line.H-alpha"])[0] # Use the Cigale version
n2ha_elow = np.array([n2ha - n2ha_elow])
n2ha_eupp = np.array([n2ha_eupp - n2ha])

ax.errorbar([n2ha],[o3hb],xerr=(n2ha_elow,n2ha_eupp),yerr=([o3hb_elow],[o3hb_eupp]),\
				ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
                marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)


###### SPECIAL SOURCES
## BOSS-EUVLG1 (Marques-Chaves et al. 2020) # 
#BOSS_O3HB = 562/61.
#BOSS_O3HB_err = 562/61. * np.sqrt( (7/562.)**2. + (5./61.)**2. )
#BOSS_O3HB_err = BOSS_O3HB_err/(np.log(10.)*BOSS_O3HB_err)
#BOSS_O3HB = np.log10(BOSS_O3HB)
#
#BOSS_N2HA = np.log10(12./190.)
#BOSS_N2HA_err = [0.2]
#ax.errorbar(BOSS_N2HA,BOSS_O3HB,xerr=(BOSS_N2HA_err),yerr=(BOSS_O3HB_err),xuplims=[True],\
#				ls="none",mec=tableau20("Green"),mfc=tableau20("Light Green"),ecolor=tableau20("Green"),\
#                marker="d",ms=8,mew=1,capsize=2,capthick=1,elinewidth=0.5)
#
#
### Ion2 (de Barros et al. 2016)
#mass = [9.5]
#err_mass = [0.2]
#Ion2_O3Hb = [22.1/1.5]
#Ion2_O3Hb_err = [22.1/1.5 * np.sqrt( (0.8/22.1)**2. + (0.8/1.5)**2. )]
#ax.errorbar(mass,Ion2_O3Hb,xerr=(err_mass),yerr=(Ion2_O3Hb_err),			
#			ls="none",mec=tableau20("Orange"),mfc=tableau20("Light Orange"),ecolor=tableau20("Orange"),\
#            marker="p",ms=8,mew=1,capsize=2,capthick=1,elinewidth=0.5)

#### Low-z Analogs
"""
# Blueberries (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18470038Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_Blueberries/Yang2017_Blueberries.fits")[1].data
Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["Hbeta"] > 0.)]
ax.plot(Ya17["mass"],Ya17["[OIII]5007"]/Ya17["Hbeta"],				
				ls="none",mec="#464196",mfc="none",\
                marker="s",ms=3,mew=0.5)

# Green Peas (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18440171Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_GreenPeas/Yang2017_GreenPeas.fits")[1].data
Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["Hbeta"] > 0.) & (Ya17["Halpha"] > 0.)  ]
ax.plot(Ya17["logM"]-np.log10(1.64),Ya17["[OIII]5007"]/Ya17["Hbeta"],\
				ls="none",mec=tableau20("Green"),mfc="none",\
                marker="p",ms=3,mew=0.5)
"""
# Izotov et al. (2018) LyC Leakers at z ~ 0 # https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.4851I/abstract
Iz18_NII = np.array([9.7,7.1,5.8,5.7,7.9])
Iz18_Ha = np.array([288.8,283.7,280.9,279.8,280.5])
Iz18_O3 = np.array([654.6,807.2,725.9,583.6,723.3])
Iz18_Hb = np.array([100.0,100.0,100.0,100.0,100.0])

ax.plot(np.log10(Iz18_NII/Iz18_Ha),np.log10(Iz18_O3/Iz18_Hb),\
				ls="none",mec=tableau20("Purple"),mfc="none",\
                marker="p",ms=3,mew=0.5,zorder=98)

# Izotov et al. (2016) LyC Leakers z ~ 0 # https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.5491I/abstract
Iz16_NII = np.array([11.7,4.7,8.4,11.1])
Iz16_Ha = np.array([282.2,276.1,280.3,280.2])
Iz16_O3 = np.array([571.1,623.2,624.4,653.8])
Iz16_Hb = np.array([100.0,100.0,100.0,100.0])

ax.plot(np.log10(Iz16_NII/Iz16_Ha),np.log10(Iz16_O3/Iz16_Hb),\
				ls="none",mec=tableau20("Purple"),mfc="none",\
                marker="s",ms=3,mew=0.5,zorder=98)


# EMPRESS # https://ui.adsabs.harvard.edu/abs/2023ApJ...952...11N/abstract
EMPRESS_NII = np.array([2.1,3.9,14.9,4.3,6.4,5.9,21.0,2.6,1.0,0.9])
EMPRESS_Ha = np.array([272.2,276.7,280.4,271.7,271.2,269.0,281.3,282.8,271.8,269.1])
EMPRESS_O3 = np.array([483.2,478.0,374.2,514.6,607.3,442.6,308.3,554.5,470.3,387.9])
EMPRESS_Hb = np.repeat(100.,len(EMPRESS_O3))

ax.plot(np.log10(EMPRESS_NII/EMPRESS_Ha),np.log10(EMPRESS_O3/EMPRESS_Hb),\
				ls="none",mfc=tableau20("Red"),mec="black",\
                marker="p",ms=5,mew=0.5,zorder=98)

# EELG (Amorin et al. 2015) -- # https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.105A/abstract
Am15 = fits.open("../../../Main Catalogs/Amorin2015_EELGs/Amorin2015_EELGs.fits")[1].data
Am15 = Am15[ (Am15["F_Hb_"] > 0.) & (Am15["F_5007_"] > 0.) & (Am15["F_Ha_"] > 0.) & (Am15["F_6584_"] > 0.) ]
ax.plot(np.log10(Am15["F_6584_"]/Am15["F_Ha_"]),np.log10(Am15["F_5007_"]/Am15["F_Hb_"]),\
				ls="none",mec=tableau20("Green"),mfc="none",\
                marker="o",ms=3,mew=0.5,zorder=97)


# FMOS (Kashino et al. 2015)
FMOS = fits.open("../../../Main Catalogs/FMOS-COSMOS/FMOS_COSMOS_DR2.fits")[1].data
FMOS = FMOS[ (FMOS["FLUX_HBETA"] > 0.) & (FMOS["FLUX_OIII5007"] > 0.) & (FMOS["FLUX_HALPHA"] > 0.) & (FMOS["FLUX_NII6584"] > 0.) ]
ax.plot(np.log10(FMOS["FLUX_NII6584"]/FMOS["FLUX_HALPHA"]),np.log10(FMOS["FLUX_OIII5007"]/FMOS["FLUX_HBETA"]),\
				ls="none",mec=tableau20("Orange"),mfc="none",\
                marker="d",ms=2,mew=0.5,zorder=96)


# MOSDEF
MOSDEF = fits.open("../../../Main Catalogs/MOSDEF/linemeas_cor.fits")[1].data
MOSDEF = MOSDEF[ (MOSDEF["HB4863_FLUX"] > 0.) & (MOSDEF["OIII5008_FLUX"] > 0.) & (MOSDEF["HA6565_FLUX"] > 0.) & (MOSDEF["NII6585_FLUX"] > 0.) & \
					(MOSDEF["HB4863_aeflag"] != 1) & (MOSDEF["OIII5008_aeflag"] !=1.) & (MOSDEF["HA6565_aeflag"] !=1.) & (MOSDEF["NII6585_aeflag"] !=1.) & \
					(MOSDEF["HB4863_slflag"] < 0.2) & (MOSDEF["OIII5008_slflag"] < 0.2) & (MOSDEF["HA6565_slflag"] < 0.2) & (MOSDEF["NII6585_slflag"] < 0.2)]
ax.plot(np.log10(MOSDEF["NII6585_FLUX"]/MOSDEF["HA6565_FLUX"]),np.log10(MOSDEF["OIII5008_FLUX"]/MOSDEF["HB4863_FLUX"]),\
				ls="none",mec=tableau20("Grey"),mfc="none",\
                marker="^",ms=2,mew=0.5,zorder=96)

## JADES (Cameron et al. 2023) # https://www.aanda.org/articles/aa/pdf/2023/09/aa46107-23.pdf
#JADES_N2 = [-0.81,-0.75,-0.39,-1.06,-1.29,-0.75,-0.80,-0.95,-0.85,-1.01,-0.83,-0.96,-0.80,-1.34,-1.13,-1.15,-0.89,-1.17]
#JADES_R3 = [0.78,0.79,0.94,0.54,0.64,0.63,0.73,0.65,0.68,0.74,0.71,0.75,0.52,0.83,0.65,0.82,0.61,0.82]
#
#ax.plot(JADES_N2,JADES_R3,\
#				ls="none",mfc=tableau20("Orange"),mec="none",\
#                marker="o",ms=3,mew=0.5,zorder=99)

### BPT SELECTION LINES

# Kewley et al. (2001)
BPT_n2ha = np.arange(-3.,0.47,0.01)
BPT_o3hb = 0.61/(BPT_n2ha - 0.47) + 1.19
plt.plot(BPT_n2ha,BPT_o3hb,ls="--",color="black")

# Kauffman et al. (2003)
BPT_n2ha = np.arange(-3.,0.05,0.01)
BPT_o3hb = 0.61/(BPT_n2ha - 0.05) + 1.3
plt.plot(BPT_n2ha,BPT_o3hb,ls=":",color="red")

# Xiao et al. (2018)
#BPT_n2ha = np.arange(-3.,0.05,0.01)
#BPT_o3hb = 0.28/(BPT_n2ha - 0.26) + 1.36
#plt.plot(BPT_n2ha,BPT_o3hb,ls=":",color=tableau20("Blue"))


#ax.text(9,8,r"\textit{MEx-AGN}",color=tableau20("Red"),fontsize=6,ha="left",va="center")
#ax.text(7.2,3,r"\textit{MEx-SF}",color=tableau20("Blue"),fontsize=6,ha="left",va="center")


# Matthee et al. (2023) 5 < z < 7 #https://ui.adsabs.harvard.edu/abs/2023ApJ...950...67M/abstract
#mass = np.array([7.5,8.2,8.9,9.5])
#mass_elow = np.array([7.5-6.8,0.7/2.,0.7/2.,0.6/2.])
#mass_eupp = np.array([0.7/2.,0.7/2.,0.6/2.,10.2-9.5])
#Ma23_O3HB = np.array([5.3, 6.7, 6.3, 5.4])
#Ma23_O3HB_elow = np.array([0.8, 0.6, 0.7, 0.4])
#Ma23_O3HB_eupp = np.array([0.9, 0.6, 0.9, 0.5])
#
#ax.errorbar(mass,Ma23_O3HB,xerr=(mass_elow,mass_eupp),yerr=(Ma23_O3HB_elow,Ma23_O3HB_eupp),\
#			ls="none",mec="black",mfc=tableau20("Red"),ecolor=tableau20("Grey"),\
#            marker="o",ms=3,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)



ax.text(-2.5,0.25,r"\textit{BPT-SFG}",color=tableau20("Blue"),fontsize=6,ha="left",va="center")
ax.text(0.,1.0,r"\textit{BPT-AGN}",color=tableau20("Red"),fontsize=6,ha="center",va="center")



handles = [Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}")]	

leg1 = ax.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg1)


handles = [	Line2D([],[],ls="none",marker="p",ms=4,mfc=tableau20("Red"),mec="black",label=r"EMPGs (Ni+23; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Purple"),mfc="none",label=r"LyC Leakers (Iz+16,18; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Green"),mfc="none",label=r"EELGs (Am+15; $z \sim 0.1 - 0.9$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Orange"),mfc="none",label=r"FMOS (Ka+19; $z \sim 1.6$)"),
			Line2D([],[],ls="none",marker="^",ms=4,mec=tableau20("Grey"),mfc="none",label=r"MOSDEF (Kr+15; $z \sim 1.2 - 2.7$)")]		

leg2 = ax.legend(handles=handles,loc="lower left",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg2)		


handles = [	Line2D([],[],ls="--",lw=1,color="black",label=r"Ke+01"),
			Line2D([],[],ls=":",lw=1,color="red",label=r"Ka+03")]		

leg3 = ax.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg3)	

plt.savefig("../../plots/BPT.png",format="png",dpi=300,bbox_inches="tight")

pdb.set_trace()
exit()



