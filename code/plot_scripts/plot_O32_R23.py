import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from astropy.io import fits
import numpy as np 
import pickle
import pdb

import sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20


def Calzetti(wave):
	wave = wave/1e4

	if wave <= 0.63:
		return 2.659*(-2.156 + 1.509/wave -0.198/wave**2. + 0.011/wave**3.) + 4.05
	if wave > 0.63:
		return 2.659*(-1.857 + 1.040/wave) + 4.05


######################################################################
#######						PREPARE THE FIGURE				   #######
######################################################################

fig = plt.figure()
fig.set_size_inches(3,3)
ax = fig.add_subplot(111)

# Define Labels
ax.set_xlabel(r"$\log_{10}$ $R_{23}$",fontsize=8)
ax.set_ylabel(r"$\log_{10}$ [O{\sc iii}]5007\AA/[O{\sc ii}]3727\AA",fontsize=8)


# Define Limits
ax.set_xlim(0.0,1.2)
ax.set_ylim(-1.0,2.0)





#### OUR SOURCE
# Load in the Line properties
with open("../../data/emline_fits/1002_line_ratios.pkl","rb") as file: data = pickle.load(file)
file.close()

# Load in the SED properties (use CIGALE for now)
mask_r23 = data["name"].index("R23")
mask_o32 = data["name"].index("O32")

r23 = data["median"][mask_r23]
r23_elow = np.log10(r23) - np.log10(r23 - data["low_1sigma"][mask_r23])
r23_eupp = np.log10(data["upp_1sigma"][mask_r23] + r23) - np.log10(r23)
r23 = np.log10(r23)

o32 = data["median"][mask_o32]
o32_elow = np.log10(o32) - np.log10(o32 - data["low_1sigma"][mask_o32])
o32_eupp = np.log10(data["upp_1sigma"][mask_o32] + o32) - np.log10(o32)
o32 = np.log10(o32)

ax.errorbar(r23,o32,
			xerr=([r23_elow],[r23_eupp]),\
				yerr=([o32_elow],[o32_eupp]),\
				ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
                marker="*",ms=12,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)


#### Low-z Analogs
# Blueberries (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18470038Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_Blueberries/Yang2017_Blueberries.fits")[1].data
Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["[OIII]4959"] > 0.) & (Ya17["[OII]3727"] > 0.) & (Ya17["Hbeta"] > 0.)]
Ya17_O32 = np.log10(Ya17["[OIII]5007"]/Ya17["[OII]3727"])#*pow(10,0.4*Ya17["E(B-V)MW"]*(Calzetti(5007.) - Calzetti(3727.))))
Ya17_R23 = np.log10((Ya17["[OIII]5007"] + Ya17["[OIII]4959"] + Ya17["[OII]3727"])/(Ya17["Hbeta"]))#*pow(10,0.4*Ya17["E(B-V)MW"]*(Calzetti(5007.) + Calzetti(4959.) + Calzetti(3727.) - Calzetti(4861.))))
ax.plot(Ya17_R23,Ya17_O32,				
				ls="none",mec="#464196",mfc="none",\
                marker="s",ms=3,mew=0.5,zorder=97)

# Green Peas (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18440171Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_GreenPeas/Yang2017_GreenPeas.fits")[1].data
Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["[OIII]4959"] > 0.) & (Ya17["[OII]3727"] > 0.) & (Ya17["Hbeta"] > 0.)]
Ya17_O32 = np.log10(Ya17["[OIII]5007"]/Ya17["[OII]3727"])#*pow(10,0.4*Ya17["E(B-V)MW"]*(Calzetti(5007.) - Calzetti(3727.))))
Ya17_R23 = np.log10((Ya17["[OIII]5007"] + Ya17["[OIII]4959"] + Ya17["[OII]3727"])/(Ya17["Hbeta"]))#*pow(10,0.4*Ya17["E(B-V)MW"]*(Calzetti(5007.) + Calzetti(4959.) + Calzetti(3727.) - Calzetti(4861.))))
ax.plot(Ya17_R23,Ya17_O32,				
				ls="none",mec=tableau20("Green"),mfc="none",\
                marker="^",ms=3,mew=0.5,zorder=98)

## Plot SDSS Sources
print ("Running SDSS")

SDSS = fits.open("../../../Main Catalogs/SDSS/DR12/portsmouth_emlinekin_full-DR12-boss.fits")[1].data
O3_5007 = SDSS["FLUX"].T[17]
O3_4959 = SDSS["FLUX"].T[16]
HB_4861 = SDSS["FLUX"].T[15]
O2_3729 = SDSS["FLUX"].T[4]
O2_3726 = SDSS["FLUX"].T[3]

O3_5007_err = SDSS["FLUX_ERR"].T[17]
O3_4959_err = SDSS["FLUX_ERR"].T[16]
HB_4861_err = SDSS["FLUX_ERR"].T[15]
O2_3729_err = SDSS["FLUX_ERR"].T[4]
O2_3726_err = SDSS["FLUX_ERR"].T[3]


O3_5007_FIT_WARNING = SDSS["FIT_WARNING"].T[17]
O3_4959_FIT_WARNING = SDSS["FIT_WARNING"].T[16]
HB_4861_FIT_WARNING = SDSS["FIT_WARNING"].T[15]
O2_3729_FIT_WARNING = SDSS["FIT_WARNING"].T[4]
O2_3726_FIT_WARNING = SDSS["FIT_WARNING"].T[3]

SDSS = SDSS[ (O3_5007 > 0.) & (O3_4959 > 0.) & (O2_3726 > 0.) & (O2_3729 > 0.) & (HB_4861 > 0.) & \
			 (O3_5007_err > 0.) & (O3_4959_err > 0.) & (O2_3726_err > 0.) & (O2_3729_err > 0.) & (HB_4861_err > 0.) & \
			 (O3_5007_FIT_WARNING == 0.) & (O3_4959_FIT_WARNING == 0.) & (O2_3726_FIT_WARNING == 0.) & (O2_3729_FIT_WARNING == 0.) & (HB_4861_FIT_WARNING == 0.)	& \
			 (O3_5007/O3_5007_err > 4.) & (O3_4959/O3_4959_err > 4.) & (O2_3726/O2_3726_err > 4.) & (O2_3729/O2_3729_err > 4.) & (HB_4861/HB_4861_err > 4.) ]

O3_5007 = SDSS["FLUX"].T[17]
O3_4959 = SDSS["FLUX"].T[16]
HB_4861 = SDSS["FLUX"].T[15]
O2_3729 = SDSS["FLUX"].T[4]
O2_3726 = SDSS["FLUX"].T[3]

O3_5007_err = SDSS["FLUX_ERR"].T[17]
O3_4959_err = SDSS["FLUX_ERR"].T[16]
HB_4861_err = SDSS["FLUX_ERR"].T[15]
O2_3729_err = SDSS["FLUX_ERR"].T[4]
O2_3726_err = SDSS["FLUX_ERR"].T[3]


O3_5007_FIT_WARNING = SDSS["FIT_WARNING"].T[17]
O3_4959_FIT_WARNING = SDSS["FIT_WARNING"].T[16]
HB_4861_FIT_WARNING = SDSS["FIT_WARNING"].T[15]
O2_3729_FIT_WARNING = SDSS["FIT_WARNING"].T[4]
O2_3726_FIT_WARNING = SDSS["FIT_WARNING"].T[3]


SDSS_O2 = O2_3729 + O2_3726
SDSS_O32 = np.log10(O3_5007/SDSS_O2)
SDSS_R23 = np.log10( (O3_5007+ O3_4959 + SDSS_O2)/(HB_4861) )#*pow(10,0.4*SDSS["EBMV"]*(Calzetti(5007.) + Calzetti(4959.) + Calzetti(3727.) - Calzetti(4861.))))

ax.plot(SDSS_R23,SDSS_O32,				
				ls="none",mfc=tableau20("Light Grey"),mec="none",\
                marker="o",ms=1,alpha=0.8,zorder=1)
print("Finished SDSS")


# MOSDEF
MOSDEF = fits.open("../../../Main Catalogs/MOSDEF/linemeas_cor.fits")[1].data
MOSDEF = MOSDEF[ (MOSDEF["HB4863_FLUX"] > 0.) & (MOSDEF["OIII5008_FLUX"] > 0.) & (MOSDEF["OIII4960_FLUX"] > 0.) & (MOSDEF["OII3727_FLUX"] > 0.) & (MOSDEF["OII3730_FLUX"] > 0.) & \
					(MOSDEF["HB4863_aeflag"] != 1) & (MOSDEF["OIII5008_aeflag"] !=1.) & (MOSDEF["OIII4960_aeflag"] !=1.) & (MOSDEF["OII3727_aeflag"] !=1.) & (MOSDEF["OII3730_aeflag"] !=1.) &\
					(MOSDEF["HB4863_slflag"] < 0.2) & (MOSDEF["OIII5008_slflag"] < 0.2) & (MOSDEF["OIII4960_slflag"] < 0.2) & (MOSDEF["OII3727_slflag"] < 0.2) & (MOSDEF["OII3730_slflag"] < 0.2)]

MOSDEF_O32 = np.log10(MOSDEF["OIII5008_FLUX"]/(MOSDEF["OII3730_FLUX"] + MOSDEF["OII3727_FLUX"]))
MOSDEF_R23 = np.log10((MOSDEF["OIII5008_FLUX"] + MOSDEF["OIII4960_FLUX"] + ( MOSDEF["OII3727_FLUX"] + MOSDEF["OII3730_FLUX"] ))/MOSDEF["HB4863_FLUX"] )
ax.plot(MOSDEF_R23,MOSDEF_O32,\
				ls="none",mec=tableau20("Light Brown"),mfc="none",\
                marker="<",ms=2,mew=0.5,zorder=96)


# Izotov et al. (2018) LyC Leakers at z ~ 0 # https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.4851I/abstract
Iz18_O3_5007 = np.array([654.6,807.2,725.9,583.6,723.3])
Iz18_O3_4959 = np.array([221.2,261.9,240.9,194.0,249.6])
Iz18_O2_3727 = np.array([81.5,29.7,53.7,49.4,44.3])
Iz18_Hb = np.array([100.0,100.0,100.0,100.0,100.0])

Iz18_R23 = np.log10( (Iz18_O3_5007 + Iz18_O3_4959 + Iz18_O2_3727) / Iz18_Hb)
Iz18_O32 = np.log10( Iz18_O3_5007 / Iz18_O2_3727)
ax.plot(Iz18_R23, Iz18_O32,\
				ls="none",mec=tableau20("Purple"),mfc="none",\
                marker="d",ms=3,mew=0.5,zorder=98)


# Izotov et al. (2016) LyC Leakers z ~ 0 # https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.3683I/abstract
Iz16_O3_5007 = np.array([571.1,623.2,624.4,653.8])
Iz16_O3_4959 = np.array([189.1,208.7,205.7,221.1])
Iz16_O2_3727 = np.array([105.8,130.2,93.6,134.3])
Iz16_Hb = np.array([100.0,100.0,100.0,100.0])

Iz16_R23 = np.log10( (Iz16_O3_5007 + Iz16_O3_4959 + Iz16_O2_3727) / Iz16_Hb)
Iz16_O32 = np.log10( Iz16_O3_5007 / Iz16_O2_3727)
ax.plot(Iz16_R23, Iz16_O32,\
				ls="none",mec=tableau20("Purple"),mfc="none",\
                marker="d",ms=3,mew=0.5,zorder=98)


# EMPRESS # https://ui.adsabs.harvard.edu/abs/2023ApJ...952...11N/abstract
EMPRESS_O3_5007 = np.array([483.2,478.0,374.2,514.6,607.3,442.6,308.3,554.5,470.3,387.9])
EMPRESS_O3_4959 = np.array([165.3,158.8,124.4,184.1,204.3,149.3,102.1,184.3,176.0,130.4])
EMPRESS_O2_3727 = np.array([30.4,74.1,73.6,64.0,61.5,69.8,91.3,27.7,13.6,14.5])
EMPRESS_O2_3729 = np.array([43.2,109.2,89.8,90.0,85.7,96.1,133.8,38.9,18.2,20.7])
EMPRESS_Hb = np.repeat(100.,len(EMPRESS_O2_3729))

EMPRESS_O32 = np.log10(EMPRESS_O3_5007/(EMPRESS_O2_3727 + EMPRESS_O2_3729))
EMPRESS_R23 = np.log10((EMPRESS_O3_5007 + EMPRESS_O3_4959 + (EMPRESS_O2_3727 + EMPRESS_O2_3729))/EMPRESS_Hb)

ax.plot(EMPRESS_R23,EMPRESS_O32,\
				ls="none",mfc=tableau20("Red"),mec="black",\
                marker="p",ms=4,mew=0.5,zorder=98)


# EELG (Amorin et al. 2015) -- # https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.105A/abstract
Am15 = fits.open("../../../Main Catalogs/Amorin2015_EELGs/Amorin2015_EELGs.fits")[1].data

Am15 = Am15[ (Am15["F_Hb_"] > 0.) & (Am15["F_5007_"] > 0.) & (Am15["F_4959_"] > 0.) & (Am15["F_3727_"] > 0.) ]
ax.plot(np.log10((Am15["F_5007_"] + Am15["F_4959_"] + Am15["F_3727_"])/Am15["F_Hb_"]),np.log10(Am15["F_5007_"]/Am15["F_3727_"]),\
				ls="none",mec=tableau20("Blue"),mfc="none",\
                marker="o",ms=3,mew=0.5,zorder=97)


# Sanders et al. (2023) -- # https://iopscience.iop.org/article/10.3847/1538-4357/acedad/pdf
Sa23_O32 = np.array([0.40,0.49,0.52,0.94])
Sa23_O32_elow = np.array([0.04,0.04,0.01,0.09])
Sa23_O32_eupp = np.array([0.02,0.03,0.07,0.10])
Sa23_R32 = np.array([0.91,0.98,0.95,0.97])
Sa23_R32_elow = np.array([0.01,0.01,0.02,0.02])
Sa23_R32_eupp = np.array([0.03,0.04,0.01,0.03])

ax.errorbar(Sa23_R32[-1:],Sa23_O32[-1:],xerr=(Sa23_R32_elow[-1:],Sa23_R32_eupp[-1:]),yerr=(Sa23_O32_elow[-1:],Sa23_O32_eupp[-1:]),\
			ls="none",mec=tableau20("Orange"),mfc=tableau20("Light Orange"),ecolor=tableau20("Grey"),\
                marker="s",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)


# Tang et al. (2023) -- # https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1657T/abstract
Ta23_O32 = np.array([17.84])
Ta23_O32_elow = np.array([1.71])/(np.log(10.)*Ta23_O32)
Ta23_O32_eupp = np.array([1.71])/(np.log(10.)*Ta23_O32)
Ta23_R32 = np.array([9.42])
Ta23_R32_elow = np.array([0.36])/(np.log(10.)*Ta23_R32)
Ta23_R32_eupp = np.array([0.36])/(np.log(10.)*Ta23_R32)

ax.errorbar(np.log10(Ta23_R32),np.log10(Ta23_O32),xerr=(Ta23_R32_elow,Ta23_R32_eupp),yerr=(Ta23_O32_elow,Ta23_O32_eupp),\
			ls="none",mec=tableau20("Pink"),mfc=tableau20("Light Pink"),ecolor=tableau20("Grey"),\
                marker="p",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)

# JADES (Cameron et al. 2023) -- # https://www.aanda.org/articles/aa/pdf/2023/09/aa46107-23.pdf
Ca23_O32 = np.array([1.21]); Ca23_O32_err = np.array([0.16])
Ca23_R23 = np.array([0.86]); Ca23_R23_err = np.array([0.02])

ax.errorbar(Ca23_R23,Ca23_O32,xerr=(Ca23_R23_err),yerr=(Ca23_O32_err),\
			ls="none",mec=tableau20("Red"),mfc=tableau20("Light Red"),ecolor=tableau20("Grey"),\
                marker="o",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)

Ca23_O32 = np.array([0.97]); Ca23_O32_err = np.array([0.13])
Ca23_R23 = np.array([0.86]); Ca23_R23_err = np.array([0.04])

ax.errorbar(Ca23_R23,Ca23_O32,xerr=(Ca23_R23_err),yerr=(Ca23_O32_err),\
			ls="none",mec=tableau20("Brown"),mfc=tableau20("Light Brown"),ecolor=tableau20("Grey"),\
                marker="o",ms=5,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)




handles = [	Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}"),
			Line2D([],[],ls="none",marker="p",ms=4,mfc=tableau20("Red"),mec="black",label=r"EMPGs (Ni+23; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec="#464196",mfc="none",label=r"Blueberries (Ya+17; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="^",ms=4,mec=tableau20("Green"),mfc="none",label=r"GPs (Ya+17; $z \sim 0.2$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Purple"),mfc="none",label=r"LyC Leakers (Iz+16,18; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Blue"),mfc="none",label=r"EELGs (Am+15; $z \sim 0.1 - 0.9$)"),
			Line2D([],[],ls="none",marker="<",ms=4,mec=tableau20("Light Brown"),mfc="none",label=r"MOSDEF (Kr+15; $z \sim 1.2 - 2.7$)"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"CEERS (Sa+23; $z \sim 7.5$)"),
			Line2D([],[],ls="none",marker="p",ms=4,mec=tableau20("Pink"),mfc=tableau20("Light Pink"),label=r"CEERS (Ta+23; $z \sim 7.7$)"),
			Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"JADES (Ca+23; $z \sim 6$)"),
			Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Brown"),mfc=tableau20("Light Brown"),label=r"JADES (Ca+23; $z \sim 8$)"),
			Line2D([],[],ls="none",marker="o",ms=4,mec="none",mfc=tableau20("Light Grey"),alpha=0.5,label=r"SDSS")
			]		

leg = ax.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=4,columnspacing=0.075)
plt.gca().add_artist(leg)	


plt.savefig("../../plots/O32_R23.png",format="png",dpi=300,bbox_inches="tight")

