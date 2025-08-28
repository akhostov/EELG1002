import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from astropy.io import fits
import numpy as np 
import pickle
import h5py

import sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
sys.path.insert(0, "..")
from colors import tableau20
import util

######################################################################
#######			ALL LITERATURE MEASUREMENTS AND PLOTING		   #######
######################################################################

def LyC_Leakers_lowz(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):
    
	# Load in Izotov et al. (2016)
	Iz16_hb, Iz16_hb_err, Iz16_o3_5007, Iz16_o3_5007_err, Iz16_nii, Iz16_nii_err, Iz16_halpha, Iz16_halpha_err, Iz16_mass = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2016.txt",unpack=True,skiprows=5,usecols=(8,9,14,15,20,21,17,18,28))

	# Load in Izotov et al. (2018)
	Iz18_hb, Iz18_hb_err, Iz18_o3_5007, Iz18_o3_5007_err, Iz18_nii, Iz18_nii_err, Iz18_halpha, Iz18_halpha_err, Iz18_mass = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2018.txt",unpack=True,skiprows=5,usecols=(8,9,14,15,20,21,17,18,28))


	# Izotov et al. (2016) & (2018) assume Salpeter IMF. Need to convert to Chabrier IMF
	Iz16_mass = Iz16_mass - 0.25
	Iz18_mass = Iz18_mass - 0.25

	# Combine all the data
	hbeta = np.concatenate([Iz16_hb,Iz18_hb])
	hbeta_err = np.concatenate([Iz16_hb_err,Iz18_hb_err])	

	o3_5007 = np.concatenate([Iz16_o3_5007,Iz18_o3_5007])
	o3_5007_err = np.concatenate([Iz16_o3_5007_err,Iz18_o3_5007_err])

	nii = np.concatenate([Iz16_nii,Iz18_nii])
	nii_err = np.concatenate([Iz16_nii_err,Iz18_nii_err])

	halpha = np.concatenate([Iz16_halpha,Iz18_halpha])
	halpha_err = np.concatenate([Iz16_halpha_err,Iz18_halpha_err])

	mass = np.concatenate([Iz16_mass,Iz18_mass])

	o3hb = o3_5007/hbeta
	o3hb_err = o3hb*np.sqrt( (o3_5007_err/o3_5007)**2. + (hbeta_err/hbeta)**2. )
	o3hb_err = o3hb_err/(np.log(10.)*o3hb)
	o3hb = np.log10(o3hb)

	if BPT:

		n2ha = nii/halpha
		n2ha_err = n2ha*np.sqrt( (nii_err/nii)**2. + (halpha_err/halpha)**2. )
		n2ha_err = n2ha_err/(np.log(10.)*n2ha)
		n2ha = np.log10(n2ha)

		ax.errorbar(n2ha,o3hb,xerr=n2ha_err,yerr=o3hb_err,
						ls="none",mec=mec,mfc=mfc,ecolor=mec,
						marker=marker,ms=ms,mew=0.5,zorder=zorder,
						capsize=2,capthick=1,elinewidth=0.5,alpha=alpha)
		return ax

	elif MEx:
		ax.errorbar(mass,o3hb,yerr=o3hb_err,
						ls="none",mec=mec,mfc=mfc,ecolor=mec,
						marker=marker,ms=ms,mew=0.5,zorder=zorder,
						capsize=2,capthick=1,elinewidth=0.5,alpha=alpha)

		return ax

	else:
		raise ValueError("Must specify either BPT or MEx")


def Blue_Berries(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

	Ya17 = fits.open("../../data/literature_measurements/Yang2017_Blueberries.fits")[1].data
	Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["Hbeta"] > 0.) ]

	Ya17["mass"] = Ya17["mass"]-np.log10(1.64) # Conversion to Chabrier IMF

	if BPT:
		return ax
	
	elif MEx:
		ax.plot(Ya17["mass"],np.log10(Ya17["[OIII]5007"]/Ya17["Hbeta"]),\
						ls="none",mec=mec,mfc=mfc,\
						marker=marker,ms=ms,mew=0.5,zorder=zorder,alpha=alpha)
		return ax
	
	else:
		raise ValueError("Must specify either BPT or MEx")


def Green_Peas(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

	Ya17 = fits.open("../../data/literature_measurements/Yang2017_GreenPeas.fits")[1].data
	Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["Hbeta"] > 0.) ]
	Ya17["logM"] = Ya17["logM"]-np.log10(1.64) # Conversion to Chabrier IMF

	o3hb = Ya17["[OIII]5007"]/Ya17["Hbeta"]
	o3hb_err = o3hb*np.sqrt( (Ya17["e_[OIII]5007"]/Ya17["[OIII]5007"])**2. + (Ya17["e_Hbeta"]/Ya17["Hbeta"])**2. )
	o3hb_err = o3hb_err/(np.log(10.)*o3hb)
	o3hb = np.log10(o3hb)

	if BPT:
		return ax
	
	elif MEx:
		ax.errorbar(Ya17["logM"],o3hb,yerr=o3hb_err,
						ls="none",mec=mec,mfc=mfc,ecolor=mec,
						marker=marker,ms=ms,mew=0.5,zorder=zorder,
						capsize=2,capthick=1,elinewidth=0.5,alpha=alpha)
		return ax
	else:
		raise ValueError("Must specify either BPT or MEx")
	

def zCOSMOS_EELGs(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

	Am15 = fits.open("../../data/literature_measurements/Amorin2015_EELGs.fits")[1].data
	Am15 = Am15[ (Am15["F_Hb_"] > 0.) & (Am15["F_5007_"] > 0.) ]

	if BPT:
		Am15 = Am15[ (Am15["F_Ha_"] > 0.) & (Am15["F_6584_"] > 0.) ]
		ax.plot(np.log10(Am15["F_6584_"]/Am15["F_Ha_"]),np.log10(Am15["F_5007_"]/Am15["F_Hb_"]),\
						ls="none",mec=mec,mfc=mfc,\
						marker=marker,ms=ms,mew=0.5,zorder=zorder,alpha=alpha)
		return ax
	
	elif MEx:
		ax.plot(Am15["logMs"],np.log10(Am15["F_5007_"]/Am15["F_Hb_"]),\
						ls="none",mec=mec,mfc=mfc,\
						marker=marker,ms=ms,mew=0.5,zorder=zorder,alpha=alpha)
		return ax
	else:
		raise ValueError("Must specify either BPT or MEx")


def EMPRESS(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

	hbeta, o3_5007, halpha, nii = np.loadtxt("../../data/literature_measurements/Nishigaki_et_al_2023.txt",unpack=True,usecols=(11,15,17,19))

	if BPT:
		ax.plot(np.log10(nii/halpha),np.log10(o3_5007/hbeta),\
					ls="none",mfc=mfc,mec=mec,\
					marker=marker,ms=ms,mew=0.5,zorder=zorder,alpha=alpha)
		return ax
	elif MEx:
		return ax
	else:
		raise ValueError("Must specify either BPT or MEx")

def FMOS(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):
	FMOS = fits.open("../../data/literature_measurements/FMOS_COSMOS_DR2.fits")[1].data

	if BPT:
		FMOS = FMOS[ (FMOS["FLUX_HBETA"] > 0.) & (FMOS["FLUX_OIII5007"] > 0.) & (FMOS["FLUX_HALPHA"] > 0.) & (FMOS["FLUX_NII6584"] > 0.) ]

		ax.plot(np.log10(FMOS["FLUX_NII6584"]/FMOS["FLUX_HALPHA"]),np.log10(FMOS["FLUX_OIII5007"]/FMOS["FLUX_HBETA"]),\
						ls="none",mec=mec,mfc=mfc,\
						marker=marker,ms=ms,mew=0.5,zorder=zorder,alpha=alpha)
		return ax
	
	elif MEx:
		return ax
	else:
		raise ValueError("Must specify either BPT or MEx")

def MOSDEF(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

	MOSDEF = fits.open("../../data/literature_measurements/MOSDEF_linemeas_cor.fits")[1].data

	if BPT:
		MOSDEF = MOSDEF[ (MOSDEF["HB4863_FLUX"] > 0.) & (MOSDEF["OIII5008_FLUX"] > 0.) & (MOSDEF["HA6565_FLUX"] > 0.) & (MOSDEF["NII6585_FLUX"] > 0.) & \
							(MOSDEF["HB4863_aeflag"] != 1) & (MOSDEF["OIII5008_aeflag"] !=1.) & (MOSDEF["HA6565_aeflag"] !=1.) & (MOSDEF["NII6585_aeflag"] !=1.) & \
							(MOSDEF["HB4863_slflag"] < 0.2) & (MOSDEF["OIII5008_slflag"] < 0.2) & (MOSDEF["HA6565_slflag"] < 0.2) & (MOSDEF["NII6585_slflag"] < 0.2)]
		
		ax.plot(np.log10(MOSDEF["NII6585_FLUX"]/MOSDEF["HA6565_FLUX"]),np.log10(MOSDEF["OIII5008_FLUX"]/MOSDEF["HB4863_FLUX"]),\
						ls="none",mec=mec,mfc=mfc,\
						marker=marker,ms=ms,mew=0.5,zorder=zorder,alpha=alpha)
		
		return ax
	
	elif MEx:
		return ax
	else:
		raise ValueError("Must specify either BPT or MEx")


def JADES(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

	N2,N2_err,N2_uplims,R3,R3_err = np.loadtxt("../../data/literature_measurements/Cameron_et_al_2023.txt",unpack=True,usecols=(2,3,4,11,12),dtype=str)
	N2 = np.double(N2); N2_err = np.double(N2_err); N2_uplims = np.where(N2_uplims == "True", True, False)
	R3 = np.double(R3); R3_err = np.double(R3_err)

	# Add a limit for arrows in uplims
	N2_err[N2_uplims] = 0.15

	# Keep only those that have N2 Detection
	ind = N2 != -99.

	if BPT:
		_,caps,_ = ax.errorbar(N2[ind],R3[ind],xerr=N2_err[ind],yerr=R3_err[ind],xuplims=N2_uplims[ind],
			  		ls="none",mec=mec,mfc=mfc,ecolor=mec,
					marker=marker,ms=ms,mew=0.5,zorder=zorder,
					capsize=2,capthick=1,elinewidth=0.5,alpha=alpha)
		
		caps[0].set_alpha(0.5)
		caps[0].set_markersize(ms)
		return ax

	elif MEx:
		return ax
	else:
		raise ValueError("Must specify either BPT or MEx")

######################################################################
#######			Load all Associated EELG1002 Data			   #######
######################################################################

#### OUR SOURCE
# Load in the Line properties
ism_props = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1]
ism_pdf = pickle.load(open("../../data/emline_fits/1002_pyneb_pdf.pkl","rb"))
line_ratios = pickle.load(open("../../data/emline_fits/1002_line_ratios.pkl","rb"))

# Extract the [OIII]/Hbeta Ratio
ind = line_ratios["name"].index("R3")
o3hb,o3hb_elow,o3hb_eupp = line_ratios["median"][ind], line_ratios["low_1sigma"][ind], line_ratios["upp_1sigma"][ind]
o3hb_elow = o3hb_elow/(np.log(10.)*o3hb)
o3hb_eupp = o3hb_eupp/(np.log(10.)*o3hb)
o3hb = np.log10(o3hb)

# Infer NII/HA from this calibration (Equation 4 of https://www.aanda.org/articles/aa/pdf/2013/11/aa21956-13.pdf)
#import pdb; pdb.set_trace()
#pdf_n2ha = np.concatenate(((ism_pdf["12+log10(O/H)"] - 9.07)/0.79,(ism_pdf["12+log10(O/H)"] - 8.743)/0.462,(ism_pdf["12+log10(O/H)"] - 8.90)/0.57))

# Infer NII/HA from Maiolino (2008) Calibration (https://ui.adsabs.harvard.edu/abs/2008A%26A...488..463M/abstract)
coeffs = np.array( [-0.7732, 1.2357, -0.2811, -0.7201, -0.3330 ])
pdf_n2ha = coeffs[0] + coeffs[1]*(ism_pdf["12+log10(O/H)"] - 8.69) + \
						coeffs[2]*(ism_pdf["12+log10(O/H)"] - 8.69)**2 + \
						coeffs[3]*(ism_pdf["12+log10(O/H)"] - 8.69)**3 + \
						coeffs[4]*(ism_pdf["12+log10(O/H)"] - 8.69)**4

n2ha_Ma08,n2ha_Ma08_elow,n2ha_Ma08_eupp = util.stats(pdf_n2ha)
n2ha_Ma08_3sigma = util.stats(pdf_n2ha,upper_limit=99.73)


# Infer NII/HA from Pettini & Pagel (2004) Calibration (https://ui.adsabs.harvard.edu/abs/2004MNRAS.348L..59P/abstract)
coeffs = np.array( [-0.7732, 1.2357, -0.2811, -0.7201, -0.3330 ])
pdf_n2ha = (ism_pdf["12+log10(O/H)"] - 8.90)/0.57

n2ha_PP04,n2ha_PP04_elow,n2ha_PP04_eupp = util.stats(pdf_n2ha)
n2ha_PP04_3sigma = util.stats(pdf_n2ha,upper_limit=99.99)


# Cigale
cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
bagpipes = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")


# Stellar Mass Measurements
cigale_mass = cigale["bayes.stellar.m_star"]
cigale_mass_err = cigale["bayes.stellar.m_star_err"]/(np.log(10.)*cigale_mass)
cigale_mass = np.log10(cigale_mass)
bagpipes_mass = bagpipes["median"][10]
bagpipes_mass_err_low,bagpipes_mass_err_up = bagpipes_mass - bagpipes["conf_int"][0][10], bagpipes["conf_int"][1][10] - bagpipes_mass








######################################################################
#######						Initialize Figure				   #######
######################################################################

fig = plt.figure()
fig.set_size_inches(3,3)
fig.subplots_adjust(wspace=0.15)




"""

######################################################################
#######						BPT Diagram 					   #######
######################################################################

# Initialize Axis
ax_BPT = fig.add_subplot(121)

# Define Labels
ax_BPT.set_xlabel(r"$\log_{10}$ [N{\sc ii}]/H$\alpha$",fontsize=8)
ax_BPT.set_ylabel(r"$\log_{10}$ [O{\sc iii}]5007\AA/H$\beta$",fontsize=8)

# Define Limits
ax_BPT.set_xlim(-3.3,0.5)
ax_BPT.set_ylim(-0.5,1.25)



#### OUR SOURCE
ax_BPT.errorbar([np.max([n2ha_Ma08_3sigma,n2ha_PP04_3sigma])],[o3hb],\
				xerr=([0.5],[0.0]),yerr=([o3hb_elow],[o3hb_eupp]),\
				xuplims=([True]),\
				ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
               marker="none",ms=3,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

#ax_BPT.fill_between([-3.0,np.max([n2ha_Ma08_3sigma,n2ha_PP04_3sigma])],o3hb-o3hb_elow,o3hb+o3hb_eupp,color=tableau20("Blue"),alpha=0.5,zorder=98)

#ax_BPT.fill_betweenx([o3hb],-3.0,np.max([n2ha_Ma08_3sigma,n2ha_PP04_3sigma]),color=tableau20("Blue"),alpha=0.2,zorder=98)

#### Literature Measurements
# Izotov et al. (2016,2018)
ax_BPT = LyC_Leakers_lowz(ax_BPT,BPT=True,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),marker="s",ms=3,zorder=98,alpha=0.5)

# EMPRESS # https://ui.adsabs.harvard.edu/abs/2023ApJ...952...11N/abstract
ax_BPT = EMPRESS(ax_BPT,BPT=True,mec="black",mfc=tableau20("Red"),marker="p",ms=5,zorder=98,alpha=0.5)

# EELG (Amorin et al. 2015) -- # https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.105A/abstract
ax_BPT = zCOSMOS_EELGs(ax_BPT,BPT=True,mec=tableau20("Orange"),mfc="none",marker="o",ms=3,zorder=97,alpha=0.5)

# FMOS (Kashino et al. 2015)
ax_BPT = FMOS(ax_BPT,BPT=True,mec=tableau20("Green"),mfc="none",marker="d",ms=2,zorder=96,alpha=0.5)

# MOSDEF
ax_BPT = MOSDEF(ax_BPT,BPT=True,mec="slategrey",mfc="none",marker="^",ms=2,zorder=96,alpha=0.5)


## JADES (Cameron et al. 2023) # https://www.aanda.org/articles/aa/pdf/2023/09/aa46107-23.pdf
ax_BPT = JADES(ax_BPT,BPT=True,mec=tableau20("Grey"),mfc=tableau20("Light Grey"),marker="o",ms=3,zorder=99,alpha=0.5)



### BPT SELECTION LINES
# Kewley et al. (2001)
BPT_n2ha = np.arange(-4.,0.47,0.01)
BPT_o3hb = 0.61/(BPT_n2ha - 0.47) + 1.19
ax_BPT.plot(BPT_n2ha,BPT_o3hb,ls="--",color="black")

# Kauffman et al. (2003)
BPT_n2ha = np.arange(-4.,0.05,0.01)
BPT_o3hb = 0.61/(BPT_n2ha - 0.05) + 1.3
ax_BPT.plot(BPT_n2ha,BPT_o3hb,ls=":",color="red")


# Add Classification Labels
ax_BPT.text(-2.8,0.25,r"\textit{BPT-SFG}",color=tableau20("Blue"),fontsize=6,ha="left",va="center")
ax_BPT.text(0.,1.0,r"\textit{BPT-AGN}",color=tableau20("Red"),fontsize=6,ha="center",va="center")


# Design the Legend
handles = [Line2D([],[],ls="none",marker="|",ms=4,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work; 5$\sigma$ limit)}}")]	

leg1 = ax_BPT.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg1)


handles = [	Line2D([],[],ls="none",marker="p",ms=4,alpha=0.5,mfc=tableau20("Red"),mec="black",label=r"EMPGs (Ni+23; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="s",ms=4,alpha=0.5,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),label=r"LyC Leakers (Iz+16,18; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="o",ms=4,alpha=0.5,mec=tableau20("Orange"),mfc="none",label=r"EELGs (Am+15; $z \sim 0.1 - 0.9$)"),
			Line2D([],[],ls="none",marker="d",ms=4,alpha=0.5,mec=tableau20("Green"),mfc="none",label=r"FMOS (Ka+19; $z \sim 1.6$)"),
			Line2D([],[],ls="none",marker="^",ms=4,alpha=0.5,mec="slategrey",mfc="none",label=r"MOSDEF (Kr+15; $z \sim 1.2 - 2.7$)"),
			Line2D([],[],ls="none",marker="o",ms=4,alpha=0.5,mec=tableau20("Grey"),mfc=tableau20("Light Grey"),label=r"JADES (Ca+23; $z \sim 5.5 - 9$)")]		

leg2 = ax_BPT.legend(handles=handles,loc="lower left",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg2)		


handles = [	Line2D([],[],ls="--",lw=1,color="black",label=r"Ke+01"),
			Line2D([],[],ls=":",lw=1,color="red",label=r"Ka+03")]		

leg3 = ax_BPT.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg3)	
"""











######################################################################
#######						MEx Diagram 					   #######
######################################################################

# Initialize the Axis
ax_MEx = fig.add_subplot(111)

# Define Labels
ax_MEx.set_xlabel(r"$\log_{10}$ Stellar Mass (M$_\odot$)",fontsize=8)


# Define Limits
ax_MEx.set_ylim(0.2,1.)
ax_MEx.set_xlim(6.5,10)

# Plot the Juneau et al. (2011) cuts
xx = np.arange(6.,12.,0.01)
yy_1 = 0.37/(xx - 10.5)+1.
yy_2 = 800.492 - 217.328*xx + 19.6431*xx**2. - 0.591349*xx**3.
yy_3 = 594.753 - 167.074*xx + 15.6748*xx**2. - 0.491215*xx**3.

#Lower Bound
yy_all_1 = np.concatenate((yy_1[xx<=9.9],yy_2[(xx>9.9) & (xx<11.2)]))

#Upper Bound
yy_all_2 = yy_3[(xx>9.9)]

# Plot the MEx selection regions
plt.plot(xx[(xx<11.2)],yy_all_1,color="black",linestyle="--")
plt.plot(xx[(xx>9.9)],yy_all_2,color="black",linestyle="--")

# Fill in Regions
lower_sel = np.concatenate((yy_all_1,yy_3[xx>11.2]))
upper_sel = np.concatenate((yy_1[xx<9.9],yy_all_2))

ax_MEx.fill_between(xx,-3.,lower_sel,color=tableau20("Light Blue"),alpha=0.15)
ax_MEx.fill_between(xx[(xx>9.9) & (xx<11.2)],yy_2[(xx>9.9) & (xx<11.2)],yy_3[(xx>9.9) & (xx<11.2)],color=tableau20("Light Green"),alpha=0.15)
ax_MEx.fill_between(xx,upper_sel,11.,color=tableau20("Light Red"),alpha=0.15)


ax_MEx.errorbar([cigale_mass],[o3hb],xerr=([cigale_mass_err]),yerr=([o3hb_elow],[o3hb_eupp]),ls="none",\
				mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),ecolor=util.color_scheme("Cigale",mec=True),\
                marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=98)

ax_MEx.errorbar([bagpipes_mass],[o3hb],xerr=([bagpipes_mass_err_low],[bagpipes_mass_err_up]),yerr=([o3hb_elow],[o3hb_eupp]),ls="none",\
                mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),ecolor=util.color_scheme("Bagpipes",mec=True),\
                marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

#### Low-z Analogs
# LyC Leakers at z ~ 0
ax_MEx = LyC_Leakers_lowz(ax_MEx,MEx=True,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),marker="s",ms=3,zorder=97)

# Blueberries (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18470038Y/abstract
ax_MEx = Blue_Berries(ax_MEx,MEx=True,mec="#464196",mfc="none",marker="v",ms=3,zorder=97)

# Green Peas (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18440171Y/abstract
ax_MEx = Green_Peas(ax_MEx,MEx=True,mec=tableau20("Green"),mfc="none",marker="p",ms=3,zorder=97)

# EELG (Amorin et al. 2015) -- # https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.105A/abstract
ax_MEx = zCOSMOS_EELGs(ax_MEx,MEx=True,mec=tableau20("Orange"),mfc="none",marker="o",ms=3,zorder=97)




ax_MEx.text(9,0.9,r"\textit{MEx-AGN}",color=tableau20("Red"),fontsize=6,ha="left",va="center")
ax_MEx.text(7.2,0.5,r"\textit{MEx-SF}",color=tableau20("Blue"),fontsize=6,ha="left",va="center")




handles = [Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),label=r"\textbf{\textit{EELG1002 (Cigale)}}"),
            Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),label=r"\textbf{\textit{EELG1002 (Bagpipes)}}"),
			Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),label=r"LyC Leakers (Iz+16,18; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="v",ms=3,mec="#464196",mfc="none",label=r"Blueberries (Ya+17; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="p",ms=3,mec=tableau20("Green"),mfc="none",label=r"GPs (Ya+17; $z \sim 0.2$)"),
			Line2D([],[],ls="none",marker="o",ms=3,mec=tableau20("Orange"),mfc="none",label=r"EELGs (Am+15; $z \sim 0.1 - 0.9$)")]			

leg1 = ax_MEx.legend(handles=handles,loc="lower left",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=0.8)
plt.gca().add_artist(leg1)


handles = [	Line2D([],[],ls="--",lw=1,color="black",label=r"Ju+11")]		

leg3 = ax_MEx.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg3)	

plt.savefig("../../plots/MEx.png",format="png",dpi=300,bbox_inches="tight")


