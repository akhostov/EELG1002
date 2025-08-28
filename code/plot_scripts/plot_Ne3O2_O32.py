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

# Yang et al. (2017)
def Blueberries(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    # Load in data
    data = fits.open("../../data/literature_measurements/Yang2017_Blueberries.fits")[1].data

    # Filter out those with no Te
    data = data[ data["12+log(O/H)"] > 0.]

    # Measure Balmer Decrement
    EBV = 2.5/(util.calzetti(4861.) - util.calzetti(6563.)) * np.log10( (data["Halpha"]/data["Hbeta"]) / 2.86 )
    EBV[EBV < 0.] = 0.

    # Dust Corr
    o32_dustcorr = 0.4*EBV*( util.calzetti(5007.) - util.calzetti(3727.) )
    ne3o2_dustcorr = 0.4*EBV*( util.calzetti(3869.) - util.calzetti(3727.) )

    # Ratios
    o32 = data["[OIII]5007"]/data["[OII]3727"]; o32_err = o32 * np.sqrt( (data["e_[OIII]5007"]/data["[OIII]5007"])**2. + (data["e_[OII]3727"]/data["[OII]3727"])**2. )
    ne3o2 = data["[NeIII]3869"]/data["[OII]3727"]; ne3o2_err = ne3o2 * np.sqrt( (data["e_[NeIII]3869"]/data["[NeIII]3869"])**2. + (data["e_[OII]3727"]/data["[OII]3727"])**2. )

    # Convet to Log-Scale
    o32_err = o32_err/(np.log(10.)*o32)
    ne3o2_err = ne3o2_err/(np.log(10.)*ne3o2) 

    # Apply Dust Corr
    o32 = np.log10(o32) + o32_dustcorr
    ne3o2 = np.log10(ne3o2) + ne3o2_dustcorr


    ax.errorbar(o32,ne3o2,xerr=o32_err,yerr=ne3o2_err,ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=1,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)

    return ax

# Izotov et al. (2016, 2018), Jaskot & Oey (2013)
def LyC_Leakers(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    # Load in Izotov et al. (2016)
    Iz16_O2, Iz16_O2_err, Iz16_Ne3, Iz16_Ne3_err, \
        Iz16_hb, Iz16_hb_err, Iz16_o3_5007, Iz16_o3_5007_err,\
                Iz16_halpha, Iz16_halpha_err = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2016.txt",unpack=True,skiprows=5,usecols=(2,3,5,6,8,9,14,15,17,18))

    # Load in Izotov et al. (2018)
    Iz18_O2, Iz18_O2_err, Iz18_Ne3, Iz18_Ne3_err, \
        Iz18_hb, Iz18_hb_err, Iz18_o3_5007, Iz18_o3_5007_err, \
            Iz18_halpha, Iz18_halpha_err = np.loadtxt("../../data/literature_measurements/Izotov_et_al_2018.txt",unpack=True,skiprows=5,usecols=(2,3,5,6,8,9,14,15,17,18))

    # Combine the data
    hb = np.concatenate((Iz16_hb,Iz18_hb))
    ha = np.concatenate((Iz16_halpha,Iz18_halpha))
    o3 = np.concatenate((Iz16_o3_5007,Iz18_o3_5007))
    o2 = np.concatenate((Iz16_O2,Iz18_O2))
    ne3 = np.concatenate((Iz16_Ne3,Iz18_Ne3))

    # Measure Balmer Decrement
    EBV = 2.5/(util.calzetti(4861.) - util.calzetti(6563.)) * np.log10( (ha/hb) / 2.86 )
    EBV[EBV < 0.] = 0.

    o32 = np.log10(o3/o2*pow(10,0.4*EBV*( util.calzetti(5007.) - util.calzetti(3727.) )))
    ne3o2 = np.log10(ne3/o2*pow(10,0.4*EBV*( util.calzetti(3869.) - util.calzetti(3727.) )))

    ax.plot(o32,ne3o2,ls="none",marker=marker,ms=ms,mew=0.5,mfc=mfc,mec=mec,zorder=zorder)


    # Load in Jaskot & Oey (2013) -- Already Corrected
    Jaskot_O2, Jaskot_O2_err, Jaskot_Ne3, Jaskot_Ne3_err, \
            Jaskot_o3_5007, Jaskot_o3_5007_err = np.loadtxt("../../data/literature_measurements/Jaskot_Oey_2013.txt",unpack=True,skiprows=5,usecols=(2,3,4,5,-6,-5))


    ax.plot(np.log10(Jaskot_o3_5007/Jaskot_O2),np.log10(Jaskot_Ne3/Jaskot_O2),ls="none",marker=marker,ms=ms,mew=0.5,mfc=mfc,mec=mec,zorder=zorder,alpha=alpha)


    return ax


# MOSDEF (Jeong et al. 2020)
def MOSDEF(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    # Load in data
    ne3o2,ne3o2_elow,ne3o2_eupp,o32,o32_elow,o32_eupp = np.loadtxt("../../data/literature_measurements/Jeong_et_al_2020.txt",unpack=True,skiprows=5,usecols=(4,5,6,13,14,15))

    ax.errorbar(o32,ne3o2,
                    xerr=(o32_elow,o32_eupp),
                    yerr=(ne3o2_elow,ne3o2_eupp),
                    ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=2,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)


    return ax

# Vanzella et al. (2020)
def Ion2(ax,mfc,mec,marker,ms,zorder=9,alpha=1):

    # Load in data
    ne3 = np.array([1.8]); ne3_err = ne3/4.0
    o2 = np.array([3.5]); o2_err = o2/6.0
    o3 = np.array([24.8]); o3_err = o3/65.0

    # Measure the Ratio and Associated Errors
    ne3o2 = ne3/o2; ne3o2_err = ne3o2*np.sqrt( (ne3_err/ne3)**2. + (o2_err/o2)**2.) 
    o32 = o3/o2; o32_err = o32*np.sqrt( (o3_err/o3)**2. + (o2_err/o2)**2.)

    # Convert to Log Scale
    ne3o2_err = ne3o2_err/(np.log(10.)*ne3o2)
    o32_err = o32_err/(np.log(10.)*o32)
    ne3o2 = np.log10(ne3o2)
    o32 = np.log10(o32)

    # Plot
    ax.errorbar(o32,ne3o2,xerr=o32_err,yerr=ne3o2_err,ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=2,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)

    return ax

# Strom et al. (2017)
def KBSS_MOSFIRE(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    # Load in data
    o32,ne3o2 = np.loadtxt("../../data/literature_measurements/Strom_et_al_2017.txt",unpack=True,skiprows=5)

    ax.plot(o32,ne3o2,ls="none",marker=marker,ms=ms,mew=0.5,mfc=mfc,mec=mec,zorder=zorder,alpha=alpha)

    return ax

# Papovich et al. (2022); 1.1 < z < 2.3
def CLEAR(ax,mfc,mec,marker,ms,zorder=98,alpha=1):

    # Define the ratios
    o32 = np.array([0.35,-0.03]); o32_err = np.array([0.11,0.11])
    ne3o2 = np.array([-0.96,-1.19]); ne3o2_err = np.array([0.11,0.11])

    # Plot
    ax.errorbar(o32,ne3o2,xerr=o32_err,yerr=ne3o2_err,ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=1,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)

    return ax

def JADES(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

    O32,O32_err,O32_lolims,Ne3O2,Ne3O2_err,Ne3O2_uplim = np.loadtxt("../../data/literature_measurements/Cameron_et_al_2023.txt",unpack=True,usecols=(-6,-5,-4,-3,-2,-1),dtype=str)
    O32 = np.double(O32); O32_err = np.double(O32_err); O32_lolims = np.where(O32_lolims == "True", True, False)
    Ne3O2 = np.double(Ne3O2); Ne3O2_err = np.double(Ne3O2_err); Ne3O2_uplim = np.where(Ne3O2_uplim == "True", True, False)

    # Add a limit for arrows in uplims
    Ne3O2_err[Ne3O2_uplim] = 0.15
    O32_err[O32_lolims] = 0.15

    # Keep only those that have N2 Detection
    ind = (O32 != -99.) & (Ne3O2 != -99.)

    _,caps,_ = ax.errorbar(O32[ind],Ne3O2[ind],xerr=O32_err[ind],yerr=Ne3O2_err[ind],
                            xlolims=O32_lolims[ind],uplims=Ne3O2_uplim[ind],
                            ls="none",mec=mec,mfc=mfc,ecolor=mec,
                            marker=marker,ms=ms,mew=0.5,zorder=zorder,
                            capsize=1,capthick=1,elinewidth=0.5,alpha=alpha)

    caps[0].set_alpha(0.5)
    caps[0].set_markersize(2)

    return ax

# Tang et al. (2023)
def CEERS(ax,mfc,mec,marker,ms,BPT=False,MEx=False,zorder=98,alpha=1):

    O32,O32_err,O32_lolims,Ne3O2,Ne3O2_err,Ne3O2_uplim = np.loadtxt("../../data/literature_measurements/Tang_et_al_2023.txt",unpack=True,skiprows=21,usecols=(4,5,6,10,11,12),dtype=str)
    O32 = np.double(O32); O32_err = np.double(O32_err); O32_lolims = np.where(O32_lolims == "True", True, False)
    Ne3O2 = np.double(Ne3O2); Ne3O2_err = np.double(Ne3O2_err); Ne3O2_uplim = np.where(Ne3O2_uplim == "True", True, False)

    # Keep only those that have N2 Detection
    ind = (O32 != -99.) & (Ne3O2 != -99.)

    O32_err[ind] = O32_err[ind]/(np.log(10.)*O32[ind])
    Ne3O2_err[ind] = Ne3O2_err[ind]/(np.log(10.)*Ne3O2[ind])
    O32[ind] = np.log10(O32[ind])
    Ne3O2[ind] = np.log10(Ne3O2[ind])

    # Add a limit for arrows in uplims
    Ne3O2_err[Ne3O2_uplim] = 0.15
    O32_err[O32_lolims] = 0.15


    _,caps,_ = ax.errorbar(O32[ind],Ne3O2[ind],xerr=O32_err[ind],yerr=Ne3O2_err[ind],
                            xlolims=O32_lolims[ind],uplims=Ne3O2_uplim[ind],
                            ls="none",mec=mec,mfc=mfc,ecolor=mec,
                            marker=marker,ms=ms,mew=0.5,zorder=zorder,
                            capsize=1,capthick=1,elinewidth=0.5,alpha=1)

    caps[0].set_alpha(0.5)
    caps[0].set_markersize(2)

    return ax

# Pharo et al. (2023)
def HALO7D(ax,mfc,mec,marker,ms,zorder=98,alpha=1):
    # Load in data
    ne3o2,ne3o2_elow,ne3o2_eupp,\
        o32,o32_elow,o32_eupp = np.loadtxt("../../data/literature_measurements/Pharo_et_al_2023.txt",unpack=True,skiprows=5,usecols=(5,6,7,11,12,13))

    ax.errorbar(o32,ne3o2,xerr=(o32_elow,o32_eupp),yerr=(ne3o2_elow,ne3o2_eupp),ls="none",
                    marker=marker,ms=ms,mfc=mfc,mec=mec,ecolor=mec,
                    mew=0.5,capsize=1,capthick=1,elinewidth=0.5,zorder=zorder,alpha=alpha)
    
    return ax

def SDSS(ax,alpha=0.8):
    ## Plot SDSS Sources
    print ("Running SDSS")

    SDSS = fits.open("../../../Main Catalogs/SDSS/DR12/portsmouth_emlinekin_full-DR12-boss.fits")[1].data
    O3_5007 = SDSS["FLUX"].T[17]
    Ne3_3869 = SDSS["FLUX"].T[5]
    O2_3729 = SDSS["FLUX"].T[4]
    O2_3726 = SDSS["FLUX"].T[3]

    O3_5007_err = SDSS["FLUX_ERR"].T[17]
    Ne3_3869_err = SDSS["FLUX_ERR"].T[5]   
    O2_3729_err = SDSS["FLUX_ERR"].T[4]
    O2_3726_err = SDSS["FLUX_ERR"].T[3]


    O3_5007_FIT_WARNING = SDSS["FIT_WARNING"].T[17]
    Ne3_3869_FIT_WARNING = SDSS["FIT_WARNING"].T[5]     
    O2_3729_FIT_WARNING = SDSS["FIT_WARNING"].T[4]
    O2_3726_FIT_WARNING = SDSS["FIT_WARNING"].T[3]

    SDSS = SDSS[ (O3_5007 > 0.) & (Ne3_3869 > 0.) & (O2_3726 > 0.) & (O2_3729 > 0.) & \
                (O3_5007_err > 0.) & (Ne3_3869_err > 0.) & (O2_3726_err > 0.) & (O2_3729_err > 0.) & \
                (O3_5007_FIT_WARNING == 0.) & (Ne3_3869_FIT_WARNING == 0.) & (O2_3726_FIT_WARNING == 0.) & (O2_3729_FIT_WARNING == 0.)	& \
                (O3_5007/O3_5007_err > 3.) & (Ne3_3869/Ne3_3869_err > 3.) & (O2_3726/O2_3726_err > 3.) & (O2_3729/O2_3729_err > 3.) ]

    O3_5007 = SDSS["FLUX"].T[17]
    Ne3_3869 = SDSS["FLUX"].T[5]
    O2_3729 = SDSS["FLUX"].T[4]
    O2_3726 = SDSS["FLUX"].T[3]

    O3_5007_err = SDSS["FLUX_ERR"].T[17]
    Ne3_3869_err = SDSS["FLUX_ERR"].T[5]   
    O2_3729_err = SDSS["FLUX_ERR"].T[4]
    O2_3726_err = SDSS["FLUX_ERR"].T[3]

    SDSS_O2 = O2_3729 + O2_3726
    SDSS_O32 = np.log10(O3_5007/SDSS_O2)
    SDSS_Ne3O2 = np.log10(Ne3_3869/SDSS_O2)   
    ax.plot(SDSS_O32,SDSS_Ne3O2,				
                    ls="none",mfc=tableau20("Light Grey"),mec="none",\
                    marker="o",ms=1,zorder=1,alpha=alpha)
    print("Finished SDSS")

    return ax

######################################################################
#######						PREPARE THE FIGURE				   #######
######################################################################

fig = plt.figure()
fig.set_size_inches(3,3)
ax = fig.add_subplot(111)

# Define Labels
ax.set_ylabel(r"$\log_{10}$ Ne3O2",fontsize=8)
ax.set_xlabel(r"$\log_{10}$ O32",fontsize=8)


# Define Limits
ax.set_xlim(-1.0,2.0)
ax.set_ylim(-2.0,1.0)


#### OUR SOURCE
# Load in the Line properties
with open("../../data/emline_fits/1002_line_ratios.pkl","rb") as file: data = pickle.load(file)
file.close()

# Load in the SED properties (use CIGALE for now)
mask_ne3o2 = data["name"].index("Ne3O2")
mask_o32 = data["name"].index("O32")

ne3o2 = data["median"][mask_ne3o2]
ne3o2_elow = np.log10(ne3o2) - np.log10(ne3o2 - data["low_1sigma"][mask_ne3o2])
ne3o2_eupp = np.log10(data["upp_1sigma"][mask_ne3o2] + ne3o2) - np.log10(ne3o2)
ne3o2 = np.log10(ne3o2)

o32 = data["median"][mask_o32]
o32_elow = np.log10(o32) - np.log10(o32 - data["low_1sigma"][mask_o32])
o32_eupp = np.log10(data["upp_1sigma"][mask_o32] + o32) - np.log10(o32)
o32 = np.log10(o32)

ax.errorbar(o32,ne3o2,
				xerr=([o32_elow],[o32_eupp]),\
			    yerr=([ne3o2_elow],[ne3o2_eupp]),\
				ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
                marker="*",ms=12,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)


# Yang et al. (2017) -- Blueberries
ax = Blueberries(ax,mec="#464196",mfc="none",marker="p",ms=5,alpha=0.8,zorder=97)

# Local LyC Leakers
ax = LyC_Leakers(ax,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),marker="s",ms=3,alpha=0.8,zorder=98)

# MOSDEF (Jeong et al. 2020)
ax = MOSDEF(ax,mec=tableau20("Red"),mfc="none",marker="o",ms=3,zorder=97,alpha=0.8)

# Ion2 (Vanzella et al. 2020)
ax = Ion2(ax,mec=tableau20("Pink"),mfc=tableau20("Light Pink"),marker="d",ms=8,zorder=98,alpha=1.0)

# KBSS (Strom et al. 2017)
ax = KBSS_MOSFIRE(ax,mec="teal",mfc="none",marker="^",ms=3,zorder=97,alpha=0.8)

# CLEAR
ax = CLEAR(ax,mec=tableau20("Brown"),mfc=tableau20("Light Brown"),marker="v",ms=6,zorder=97,alpha=0.8)

# JADES (Cameron et al. 2023)
ax = JADES(ax,mec=tableau20("Red"),mfc=tableau20("Light Red"),marker="<",ms=3,zorder=97,alpha=0.8)

# CEERS (Tang et al. 2023)
ax = CEERS(ax,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),marker=">",ms=3,zorder=97,alpha=0.8)

# HALO7D (Pharo et al. 2023)
ax = HALO7D(ax,mec="black",mfc="none",marker="h",ms=6,zorder=97,alpha=0.8)

ax = SDSS(ax)


handles = [	Line2D([],[],ls="none",marker="*",ms=8,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}"),
           	Line2D([],[],ls="none",marker="s",ms=4,mec=tableau20("Purple"),mfc=tableau20("Light Purple"),label=r"LyC Leakers (Iz+16,18; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="v",ms=4,mec=tableau20("Brown"),mfc=tableau20("Light Brown"),label=r"CLEAR (Pa+23; $z \sim 1 - 2$)"),
			Line2D([],[],ls="none",marker="d",ms=4,mec=tableau20("Pink"),mfc=tableau20("Light Pink"),label=r"\textit{Ion2} (Va+20; $z \sim 3.2$)"),
			Line2D([],[],ls="none",marker="<",ms=4,mec=tableau20("Red"),mfc=tableau20("Light Red"),label=r"JADES (Ca+23; $z \sim 6$)"),
			Line2D([],[],ls="none",marker=">",ms=4,mec=tableau20("Orange"),mfc=tableau20("Light Orange"),label=r"CEERS (Ta+23; $z \sim 7.7$)") 
			]		

leg = ax.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=4,columnspacing=0.075,labelspacing=0.8)
plt.gca().add_artist(leg)	


handles = [ Line2D([],[],ls="none",marker="o",ms=4,mec="none",mfc=tableau20("Light Grey"),label=r"SDSS"),
			Line2D([],[],ls="none",marker="p",ms=4,mec="#464196",mfc="none",label=r"Blueberries (Ya+17; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="h",ms=4,mec="black",mfc="none",label=r"HALO7D (Ph+23; $z \sim 1$)"),
			Line2D([],[],ls="none",marker="o",ms=4,mec=tableau20("Red"),mfc="none",label=r"MOSDEF (Je+20; $z \sim 1.2 - 2.7$)"),
			Line2D([],[],ls="none",marker="^",ms=4,mec="teal",mfc="none",label=r"KBSS (St+17; $z \sim 2 - 3$)"),
			]		

leg = ax.legend(handles=handles,loc="lower right",frameon=False,ncol=1,numpoints=1,fontsize=4,columnspacing=0.075,labelspacing=0.8)
plt.gca().add_artist(leg)	





#########################################################################################################
#########                       OVERPLOT THE CLOUDY GRID SPECIFIC TO OUR SOURCE                 #########
#########################################################################################################
# Load in the Cloudy Grid
data = fits.open("/Users/aahsps/Downloads/bagpipes_nebular_line_grids.fits")

# ZMET
zmet_ind = np.arange(1,208,13,dtype=int)

# Line Index in Grid
ne3_ind = 85
o2_ind_blue = 69
o2_ind_red = 70
o3_ind = 76

# Sort through only the one that has ZMET = 0.05 ZSOL consistent with our source
cloudy_ne3o2,cloudy_o32,logU =  [],[],[]
for ii in np.arange(1+2,261+2,13,dtype=int):
    line_data = data[ii].data.T
    mod_ne3o2 = line_data[ne3_ind]/(line_data[o2_ind_blue]+line_data[o2_ind_red])
    mod_o32 = line_data[o3_ind]/(line_data[o2_ind_blue]+line_data[o2_ind_red])
    cloudy_ne3o2.append(mod_ne3o2)
    cloudy_o32.append(mod_o32)
    logU.append(float(data[ii].header["EXTNAME"].split("_")[-1]))

cloudy_ne3o2 = np.array(cloudy_ne3o2).T
cloudy_o32 = np.array(cloudy_o32).T
logU = np.asarray(logU)
4
# Plot only those between -3.5 < log10 < -1
these = (logU >= -3.5) & (logU <= -1.) 
ax.plot(np.log10(cloudy_o32[1][these]),np.log10(cloudy_ne3o2[1][these]),marker="o",ms=4,mfc=tableau20("Green"),mec="none",color=tableau20("Green"),ls="-",lw=0.5,zorder=99,alpha=1)


ax.text(0.6,0.88,r"\texttt{Cloudy}$+$\texttt{BPASS} (1 Myr Burst)",fontsize=4,ha="left",va="center",color=tableau20("Green"))
ax.text(0.6,0.78,r"0.05 $Z_{\odot}$; $n_e = 800$ cm$^{-3}$",fontsize=4,ha="left",va="center",color=tableau20("Green"))
ax.text(0.6,0.68,r"$-3.5 < \log_{10} U < -1.0$",fontsize=4,ha="left",va="center",color=tableau20("Green"))
ax.text(0.6,0.58,r"increments of 0.2",fontsize=4,ha="left",va="center",color=tableau20("Green"))

### Save Figure
plt.savefig("../../plots/Ne3O2_O32.png",format="png",dpi=300,bbox_inches="tight")

