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
import os

# Define Cosmology
cosmo = util.get_cosmology()


######################################################################
#######					  LOAD EELG1002 RESULTS			       #######
######################################################################

cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
cigale_mass = cigale["bayes.stellar.m_star"]/1e7
cigale_mass_err = cigale["bayes.stellar.m_star_err"]/1e7
cigale_SFR10Myrs = cigale["bayes.sfh.sfr10Myrs"]
cigale_SFR10Myrs_err = cigale["bayes.sfh.sfr10Myrs_err"]

bagpipes_sfh = fits.open("../../data/SED_results/bagpipes_results/best_fit_SFH_sfh_continuity_spec_BPASS.fits")[1].data
bagpipes_results = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")

# Stellar Mass
bagpipes_mass = pow(10,bagpipes_results["median"][10])
bagpipes_mass_err_low,bagpipes_mass_err_up = (bagpipes_mass - pow(10,bagpipes_results["conf_int"][0][10]))/1e7, (pow(10,bagpipes_results["conf_int"][1][10]) - bagpipes_mass)/1e7
bagpipes_mass = bagpipes_mass/1e7

# Gas-Phase Metallicity
pyneb_stat = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1].data
metal_med = pow(10,pyneb_stat["12+log10(O/H)_med"]-8.69)#,pyneb_stat["12+log10(O/H)_err_up"],pyneb_stat["12+log10(O/H)_err_low"]

# Age at z = 0.8275
time_EELG1002 = np.array(cosmo.lookback_time(0.8275).value)

######################################################################
#######					ANALOG 815206 of TNG300-1			   #######
######################################################################

# Initialize the Figure
fig = plt.figure()
fig.set_size_inches(3,7)
fig.subplots_adjust(hspace=0.05)

# Define the Subplots
ax_sfr = fig.add_subplot(311)
ax_mass = fig.add_subplot(312)
ax_metal = fig.add_subplot(313)

# Set Limits
ax_sfr.set_xlim(0.,12.)
ax_mass.set_xlim(0.,12.)
ax_metal.set_xlim(0.,12.)

# Remove the xtick labels in the SFR and Mass plots
ax_sfr.set_xticklabels([])
ax_mass.set_xticklabels([])

# Define the Axis Labels
ax_sfr.set_ylabel(r"SFR($t$) (M$_\odot$ yr$^{-1}$)")
ax_mass.set_ylabel(r"Mass($t$) ($10^7$ M$_\odot$)")
ax_metal.set_ylabel(r"$Z(t)$ ($Z_\odot$)")
ax_metal.set_xlabel(r"Lookback Time (Gyr)")

# Define Scaling
ax_mass.set_yscale("log")
ax_metal.set_yscale("log")   



# Load in the 815206 Analog
#id_analog = "TNG300-2_182200"
#id_analog = "TNG300-2_119294"
#id_analog = "TNG300-2_37967"
#id_analog = "TNG300-2_125608"
id_analog = "TNG300-1_815206"
analog = pickle.load(open(f"../../data/Illustris_Analogs/{id_analog}_analogs_with_histories.pkl","rb"))
#analog = pickle.load(open(f"../../data/Illustris_Analogs/TNG300-2_182200_analogs_with_histories.pkl","rb"))
#analog = pickle.load(open(f"../../data/Illustris_Analogs/TNG300-2_119294_analogs_with_histories.pkl","rb"))
#analog = pickle.load(open(f"../../data/Illustris_Analogs/TNG300-2_37967_analogs_with_histories.pkl","rb"))
#analog = pickle.load(open(f"../../data/Illustris_Analogs/TNG300-2_125608_analogs_with_histories.pkl","rb"))

# Star Formation History
ax_sfr.plot(cosmo.lookback_time(analog["redshift"]).value,analog["sfr"],color=tableau20("Red"))
ax_sfr.plot(cosmo.age(0.).value - bagpipes_sfh["Lookback Time"], bagpipes_sfh["SFH_median"],color=tableau20("Blue"),ls="--",alpha=0.5)
ax_sfr.errorbar(time_EELG1002,bagpipes_sfh["SFH_median"][0], yerr = ([bagpipes_sfh["SFH_median"][0] - bagpipes_sfh["SFH_1sig_low"][0]], [bagpipes_sfh["SFH_1sig_upp"][0] - bagpipes_sfh["SFH_median"][0]]),
                    mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),ecolor=util.color_scheme("Bagpipes",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)
ax_sfr.errorbar(time_EELG1002,cigale_SFR10Myrs, yerr = cigale_SFR10Myrs_err,
                    mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),ecolor=util.color_scheme("Cigale",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

# Mass Assembly History
ax_mass.plot(cosmo.lookback_time(analog["redshift"]).value,analog["mass_stars"]*1e3,color=tableau20("Red"))
ax_mass.plot(cosmo.lookback_time(analog["redshift"]).value,analog["mass_gas"]*1e3,color=tableau20("Red"),ls="--",alpha=0.5)    
ax_mass.errorbar(time_EELG1002,[bagpipes_mass], yerr = ([bagpipes_mass_err_low],[bagpipes_mass_err_up]),
                    mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),ecolor=util.color_scheme("Bagpipes",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

ax_mass.errorbar(time_EELG1002,cigale_mass, yerr = cigale_mass_err,
                    mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),ecolor=util.color_scheme("Cigale",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

# Metallicity History
ax_metal.plot(cosmo.lookback_time(analog["redshift"]).value,analog["gasmetallicitysfrweighted"]/0.02,color=tableau20("Red"))
ax_metal.plot(cosmo.lookback_time(analog["redshift"]).value,analog["starmetallicity"]/0.02,color=tableau20("Red"),ls="--",alpha=0.5)
ax_metal.plot([cosmo.lookback_time(0.8275).value],[metal_med],marker="*",ms=15,mew=1,ls="none",mec=tableau20("Green"),mfc=tableau20("Light Green"),zorder=99)




# Define Legend
ax_sfr.text(0.98,0.92,r"\textbf{%s}" % (id_analog.split("_")[0]) + '\n' + r"\textbf{\#%s}" % (id_analog.split("_")[1]),color=tableau20("Red"), ha="right",va="center",transform=ax_sfr.transAxes,fontsize=5)
            
handles = [Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),label=r"\textbf{\textit{EELG1002 (Cigale)}}"),
            Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),label=r"\textbf{\textit{EELG1002 (Bagpipes)}}"),
            Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Green"),mfc=tableau20("Light Green"),label=r"\textbf{\textit{EELG1002 (GMOS)}}")]			
leg1 = ax_sfr.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=0.8)
#plt.gca().add_artist(leg1)

handles = [Line2D([],[],ls="-",color=tableau20("Red"),label=r"Stellar"),
            Line2D([],[],ls="--",color=tableau20("Red"),alpha=0.5,label=r"Gas")]   	

leg2 = ax_mass.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=0.8)
#plt.gca().add_artist(leg2)

handles = [Line2D([],[],ls="-",color=tableau20("Red"),label=r"Gas"),
            Line2D([],[],ls="--",color=tableau20("Red"),alpha=0.5,label=r"Stellar")]   	

leg3 = ax_metal.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=0.8)
#plt.gca().add_artist(leg3)

fig.savefig("../../plots/%s_Snapshot_55_ID_%s.png" % (id_analog.split("_")[0],id_analog.split("_")[1]),format="png",dpi=300,bbox_inches="tight")





exit()
analog_37967 = pickle.load(open(f"../../data/Illustris_Analogs/TNG300-2_37967_analogs_with_histories.pkl","rb"))
analog_182200 = pickle.load(open(f"../../data/Illustris_Analogs/TNG300-2_182200_analogs_with_histories.pkl","rb"))





"""
######################################################################
#######						Initialize Figure				   #######
######################################################################

files = os.listdir("../../data/Illustris_Analogs/")

for ff in files:

    if (ff == 'TNG300-2_analogs_with_histories.pkl') or (ff == 'TNG300-1_analogs_with_histories.pkl'):
        continue

    if ".pkl" in ff:
        fig = plt.figure()
        fig.set_size_inches(4,9)
        ax_sfr = fig.add_subplot(311)
        ax_mass = fig.add_subplot(312)
        ax_metal = fig.add_subplot(313)


        ######################################################################
        #######			Load all Associated Illustris Data			   #######
        ######################################################################


        tng300 = pickle.load(open(f"../../data/Illustris_Analogs/{ff}","rb"))
        plots(ax_sfr,ax_mass,ax_metal,tng300)

        ax_sfr.set_ylabel("SFR")
        ax_mass.set_ylabel("mass"); ax_mass.set_yscale("log")
        ax_metal.set_ylabel("gas_metal")    
        ax_sfr.set_title(f"{ff}")

        fig.savefig("../../plots/illustris/%s.png" % (ff.split(".pkl")[0]), bbox_inches='tight',dpi=300, format="png")
"""
#plt.show()
#print("hit")