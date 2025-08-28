import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.cosmology import z_at_value
import astropy.units as u
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

# Only used to ensure continuity in plots. No physical meaning
def log10_with_dummy_numbers(prop, dummy=-99.):
    prop = np.log10(prop)
    prop[~np.isfinite(prop)] = dummy
    
    return prop

def set_top_axis(ax,ticklabels=True):

    #### Now Let's Define the Redshift Axis
    # Create a twin x-axis to display redshift
    ax_redshift = ax.twiny()

    # Convert lookback time to redshift
    redshifts = np.array([0.,0.1,0.5,1.0,1.5])
    redshifts_minor = np.arange(0.,1.5,0.1)
    lookback_time = cosmo.lookback_time(redshifts).value
    lookback_time_minor = cosmo.lookback_time(redshifts_minor).value

    # Set ticks and labels for the redshift axis
    ax_redshift.set_xticks(lookback_time)
    ax_redshift.set_xticks(lookback_time_minor,minor=True)
    
    if ticklabels:
        ax_redshift.set_xticklabels([f'{z:.1f}' for z in redshifts])
        ax_redshift.set_xlabel(r'Redshift')
    else:
        ax_redshift.set_xticklabels([])

    # Set limits for the secondary axis (redshift axis)
    ax_redshift.set_xlim(ax_sfr.get_xlim())

    return ax_redshift

######################################################################
#######					  LOAD EELG1002 RESULTS			       #######
######################################################################

cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
cigale_mass = cigale["bayes.stellar.m_star"]
cigale_mass_err = cigale["bayes.stellar.m_star_err"]/(np.log(10.)*cigale_mass)
cigale_mass = np.log10(cigale_mass)
cigale_SFR10Myrs = cigale["bayes.sfh.sfr10Myrs"]
cigale_SFR10Myrs_err = cigale["bayes.sfh.sfr10Myrs_err"]

bagpipes_sfh = fits.open("../../data/SED_results/bagpipes_results/best_fit_SFH_sfh_continuity_spec_BPASS.fits")[1].data
bagpipes_results = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")

# Stellar Mass
bagpipes_mass = bagpipes_results["median"][10]
bagpipes_mass_err_low,bagpipes_mass_err_up = (bagpipes_mass - bagpipes_mass), (bagpipes_results["conf_int"][1][10] - bagpipes_mass)

# Gas-Phase Metallicity
pyneb_stat = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1].data
metal_med = pyneb_stat["12+log10(O/H)_med"]-8.69

# Age at z = 0.8275
time_EELG1002 = np.array(cosmo.lookback_time(0.8275).value)

######################################################################
#######				    	Prepare the Figure			       #######
######################################################################

# Initialize the Figure
fig = plt.figure()
fig.set_size_inches(3,7)
fig.subplots_adjust(hspace=0.08)

######################################################################
#######				    	ANALOG TNG300-2 182200			   #######
######################################################################

# Define the Subplots
ax_sfr = fig.add_subplot(311)
ax_mass = fig.add_subplot(312)
ax_metal = fig.add_subplot(313)

# Define the SFR zoom-in plot
left, bottom, width, height = [0.19, 0.74, 0.3, 0.075]
ax_sfr_zoomin = fig.add_axes([left, bottom, width, height])
ax_sfr_zoomin.tick_params(labelsize=5)
#ax_sfr_zoomin.set_xlabel(r'Lookback Time (Gyr)',fontsize=5)
#ax_sfr_zoomin.set_ylabel(r'SFR (M$_\odot$ yr$^{-1}$)',fontsize=5)


# Set Limits
ax_sfr.set_xlim(0.,8.5)
ax_sfr_zoomin.set_xlim(6.5,8.5)
ax_mass.set_xlim(0.,8.5)
ax_metal.set_xlim(0.,8.5)
ax_sfr_zoomin.set_ylim(0.,6.)
ax_mass.set_ylim(7.,12.)
ax_metal.set_ylim(-1.5,0.5)

# Remove the xtick labels in the SFR and Mass plots
ax_sfr.set_xticklabels([])
ax_mass.set_xticklabels([])

# Define the Axis Labels
ax_sfr.set_ylabel(r"SFR (M$_\odot$ yr$^{-1}$)")
ax_mass.set_ylabel(r"$\log_{10} M$ (M$_\odot$)")
ax_metal.set_ylabel(r"$\log_{10} Z$ ($Z_{\odot}$)")
ax_metal.set_xlabel(r"Lookback Time (Gyr)")


#### Now Let's Define the Redshift Axis
_ = set_top_axis(ax_sfr,ticklabels=True)
_ = set_top_axis(ax_mass,ticklabels=False)
_ = set_top_axis(ax_metal,ticklabels=False)


# Highlight the zoomed in region
mark_inset(ax_sfr, ax_sfr_zoomin, loc1=1,loc2=3,edgecolor="black", linewidth=0.5,ls="--",facecolor="none")
"""
rect, lines = ax_sfr.indicate_inset_zoom(ax_sfr_zoomin,edgecolor="black",linewidth=0.5,ls="--",facecolor="none")
for line in lines: 
	line.set_linestyle("--")
	line.set_linewidth(0.5)
	line.set_zorder(1)
import pdb; pdb.set_trace()
lines.connectors = (True,False,False,True)
"""


# Load in the 182200 Analog
id_analog_1 = "TNG300-2_182200"
analog = pickle.load(open(f"../../data/Illustris_Analogs/{id_analog_1}_analogs_with_histories.pkl","rb"))

# Star Formation History
ax_sfr.plot(cosmo.lookback_time(analog["redshift"]).value,analog["sfr"],color=tableau20("Red"))
ax_sfr_zoomin.plot(cosmo.lookback_time(analog["redshift"]).value,analog["sfr"],lw=1,color=tableau20("Red"))

# Mass Assembly History
mstar = log10_with_dummy_numbers(analog["mass_stars"]*1e10)
mgas = log10_with_dummy_numbers(analog["mass_gas"]*1e10)
ax_mass.plot(cosmo.lookback_time(analog["redshift"]).value,mstar,color=tableau20("Red"))
ax_mass.plot(cosmo.lookback_time(analog["redshift"]).value,mgas,color=tableau20("Red"),ls="--",alpha=0.5)    

# Metallicity History
zgas = log10_with_dummy_numbers(analog["gasmetallicity"]/0.02)
zstar = log10_with_dummy_numbers(analog["starmetallicity"]/0.02)
ax_metal.plot(cosmo.lookback_time(analog["redshift"]).value,zgas,color=tableau20("Red"))
ax_metal.plot(cosmo.lookback_time(analog["redshift"]).value,zstar,color=tableau20("Red"),ls="--",alpha=0.5)

# Now Let's Highlgiht the Merging events
merging_snapnum = []




# Load in the 119294 Analog
id_analog_2 = "TNG300-2_119294"
analog = pickle.load(open(f"../../data/Illustris_Analogs/{id_analog_2}_analogs_with_histories.pkl","rb"))

# Star Formation History
ax_sfr.plot(cosmo.lookback_time(analog["redshift"]).value,analog["sfr"],color=tableau20("Green"))
ax_sfr_zoomin.plot(cosmo.lookback_time(analog["redshift"]).value,analog["sfr"],lw=1,color=tableau20("Green"))

# Mass Assembly History
mstar = log10_with_dummy_numbers(analog["mass_stars"]*1e10)
mgas = log10_with_dummy_numbers(analog["mass_gas"]*1e10)
ax_mass.plot(cosmo.lookback_time(analog["redshift"]).value,mstar,color=tableau20("Green"))
ax_mass.plot(cosmo.lookback_time(analog["redshift"]).value,mgas,color=tableau20("Green"),ls="--",alpha=0.5)    

# Metallicity History
zgas = log10_with_dummy_numbers(analog["gasmetallicity"]/0.02)
zstar = log10_with_dummy_numbers(analog["starmetallicity"]/0.02)
ax_metal.plot(cosmo.lookback_time(analog["redshift"]).value,zgas,color=tableau20("Green"))
ax_metal.plot(cosmo.lookback_time(analog["redshift"]).value,zstar,color=tableau20("Green"),ls="--",alpha=0.5)






#### PLOT EELG1002
ax_sfr.errorbar(time_EELG1002,bagpipes_sfh["SFH_median"][0], yerr = ([bagpipes_sfh["SFH_median"][0] - bagpipes_sfh["SFH_1sig_low"][0]], [bagpipes_sfh["SFH_1sig_upp"][0] - bagpipes_sfh["SFH_median"][0]]),
                    mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),ecolor=util.color_scheme("Bagpipes",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)
ax_sfr.errorbar(time_EELG1002,cigale_SFR10Myrs, yerr = cigale_SFR10Myrs_err,
                    mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),ecolor=util.color_scheme("Cigale",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)
ax_sfr.plot(cosmo.age(0.).value - bagpipes_sfh["Lookback Time"], bagpipes_sfh["SFH_median"],color=tableau20("Blue"),lw=1,ls="--")



ax_sfr_zoomin.errorbar(time_EELG1002,bagpipes_sfh["SFH_median"][0], yerr = ([bagpipes_sfh["SFH_median"][0] - bagpipes_sfh["SFH_1sig_low"][0]], [bagpipes_sfh["SFH_1sig_upp"][0] - bagpipes_sfh["SFH_median"][0]]),
                    mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),ecolor=util.color_scheme("Bagpipes",mec=True),
                    ls="none",marker="*",ms=10,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)
ax_sfr_zoomin.errorbar(time_EELG1002,cigale_SFR10Myrs, yerr = cigale_SFR10Myrs_err,
                    mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),ecolor=util.color_scheme("Cigale",mec=True),
                    ls="none",marker="*",ms=10,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)
ax_sfr_zoomin.plot(cosmo.age(0.).value - bagpipes_sfh["Lookback Time"], bagpipes_sfh["SFH_median"],color=tableau20("Blue"),ls="--")


ax_mass.errorbar(time_EELG1002,[bagpipes_mass], yerr = ([bagpipes_mass_err_low],[bagpipes_mass_err_up]),
                    mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),ecolor=util.color_scheme("Bagpipes",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

ax_mass.errorbar(time_EELG1002,cigale_mass, yerr = cigale_mass_err,
                    mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),ecolor=util.color_scheme("Cigale",mec=True),
                    ls="none",marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5,zorder=99)

ax_metal.plot([cosmo.lookback_time(0.8275).value],[metal_med],marker="*",ms=15,mew=1,ls="none",mec=tableau20("Green"),mfc=tableau20("Light Green"),zorder=99)




######## Define Legend
ax_sfr.text(0.95,0.90,r"\textbf{%s}" % (id_analog_1.split("_")[0]) + '\n' + r'\textbf{Snap = 55}',color="black", ha="right",va="center",transform=ax_sfr.transAxes,fontsize=5)
ax_sfr.text(0.95,0.82,r"\textbf{ID = %s}" % (id_analog_1.split("_")[1]),color=tableau20("Red"), ha="right",va="center",transform=ax_sfr.transAxes,fontsize=5)
ax_sfr.text(0.95,0.75,r"\textbf{ID = %s}" % (id_analog_2.split("_")[1]),color=tableau20("Green"), ha="right",va="center",transform=ax_sfr.transAxes,fontsize=5)

# Goes to Top Panel: SFR            
handles = [Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Cigale",mec=True),mfc=util.color_scheme("Cigale",mfc=True),label=r"\textbf{\textit{EELG1002 (Cigale)}}"),
            Line2D([],[],ls="none",marker="*",ms=10,mec=util.color_scheme("Bagpipes",mec=True),mfc=util.color_scheme("Bagpipes",mfc=True),label=r"\textbf{\textit{EELG1002 (Bagpipes)}}"),
            Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Green"),mfc=tableau20("Light Green"),label=r"\textbf{\textit{EELG1002 (GMOS)}}")]			
leg1 = ax_sfr.legend(handles=handles,loc="upper left",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=0.8)

# Goes to Middle Panel: Mass Growth     
handles = [Line2D([],[],ls="-",color=tableau20("Grey"),label=r"Stellar"),
            Line2D([],[],ls="--",color=tableau20("Grey"),alpha=0.5,label=r"Gas")]   	

leg2 = ax_mass.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=0.8)

# Goes to Bottom Panel: Chemical Enrichment 
handles = [Line2D([],[],ls="-",color=tableau20("Grey"),label=r"Gas"),
            Line2D([],[],ls="--",color=tableau20("Grey"),alpha=0.5,label=r"Stellar")]   	

leg3 = ax_metal.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5,labelspacing=0.8)



######## Save the Figure
fig.savefig("../../plots/Illustris_Analogs.png",format="png",dpi=300,bbox_inches="tight")



