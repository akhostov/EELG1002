import matplotlib as mpl
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pdb

import os,sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20


##################################################################################################
########						PREP WORK TO GET SPECTRA READY FOR PLOTTING			    ##########
##################################################################################################

# Load in the best-fit SED
cigale = fits.open("../../data/cigale_results/sfh_delayed_nodust/1002_best_model.fits")[1].data

# Load in the results that include best-model fits
model_phot = fits.open("../../data/cigale_results/sfh_delayed_nodust/results.fits")[1].data

# Load in the observations
obs = fits.open("../../data/cigale_results/sfh_delayed_nodust/observations.fits")[1].data

# Define the central wavelengths
lib_wave = [("CFHT_u",3709.,518.),
			("subaru.hsc.g",4847.,1383.),("subaru.hsc.r",6219.,1547.),("subaru.hsc.i",7699.,1471.),("subaru.hsc.z",8894.,766.),("subaru.hsc.y",9761.,786.),
			("SUBARU_B",4488.,892.),("g_prime",4804.,1265.),("subaru.suprime.V",5487.,954.),("subaru.suprime.r",6305.,1376.),("subaru.suprime.i",7693.,1497.),("SUBARU_z",8978.,847.),("subaru.suprime.zpp",9063.,1335.),
			("subaru.suprime.IB427",4266.,207.),("subaru.suprime.IB464",4635.,218.),("subaru.suprime.IB484",4851.,229.),("subaru.suprime.IB505",5064.,231.),("subaru.suprime.IB527",5261.,243.),("subaru.suprime.IB574",5766.,273.),
			("subaru.suprime.IB624",6232.,300.),("subaru.suprime.IB679",6780.,336.),("subaru.suprime.IB709",7073.,316.),("subaru.suprime.IB738",7361.,324.),("subaru.suprime.IB767",7694.,365.),("subaru.suprime.IB827",8243.,343.),
			("subaru.suprime.NB711",7121.,72.),("subaru.suprime.NB816",8150.,120.),
			("galex.NUV",2300.,795.),("galex.FUV",1535.08,233.93),
			("spitzer.irac.ch1",35378.41,7431.71),("spitzer.irac.ch2",44780.49,10096.82),("spitzer.irac.ch3",56961.78,13911.89),("spitzer.irac.ch4",77978.40,28311.77),
			("hst.wfc.F814W",8333.,2511.),("hst.wfc3.F140W",13923.21,3933.32),("cfht.wircam.H",16243.54,2911.26),("cfht.wircam.Ks",21434.00,3270.46)]

lib_wave = np.transpose(lib_wave)

# Match the filters and extract wavelengths
filters = np.array([ii for ii in obs.columns.names if not ii.endswith("_err") and not any(ii.startswith(prefix) for prefix in ("id","redshift","line","line.H-alpha"))])

index_map = {element: index for index, element in enumerate(lib_wave[0])}
match_ind = [index_map[element] for element in filters]

obs_wave = np.double(lib_wave[1][match_ind])
obs_fwhm = np.double(lib_wave[2][match_ind])
obs_fluxes = np.asarray([obs[ii] for ii in filters]).T[0]*1e-3*1e-23*3e18/pow(obs_wave,2.)
obs_ferr = np.asarray([obs[ii+"_err"] for ii in filters]).T[0]*1e-3*1e-23*3e18/pow(obs_wave,2.)
obs_band_ids = lib_wave[0][match_ind]

results_fluxes = np.asarray([model_phot["bayes."+ii] for ii in filters]).T[0]*1e-3*1e-23*3e18/pow(obs_wave,2.)
results_ferr = np.asarray([model_phot["bayes."+ii+"_err"] for ii in filters]).T[0]*1e-3*1e-23*3e18/pow(obs_wave,2.)



##################################################################################################
########						PREPARE THE FIGURE GRID SPACE						    ##########
##################################################################################################

# Define the Figure Size and Grid Space
fig = plt.figure()
fig.set_size_inches(7,3)
#fig.subplots_adjust(hspace=0.1, wspace=0.1)
#gs_top = mpl.gridspec.GridSpec(8, 3,  height_ratios=[1.0, 2., 0.4, 1.0, 2., 0.4, 1.0, 2.0],hspace=0.)
#gs_bot = mpl.gridspec.GridSpec(8, 3,  height_ratios=[1.0, 2., 0.4, 1.0, 2., 0.4, 1.0, 2.0],hspace=0.)
ax = fig.add_subplot(111)

# Labels
#fig.text(0.5, 0.07, r'Rest-Frame Wavelength (\AA)', ha='center')
#fig.text(0.04, 0.5, r'$f_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)', va='center', rotation='vertical')


ax.set_xscale("log")
ax.set_yscale("log")

#ax.set_xticks(np.arange(0.0,9.,1.))
#ax.set_xticks(np.arange(0.0,9.,0.5),minor=True)

ax.set_ylim(0.1,1e3)
ax.set_xlim(0.3,3.0)

ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

ax.set_xticks([0.3,0.4,0.5,0.6,0.8,1,2,3])
ax.set_xticks([0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.25,2.5,2.75,3],minor=True)
ax.set_xticklabels(["0.3","0.4","0.5","0.6","0.8","1","2","3"])

##################################################################################################
########						NOW DO THE PLOTTING STUFF HERE						    ##########
##################################################################################################


# Plot the Best-Fit SED
ax.plot(cigale["wavelength"]/1e3,cigale["fnu"]*1e-3*1e-23*3e18/pow(cigale["wavelength"]*10.,2.)*1e19,color="black",lw=0.5)

# Keep only Detections
keep = (obs_ferr > 0.) & (obs_fluxes > 0.)
obs_band_ids = obs_band_ids[keep]

# Plot the Model
ax.plot(obs_wave[keep]/1e4,results_fluxes[keep]*1e19,\
				ls="none",mfc="none",mec=tableau20("Red"),\
				marker="s",ms=5,mew=0.5)

# Detections 
cfht = [ind for ind,band in enumerate(obs_band_ids) if "CFHT_u" in band] #purple
scam_BB = [ind for ind,band in enumerate(obs_band_ids) if ("subaru" in band) and ("IB" not in band) and ("NB" not in band) ] # blue
scam_IB = [ind for ind,band in enumerate(obs_band_ids) if ("subaru" in band) and ("IB" in band) and ("NB" not in band) ] # blue
scam_NB = [ind for ind,band in enumerate(obs_band_ids) if ("subaru" in band) and ("IB" not in band) and ("NB" in band) ] # blue

hsc = [ind for ind,band in enumerate(obs_band_ids) if "hsc" in band] # green
F814W = [ind for ind,band in enumerate(obs_band_ids) if "F814W" in band] # olive
F140W = [ind for ind,band in enumerate(obs_band_ids) if "F140W" in band] # orange
wircam = [ind for ind,band in enumerate(obs_band_ids) if "wircam" in band] # light red
spitzer = [ind for ind,band in enumerate(obs_band_ids) if "spitzer" in band] # red

# plot
ms = 4
ax.errorbar(obs_wave[keep][cfht]/1e4,obs_fluxes[keep][cfht]*1e19,xerr=obs_fwhm[keep][cfht]/(1e4*2.),yerr=obs_ferr[keep][cfht]*1e19,\
				ls="none",mec="black",mfc=tableau20("Purple"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

ax.errorbar(obs_wave[keep][scam_BB]/1e4,obs_fluxes[keep][scam_BB]*1e19,xerr=obs_fwhm[keep][scam_BB]/(1e4*2.),yerr=obs_ferr[keep][scam_BB]*1e19,\
				ls="none",mec="black",mfc=tableau20("Blue"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

ax.errorbar(obs_wave[keep][scam_IB]/1e4,obs_fluxes[keep][scam_IB]*1e19,xerr=obs_fwhm[keep][scam_IB]/(1e4*2.),yerr=obs_ferr[keep][scam_IB]*1e19,\
				ls="none",mec="black",mfc=tableau20("Light Blue"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

ax.errorbar(obs_wave[keep][scam_NB]/1e4,obs_fluxes[keep][scam_NB]*1e19,xerr=obs_fwhm[keep][scam_NB]/(1e4*2.),yerr=obs_ferr[keep][scam_NB]*1e19,\
				ls="none",mec="black",mfc=tableau20("Sky Blue"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)


ax.errorbar(obs_wave[keep][hsc]/1e4,obs_fluxes[keep][hsc]*1e19,xerr=obs_fwhm[keep][hsc]/(1e4*2.),yerr=obs_ferr[keep][hsc]*1e19,\
				ls="none",mec="black",mfc=tableau20("Green"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

ax.errorbar(obs_wave[keep][F814W]/1e4,obs_fluxes[keep][F814W]*1e19,xerr=obs_fwhm[keep][F814W]/(1e4*2.),yerr=obs_ferr[keep][F814W]*1e19,\
				ls="none",mec="black",mfc=tableau20("Olive"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

ax.errorbar(obs_wave[keep][F140W]/1e4,obs_fluxes[keep][F140W]*1e19,xerr=obs_fwhm[keep][F140W]/(1e4*2.),yerr=obs_ferr[keep][F140W]*1e19,\
				ls="none",mec="black",mfc=tableau20("Orange"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

ax.errorbar(obs_wave[keep][wircam]/1e4,obs_fluxes[keep][wircam]*1e19,xerr=obs_fwhm[keep][wircam]/(1e4*2.),yerr=obs_ferr[keep][wircam]*1e19,\
				ls="none",mec="black",mfc=tableau20("Red"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)


ax.text(0.02,0.19,r"CFHT/MegaCam",color=tableau20("Purple"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
ax.text(0.02,0.12,r"CFHT/WirCam",color=tableau20("Red"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
ax.text(0.20,0.26,r"Subaru/SCam BB",color=tableau20("Blue"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
ax.text(0.20,0.19,r"Subaru/SCam IB",color=tableau20("Light Blue"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
ax.text(0.20,0.12,r"Subaru/SCam NB",color=tableau20("Sky Blue"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
ax.text(0.20,0.05,r"Subaru/HSC BB",color=tableau20("Green"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
ax.text(0.42,0.19,r"HST/ACS F814W",color=tableau20("Olive"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
ax.text(0.42,0.12,r"HST/WFC3 F140W",color=tableau20("Orange"),ha="left",va="center",fontsize=8,transform=ax.transAxes)

#ax.text(0.05,0.05,r"Subaru/HSC",color=tableau20("Light Green"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
#ax.text(0.20,0.19,r"HST",color=tableau20("Orange"),ha="left",va="center",fontsize=8,transform=ax.transAxes)
#ax.text(0.20,0.12,r"CFHT/Wcam",color=tableau20("Light Red"),ha="left",va="center",fontsize=8,transform=ax.transAxes)


#ax.text(np.mean([0.2060,0.2457]),10.,"A",color=tableau20("Grey"),ha="center",va="center",fontsize=8)
#ax.text(np.mean([0.3162,0.3560]),10.,"B",color=tableau20("Grey"),ha="center",va="center",fontsize=8)


"""
# Detections 
ax.errorbar(obs_wave[keep]/1e4,obs_fluxes[keep]*1e19,xerr=obs_fwhm[keep]/(1e4*2.),yerr=obs_ferr[keep]*1e19,\
				ls="none",mec="black",mfc=tableau20("Blue"),ecolor=tableau20("Grey"),\
				marker="o",ms=3,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)

# Plot the Lyman Limit
ax.plot([0.166,0.166],[1e-5,1e6],color=tableau20("Blue"),alpha=0.3,lw=1,ls="--")
ax.fill_between([0.1,0.166],1e-5,1e6,color=tableau20("Blue"),alpha=0.3)

# Plot the >1 um restframe Population
ax.plot([1.8275,1.8275],[1e-5,1e6],color=tableau20("Red"),alpha=0.3,lw=1,ls="--")
ax.fill_between([1.8275,10.],1e-5,1e6,color=tableau20("Red"),alpha=0.3)

# Plot the Spectra Ranges
ax.fill_between([0.365*1.8275,0.51*1.8275],1,1e3,color=tableau20("Grey"),alpha=0.3)
"""

custom_leg = [Line2D([0], [0], ls="none", marker="o",mfc=tableau20("Blue"),mec="black",label="Observations",mew=0.2),
				Line2D([0], [0], ls="none", marker="s",mfc="none",mec=tableau20("Red"),label=r"{\sc Cigale} $\chi^2_\textrm{red} = %0.2f$" % (model_phot["best.reduced_chi_square"]))]

ax.legend(handles=custom_leg,frameon=False,ncol=1,loc="upper right",fontsize=8)




ax.set_xlabel(r"Observed Wavelength ($\mu$m)",fontsize=8)
ax.set_ylabel(r"$f_\lambda$ (10$^{-19}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)",fontsize=8)

fig.savefig("../../plots/EELG1002_SED.png",format="png",dpi=300,bbox_inches="tight")



pdb.set_trace()