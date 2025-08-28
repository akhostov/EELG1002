import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import imshow_norm,LinearStretch,ZScaleInterval,ImageNormalize

import sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20
sys.path.insert(0,"..")
import util

##################################################################################################
########						PREPARE THE FIGURE GRID SPACE						    ##########
##################################################################################################

# Define the Figure Size and Grid Space
fig = plt.figure()
fig.set_size_inches(7,4)
fig.subplots_adjust(hspace=0.2,wspace=0.2)

mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)

# Set global labels
fig.supxlabel(r'Rest-Frame Wavelength (\AA)', fontsize=10)
fig.subplots_adjust(left=0.08)
fig.supylabel(r'$f_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)', fontsize=10)

##################################################################################################
########						PREP WORK TO GET SPECTRA READY FOR PLOTTING			    ##########
##################################################################################################

# Load in the Line Fits
lfits = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data
lwave = Table.read("../../data/emline_fits/1002.fits")

# Load in the 1D and 2D spectra
spec1d = fits.open("../../data/flux_corr_spectra/43158747673038238/1002.fits")[1].data
spec2d = fits.open("../../data/Science_coadd/spec2d_43158747673038238.fits")

flux_2D = spec2d[1].data - spec2d[3].data # EXT=1 SCIIMG and EXT=2 SKYMODEL
lambda_2D = spec2d[8].data # EXT8 WAVEIMG
ivar_2D = spec2d[2].data  #EXT=2 IVARRAW and EXT=5 IVARMODEL

# Find where slit_id is in the 2D data
ind = spec2d[10].data["SPAT_ID"] == 188

# Extract the slit using the left and right boundaries of the slit
left_init = np.squeeze(spec2d[10].data["left_init"][ind])
right_init = np.squeeze(spec2d[10].data["right_init"][ind])

# Convert to Integer
left_init = int(left_init[0])
right_init = int(right_init[0])

# Extract the 2D
flux_2D = flux_2D[:,left_init:right_init].T*(1.8275)
lambda_2D = lambda_2D[:,left_init:right_init].T/(1.8275)
ivar_2D = ivar_2D[:,left_init:right_init].T*(1.8275)**2.

# Now let's remove the zeros at the end
index_2D = np.apply_along_axis(lambda row: np.flatnonzero(row[::-1] == 0)[::-1], axis=1, arr=lambda_2D)
flux_2D = np.delete(flux_2D,-index_2D-1,axis=1)
ivar_2D = np.delete(ivar_2D,-index_2D-1,axis=1)
lambda_2D = np.delete(lambda_2D,-index_2D-1,axis=1)


##################################################################################################
########								SPECTRA PLOTTING							    ##########
##################################################################################################

wave = spec1d["OPT_WAVE"]/(1.8275)
flam = spec1d["OPT_FLAM"]*(1.8275)
sigm = spec1d["OPT_FLAM_SIG"]*(1.8275)


wave_min = 3650
wave_max = 5100
these_1D = (wave > wave_min) & (wave < wave_max)
these_2D = (lambda_2D > wave_min) & (lambda_2D < wave_max)

wave = wave[these_1D] 
flam = flam[these_1D]
sigm = sigm[these_1D]


##################################################################################################
########						LINE FIT PLOTTING STARTS HERE						    ##########
##################################################################################################

# Plot [OIII]
scales = [ll for ll in lwave.colnames if ("_scale" in ll) and ("_err" not in ll)]
centerwaves = [ll for ll in lwave.colnames if ("_centerwave" in ll) and ("_err" not in ll)]
sigmas = [ll for ll in lwave.colnames if ("_sigma" in ll) and ("_err" not in ll) and ("_br_" not in ll)]

wave_line = np.linspace(3700,5100,5000)
cont = float(lwave["PL_norm"])*(wave_line/3000.)**(-1.*float(lwave["PL_slope"])) #NOTE: The continuum is really really faint and uncertainties are large on PL_norm and PL_slope. Not suitable for EW measurements


##### Plot the Spectra Per Line
## [OIII]5007
fitted_line = float(lwave["OIII5007c_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["OIII5007c_1_centerwave"]))/float(lwave["OIII5007c_1_sigma"]),2.))

ax_OIII5007=fig.add_subplot(2,4,1)
ax_OIII5007.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_OIII5007.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_OIII5007.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")
ax_OIII5007.set_xlim(np.exp(float(lwave["OIII5007c_1_centerwave"]) - 5.*float(lwave["OIII5007c_1_sigma"])),np.exp(float(lwave["OIII5007c_1_centerwave"]) + 5.*float(lwave["OIII5007c_1_sigma"])))
ax_OIII5007.set_ylim(-0.1,fitted_line.max()*1.1)

# [OIII]5007
ax_OIII5007.plot([5007.,5007.],ax_OIII5007.get_ylim(),color="black",lw=0.5,ls=":")
ax_OIII5007.text(0.05,0.90,r'[O{\sc iii}]5007',color="black",ha="left",va="top",fontsize=7,rotation=0,transform=ax_OIII5007.transAxes)



## [OIII]4959
fitted_line = float(lwave["OIII4959c_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["OIII4959c_1_centerwave"]))/float(lwave["OIII4959c_1_sigma"]),2.))

ax_OIII4959=fig.add_subplot(2,4,2)
ax_OIII4959.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_OIII4959.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_OIII4959.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")
ax_OIII4959.set_xlim(np.exp(float(lwave["OIII4959c_1_centerwave"]) - 5.*float(lwave["OIII4959c_1_sigma"])),np.exp(float(lwave["OIII4959c_1_centerwave"]) + 5.*float(lwave["OIII4959c_1_sigma"])))
ax_OIII4959.set_ylim(-0.1,fitted_line.max()*1.2)

# [OIII]4959
ax_OIII4959.plot([4959.,4959.],ax_OIII4959.get_ylim(),color="black",lw=0.5,ls=":")
ax_OIII4959.text(0.05,0.90,r'[O{\sc iii}]4959',color="black",ha="left",va="top",fontsize=7,rotation=0,transform=ax_OIII4959.transAxes)

## Hbeta
fitted_line = float(lwave["Hb_na_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["Hb_na_1_centerwave"]))/float(lwave["Hb_na_1_sigma"]),2.))

ax_Hb=fig.add_subplot(2,4,3)
ax_Hb.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_Hb.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_Hb.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")
ax_Hb.set_xlim(np.exp(float(lwave["Hb_na_1_centerwave"]) - 5.*float(lwave["Hb_na_1_sigma"])),np.exp(float(lwave["Hb_na_1_centerwave"]) + 5.*float(lwave["Hb_na_1_sigma"])))
ax_Hb.set_ylim(-0.1,fitted_line.max()*1.5)

# Hbeta
ax_Hb.plot([4861.,4861.],ax_Hb.get_ylim(),color="black",lw=0.5,ls=":")
ax_Hb.text(0.05,0.90,r'H$\beta$',color="black",ha="left",va="top",fontsize=7,rotation=0,transform=ax_Hb.transAxes)



## Hgamma+[OIII]
fitted_line = float(lwave["Hg_na_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["Hg_na_1_centerwave"]))/float(lwave["Hg_na_1_sigma"]),2.)) + \
						float(lwave["OIII4363_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["OIII4363_1_centerwave"]))/float(lwave["OIII4363_1_sigma"]),2.))

ax_HgO3=fig.add_subplot(2,4,4)
ax_HgO3.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_HgO3.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_HgO3.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")
ax_HgO3.set_xlim(np.exp(float(lwave["Hg_na_1_centerwave"]) - 5.*float(lwave["Hg_na_1_sigma"])),np.exp(float(lwave["OIII4363_1_centerwave"]) + 5.*float(lwave["OIII4363_1_sigma"])))
ax_HgO3.set_ylim(-0.1,fitted_line.max()*1.4)

# Hgamma
ax_HgO3.plot([4341.,4341.],ax_HgO3.get_ylim(),color="black",lw=0.5,ls=":")
ax_HgO3.text(4341-5.,1.7,r'H$\gamma$',color="black",ha="right",va="top",fontsize=7,rotation=90)

# [OIII]4363
ax_HgO3.plot([4363.,4363.],ax_HgO3.get_ylim(),color="black",lw=0.5,ls=":")
ax_HgO3.text(4363+5.,1.7,r'[O{\sc iii}]4363',color="black",ha="right",va="top",fontsize=7,rotation=90)





## Hdelta
fitted_line = float(lwave["Hd_na_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["Hd_na_1_centerwave"]))/float(lwave["Hd_na_1_sigma"]),2.))

ax_Hd=fig.add_subplot(2,4,5)
ax_Hd.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_Hd.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_Hd.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")
ax_Hd.set_xlim(np.exp(float(lwave["Hd_na_1_centerwave"]) - 5.*float(lwave["Hd_na_1_sigma"])),np.exp(float(lwave["Hd_na_1_centerwave"]) + 5.*float(lwave["Hd_na_1_sigma"])))
ax_Hd.set_ylim(0.0,fitted_line.max()*1.3)

# Hdelta
ax_Hd.plot([4101.,4101.],ax_Hd.get_ylim(),color="black",lw=0.5,ls=":")
ax_Hd.text(0.05,0.90,r'H$\delta$',color="black",ha="left",va="top",fontsize=7,rotation=0,transform=ax_Hd.transAxes)



## NeIII + Hepsilon
NeIII3968 = float(lwave["NeIII3968_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["NeIII3968_1_centerwave"]))/float(lwave["NeIII3968_1_sigma"]),2.))
Heps = float(lwave["Hep_na_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["Hep_na_1_centerwave"]))/float(lwave["Hep_na_1_sigma"]),2.))
fitted_line = NeIII3968 + Heps
ax_NeIIIHeps=fig.add_subplot(2,4,6)
ax_NeIIIHeps.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_NeIIIHeps.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_NeIIIHeps.plot(wave_line,NeIII3968+cont,ls="--",lw=0.5,color=tableau20("Purple"))	
ax_NeIIIHeps.plot(wave_line,Heps+cont,ls="--",lw=0.5,color=tableau20("Red"))
ax_NeIIIHeps.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")


ax_NeIIIHeps.set_xlim(np.exp(float(lwave["NeIII3968_1_centerwave"]) - 5.*float(lwave["NeIII3968_1_sigma"])),np.exp(float(lwave["Hep_na_1_centerwave"]) + 5.*float(lwave["Hep_na_1_sigma"])))
ax_NeIIIHeps.set_ylim(0.0,1.4)


# [NeIII]
ax_NeIIIHeps.plot([3967.5,3967.5],ax_NeIIIHeps.get_ylim(),color=tableau20("Purple"),lw=0.5,ls=":")
ax_NeIIIHeps.text(3967.5-2.,1.3,r'[Ne{\sc iii}]3968',color=tableau20("Purple"),ha="right",va="top",fontsize=7,rotation=90)

# Hepsilon
ax_NeIIIHeps.plot([3970.,3970.],ax_NeIIIHeps.get_ylim(),color=tableau20("Red"),lw=0.5,ls=":")
ax_NeIIIHeps.text(3970+2.,1.3,r'H$\epsilon$',color=tableau20("Red"),ha="right",va="top",fontsize=7,rotation=90)






## NeIII + H8 + H0
NeIII3869 = float(lwave["NeIII3869_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["NeIII3869_1_centerwave"]))/float(lwave["NeIII3869_1_sigma"]),2.))
H8 = float(lwave["H8_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["H8_1_centerwave"]))/float(lwave["H8_1_sigma"]),2.))
H9 = float(lwave["H9_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["H9_1_centerwave"]))/float(lwave["H9_1_sigma"]),2.))

fitted_line = NeIII3869 + H8 + H9
ax_NeIIIH8H9=fig.add_subplot(2,4,7)
ax_NeIIIH8H9.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_NeIIIH8H9.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_NeIIIH8H9.plot(wave_line,H9+cont,ls="--",lw=0.5,color=tableau20("Purple"))	
ax_NeIIIH8H9.plot(wave_line,NeIII3869+cont,ls="--",lw=0.5,color=tableau20("Green"))
ax_NeIIIH8H9.plot(wave_line,H8+cont,ls="--",lw=0.5,color=tableau20("Red"))
ax_NeIIIH8H9.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")


ax_NeIIIH8H9.set_xlim(np.exp(float(lwave["H9_1_centerwave"]) - 5.*float(lwave["H9_1_sigma"])),np.exp(float(lwave["H8_1_centerwave"]) + 5.*float(lwave["H8_1_sigma"])))
ax_NeIIIH8H9.set_ylim(-0.1,fitted_line.max()*1.8)

# H9
ax_NeIIIH8H9.plot([3835.4,3835.4],ax_NeIIIH8H9.get_ylim(),color="black",lw=0.5,ls=":")
ax_NeIIIH8H9.text(3835.4 - 2.,2.5,r'H9',color="black",ha="right",va="top",fontsize=7,rotation=90)

# [NeIII]
ax_NeIIIH8H9.plot([3869.,3869.],ax_NeIIIH8H9.get_ylim(),color="black",lw=0.5,ls=":")
ax_NeIIIH8H9.text(3869-2.,2.5,r'[Ne{\sc iii}]3869',color="black",ha="right",va="top",fontsize=7,rotation=90)
		
# H8
ax_NeIIIH8H9.plot([3889.0,3889.0],ax_NeIIIH8H9.get_ylim(),color="black",lw=0.5,ls=":")
ax_NeIIIH8H9.text(3889.0 + 0., 2.5,r'H8',color="black",ha="right",va="top",fontsize=7,rotation=90)



## [OII]
OII_3726 = float(lwave["OII3726_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["OII3726_1_centerwave"]))/float(lwave["OII3726_1_sigma"]),2.))
OII_3728 = float(lwave["OII3728_1_scale"])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave["OII3728_1_centerwave"]))/float(lwave["OII3728_1_sigma"]),2.))

fitted_line = OII_3726 + OII_3728
ax_OII=fig.add_subplot(2,4,8)
ax_OII.step(wave,flam,color=tableau20("Blue"),where="mid")
ax_OII.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")
ax_OII.plot(wave_line,OII_3726+cont,ls="--",lw=0.5,color=tableau20("Purple"))	
ax_OII.plot(wave_line,OII_3728+cont,ls="--",lw=0.5,color=tableau20("Red"))
ax_OII.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")


ax_OII.set_xlim(np.exp(float(lwave["OII3726_1_centerwave"]) - 5.*float(lwave["OII3726_1_sigma"])),np.exp(float(lwave["OII3728_1_centerwave"]) + 5.*float(lwave["OII3728_1_sigma"])))
ax_OII.set_ylim(0.0,fitted_line.max()*1.4)

ax_OII.plot([3726.,3726.],ax_OII.get_ylim(),color=tableau20("Purple"),lw=0.5,ls=":")
ax_OII.text(3726.-3.,1.5,r'[O{\sc ii}]3726',color=tableau20("Purple"),ha="right",va="top",fontsize=5,rotation=90)

ax_OII.plot([3729,3729],ax_OII.get_ylim(),color=tableau20("Red"),lw=0.5,ls=":")
ax_OII.text(3729 +3.,1.5,r'[O{\sc ii}]3729',color=tableau20("Red"),ha="right",va="top",fontsize=5,rotation=90)








fig.savefig("../../plots/emission_line_profiles.png",format="png",dpi=300,bbox_inches="tight")
exit()



for ii in range(len(scales)):
	
	keep = np.abs(np.log(wave) - float(lwave[centerwaves[ii]])) < 2.*float(lwave[sigmas[ii]])
	fitted_line = float(lwave[scales[ii]])*np.exp(-0.5*pow((np.log(wave_line) - float(lwave[centerwaves[ii]]))/float(lwave[sigmas[ii]]),2.))

	ax=fig.add_subplot(2,4,jj+1)

	ax.step(wave,flam,color=tableau20("Blue"),where="mid")
	ax.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Blue"),alpha=0.5,step="mid")

	ax.plot(wave_line,fitted_line+cont,ls="--",lw=0.5,color="black")
	
	jj += 1

	import pdb; pdb.set_trace()


# [OII] Doublet
#o2_3726 = float(lwave["OII3726_1_scale"])*np.exp(-0.5*pow((np.log(wave) - float(lwave["OII3726_1_centerwave"]))/float(lwave["OII3726_1_sigma"]),2.))
#o2_3728 = float(lwave["OII3728_1_scale"])*np.exp(-0.5*pow((np.log(wave) - float(lwave["OII3728_1_centerwave"]))/float(lwave["OII3728_1_sigma"]),2.))
#ax_O2.plot(wave,o2_3726+cont,ls="--",lw=0.5,color=tableau20("Purple"))
#ax_O2.plot(wave,o2_3728+cont,ls="--",lw=0.5,color=tableau20("Orange"))

##################################################################################################
########							SED PLOT STARTS HERE							    ##########
##################################################################################################


# Plot the Best-Fit SED
ax_SED.plot(cigale["wavelength"]/1e3,cigale["fnu"],color=tableau20("Green"),ls="-",lw=0.5)
ax_SED.plot(pow(10,bagpipes["log_wave"])/1e4,bagpipes["SED_median"]*1e-18*pow(pow(10,bagpipes["log_wave"]),2.)/3e18*1e23*1e3,color=tableau20("Red"),ls="--",lw=0.5)

# Keep only Detections
keep = (obs_ferr > 0.) & (obs_fluxes > 0.)
obs_band_ids = obs_band_ids[keep]


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
ax_SED.errorbar(obs_wave[keep][cfht]/1e4,obs_fluxes[keep][cfht],xerr=obs_fwhm[keep][cfht]/(1e4*2.),yerr=obs_ferr[keep][cfht],\
				ls="none",mec="black",mfc=tableau20("Purple"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)

ax_SED.errorbar(obs_wave[keep][scam_BB]/1e4,obs_fluxes[keep][scam_BB],xerr=obs_fwhm[keep][scam_BB]/(1e4*2.),yerr=obs_ferr[keep][scam_BB],\
				ls="none",mec="black",mfc=tableau20("Blue"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)

ax_SED.errorbar(obs_wave[keep][scam_IB]/1e4,obs_fluxes[keep][scam_IB],xerr=obs_fwhm[keep][scam_IB]/(1e4*2.),yerr=obs_ferr[keep][scam_IB],\
				ls="none",mec="black",mfc=tableau20("Light Blue"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)

ax_SED.errorbar(obs_wave[keep][scam_NB]/1e4,obs_fluxes[keep][scam_NB],xerr=obs_fwhm[keep][scam_NB]/(1e4*2.),yerr=obs_ferr[keep][scam_NB],\
				ls="none",mec="black",mfc=tableau20("Sky Blue"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)


ax_SED.errorbar(obs_wave[keep][hsc]/1e4,obs_fluxes[keep][hsc],xerr=obs_fwhm[keep][hsc]/(1e4*2.),yerr=obs_ferr[keep][hsc],\
				ls="none",mec="black",mfc=tableau20("Green"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)

ax_SED.errorbar(obs_wave[keep][F814W]/1e4,obs_fluxes[keep][F814W],xerr=obs_fwhm[keep][F814W]/(1e4*2.),yerr=obs_ferr[keep][F814W],\
				ls="none",mec="black",mfc=tableau20("Olive"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)

ax_SED.errorbar(obs_wave[keep][F140W]/1e4,obs_fluxes[keep][F140W],xerr=obs_fwhm[keep][F140W]/(1e4*2.),yerr=obs_ferr[keep][F140W],\
				ls="none",mec="black",mfc=tableau20("Orange"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)

ax_SED.errorbar(obs_wave[keep][wircam]/1e4,obs_fluxes[keep][wircam],xerr=obs_fwhm[keep][wircam]/(1e4*2.),yerr=obs_ferr[keep][wircam],\
				ls="none",mec="black",mfc=tableau20("Red"),ecolor=tableau20("Grey"),\
				marker="o",ms=ms,mew=0.2,capsize=1,capthick=1,elinewidth=0.2)


start = 0.95
increment = 0.07
ax_SED.text(0.02,start              ,r"CFHT/MegaCam",color=tableau20("Purple"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
ax_SED.text(0.02,start - increment  ,r"CFHT/WirCam",color=tableau20("Red"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
ax_SED.text(0.02,start - increment*2,r"Subaru/SCam BB",color=tableau20("Blue"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
ax_SED.text(0.02,start - increment*3,r"Subaru/SCam IB",color=tableau20("Light Blue"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
ax_SED.text(0.02,start - increment*4,r"Subaru/SCam NB",color=tableau20("Sky Blue"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
ax_SED.text(0.02,start - increment*5,r"Subaru/HSC BB",color=tableau20("Green"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
ax_SED.text(0.02,start - increment*6,r"HST/ACS F814W",color=tableau20("Olive"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
ax_SED.text(0.02,start - increment*7,r"HST/WFC3 F140W",color=tableau20("Orange"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)



# Highlight the zoomed in region
x0 = ax_1D_Spectra.get_xlim()[0]*1.8275/1e4
width = ax_1D_Spectra.get_xlim()[1]*1.8275/1e4 - x0
y0 = 4e-4
height = 0.2 - y0

rect = Rectangle((x0,y0),width,height,facecolor="none",edgecolor="black",linewidth=0.5,ls="--",zorder=1)
ax_SED.add_patch(rect)
fig.add_artist(Line2D([0.395, 0.124], [0.499, 0.54],lw=0.5,ls="--",color="black"))
fig.add_artist(Line2D([0.507, 0.90], [0.499, 0.54],lw=0.5,ls="--",color="black"))

custom_leg = [Line2D([0], [0], ls="none", marker="o",mfc=tableau20("Grey"),mec="black",label="Observations",mew=0.2),
				Line2D([0], [0], ls="-", color=tableau20("Green"),label=r"\texttt{Cigale}"),
                Line2D([0], [0], ls="--", color=tableau20("Red"),label=r"\texttt{Bagpipes}")]

ax_SED.legend(handles=custom_leg,frameon=False,ncol=1,loc="upper right",fontsize=8)




ax_SED.set_xlabel(r"Observed Wavelength ($\mu$m)",fontsize=8)
ax_SED.set_ylabel(r"$f_\nu$ (mJy)",fontsize=8)

fig.savefig("../../plots/EELG1002_SED.png",format="png",dpi=300,bbox_inches="tight")



