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
########						PREP WORK TO GET SPECTRA READY FOR PLOTTING			    ##########
##################################################################################################

# Load in the best-fit SED
cigale = fits.open("../../data/SED_results/cigale_results/1002_best_model.fits")[1].data
bagpipes = fits.open("../../data/SED_results/bagpipes_results/best_fit_SED_sfh_continuity_spec_BPASS.fits")[1].data

# Load in the results that include best-model fits
model_phot = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data

# Load in the observations
obs = fits.open("../../data/SED_results/cigale_results/observations.fits")[1].data

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
obs_fluxes = np.asarray([obs[ii] for ii in filters]).T[0]
obs_ferr = np.asarray([obs[ii+"_err"] for ii in filters]).T[0]
obs_band_ids = lib_wave[0][match_ind]

results_fluxes = np.asarray([model_phot["bayes."+ii] for ii in filters]).T[0]
results_ferr = np.asarray([model_phot["bayes."+ii+"_err"] for ii in filters]).T[0]



##################################################################################################
########						PREPARE THE FIGURE GRID SPACE						    ##########
##################################################################################################

# Define the Figure Size and Grid Space
fig = plt.figure()
fig.set_size_inches(7,5)

plt.subplots_adjust(hspace=0.05)

outer = gridspec.GridSpec(2, 1, height_ratios = [4.5, 5.5]) 

gs_top = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[2.5,0.5],subplot_spec=outer[0],hspace=0.0)
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])


ax_SED = fig.add_subplot(gs_bot[0])
ax_2D_Spectra = fig.add_subplot(gs_top[1])
ax_1D_Spectra = fig.add_subplot(gs_top[0])

# Set the xlabels in the 1D spectra on top
ax_1D_Spectra.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


# Define the Scaling
ax_SED.set_xscale("log")
ax_SED.set_yscale("log")

# Set the Limits
ax_SED.set_ylim(3e-4,0.3)
ax_SED.set_xlim(0.3,3.0)

# Set Tick Formating
ax_SED.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
ax_SED.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax_SED.set_xticks([0.3,0.4,0.5,0.6,0.8,1,2,3])
ax_SED.set_xticks([0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.25,2.5,2.75,3],minor=True)
ax_SED.set_xticklabels(["0.3","0.4","0.5","0.6","0.8","1","2","3"])


##################################################################################################
########						PREP WORK TO GET SPECTRA READY FOR PLOTTING			    ##########
##################################################################################################

# Load in the Line Fits
lfits = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data
lwave = Table.read("../../data/emline_fits/1002.fits")

# Load in the 1D and 2D spectra
spec1d = fits.open("../../data/flux_corr_spectra/43158747673038238/1002.fits")[1].data
spec2d = fits.open("../../data/Science_coadd/spec2d_43158747673038238.fits")

flux_2D = spec2d[1].data - spec2d[3].data # EXT=1 SCIIMG and EXT=3 SKYMODEL
lambda_2D = spec2d[8].data # EXT8 WAVEIMG
ivar_2D = spec2d[2].data #EXT=2 IVARRAW 

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


ax_1D_Spectra.set_xlabel(r'Rest-Frame Wavelength (\AA)',fontsize=8)
ax_1D_Spectra.xaxis.set_label_position('top') 
ax_1D_Spectra.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=8)
ax_2D_Spectra.set_yticks([])
ax_2D_Spectra.set_xticklabels([])
ax_1D_Spectra.minorticks_on()

ax_1D_Spectra.tick_params(labelsize=8)

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

# Plot the 1D spectra Here
ax_1D_Spectra.step(wave,flam,where="mid",lw=0.5,zorder=2,color=tableau20("Blue"))
ax_1D_Spectra.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Light Grey"),alpha=0.8,step="mid")

# Set the Limits of the Top Axis
ax_1D_Spectra.set_ylim(np.min(flam)*0.98,np.max(flam)*1.3)
ax_1D_Spectra.set_xlim(np.min(wave[np.isfinite(flam)]),np.max(wave[np.isfinite(flam)]))

# Reset the Limits of the Top Axis
ax_1D_Spectra.set_ylim(np.min(flam)*0.98,np.max(flam)*1.1)
ax_1D_Spectra.set_xlim(np.min(wave[np.isfinite(flam)]),np.max(wave[np.isfinite(flam)]))

# Zoom-in of the weaker lines
left, bottom, width, height = [0.20, 0.70, 0.5, 0.16]
ax_O2 = fig.add_axes([left, bottom, width, height])
ax_O2.tick_params(labelsize=5)
ax_O2.set_xlabel(r"Rest-Frame Wavelength (\AA)",fontsize=5)
ax_O2.set_ylabel(r'$f_\lambda$ (same units)',fontsize=5)

keep = (wave > 3700.) & (wave < 4400.)
ax_O2.step(wave[keep],flam[keep],where="mid",lw=0.5,zorder=2)
ax_O2.fill_between(wave[keep],flam[keep]-sigm[keep],flam[keep]+sigm[keep],color=tableau20("Light Grey"),alpha=0.8,step="mid")

ax_O2.set_xlim(3700.,4400.)
ax_O2.set_ylim(-0.2,1.8)

# Highlight the zoomed in region
rect, lines = ax_1D_Spectra.indicate_inset_zoom(ax_O2,edgecolor="black",linewidth=0.5,ls="--",facecolor="none")
for line in lines: 
	line.set_linestyle("--")
	line.set_linewidth(0.5)
	line.set_zorder(1)


################ Add the line labels
ax_O2.plot([3727.,3727.],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(3727+18.,1.65,r'[O{\sc ii}]d',color="black",ha="right",va="top",fontsize=5,rotation=90)

# H9
ax_O2.plot([3835.4,3835.4],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(3835.4 - 7.,1.65,r'H9',color="black",ha="right",va="top",fontsize=5,rotation=90)

# [NeIII]
ax_O2.plot([3869.,3869.],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(3869-7.,1.65,r'[Ne{\sc iii}]3869',color="black",ha="right",va="top",fontsize=5,rotation=90)
		
# H8
ax_O2.plot([3889.0,3889.0],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(3889.0 + 18.,1.65,r'H8',color="black",ha="right",va="top",fontsize=5,rotation=90)

# [NeIII]
ax_O2.plot([3967.5,3967.5],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(3967.5-7.,1.65,r'[Ne{\sc iii}]3968',color="black",ha="right",va="top",fontsize=5,rotation=90)

# Hepsilon
ax_O2.plot([3970.,3970.],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(3970+18.,1.65,r'H$\epsilon$',color="black",ha="right",va="top",fontsize=5,rotation=90)

# Hdelta
ax_O2.plot([4101.,4101.],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(4101-7.,1.65,r'H$\delta$',color="black",ha="right",va="top",fontsize=5,rotation=90)

# Hgamma
ax_O2.plot([4341.,4341.],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(4337,1.65,r'H$\gamma$',color="black",ha="right",va="top",fontsize=5,rotation=90)

# [OIII]4363
ax_O2.plot([4363.,4363.],ax_O2.get_ylim(),color="black",lw=0.5,ls=":")
ax_O2.text(4363+18.,1.65,r'[O{\sc iii}]4363',color="black",ha="right",va="top",fontsize=5,rotation=90)

# Hbeta
ax_1D_Spectra.plot([4861.,4861.],ax_1D_Spectra.get_ylim(),color="black",lw=0.5,ls=":")
ax_1D_Spectra.text(4861-7.,16.,r'H$\beta$',color="black",ha="right",va="top",fontsize=5,rotation=90)

# [OIII]4959
ax_1D_Spectra.plot([4959.,4959.],ax_1D_Spectra.get_ylim(),color="black",lw=0.5,ls=":")
ax_1D_Spectra.text(4959-7.,16.,r'[O{\sc iii}]4959',color="black",ha="right",va="top",fontsize=5,rotation=90)

# [OIII]5007
ax_1D_Spectra.plot([5007.,5007.],ax_1D_Spectra.get_ylim(),color="black",lw=0.5,ls=":")
ax_1D_Spectra.text(5007.-7.,16.,r'[O{\sc iii}]5007',color="black",ha="right",va="top",fontsize=5,rotation=90)
################ Finished adding the line labels


# Plot the 2D spectra in lower panel
# Start By Preparing the Z-Axis interval and stretch (similar to how we show FITS images in DS9)
norm = ImageNormalize(flux_2D[0],interval=ZScaleInterval(),stretch=LinearStretch())

# Because the Wavelength Calibration has dLambda as non-linear, we need to use the NonUniformImage function rather than imshow (imshow_norm)\
# The latter is with the assumption that the extent is linear from origin to maximum x-extent. This is not true for the DEIMOS images
# which is why we need to use NonUniformImage
im = NonUniformImage(ax_2D_Spectra, interpolation='nearest',origin="lower",extent=(np.min(wave),np.max(wave),0,1),cmap="gray",norm=norm)

# Now we define the x-axis for the image using the actual 2D spectra wavelength information
im.set_data(lambda_2D[0], np.arange(0,1,1/flux_2D.shape[0]), flux_2D)


# Add the Image and Set Matching X-limits
ax_2D_Spectra.add_image(im)
ax_2D_Spectra.set_xlim(ax_1D_Spectra.get_xlim())


##################################################################################################
########						LINE FIT PLOTTING STARTS HERE						    ##########
##################################################################################################

# Plot [OIII]
scales = [ll for ll in lwave.colnames if ("_scale" in ll) and ("_err" not in ll)]
centerwaves = [ll for ll in lwave.colnames if ("_centerwave" in ll) and ("_err" not in ll)]
sigmas = [ll for ll in lwave.colnames if ("_sigma" in ll) and ("_err" not in ll) and ("_br_" not in ll)]

fitted_lines = 0
cont = float(lwave["PL_norm"])*(wave/3000.)**(-1.*float(lwave["PL_slope"])) #NOTE: The continuum is really really faint and uncertainties are large on PL_norm and PL_slope. Not suitable for EW measurements
for ii in range(len(scales)):
	fitted_lines += float(lwave[scales[ii]])*np.exp(-0.5*pow((np.log(wave) - float(lwave[centerwaves[ii]]))/float(lwave[sigmas[ii]]),2.))

ax_1D_Spectra.plot(wave,fitted_lines+cont,ls="--",lw=0.5,color="black")
ax_O2.plot(wave,fitted_lines+cont,ls="--",lw=0.5,color="black")


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
ax_SED.text(0.02,start - increment  ,r"CFHT/WIRCam",color=tableau20("Red"),ha="left",va="center",fontsize=7,transform=ax_SED.transAxes)
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



