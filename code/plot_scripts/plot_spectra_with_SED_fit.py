import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
from astropy.io import fits
from astropy.visualization import imshow_norm,LinearStretch,ZScaleInterval,HistEqStretch,ManualInterval,ImageNormalize
from matplotlib.image import NonUniformImage
import pdb

import os,sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20

##################################################################################################
########								SOME NEEDED FUNCTIONS						    ##########
##################################################################################################

def gaussian(xx,scale,cwave,sigma):
	return scale/np.sqrt(2.*np.pi*sigma**2.)*np.exp(-0.5*pow((xx-cwave)/sigma,2.))


##################################################################################################
########						PREP WORK TO GET SPECTRA READY FOR PLOTTING			    ##########
##################################################################################################

# Load in the Line Fits
lfits = fits.open("../../data/emline_fits/43158747673038238/1002_lineprops.fits")[1].data
lwave = fits.open("../../data/emline_fits/43158747673038238/1002.fits")[1].data

# Load in the 1D and 2D spectra
spec1d = fits.open("../../data/flux_corr_spectra/43158747673038238/1002.fits")[1].data
spec2d = fits.open("../../data/Science_coadd/spec2d_43158747673038238.fits")

flux_2D = spec2d[1].data - spec2d[3].data # EXT=1 SCIIMG and EXT=2 SKYMODEL
lambda_2D = spec2d[8].data # EXT8 WAVEIMG
ivar_2D = spec2d[2].data #1./(1./data_2D[2].data + 1./data_2D[5].data) #EXT=2 IVARRAW and EXT=5 IVARMODEL

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
########						GET THE SED FITTING BEST RESULTS					    ##########
##################################################################################################

# Cigale
cigale = fits.open("../../data/cigale_results/sfh_delayed_nodust/1002_best_model.fits")[1].data
cigale_results = fits.open("../../data/cigale_results/sfh_delayed_nodust/results.fits")[1].data

cigale_wave = cigale["wavelength"]*10./1.8275
cigale_flam = cigale["fnu"]*1e-26*1.8275*3e18/(cigale["wavelength"]*10.)**2.


##################################################################################################
########						PREPARE THE FIGURE GRID SPACE						    ##########
##################################################################################################

# Define the Figure Size and Grid Space
fig = plt.figure()
fig.set_size_inches(13.3,2.5)#(30,3)
#fig.subplots_adjust(hspace=0.1, wspace=0.1)
gs_top = mpl.gridspec.GridSpec(2, 1,  height_ratios=[1.0, 3.],hspace=0.)
gs_bot = mpl.gridspec.GridSpec(2, 1,  height_ratios=[1.0, 3.],hspace=0.)

# Labels

##################################################################################################
########						NOW DO THE PLOTTING STUFF HERE						    ##########
##################################################################################################


ax1 = fig.add_subplot(gs_bot[0])
ax2 = fig.add_subplot(gs_bot[1])		

ax2.set_xlabel(r'Rest-Frame Wavelength (\AA)',fontsize=10)
ax2.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=10)
ax1.set_yticks([])
ax1.set_xticklabels([])
ax2.minorticks_on()

ax2.tick_params(labelsize=10)

wave = spec1d["OPT_WAVE"]/(1.8275)
flam = spec1d["OPT_FLAM"]*(1.8275) #- np.double(lwave["PL_norm"])*(wave/3000)**(-np.double(lwave["PL_slope"]))
sigm = spec1d["OPT_FLAM_SIG"]*(1.8275)


# Fill Line Areas

# [OII]
ax2.fill_between([3727-7,3727+7],-1,100.,color=tableau20("Green"),alpha=0.2)

# [OIII]
ax2.fill_between([5007-8,5007+8],-1,100.,color=tableau20("Purple"),alpha=0.2)
ax2.fill_between([4959-7,4959+7],-1,100.,color=tableau20("Purple"),alpha=0.2)
ax2.fill_between([4363-5,4363+5],-1,100.,color=tableau20("Purple"),alpha=0.2)

# Balmer lines
ax2.fill_between([4861-5,4861+5],-1,100.,color=tableau20("Red"),alpha=0.2)
ax2.fill_between([4341-5,4341+5],-1,100.,color=tableau20("Red"),alpha=0.2)
ax2.fill_between([4101-5,4101+5],-1,100.,color=tableau20("Red"),alpha=0.2)
ax2.fill_between([3970-5,3970+5],-1,100.,color=tableau20("Red"),alpha=0.2)
ax2.fill_between([3889-5,3889+5],-1,100.,color=tableau20("Red"),alpha=0.2)
ax2.fill_between([3835-5,3835+5],-1,100.,color=tableau20("Red"),alpha=0.2)

# Neon Lines
ax2.fill_between([3869-5,3869+5],-1,100.,color=tableau20("Orange"),alpha=0.2)
ax2.fill_between([3960-5,3960+5],-1,100.,color=tableau20("Orange"),alpha=0.2)


wave_min = 3650 #4300
wave_max = 5100 #4400
these_1D = (wave > wave_min) & (wave < wave_max)
these_2D = (lambda_2D > wave_min) & (lambda_2D < wave_max)

wave = wave[these_1D] 
flam = flam[these_1D]
sigm = sigm[these_1D]

# Plot the 1D spectra Here
ax2.step(wave,flam,where="mid",lw=0.5,zorder=2)
ax2.fill_between(wave,flam-sigm,flam+sigm,color=tableau20("Grey"),alpha=0.3,step="mid")

# Plot the Cigale Spectra
these_cigale = (cigale_wave > wave_min) & (cigale_wave < wave_max)
cigale_wave = cigale_wave[these_cigale]
cigale_flam = cigale_flam[these_cigale]*1e17
ax2.plot(cigale_wave,cigale_flam,color=tableau20("Red"),alpha=0.8,ls="--",lw=1)

# Set the Limits of the Top Axis
ax2.set_ylim(np.min(flam)*0.98,np.max(flam)*1.3)
ax2.set_xlim(np.min(wave[np.isfinite(flam)]),np.max(wave[np.isfinite(flam)]))


# Reset the Limits of the Top Axis
ax2.set_ylim(np.min(flam)*0.98,np.max(flam)*1.1)
ax2.set_xlim(np.min(wave[np.isfinite(flam)]),np.max(wave[np.isfinite(flam)]))

custom_lines = [Line2D([0], [0], color=tableau20("Red"), lw=1, alpha=0.8,ls="--")]


ax2.legend(custom_lines, [r'{\sc Cigale}-best fit ($\log_{10} U = %0.2f\pm%0.2f$, $12+\log_{10}$(O/H)$ = 7.4$)' % (cigale_results["bayes.nebular.logU"],cigale_results["bayes.nebular.logU_err"])],frameon=False,loc="upper left",fontsize=10)

"""
# Plot Each Line Fit
xx = np.arange(3000.,9300.,0.1)
if any("OII372" in line for line in lines[ind]):

	this_1 = lfits["line_ID"] == "OII3728_1"
	this_2 = lfits["line_ID"] == "OII3726_1"
	this_3 = lfits["line_ID"] == "[OII]"

	lcent_1 = np.exp(np.double(lwave["OII3728_1_centerwave"]))
	lcent_2 = np.exp(np.double(lwave["OII3726_1_centerwave"]))
	lcent_3 = 3727.

	sigma_1 = lfits["linesigma_med"][this_1]/3e5*lcent_1
	sigma_2 = lfits["linesigma_med"][this_2]/3e5*lcent_2
	sigma_3 = lfits["linesigma_med"][this_2]/3e5*lcent_2

	model_1 = gaussian(xx,lfits["lineflux_med"][this_1],lcent_1,sigma_1)
	model_2 = gaussian(xx,lfits["lineflux_med"][this_2],lcent_2,sigma_2)
	model_3 = gaussian(xx,lfits["lineflux_med"][this_3],lcent_3,sigma_3)

	ax2.plot(xx,model_1,color=tableau20("Red"),lw=0.7)
	ax2.plot(xx,model_2,color=tableau20("Purple"),lw=0.7)
	ax2.plot(xx,model_3,color=tableau20("Green"),lw=0.7)

	ax2.text(0.025,0.93,labels[0][0],fontsize=10,color=tableau20("Green"),transform=ax2.transAxes,va="center",ha="left")
	ax2.text(0.025,0.82,labels[0][1],fontsize=10,color=tableau20("Purple"),transform=ax2.transAxes,va="center",ha="left")
	ax2.text(0.025,0.71,labels[0][2],fontsize=10,color=tableau20("Red"),transform=ax2.transAxes,va="center",ha="left")

else:
	start = 0.93
	ii = 0
	color = [tableau20("Green"),tableau20("Red"),tableau20("Purple")]
	for ll in lines[ind]:
		this = lfits["line_ID"] == ll

		lcent = np.exp(np.double(lwave[ll+"_centerwave"]))

		sigma = lfits["linesigma_med"][this]/3e5*lcent

		model = gaussian(xx,lfits["lineflux_med"][this],lcent,sigma)

		if ind in (3,4,5):
			ha="right"
			xpos = 0.95
		else:
			ha="left"
			xpos = 0.04

		if ll == "NeIII3968_1":
			ax2.plot(xx,model,color=color[2],lw=0.7)
			model_ne3_3968 = model
			ax2.text(xpos,0.82,labels[ind][ii],fontsize=9,color=color[2],transform=ax2.transAxes,va="center",ha=ha)

		elif ll == "Hep_na_1":
			ax2.plot(xx,model,color=color[1],lw=0.7)
			ax2.plot(xx,model + model_ne3_3968,color=color[0],lw=0.7)

			ax2.text(xpos,0.71,labels[ind][ii],fontsize=9,color=color[1],transform=ax2.transAxes,va="center",ha=ha)
			ax2.text(xpos,0.93,"Combined",fontsize=10,color=color[0],transform=ax2.transAxes,va="center",ha=ha)

		else:
			ax2.plot(xx,model,color=color[ii],lw=0.7)
			ax2.text(xpos,start,labels[ind][ii],fontsize=9,color=color[ii],transform=ax2.transAxes,va="center",ha=ha)

		start = start - 0.11
		ii = ii+1
		#ax2.text(0.025,0.85,labels[1],fontsize=10,color=tableau20("Purple"),transform=ax2.transAxes,va="center",ha="left")
		#ax2.text(0.025,0.77,labels[2],fontsize=10,color=tableau20("Red"),transform=ax2.transAxes,va="center",ha="left")

"""
# Plot the 2D spectra in lower panel
# Start By Preparing the Z-Axis interval and stretch (similar to how we show FITS images in DS9)
norm = ImageNormalize(flux_2D[0],interval=ZScaleInterval(),stretch=LinearStretch())

# Because the Wavelength Calibration has dLambda as non-linear, we need to use the NonUniformImage function rather than imshow (imshow_norm)\
# The latter is with the assumption that the extent is linear from origin to maximum x-extent. This is not true for the DEIMOS images
# which is why we need to use NonUniformImage
im = NonUniformImage(ax1, interpolation='nearest',origin="lower",extent=(np.min(wave),np.max(wave),0,1),cmap="gray",norm=norm)

# Now we define the x-axis for the image using the actual 2D spectra wavelength information
im.set_data(lambda_2D[0], np.arange(0,1,1/flux_2D.shape[0]), flux_2D)

#pdb.set_trace()
# Add the Image and Set Matching X-limits
ax1.add_image(im)
ax1.set_xlim(ax2.get_xlim())
#axes[ind][1].images.append(im)
#axes[ind][1].set_xlim(axes[ind][0].get_xlim())
#plt.show()
# Save the Figure in Outfile
#plt.savefig(self.outfile,format=out_format,dpi=300,bbox_inches="tight")
#fig.clear()
#plt.close(fig)



#pdb.set_trace()
#plt.show()

fig.savefig("../../plots/1002_full_spectra_AAS_%i.png" % (wave_max),format="png",dpi=300,bbox_inches="tight")

#os.system("open ../../plots/1002_spectra.png")