import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from astropy.io import fits
from astropy.visualization import imshow_norm,LinearStretch,ZScaleInterval
from astropy.visualization.wcsaxes import add_scalebar
from astropy.wcs import WCS
import astropy.units as u
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
import numpy as np
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0,"../../../My_Modules/color_lib/")
from colors import tableau20


##################
### Load in the Images
hst = fits.open("../../data/cutouts/ACS/EELG1002_10arcsec_HST_ACS_F814W_sci.fits")[0]
hsc = fits.open("../../data/cutouts/Subaru/HSC/pdr2_dud_150.1346_2.8531_5.00arcsec_i.fits")[1]
galex = fits.open("../../data/cutouts/GALEX/EELG1002_GALEX_NUV.fits")[0]

### Define the WCS
wcs_hst = WCS(hst.header)
wcs_hsc = WCS(hsc.header)
wcs_galex = WCS(galex.header)

### Redefine the WCS to ensure North is pointing up
refined_wcs_hst, shape_out = find_optimal_celestial_wcs([(hst.data, wcs_hst)], auto_rotate=False)
reprojected_hst, _ = reproject_interp((hst.data, wcs_hst), refined_wcs_hst, shape_out=shape_out)

refined_wcs_hsc, shape_out = find_optimal_celestial_wcs([(hsc.data, wcs_hsc)], auto_rotate=False)
reprojected_hsc, _ = reproject_interp((hsc.data, wcs_hsc), refined_wcs_hsc, shape_out=shape_out)

refined_wcs_galex, shape_out = find_optimal_celestial_wcs([(galex.data, wcs_galex)], auto_rotate=False)
reprojected_galex, _ = reproject_interp((galex.data, wcs_galex), refined_wcs_galex, shape_out=shape_out)


##################
# Load in the Mask Design
mask = fits.open("../../data/mask_design/GS2017AFT009-03.fits")[1].data
this = mask["ID"] == 1002
mask = mask[this] # This will only keep EELG1002




####################
### Plotting Starts Here

# Initialize Figure and Set Size
fig = plt.figure()
fig.set_size_inches(4,4)

# Define the Axis
ax = fig.add_subplot(111,projection=refined_wcs_hst)

# Set the format of the ticks to decimal degrees
lon = ax.coords['ra']
lat = ax.coords['dec']
lon.set_major_formatter('d.ddd')
lat.set_major_formatter('d.ddd')

# Define the Label
ax.set_xlabel("Right Ascension (deg)",fontsize=12)
ax.set_ylabel("Declination (deg)",fontsize=12)

# Plot the HST Image
imshow_norm(reprojected_hst,ax,origin="lower",stretch=LinearStretch(),interval=ZScaleInterval(),cmap="Greys")

# Add HSC Contours with Gaussian Smoothing
smoothed_hsc = gaussian_filter(reprojected_hsc, sigma=1.2)
contour_levels = [0.025,0.0445033,0.0717549,0.5,np.max(reprojected_hsc)*0.9]
ax.contour(smoothed_hsc, levels=contour_levels, transform=ax.get_transform(refined_wcs_hsc), colors='darkred', alpha=0.6)


# Add GALEX Contours with Gaussian Smoothing
smoothed_galex = gaussian_filter(reprojected_galex, sigma=1.2)
contour_levels = [0.004,0.00425,0.0045,0.00475,0.005]
ax.contour(smoothed_galex, levels=contour_levels, transform=ax.get_transform(refined_wcs_galex), colors='purple', alpha=0.6)


##########################
### Plotting the GMOS Slit

# Slit Info
ra_center = mask["RA"]*15. # hours to degrees
dec_center = mask["DEC"] # degrees
width_deg = mask["slitsize_x"]/3600. # arcsec to degrees
height_deg = mask["slitsize_y"]/3600. # arcsec to degrees
pa_deg = 136.17185 # degrees; Taken from the Raw Science Images (Header File --> CRPA)

# Get Slit Position in Physical Space
x_center, y_center = refined_wcs_hst.world_to_pixel_values(ra_center,dec_center)

# Calculate width and height in pixels
x_width, _ = refined_wcs_hst.world_to_pixel_values(ra_center + width_deg, dec_center)
_, y_height = refined_wcs_hst.world_to_pixel_values(ra_center, dec_center + height_deg)
width_pix = abs(x_width - x_center)
height_pix = abs(y_height - y_center)

# Create the rectangle centered at (x_center, y_center)
slit = Rectangle((x_center - width_pix / 2, y_center - height_pix / 2), width_pix, height_pix, 
                 linewidth=1.5, edgecolor="darkblue", facecolor="none", transform=ax.transData)

# Apply rotation using Affine2D transformation
t2 = mpl.transforms.Affine2D().rotate_deg_around(x_center, y_center, pa_deg) + ax.transData
slit.set_transform(t2)

# Add the rectangle to the plot
ax.add_patch(slit)




###############################
### Finalize with label setting

# Add a Frame Background
background = FancyBboxPatch((0.90,0.99),0.99-0.2,0.99-0.2,facecolor="white",alpha=0.8,edgecolor="black",transform=ax.transAxes)
ax.add_patch(background)

# Add in the Labels for the Data Used
ax.text(0.98,0.96,r"\textbf{GMOS Slit ($0.5''$)}",color="darkblue",ha="right",va="top",fontsize=8,transform=ax.transAxes)
ax.text(0.98,0.90,r"\textbf{\textit{HST}/ACS F814W}",color="black",ha="right",va="top",fontsize=8,transform=ax.transAxes)
ax.text(0.98,0.84,r"\textbf{Subaru/HSC $i$}",color="darkred",ha="right",va="top",fontsize=8,transform=ax.transAxes)
ax.text(0.98,0.78,r"\textbf{\textit{GALEX} NUV}",color="purple",ha="right",va="top",fontsize=8,transform=ax.transAxes)

# Add in annotation for which source is which
ax.annotate(r'\textbf{EELG1002}' + '\n' + r'$\bm{z_{s} = 0.8275}$', 
             xy=(0.52, 0.48),
             xycoords=ax.transAxes,
             xytext=(0.8, 0.3),
             textcoords=ax.transAxes,
             ha="center",
             va="center",
             arrowprops=dict(arrowstyle= '-|>',
                             color='black',
                             lw=1.0,
                             ls='--')
           )

ax.annotate(r'\textbf{COS20-1502912}' + '\n' + r'$\bm{z_{p} \sim 1.5}$', 
             xy=(0.25, 0.48),
             xycoords=ax.transAxes,
             xytext=(0.18, 0.73),
             textcoords=ax.transAxes,
             ha="center",
             va="center",
             fontsize=8,
             arrowprops=dict(arrowstyle= '-|>',
                             color='black',
                             lw=1.0,
                             ls='--')
           )

ax.annotate(r'\textbf{COS20-1501918}' + '\n' + r'$\bm{z_{p} \sim 0.7}$', 
             xy=(0.45, 0.15),
             xycoords=ax.transAxes,
             xytext=(0.65, 0.05),
             textcoords=ax.transAxes,
             ha="center",
             va="center",
             fontsize=8,
             arrowprops=dict(arrowstyle= '-|>',
                             color='black',
                             lw=1.0,
                             ls='--')
           )

# Add a Scale Bar set to 1 arcsecond
add_scalebar(ax=ax,length=1.*u.arcsec,label=r"$1''$",corner="bottom right",frame=False)




########################
### Write out the figure
fig.savefig("../../plots/slit_pos.png",format="png",dpi=300,bbox_inches="tight")
