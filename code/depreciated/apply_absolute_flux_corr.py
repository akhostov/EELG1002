import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.integrate import simps
from scipy.interpolate import interp1d
import os
import pdb


from astropy.convolution import Gaussian2DKernel,convolve
FWHM_ground = 0.9/0.03 #0.9'' seeing with a 0.03''/pix scale
FWHM_F814W = 0.09/0.03 # arcsec
FWHM = np.sqrt(FWHM_ground**2. - FWHM_F814W**2.)
gaussian_2D_kernel = Gaussian2DKernel(FWHM/2.634) #0.9'' seeing with a 0.03''/pix scale

"""
plt.imshow(gaussian_2D_kernel, interpolation='none', origin='lower')
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.colorbar()
plt.show()
"""

# Load the image
data = fits.open("../data/cutouts/ACS/1002_HST_ACS_F814W_10arcsec_unrot_sci.fits")
wht = fits.open("../data/cutouts/ACS/1002_HST_ACS_F814W_10arcsec_unrot_RMS.fits")


# Now let's do Aperture Photometry with Sextractor
os.system("cd ../data/cutouts/ACS/; sex 1002_HST_ACS_F814W_10arcsec_unrot_sci.fits -c original_HST.sex")

original_flux = pow(10,-0.4*(fits.open("../data/cutouts/ACS/original_HST.fits")[1].data["MAG_APER"]+48.6))*1e6*1e23


# Range of the GMOS Slit
dx = (168-round(0.25/0.03),168+round(0.25/0.03))
dy = (168-round(2.5/0.03),168+round(2.5/0.03))

# Smooth out the images
data[0].data = convolve(data[0].data, gaussian_2D_kernel)
data.writeto("../data/cutouts/ACS/1002_HST_ACS_F814W_10arcsec_unrot_sci_ground_PSF.fits",overwrite=True)

# Calculate Original Flux
mag_new = -2.5*np.log10(np.sum(data[0].data[dx[0]:dx[1],dy[0]:dy[1]])) + 25.936
flux_new = pow(10,-0.4*(mag_new+48.6))*1e6*1e23


corr_factor = 1. - flux_new/original_flux


print(f"Original Flux: {original_flux} uJy")
print(f"Smoothed Flux: {flux_new} uJy")
print(f"Correction: {corr_factor}")


# Load in the Spectra
spec1d = fits.open("../data/Science_coadd/spec1d_43158747673038238.fits")
hdr = spec1d[5].header
spec1d = spec1d[5].data

spec1d["OPT_FLAM"] = spec1d["OPT_FLAM"]*(1.+corr_factor)
spec1d["OPT_FLAM_SIG"] = spec1d["OPT_FLAM_SIG"]*(1.+corr_factor)



table_hdu = fits.BinTableHDU(spec1d)
table_hdu.header = hdr

new_hdulist = fits.HDUList([fits.PrimaryHDU(), table_hdu])
new_hdulist.writeto("../data/flux_corr_spectra/43158747673038238/1002.fits", overwrite=True)

exit()

"""

pdb.set_trace()

wht[0].data = convolve(wht[0].data, gaussian_2D_kernel)
wht.writeto("../data/cutouts/ACS/1002_HST_ACS_F814W_10arcsec_unrot_RMS_ground_PSF.fits",overwrite=True)









# Now let's do Aperture Photometry with Sextractor
os.system("cd ../data/cutouts/ACS/; sex 1002_HST_ACS_F814W_10arcsec_unrot_sci.fits -c original_HST.sex")
os.system("cd ../data/cutouts/ACS/; sex 1002_HST_ACS_F814W_10arcsec_unrot_sci_ground_PSF.fits -c smoothed_ground_PSF.sex")


original = fits.open("../data/cutouts/ACS/original_HST.fits")[1].data
smoothed = fits.open("../data/cutouts/ACS/smoothed_ground_HST.fits")[1].data


original_flux = pow(10,-0.4*(original["MAG_APER"]+48.6))*1e6*1e23
smoothed_flux = pow(10,-0.4*(smoothed["MAG_APER"]+48.6))*1e6*1e23


corr_factor = original_flux/smoothed_flux

print(f"Original Flux: {original_flux} uJy")
print(f"Smoothed Flux: {smoothed_flux} uJy")
print(f"Correction: {corr_factor}")



# Load in the Spectra
spec1d = fits.open("../data/Science_coadd/spec1d_43158747673038238.fits")
hdr = spec1d[5].header
spec1d = spec1d[5].data

spec1d["OPT_FLAM"] = spec1d["OPT_FLAM"]*corr_factor
spec1d["OPT_FLAM_SIG"] = spec1d["OPT_FLAM_SIG"]*corr_factor



table_hdu = fits.BinTableHDU(spec1d)
table_hdu.header = hdr

new_hdulist = fits.HDUList([fits.PrimaryHDU(), table_hdu])
new_hdulist.writeto("../data/flux_corr_spectra/43158747673038238/1002.fits", overwrite=True)



"""

"""

sys.path.append('/home/fitsfiles')     #not sure if this does anything/is correct

def sex(image, output, sexdir='/home/sextractor-2.5.0', check_img=None,config=None, l=None) : 
  '''Construct a sextractor command and run it.'''
  #creates a sextractor line e.g sex img.fits -catalog_name -checkimage_name
  q="/home/fitsfiles/"+ "01" +".fits"
  com = [ "sex ", q, " -CATALOG_NAME " + output]
  s0=''
  com = s0.join(com)
  res = os.system(com)
  return res

img_name=sys.argv[0]
output=img_name[0:1]+'_star_catalog.fits'
t=sex(img_name,output)

print '----done !---'




for arg in len(sys.argv):
    filename = arg.split('/')[-1].strip('.fits')
    t = sex(arg, filename +'_star_catalog.fits')
    # Whatever else
"""








def get_fnu(spec1d,filt):
	wave_z,trans_z = np.loadtxt(filt,unpack=True)
	fx = interp1d(wave_z,trans_z,fill_value=0.,bounds_error=False)
	pdb.set_trace()
	return (simps(spec1d["OPT_FLAM"]*1e-17*fx(spec1d["OPT_WAVE"])*spec1d["OPT_WAVE"],x=spec1d["OPT_WAVE"])/simps(fx(spec1d["OPT_WAVE"])/spec1d["OPT_WAVE"],x=spec1d["OPT_WAVE"])*1/3e18*1e23*1e6,\
			simps(spec1d["OPT_WAVE"]*fx(spec1d["OPT_WAVE"]),x=spec1d["OPT_WAVE"])/simps(fx(spec1d["OPT_WAVE"]),x=spec1d["OPT_WAVE"]))

# Filter out zeros
these  = spec1d["OPT_WAVE"] > 0.
spec1d = spec1d[these]

path = "../../Filters/Subaru"
fnu,fwave = get_fnu(spec1d,f"{path}/Subaru_HSC.z.dat")

cosmos = fits.open("../data/catalogs/COSMOS2020_Classical_GMOS_OIII_EELG.fits")[1].data

# Filter the object of interest
this = cosmos["ID"] == "1002"
fnu_z = cosmos["HSC_z_FLUX_AUTO"][this]

pdb.set_trace()
"""
# Load in the Spectra
spec1d = fits.open("../data/Science_coadd/spec1d_43158747673038238.fits")
hdr = spec1d[5].header
spec1d = spec1d[5].data

# Filter out zeros
these  = spec1d["OPT_WAVE"] > 0.
spec1d = spec1d[these]

# Load in the catalog
cosmos = fits.open("../data/catalogs/COSMOS2020_Classical_GMOS_OIII_EELG.fits")[1].data

# Filter the object of interest
this = cosmos["ID"] == "1002"

# Now get the BBs




# Run Correction Measurement for Subaru SuprimeCam
path = "../../Filters/Subaru"
filters = [(f"{path}/IA827.SuprimeCam.txt","SC_IB827_FLUX_AUTO","SC_IB827_FLUXERR_AUTO"),
			(f"{path}/IA767.SuprimeCam.txt","SC_IA767_FLUX_AUTO","SC_IA767_FLUXERR_AUTO"),
			(f"{path}/IA738.SuprimeCam.txt","SC_IA738_FLUX_AUTO","SC_IA738_FLUXERR_AUTO"),
			(f"{path}/IA709.SuprimeCam.txt","SC_IB709_FLUX_AUTO","SC_IB709_FLUXERR_AUTO"),
			(f"{path}/IA679.SuprimeCam.txt","SC_IA679_FLUX_AUTO","SC_IA679_FLUXERR_AUTO"),
			(f"{path}/IA624.SuprimeCam.txt","SC_IA624_FLUX_AUTO","SC_IA624_FLUXERR_AUTO"),
			(f"{path}/IA574.SuprimeCam.txt","SC_IB574_FLUX_AUTO","SC_IB574_FLUXERR_AUTO"),
			(f"{path}/Subaru_NB816.txt","SC_NB816_FLUX_AUTO","SC_NB816_FLUXERR_AUTO"),
			(f"{path}/Subaru_NB711.txt","SC_NB711_FLUX_AUTO","SC_NB711_FLUXERR_AUTO"),
			(f"{path}/Subaru-Suprime.ip.dat","SC_ip_FLUX_AUTO","SC_ip_FLUXERR_AUTO"),
			(f"{path}/Subaru-Suprime.zp.dat","SC_zp_FLUX_AUTO","SC_zp_FLUXERR_AUTO"),
			(f"{path}/Subaru-Suprime.z_FD.dat","SC_zpp_FLUX_AUTO","SC_zpp_FLUXERR_AUTO"),
			(f"{path}/Subaru-Suprime.rp.dat","SC_rp_FLUX_AUTO","SC_rp_FLUXERR_AUTO")]

corr,corr_err,fwave_all = [],[],[]
for ii in range(len(np.transpose(filters)[0])):

	fnu,fwave = get_fnu(spec1d,filters[ii][0])
	fwave_all.append(fwave)
	corr.append(cosmos[filters[ii][1]][this]/fnu)
	corr_err.append(cosmos[filters[ii][2]][this]/fnu)

# Run Correction Measurement for Subaru HyperSuprimeCam
path = "../../Filters/Subaru/HSC"
filters = [(f"{path}/Subaru_HSC.z.dat","HSC_z_FLUX_AUTO","HSC_z_FLUXERR_AUTO"),
			(f"{path}/Subaru_HSC.i.dat","HSC_i_FLUX_AUTO","HSC_i_FLUXERR_AUTO"),
			(f"{path}/Subaru_HSC.r.dat","HSC_r_FLUX_AUTO","HSC_r_FLUXERR_AUTO")]

for ii in range(len(np.transpose(filters)[0])):

	fnu,fwave = get_fnu(spec1d,filters[ii][0])
	fwave_all.append(fwave)
	corr.append(cosmos[filters[ii][1]][this]/fnu)
	corr_err.append(cosmos[filters[ii][2]][this]/fnu)

deg=2
params = np.polyfit(np.asarray(fwave_all),np.asarray(corr),deg)#,w=1./np.hstack(corr_err)**2.)

spec1d["OPT_FLAM"] = spec1d["OPT_FLAM"]*np.polyval(params,spec1d["OPT_WAVE"])
spec1d["OPT_FLAM_SIG"] = spec1d["OPT_FLAM_SIG"]*np.polyval(params,spec1d["OPT_WAVE"])


plt.errorbar(np.hstack(fwave_all),np.hstack(corr),yerr=np.hstack(corr_err),ls="None",marker="o")

xx = np.arange(6000,9000,0.1)
plt.plot(xx,np.polyval(params,xx))
plt.show()

keep = (spec1d["OPT_WAVE"] > 6000.) & (spec1d["OPT_WAVE"]< 9500.)
plt.plot(spec1d["OPT_WAVE"][keep],spec1d["OPT_FLAM"][keep]*np.polyval(params,spec1d["OPT_WAVE"][keep]))
plt.show()




table_hdu = fits.BinTableHDU(spec1d)
table_hdu.header = hdr

new_hdulist = fits.HDUList([fits.PrimaryHDU(), table_hdu])
new_hdulist.writeto("../data/flux_corr_spectra/43158747673038238/1002.fits", overwrite=True)

pdb.set_trace()
"""
