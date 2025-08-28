import numpy as np 
import bagpipes as pipes
from astropy.io import fits

def load_obs(ID):

    # load up the relevant columns from the catalogue.
    cat = np.loadtxt("/project/rlsa239_uksr/ali_ahmad/EELG1002/data/catalogs/EELG_OIII_GMOS_BAGPIPES.txt", delimiter=" ")

    ID = cat[0]
    redshift = cat[1]

    # Extract the object we want from the catalogue.
    fluxes = cat[2::2]*1e3 #uJy
    fluxerrs = cat[3::2]*1e3 #uJy

    # Turn these into a 2D array.
    photometry = np.c_[fluxes, fluxerrs]

    # blow up the errors associated with any missing fluxes.
    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0) or (np.isnan(photometry[i,0])):
            photometry[i,:] = [0., 9.9*10**99.]

    # Add 10% Uncertainty to all Flux Measurements in Quadrature
    for i in range(len(photometry)):
        photometry[i, 1] = np.sqrt( photometry[i, 1]**2. + (0.1*photometry[i, 0])**2. )

    return photometry

def bin(spectrum, binn):
    """ Bins up two or three column spectral data by a specified factor. """

    binn = int(binn)
    nbins = len(spectrum) // binn
    binspec = np.zeros((nbins, spectrum.shape[1]))

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]
        binspec[i, 0] = np.mean(spec_slice[:, 0])
        binspec[i, 1] = np.mean(spec_slice[:, 1])

        if spectrum.shape[1] == 3:
            binspec[i,2] = (1./float(binn)
                            *np.sqrt(np.sum(spec_slice[:, 2]**2)))

    return binspec

def load_spectrum(ID):

    hdulist = fits.open("/project/rlsa239_uksr/ali_ahmad/EELG1002/data/flux_corr_spectra/EELG1002_1dspec_slitloss_corrected.fits")

    spectrum = np.c_[hdulist[1].data["OPT_WAVE"],
                     hdulist[1].data["OPT_FLAM"]*1e-17,
                     hdulist[1].data["OPT_FLAM_SIG"]*1e-17]

    mask = (spectrum[:,0] > 6700.) & (spectrum[:,0] < 9300.) 

    return spectrum[mask]
    #return bin(spectrum[mask],2)


def load_both(ID):
    
    photometry = load_obs(ID)
    spectrum = load_spectrum(ID)

    return spectrum, photometry

def save_SFH(fit,outname):

    fit.posterior.get_advanced_quantities()

    # Get Redshift
    redshift = fit.fitted_model.model_components["redshift"]

    # Age of Universe
    age_of_universe = np.interp(redshift, pipes.utils.z_array, pipes.utils.age_at_z)

    # Calculate median and confidence interval for SFH posterior
    post = np.percentile(fit.posterior.samples["sfh"], (16, 50, 84), axis=0).T
    SFH_1sig_low,SFH_median,SFH_1sig_upp = post[:,0],post[:,1],post[:,2]

    # Get the SFH
    time = (age_of_universe - fit.posterior.sfh.ages*10**-9) #note the model starts at 1 Myr age!
    
    # Append a time of 0 Myr to the beginning
    time = np.insert(time, 0, time[0]+1e-3)
    SFH_median = np.insert(SFH_median, 0, SFH_median[0])
    SFH_1sig_low = np.insert(SFH_1sig_low, 0, SFH_1sig_low[0])
    SFH_1sig_upp = np.insert(SFH_1sig_upp, 0, SFH_1sig_upp[0])

    col_time = fits.Column(name='Lookback Time', array=time, format='D',unit="Gyr")
    col_median = fits.Column(name='SFH_median', array=SFH_median, format='D', unit="Msol yr-1")
    col_low = fits.Column(name='SFH_1sig_low', array=SFH_1sig_low, format='D', unit="Msol yr-1")
    col_upp = fits.Column(name='SFH_1sig_upp', array=SFH_1sig_upp, format='D', unit="Msol yr-1")

    t = fits.BinTableHDU.from_columns([col_time,col_median,col_low,col_upp])
    t.writeto(outname,overwrite=True)


def save_SED(fit,outname):

    fit.posterior.get_advanced_quantities()

    mask = (fit.galaxy.photometry[:, 1] > 0.)
    upper_lims = fit.galaxy.photometry[:, 1] + fit.galaxy.photometry[:, 2]
    ymax = 1.05*np.max(upper_lims[mask])


    y_scale = float(int(np.log10(ymax))-1)

    redshift = fit.fitted_model.model_components["redshift"]

    # Get the posterior photometry and full spectrum.
    log_wavs = np.log10(fit.posterior.model_galaxy.wavelengths*(1.+redshift))
    log_eff_wavs = np.log10(fit.galaxy.filter_set.eff_wavs)

    spec_post = np.percentile(fit.posterior.samples["spectrum_full"],
                                (2.5, 16, 50, 84, 97.5), axis=0)*10**-y_scale

    phot_post = np.percentile(fit.posterior.samples["photometry"],
                              (2.5, 16, 50, 84, 97.5), axis=0)

    col_wave = fits.Column(name='log_wave', array=log_wavs, format='D')
    col_median = fits.Column(name='SED_median', array=spec_post[2], format='D')
    col_1low = fits.Column(name='SED_1sig_low', array=spec_post[2] - spec_post[1], format='D')
    col_1upp = fits.Column(name='SED_1sig_upp', array=spec_post[3] - spec_post[2], format='D')
    col_2low = fits.Column(name='SED_2sig_low', array=spec_post[2] - spec_post[0], format='D')
    col_2upp = fits.Column(name='SED_2sig_upp', array=spec_post[4] - spec_post[2], format='D')
    t = fits.BinTableHDU.from_columns([col_wave,col_median,col_1low,col_1upp,col_2low,col_2upp])
    t.writeto(outname,overwrite=True)



###########################################################################
#####                          FITTING PARAMETERS                     #####
###########################################################################
fit_info = {}                            # The fit instructions dictionary

###########     GENERAL PROPERTY
fit_info["t_bc"] = 0.01 # Gyr
fit_info["redshift"] = 0.8275
fit_info["veldisp"] = (1., 1000.)   #km/s


###########     STAR FORMATION HISTORY
# Nonparamteric SFH similar to Leja et al. (2019)
continuity = {}
continuity["massformed"] = (1., 13.)
continuity["metallicity"] = (0.0005,2.) # ZSOL
continuity["metallicity_prior"] = "log_10"
continuity["bin_edges"] = [0., 3., 10., 30., 100., 300.,
                           1000., 3000., 6000.]

for i in range(1, len(continuity["bin_edges"])-1):
    continuity["dsfr" + str(i)] = (-10., 10.)
    continuity["dsfr" + str(i) + "_prior"] = "student_t"

fit_info["continuity"] = continuity 


###########     DUST COMPONENT
# Dust Attenuation Parameters
dust = {}                           
dust["type"] = "Calzetti"
dust["Av"] = 0. # Hbeta/Hgamma ratio suggests no dust attenuation
dust["eta"] = 1.0
dust["delta"] = 0.0


# Dust emission parameters
dust["qpah"] = 2.5         # PAH mass fraction
dust["umin"] = 1.0          # Lower limit of starlight intensity distribution
dust["gamma"] = 0.1       # Fraction of stars at umin

fit_info["dust"] = dust


###########     NEBULAR COMPONENT
nebular = {}
nebular["logU"] = (-4.,0.0)
nebular["logU_prior"] = "uniform"
nebular["metallicity"] = pow(10,7.522-8.71)  # fixed to direct T_e measurement (using Zsol based on Andres & Grevesse 1989; needed for Bagpipes)

fit_info["nebular"] = nebular


###########     SPECTRAL FITTING AND RECALIBRATION
calib = {}
calib["type"] = "polynomial_bayesian"

calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
calib["0_prior"] = "Gaussian"
calib["0_prior_mu"] = 1.0
calib["0_prior_sigma"] = 0.10

calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
calib["1_prior"] = "Gaussian"
calib["1_prior_mu"] = 0.
calib["1_prior_sigma"] = 0.10 

calib["2"] = (-0.5, 0.5)
calib["2_prior"] = "Gaussian"
calib["2_prior_mu"] = 0.
calib["2_prior_sigma"] = 0.10

fit_info["calib"] = calib


mlpoly = {}
mlpoly["type"] = "polynomial_max_like"
mlpoly["order"] = 2
fit_info["mlpoly"] = mlpoly


noise = {}
noise["type"] = "white_scaled"
noise["scaling"] = (1., 10.)
noise["scaling_prior"] = "log_10"
fit_info["noise"] = noise




###########################################################################
#####                    ACTUAL SED FITTING PROCESS                   #####
###########################################################################
filt_list = np.loadtxt("filters/filt_list_wo_GALEX_and_Spitzer.txt", dtype="str")
galaxy = pipes.galaxy("1002", load_both, filt_list=filt_list,phot_units="mujy",photometry_exists=True,spectrum_exists=True)

fit = pipes.fit(galaxy, fit_info, run="sfh_continuity_spec_BPASS")
#fit.fit(verbose=False,n_live=2000,use_MPI=True)

# make diagnostic plots
fig = fit.plot_1d_posterior(save=True, show=False)
# fig = fit.plot_spectrum_posterior(save=True, show=False)
# fig = fit.plot_sfh_posterior(save=True, show=False)
# fig = fit.plot_corner(save=True, show=False)
# try:
#     fig = fit.plot_calibration(save=True, show=False)
# except:
#     None

# print(np.percentile(fit.posterior.samples["chisq_phot"], (16, 50, 84)))

# save the best fits for the SFH and SED
save_SFH(fit,"best_fit_SFH_sfh_continuity_spec_BPASS.fits")
save_SED(fit,"best_fit_SED_sfh_continuity_spec_BPASS.fits")
