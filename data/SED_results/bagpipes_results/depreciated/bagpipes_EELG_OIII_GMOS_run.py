import numpy as np 
import bagpipes as pipes

from astropy.io import fits

def load_obs(ID):

    # load up the relevant columns from the catalogue.
    cat = np.loadtxt("../../EELG_OIII_GMOS_BAGPIPES.txt", delimiter=" ")

    ID = cat[0]
    redshift = cat[1]

    # Extract the object we want from the catalogue.
    fluxes = cat[2::2]*1e3
    fluxerrs = cat[3::2]*1e3

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

    hdulist = fits.open("../../1002_fluxcorr_aper_corrected_spectrum.fits")

    spectrum = np.c_[hdulist[1].data["OPT_WAVE"],
                     hdulist[1].data["OPT_FLAM"]*1e-17,
                     hdulist[1].data["OPT_FLAM_SIG"]*1e-17]

    mask = (spectrum[:,0] > 6700.) & (spectrum[:,0] < 9300.) 

    return bin(spectrum[mask],2)


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

    # Get the SFH
    time = age_of_universe - fit.posterior.sfh.ages*10**-9

    col_time = fits.Column(name='Lookback Time', array=time, format='D',unit="Gyr")
    col_median = fits.Column(name='SFH_median', array=post.T[1], format='D', unit="Msol yr-1")
    col_low = fits.Column(name='SFH_1sig_low', array=post.T[0], format='D', unit="Msol yr-1")
    col_upp = fits.Column(name='SFH_1sig_upp', array=post.T[2], format='D', unit="Msol yr-1")

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
    #c3 = fits.Column(name='photometric_model', array=phot_post, format='D')
    t = fits.BinTableHDU.from_columns([col_wave,col_median,col_1low,col_1upp,col_2low,col_2upp])
    t.writeto(outname,overwrite=True)

    """
    import pdb; pdb.set_trace()

    for j in range(fit.galaxy.photometry.shape[0]):

        if skip_no_obs and fit.galaxy.photometry[j, 1] == 0.:
            continue

        phot_band = fit.posterior.samples["photometry"][:, j]
        mask = (phot_band > phot_post[j, 0]) & (phot_band < phot_post[j, 1])
        phot_1sig = phot_band[mask]*10**-y_scale
        wav_array = np.zeros(phot_1sig.shape[0]) + log_eff_wavs[j]

        if phot_1sig.min() < ymax*10**-y_scale:
            ax.scatter(wav_array, phot_1sig, color=color2,
                       zorder=zorder, alpha=0.05, s=100, rasterized=True)
    """

# Fitting 

#### STAR FORMATION HISTORY


# Similar to Leja et al. (2019)
continuity = {}
continuity["massformed"] = (1., 13.)
continuity["metallicity"] = 0.065 #(0.001, 5.)
#continuity["metallicity_prior"] = "log_10"
#continuity["bin_edges"] = [0., 1., 2.5, 5., 10., 100., 250., 500.,
#                           1000., 2500., 5000.]
continuity["bin_edges"] = [0., 3., 10., 30., 100., 300.,
                           1000., 3000., 6000.]

for i in range(1, len(continuity["bin_edges"])-1):
    continuity["dsfr" + str(i)] = (-10., 10.)
    continuity["dsfr" + str(i) + "_prior"] = "student_t"
    #continuity["dsfr" + str(i) + "_prior_scale"] = 0.3  # Defaults to this value as in Leja19, but can be set
    #continuity["dsfr" + str(i) + "_prior_df"] = 2       # Defaults to this value as in Leja19, but can be set

"""
delayed = {}                                  
delayed["age"] = (0.01, 6.6) # Gyr
delayed["tau"] = (0.01, 0.1) # Gyr
delayed["massformed"] = (1., 15.)
delayed["metallicity"] = 0.065 # Zsol

burst1 = {}                                  
burst1["age"] = (0.01, 0.1) # Gyr
burst1["tau"] = (0.05, 1.) # Gyr
burst1["massformed"] = (1., 15.)
burst1["metallicity"] = 0.065 #Zsol
"""

dust = {}                           
dust["type"] = "Calzetti"
dust["Av"] = (0., 10.)
#dust["eta"] = (0.8,10.0)
dust["eta"] = 1.0
#dust["eta_prior"] = "uniform"
dust["delta"] = 0.#(-4.,4.)
#dust["delta_prior"] = "uniform"



# Dust emission parameters
dust["qpah"] = 2.5         # PAH mass fraction
dust["umin"] = 1.0          # Lower limit of starlight intensity distribution
dust["gamma"] = 0.1       # Fraction of stars at umin

# Nebular Component
nebular = {}
nebular["logU"] = (-4.,-1.)
nebular["logU_prior"] = "uniform"
#nebular["logU_prior_mu"] = -2.2
#nebular["logU_prior_sigma"] = 0.3

# AGN Component
agn = {}
agn["alphalam"] = (-2.0,2.0) # power law slope < 5000A
agn["alphalam_prior"] = "Gaussian"
agn["alphalam_prior_mu"] = -1.5
agn["alphalam_prior_sigma"] = -0.5
agn["betalam"] = (-2.0,2.0) # power law slope > 5000A
agn["betalam_prior"] = "Gaussian"
agn["betalam_prior_mu"] = 0.5
agn["betalam_prior_sigma"] = 0.5
agn["hanorm"] = (0.,2.5e-17) # cgs
agn["hanorm_prior"] = "uniform"
agn["sigma"] = (1.,1000.) # km/s
agn["sigma_prior"] = "log_10"
agn["f5100A"] = (0., 1e-18)
agn["f5100A_prior"] = "uniform"

fit_info = {}                            # The fit instructions dictionary
fit_info["t_bc"] = 0.01
fit_info["redshift"] = 0.8275
#fit_info["veldisp"] = 300.          # Velocity dispersion: km/s
fit_info["veldisp"] = (1., 1000.)   #km/s
fit_info["veldisp_prior"] = "log_10"

#fit_info["delayed"] = delayed
#fit_info["burst1"] = burst1
#fit_info["dblplaw"] = dblplaw
#fit_info["iyer"] = iyer
fit_info["continuity"] = continuity 
#fit_info["psb_wild2020"] = psb_wild2020
#fit_info["constant"] = constant
fit_info["dust"] = dust
fit_info["nebular"] = nebular
#fit_info["agn"] = agn



# Spectral Fitting
calib = {}
calib["type"] = "polynomial_bayesian"

calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
calib["0_prior"] = "Gaussian"
calib["0_prior_mu"] = 1.0
calib["0_prior_sigma"] = 0.10#0.25

calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
calib["1_prior"] = "Gaussian"
calib["1_prior_mu"] = 0.
calib["1_prior_sigma"] = 0.10 #0.25

calib["2"] = (-0.5, 0.5)
calib["2_prior"] = "Gaussian"
calib["2_prior_mu"] = 0.
calib["2_prior_sigma"] = 0.10 #0.25

fit_info["calib"] = calib

mlpoly = {}
mlpoly["type"] = "polynomial_max_like"
mlpoly["order"] = 2


noise = {}
noise["type"] = "white_scaled"
noise["scaling"] = (1., 10.)
noise["scaling_prior"] = "log_10"
fit_info["noise"] = noise



filt_list = np.loadtxt("filters/filt_list_wo_GALEX_and_Spitzer.txt", dtype="str")
galaxy = pipes.galaxy("1002", load_both, filt_list=filt_list,phot_units="mujy",photometry_exists=True,spectrum_exists=True)


fit = pipes.fit(galaxy, fit_info, run="sfh_continuity_spec_BPASS")
import pdb; pdb.set_trace()
fit.fit(verbose=False,n_live=1000,use_MPI=True)

#save_SED(fit)
#import pdb; pdb.set_trace()
fig = fit.plot_spectrum_posterior(save=True, show=False)
fig = fit.plot_sfh_posterior(save=True, show=False)
fig = fit.plot_corner(save=True, show=False)
fig = fit.plot_calibration(save=True, show=False)

print(np.percentile(fit.posterior.samples["chisq_phot"], (16, 50, 84)))

save_SFH(fit,"best_fit_SFH_sfh_continuity_spec_BPASS.fits")
save_SED(fit,"best_fit_SED_sfh_continuity_spec_BPASS.fits")
#

"""
fig = fit.plot_spectrum_posterior(save=False, show=True)
fig = fit.plot_sfh_posterior(save=False, show=True)
fig = fit.plot_corner(save=False, show=True)
"""
#import pdb; pdb.set_trace()
