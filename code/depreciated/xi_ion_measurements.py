from astropy.io import fits
import h5py
from scipy.integrate import simps
import numpy as np
import pickle
import util

def get_1500A_luminosity(wave, flam, redshift=0.8275):
    
    mask = (wave > 1450.) & (wave < 1550.)
    wave = wave[mask]
    flam = flam[mask]
    
    fnu = (1./3e18) * simps(flam*wave, x=wave)/simps(1./wave, x=wave)

    return util.lineFlux_to_Luminosity(fnu,redshift)

def get_xi_ion(uv_lum, ha_lum):
    return ha_lum/(1.36e-12 * uv_lum)

def sampling(central, sigma, n_samples=10000):
    if isinstance(sigma,tuple):
        
        samples = np.random.normal(loc=0., scale=sigma[0], size=n_samples)

        mask = samples < 0.

        samples[mask] = -1.*np.abs(np.random.normal(loc=0., scale=sigma[1], size=np.sum(mask)))

        return samples + central
    
    else:

        return np.random.normal(loc=central, scale=sigma, size=n_samples)

def get_xi_ion_bagpipes(gmos, bagpipes_SED, bagpipes_params):

    # Get Best-Fit EBV. Will be used for both emission line and continuum
    EBV = bagpipes_params["median"][11]/4.05
    EBV_elow = EBV - bagpipes_params["conf_int"][0][11]/4.05 
    EBV_eupp = bagpipes_params["conf_int"][1][11]/4.05 - EBV

    # Calculate the 1500A Luminosity
    wave = pow(10,bagpipes_SED["log_wave"])/1.8275
    flam = bagpipes_SED["SED_median"]*1.8275*1e-18

    EBV_pdf = sampling(EBV,(EBV_elow,EBV_eupp))
    EBV_pdf = np.repeat(0.,len(EBV_pdf))
    HB_pdf = sampling(gmos["lineflux_med"],(gmos["lineflux_elow"],gmos["lineflux_eupp"]))

    Lnu_1500_pdf = get_1500A_luminosity(wave,flam=flam)*pow(10,0.4*EBV_pdf*util.calzetti(1500.))
    lum_HB_pdf = util.lineFlux_to_Luminosity(HB_pdf,redshift=0.8275)*pow(10,0.4*EBV_pdf*util.calzetti(4861.))

    # Xi Ion Measurement
    xi_ion_pdf = get_xi_ion(uv_lum = Lnu_1500_pdf, ha_lum = lum_HB_pdf*2.86)

    # UV Luminosity Measurement
    M_UV_pdf_corr = -2.5*np.log10(Lnu_1500_pdf/(4.*np.pi*(10*3.08e18)**2.)) - 48.6
    return (xi_ion_pdf, M_UV_pdf_corr)

def get_xi_ion_cigale(gmos, cigale_SED, cigale_params):

    # Get Best-Fit EBV. Will be used for both emission line and continuum
    EBV = cigale_params["bayes.attenuation.E_BV_lines"]
    EBV_err = cigale_params["bayes.attenuation.E_BV_lines_err"]

    # 1500A Luminosity
    flam = cigale_SED["Fnu"]*1e-26*3e18/(cigale_SED["wavelength"]*10.)**2.*1.8275
    wave = cigale_SED["wavelength"]*10./1.8275

    EBV_pdf = np.repeat(0.,10000)#sampling(EBV,EBV_err)
    HB_pdf = sampling(gmos["lineflux_med"],(gmos["lineflux_elow"],gmos["lineflux_eupp"]))

    Lnu_1500_pdf = get_1500A_luminosity(wave,flam=flam)*pow(10,0.4*EBV_pdf*util.calzetti(1500.))
    lum_HB_pdf = util.lineFlux_to_Luminosity(HB_pdf,redshift=0.8275)*pow(10,0.4*EBV_pdf*util.calzetti(4861.))

    # Xi Ion Measurement
    xi_ion_pdf = get_xi_ion(uv_lum = Lnu_1500_pdf, ha_lum = lum_HB_pdf*2.86)

    # UV Luminosity Measurement
    M_UV_pdf_corr = -2.5*np.log10(Lnu_1500_pdf/(4.*np.pi*(10*3.08e18)**2.)) - 48.6

    return (xi_ion_pdf, M_UV_pdf_corr)

def main():

    output = {}

    # Load in the GMOS Line Properties
    gmos = fits.open("../data/emline_fits/1002_lineprops.fits")[1].data
    gmos = gmos[ gmos["line_ID"] == "Hb_na" ]

    # Load in the cigale sed fit and results
    cigale_SED = fits.open("../data/SED_results/cigale_results/1002_best_model.fits")[1].data
    cigale_params = fits.open("../data/SED_results/cigale_results/results.fits")[1].data

    # Load in the bagpipes sed fit and results
    bagpipes_SED = fits.open("../data/SED_results/bagpipes_results/best_fit_SED_sfh_continuity_spec_BPASS_new.fits")[1].data
    bagpipes_params = h5py.File("../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")

    # Load in the Line Prop PDF
    with open("../data/emline_fits/1002_EW_pdfs.pkl","rb") as f:
        lineprops_pdf = pickle.load(f)
    
    # Measure Xi_Ion with Bagpipes SED
    bagpipes_xi_ion_pdf,bagpipes_M_UV_pdf_corr = get_xi_ion_bagpipes(gmos=gmos,bagpipes_SED=bagpipes_SED,bagpipes_params=bagpipes_params)

    # Measure [OIII]+Hb EW with Bagpipes SED
    EW_O3HB_bagpipes_pdf = lineprops_pdf["OIII5007c_EW_bagpipes_pdf"] + lineprops_pdf["OIII4959c_EW_bagpipes_pdf" ] + lineprops_pdf["Hb_na_EW_bagpipes_pdf"]

    output["bagpipes_xi_ion"] = util.stats(bagpipes_xi_ion_pdf)
    output["bagpipes_M_UV_corr"] = util.stats(bagpipes_M_UV_pdf_corr)
    output["bagpipes_xi_ion_pdf"] = bagpipes_xi_ion_pdf
    output["bagpipes_M_UV_corr_pdf"] = bagpipes_M_UV_pdf_corr
    output["bagpipes_O3HB_EW_pdf"] = EW_O3HB_bagpipes_pdf
    output["bagpipes_O3HB_EW"] = util.stats(EW_O3HB_bagpipes_pdf)
    

    # Cigale Xi_ion
    xi_ion_pdf, M_UV_pdf_corr = get_xi_ion_cigale(gmos, cigale_SED, cigale_params)

    # Measure [OIII]+Hb EW with Cigale SED
    EW_O3HB_cigale_pdf = lineprops_pdf["OIII5007c_EW_cigale_pdf"] + lineprops_pdf["OIII4959c_EW_cigale_pdf" ] + lineprops_pdf["Hb_na_EW_cigale_pdf"]

    output["cigale_xi_ion"] = util.stats(xi_ion_pdf)
    output["cigale_M_UV_corr"] = util.stats(M_UV_pdf_corr)
    output["cigale_xi_ion_pdf"] = xi_ion_pdf
    output["cigale_M_UV_corr_pdf"] = M_UV_pdf_corr
    output["cigale_O3HB_EW_pdf"] = EW_O3HB_cigale_pdf
    output["cigale_O3HB_EW"] = util.stats(EW_O3HB_cigale_pdf)
    
    with open("../data/xi_ion_measurements.pkl","wb") as f:
        pickle.dump(output,f)

if __name__ == "__main__":
    main()