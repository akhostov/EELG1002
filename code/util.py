import numpy as np
from scipy.integrate import simps,quad
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

def get_cosmology(H0=70,Om0=0.3):
    """
    Returns a FlatLambdaCDM cosmology object.

    Parameters:
        H0 (float, optional): The Hubble constant. Defaults to 70.
        Om0 (float, optional): The matter density parameter. Defaults to 0.3.

    Returns:
        FlatLambdaCDM: A FlatLambdaCDM cosmology object.
    """
    return FlatLambdaCDM(H0=H0,Om0=Om0)

def stats(pdf,upper_limit=False): 
    """
    Calculate statistics of a probability density function (PDF).

    Parameters:
        pdf (array-like): The probability density function.
        upper_limit (bool, optional): If True, returns the upper limit of the PDF. Defaults to False.

    Returns:
        float or tuple: If upper_limit is True, returns the upper limit of the PDF. Otherwise, returns a tuple of three values: the median of the PDF, the difference between the median and the 16th percentile of the PDF, and the difference between the 84th percentile and the median of the PDF.
    """

    if upper_limit:
        return np.nanpercentile(pdf,upper_limit)
    else:
        return (np.nanmedian(pdf),np.nanmedian(pdf) - np.nanpercentile(pdf,16.),np.nanpercentile(pdf,84.) - np.nanmedian(pdf))

def calzetti(lam,unit="Angstrom"):
    """
    Calculate the Calzetti extinction curve for a given wavelength.

    Parameters:
        lam (float): The wavelength for which to calculate the extinction curve.
        unit (str, optional): The unit of the wavelength. Defaults to "Angstrom".

    Returns:
        float: The calculated extinction curve value.

    Raises:
        ValueError: If the input unit is not 'Angstrom' or 'micron', or if the input wavelength is not between 0.12 and 2.2 microns.
    """

    if unit == "Angstrom":
        lam = lam/1e4
    elif unit == "micron":
        pass
    else:
        raise ValueError("Input unit must be 'Angstrom' or 'micron'.")
    
    if (lam > 0.12) & (lam < 0.63):
        return 2.659*( -2.156 + 1.509/lam - 0.198/(lam**2.) + 0.011/(lam**3.) ) + 4.05
    elif (lam > 0.63) & (lam < 2.2):
        return 2.659*( -1.857 + 1.040/lam ) + 4.05
    else:
        raise ValueError("Input wavelength must be between 0.12 and 2.2 microns.") 

def lineFlux_to_Luminosity(lineFlux,redshift,cosmo=FlatLambdaCDM(H0=70.,Om0=0.3)):
    """
    Calculate the luminosity of a line in a given redshift.

    Parameters:
        lineFlux (float): The flux of the line in cgs (erg/s/cm2)
        redshift (float): The redshift of the line.
        cosmo (astropy.cosmology.FLRW, optional): The cosmology to use. Defaults assumes FlatLambdaCDM(H0=70.,Om0=0.3). 

    Returns:
        float: The luminosity of the line in cgs (erg/s)
    """
    
    return lineFlux*4.*np.pi*(cosmo.luminosity_distance(redshift).value*3.086e24)**2.

def fnu_to_ab_mag(fnu, fnuerr = None, unit="cgs", ZP = None):
    """
    Converts a flux density to an AB magnitude.

    Parameters:
        fnu (float or array-like): The flux density to be converted.
        fnuerr (float or array-like, optional): The error in the flux density. Defaults to None.
        unit (str, optional): The unit of the flux density. Defaults to "cgs".
        ZP (float, optional): The zero point of the magnitude system. Defaults to None.

    Returns:
        float or tuple: The converted AB magnitude, or a tuple containing the magnitude and its error if fnuerr is not None.

    Raises:
        ValueError: If the unit is not one of "mJy", "uJy", "Jy", "cgs", "SI", or "custom".   
        ValueError: If custom is used and ZP is not specified.
    """
    
    # Check if the input unit is valid
    valid_units = ["mJy", "uJy", "Jy", "cgs", "SI", "custom"]
    if unit not in valid_units:
        raise ValueError(f"Unit must be one of {valid_units}")

    if unit == "cgs":
        ZP = -48.6
    if unit == "SI":
        ZP =  -56.1
    if "Jy" in unit:
        ZP = 8.90
        if unit == "mJy":
            ZP += 7.5
        elif unit == "uJy":
            ZP += 15.0       

    # Ensure that the Zero Point is specified if the unit is 'custom'
    if "unit" == "custom" and ZP is None:
        raise ValueError("ZP must be specified if unit is 'custom'")

    # Convert flux density to AB magnitude
    mag = -2.5*np.log10(fnu) + ZP

    if fnuerr is not None:
        # Convert error in flux density to error in AB magnitude
        magerr = 2.5/np.log(10) * fnuerr / fnu
    else:
        magerr = None
    
    return (mag,magerr) if magerr is not None else mag

def ab_mag_to_fnu(mag, magerr = None, unit="mJy"):
    """
    Converts an AB magnitude to flux density.

    Args:
        mag (float or array-like): The AB magnitude(s) to be converted.
        magerr (float or array-like, optional): The error in the AB magnitude(s). Defaults to None.
        unit (str, optional): The unit of the flux density. Defaults to "mJy".

    Returns:
        tuple: A tuple containing the converted flux density and its error.

    Raises:
        ValueError: If the unit is not one of "mJy", "uJy", "Jy", "cgs", or "SI".
    """

    # Check if the input unit is valid
    valid_units = ["mJy", "uJy", "Jy", "cgs", "SI"]
    if unit not in valid_units:
        raise ValueError(f"Unit must be one of {valid_units}")

    # Convert magnitude to flux density in Jy
    fnu = pow(10, -0.4 * (mag - 8.9))

    # Convert flux density to different units
    if "Jy" in unit:
        if unit == "mJy":
            fnu *= 1e3
        elif unit == "uJy":
            fnu *= 1e6
    elif "cgs" in unit:
        fnu = pow(10, -0.4 * (mag + 48.60))
    elif "SI" in unit:
        fnu = pow(10, -0.4 * (mag + 56.1))


    # Calculate the error only if magerr is defined
    if magerr is not None:
        dfnu = 0.4*np.log(10)*magerr*fnu
    else:
        dfnu = None

    # Handle non-detections
    try:
        fnu[np.abs(mag) > 80.] = -99.
        dfnu[np.abs(mag) > 80.] = -99.
    except:
        pass

    return (fnu, dfnu) if dfnu is not None else fnu


def write_with_newline(table, text):
    """
    Used in multiple table scripts. This will write the given text to a file object ("table") with a newline character appended.
    Saves the hassle of adding a newline character at the end of the file.

    Parameters:
        table (Table): The table object to write to.
        text (str): The text to write to the table.

    Returns:
        None
    """
    table.write(text + "\n")

def calc_SFR(time, sfr, max_time_limit = None, sigma=None):
    """
    Calculate the average SFR (Star Formation Rate) over a given time period.

    Parameters:
        time (array-like): An array of time values.
        sfr (array-like): An array of SFR values.
        max_time_limit (float, optional): The maximum time limit for integration. Default is None.
        sigma (array-like, tuple, optional): The error in SFR. If a single array is provided, it is considered as the symmetrical error.
                                              If a tuple is provided, it is considered as the low and upper errors in that order. Default is None.
    Returns:
        tuple or float: If sigma is provided, a tuple containing the average SFR, the lower error, and the upper error.
                        If sigma is not provided, the average SFR.
    Raises:
        ValueError: If sigma is a tuple and it does not contain exactly two arrays.

    """

    # Check to make sure that the tuple is of size 2 (low, upp)
    if isinstance(sigma,tuple):
        if len(sigma) != 2:
            raise ValueError("Sigma tuple must contain exactly two arrays: (sigma_low, sigma_upp)")
        sigma_low, sigma_upp = sigma

    # Limit the integration from 0 to max_time_limit Gyr
    if max_time_limit is not None:
        mask = time < max_time_limit

        # Make sure that the mask is not empty
        if not np.any(mask):
            raise ValueError("The max_time_limit is too small resulting in an empty array. Can't Integrate. Adjust your max_time_limit accordingly.")

        time = time[mask]
        sfr = sfr[mask]

        if sigma is not None:
            if isinstance(sigma,tuple):
                sigma_low, sigma_upp = sigma_low[mask], sigma_upp[mask]
            else:
                sigma = sigma[mask]

    # Calculate time-averaged SFR
    fx_sfr = interp1d(time*1e9,sfr,bounds_error=False,fill_value=(sfr[0],sfr[-1]))
    sfr_avg = quad(fx_sfr,0,max_time_limit*1e9)[0]/(max_time_limit*1e9)

    # Calculate the error in the time-averaged SFR
    if sigma is not None:
        if isinstance(sigma, tuple):
            fx_sigma_low = interp1d(time*1e9,sigma_low**2.,bounds_error=False,fill_value=(sigma_low[0]**2.,sigma_low[-1]**2.))
            fx_sigma_upp = interp1d(time*1e9,sigma_upp**2.,bounds_error=False,fill_value=(sigma_upp[0]**2.,sigma_upp[-1]**2.))
            err_sfr_avg_low = np.sqrt(quad(fx_sigma_low,0,max_time_limit*1e9)[0]/(max_time_limit*1e9))
            err_sfr_avg_upp = np.sqrt(quad(fx_sigma_upp,0,max_time_limit*1e9)[0]/(max_time_limit*1e9))

            #err_sfr_avg_low = np.sqrt(simps(sigma_low**2.,x=time))
            #err_sfr_avg_upp = np.sqrt(simps(sigma_upp**2.,x=time))
            return sfr_avg, err_sfr_avg_low, err_sfr_avg_upp
        
        else:
            fx_sigma = interp1d(time*1e9,sigma**2.,bounds_error=False,fill_value=(sigma[0]**2.,sigma[-1]**2.))
            err_sfr_avg = np.sqrt(quad(fx_sigma,0,max_time_limit*1e9)[0]/(max_time_limit*1e9))

            #err_sfr_avg = np.sqrt(simps(sigma**2.,x=time))
            return sfr_avg,err_sfr_avg
        
    return sfr_avg


def sampling(central, sigma, n_samples=10000):
    """
    Generates a set of random samples from a normal and asymmetrically distributed normal distribution.

    Parameters:
        central (float): The mean of the normal distribution.
        sigma (float or tuple): The standard deviation of the normal distribution. If a tuple, the first element is +sigma and the second element is -sigma.
        n_samples (int): The number of samples to generate. Defaults to 10000.

    Returns:
        numpy.ndarray: An array of random samples from the specified normal distribution.
    """
    if isinstance(sigma,tuple):
        
        samples = np.random.normal(loc=0., scale=sigma[0], size=n_samples)

        mask = samples < 0.

        samples[mask] = -1.*np.abs(np.random.normal(loc=0., scale=sigma[1], size=np.sum(mask)))

        return samples + central
    
    else:

        return np.random.normal(loc=central, scale=sigma, size=n_samples)

def color_scheme(sed,mfc=False,mec=False):
    if sed == "Cigale":
        if mfc == True: return "#AEC7E8"
        if mec == True: return "#1F77B4"
    elif sed == "Bagpipes":
        if mfc == True: return "#acb7dc"
        if mec == True: return "#003366"
