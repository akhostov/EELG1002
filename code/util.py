import numpy as np


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

def calzetti(lam):
    """
    Calculate the Calzetti Dust Atteunation Curve for a given wavelength.

    Parameters:
        lam (float or array-like): The wavelength(s) in micrometers.

    Returns:
        float or array-like: The Calzetti extinction value(s) corresponding to the input wavelength(s).
    """
    lam = lam/1e4
    return (2.659*(-2.156 + 1.509/lam - 0.198/(lam**2.) + 0.011/(lam**3.)) + 4.05)

def fnu_to_ab_mag(fnu, fnuerr = None, unit="cgs", ZP = None):
    
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