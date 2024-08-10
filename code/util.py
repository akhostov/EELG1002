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

def ab_mag_to_fnu(mag, magerr, unit="mJy"):
    """
    Converts an AB magnitude to flux density.

    Args:
        mag (float or array-like): The AB magnitude(s) to be converted.
        magerr (float or array-like): The error in the AB magnitude(s).
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
    dfnu = 0.4 * np.log(10) * magerr * fnu

    # Convert flux density to different units
    if "Jy" in unit:
        if unit == "mJy":
            fnu *= 1e3
            dfnu *= 1e3
        elif unit == "uJy":
            fnu *= 1e6
            dfnu *= 1e6
    elif "cgs" in unit:
        fnu = pow(10, -0.4 * (mag + 48.60))
        dfnu = 0.4 * np.log(10) * magerr * fnu
    elif "SI" in unit:
        fnu = pow(10, -0.4 * (mag + 56.1))
        dfnu = 0.4 * np.log(10) * magerr * fnu

    # Handle non-detections
    try:
        fnu[np.abs(mag) > 80.] = -99.
        dfnu[np.abs(mag) > 80.] = -99.
    except:
        None

    return (fnu, dfnu)


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