from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.psf import IntegratedGaussianPRF,PSFPhotometry

psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
fit_shape = (5, 5)
finder = DAOStarFinder(6.0, 2.0)
psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                        aperture_radius=4)


# Load in the Image
data = fits.open("../data/added_F140W_hst_data/F140W_HST_15115/psf_F140W_raw_cutout.fits")[0].data

from astropy.table import QTable
init_params = QTable()
init_params['x'] = [20]
init_params['y'] = [22]
phot = psfphot(data, init_params=init_params)

phot['x_fit'].info.format = '.4f'  # optional format
phot['y_fit'].info.format = '.4f'
phot['flux_fit'].info.format = '.4f'
print(phot[('id', 'x_fit', 'y_fit', 'flux_fit')])  