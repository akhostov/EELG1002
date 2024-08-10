import os
import io
import sys
import json

import numpy as np

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
from astropy.utils.data import download_file, clear_download_cache

import matplotlib.pyplot as plt

from unagi import hsc
from unagi import task
from unagi import query
from unagi import config
from unagi import plotting

pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_dud')
coord = SkyCoord(150.13463, 2.85315, frame='icrs', unit='deg')

# Angular size
s_ang = 5.0 * u.arcsec

# Physical size
s_phy = 100.0 * u.kpc
redshift = 0.25

# Filters
filters = 'grizy'

# Output dir
output_dir = '../data/cutouts/Subaru/HSC/'


cutout_test = task.hsc_cutout(coord, cutout_size=s_ang, filters=filters, archive=pdr2, 
                         use_saved=False, output_dir=output_dir, verbose=True, 
                         save_output=True)
w = wcs.WCS(cutout_test[1].header)

_ = plotting.display_single(cutout_test[1].data)

plt.show()

cutout_test.close()

#sql_file = 'HSC_SQL_query.sql'
#result_test = pdr3.sql_query(sql_file, from_file=True, verbose=True)

#print(result_test)