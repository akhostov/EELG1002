import numpy as np
from astropy.io import fits
import argparse
import os


# This will check to see if the filename does exist
def check_path(filename):
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(f"{filename} does not exist.")
    return filename

# This
def check_same_values(arr):
	unique_vals = np.unique(arr)

	if len(unique_vals) > 1:
		raise ValueError("There is a tilt in the 2D spectra. Wrapper can't take this into acccount at the moment")
	else:
		return True


# Read in the User Input
parser = argparse.ArgumentParser(description='This is a wrapper for SpecPro')
parser.add_argument('filename_1D', type=check_path, help='Path and Filename for the 1D spectra')
parser.add_argument('filename_2D', type=check_path, help='Path and Filename for the 2D spectra')
parser.add_argument("--outdir", "-o", help="Path To Store the Output Files", nargs='?',default="./")

# Parse the Arguments
args = parser.parse_args()
fname_1D = args.filename_1D
fname_2D = args.filename_2D
outdir = args.outdir

# Load the 1D and 2D FITS files
data_1D = fits.open(fname_1D)
data_2D = fits.open(fname_2D)

# Get the total numnber of nspecs
nspec = data_1D[0].header["NSPEC"]
slit_ids = [hdu.name for hdu in data_1D[1:-1]] # the cutoff at the end is for "MSC01-DETECTOR" todo: does this apply for all versions of pypeit reduced spectra?

# Define Mask Name
maskname = data_2D[0].header["TARGET"]

# Define then the 2D spectra
flux_2D = data_2D[1].data - data_2D[3].data # EXT=1 SCIIMG and EXT=2 SKYMODEL
lambda_2D = data_2D[8].data # EXT8 WAVEIMG
ivar_2D = data_2D[2].data #1./(1./data_2D[2].data + 1./data_2D[5].data) #EXT=2 IVARRAW and EXT=5 IVARMODEL
for ii in range(nspec):

	slit_id = int(slit_ids[ii].split("-")[1].split("SLIT")[1])
	slitname = data_1D[ii+1].header["MASKDEF_OBJNAME"]

	# Find where slit_id is in the 2D data
	ind = data_2D[10].data["SPAT_ID"] == slit_id

	print("Running for Object: %s" % slitname)
	# Extract the slit using the left and right boundaries of the slit
	left_init = np.squeeze(data_2D[10].data["left_init"][ind])
	right_init = np.squeeze(data_2D[10].data["right_init"][ind])

	# Check if the there is a curvature in the 2D Spectra
	check_same_values(left_init)
	check_same_values(right_init)

	# Convert to Integer
	left_init = int(left_init[0])
	right_init = int(right_init[0])

	# Extract the 2D
	flux_2D_source = flux_2D[:,left_init:right_init].T
	lambda_2D_source = lambda_2D[:,left_init:right_init].T
	ivar_2D_source = ivar_2D[:,left_init:right_init].T

	# Now let's remove the zeros at the end
	index_2D = np.apply_along_axis(lambda row: np.flatnonzero(row[::-1] == 0)[::-1], axis=1, arr=lambda_2D_source)
	flux_2D_source = np.delete(flux_2D_source,-index_2D-1,axis=1)
	ivar_2D_source = np.delete(ivar_2D_source,-index_2D-1,axis=1)
	lambda_2D_source = np.delete(lambda_2D_source,-index_2D-1,axis=1)

	# Apply A Limit
	#cutoff = np.percentile(flux_2D_source,99.)
	mask = np.abs(flux_2D_source) > np.std(np.abs(flux_2D_source))
	flux_2D_source[mask] = np.nan
	ivar_2D_source[mask] = np.nan

	# Define Sizes For Formatting Purposes
	shape_2D = flux_2D_source.shape[::-1]
	nele = flux_2D_source.size

	# Prepare the output 2D spectra
	dim_size = np.prod(flux_2D_source.shape)
	table_2D = fits.BinTableHDU.from_columns(fits.ColDefs([
					fits.Column(name='flux', format=f'{nele}D', array=flux_2D_source.reshape((1,-1)),dim='(%i,%i)' % shape_2D),
					fits.Column(name='lambda', format=f'{nele}D', array=lambda_2D_source.reshape((1,-1)),dim='(%i,%i)' % shape_2D),
					fits.Column(name='ivar', format=f'{nele}D', array=ivar_2D_source.reshape((1,-1)),dim='(%i,%i)' % shape_2D)
	]))

	# Write out the 2D spectra file
	out_2dspec = f"spec2d.{maskname}.{ii:03d}.{slitname}.fits"	
	table_2D.writeto(outdir+out_2dspec, overwrite=True)


	# Now Let's Work on the 1D which is much easier
	spec1d = data_1D[ii+1].data

	# COUNTS NOT FLUX CALIBRATED SPECTRA WHICH CAN CAUSE SOME UNUSUAL YLIMS
	lambda_1D = spec1d["OPT_WAVE"]
	flux_1d = spec1d["OPT_COUNTS"]
	ivar_1d = spec1d["OPT_COUNTS_IVAR"]

	# Remove the tail end of lambda where it is set to 0
	index = np.flatnonzero(lambda_1D[::-1] != 0)[::-1]
	flux_1d = flux_1d[:(len(lambda_1D) - index[-1])]
	ivar_1d = ivar_1d[:(len(lambda_1D) - index[-1])]
	lambda_1D = lambda_1D[:(len(lambda_1D) - index[-1])]


	# Write it out in a table
	table_1D = fits.BinTableHDU.from_columns([
	    fits.Column(name='flux', format='D', array=flux_1d),
	    fits.Column(name='lambda', format='D', array=lambda_1D),
	    fits.Column(name='ivar', format='D', array=ivar_1d)
	])

	# Save the FITS table to a file
	out_1dspec = f"spec1d.{maskname}.{ii:03d}.{slitname}.fits"
	table_1D.writeto(outdir+out_1dspec, overwrite=True)


	### Now let's create the extraction info of this object
	outinfoname = outdir+f"info.{maskname}.{ii:03d}.{slitname}.dat"	
	with open(outinfoname, 'w') as file:
		head = "# info file for %s" % out_1dspec
		file.write(head + '\n')
		file.write('ID {}\n'.format(data_1D[ii+1].header["MASKDEF_ID"]))
		file.write('name {}\n'.format(data_1D[ii+1].header["MASKDEF_OBJNAME"]))
		file.write('slitno {}\n'.format(ii))
		file.write('extractpos {}\n'.format(data_1D[ii+1].header["SPAT_FRACPOS"]*(right_init - left_init)))
		file.write('extractwidth {}\n'.format(data_1D[ii+1].header["FWHM"]))
		file.write('RA {}\n'.format(data_1D[ii+1].header["RA"]))
		file.write('DEC {}\n'.format(data_1D[ii+1].header["DEC"]))

data_1D.close()
data_2D.close()