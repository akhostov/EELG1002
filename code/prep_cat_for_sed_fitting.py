from astropy.io import fits
import numpy as np
import util



def load_catalog_and_upper_limits():
	# USES COSMOS2020 Classical Matched Catalog
	cat = fits.open("../data/catalogs/COSMOS2020_Classical_GMOS_OIII_EELG.fits")[1].data
	cat = cat[cat["ID"] == "1002"]

	# Upper limits (last data point per tuple) are 3 sigma and based on 2" apertures
	info = [("CFHT_ustar_FLUX_AUTO","CFHT_ustar_FLUXERR_AUTO","CFHT_u",27.7),
			("HSC_g_FLUX_AUTO","HSC_g_FLUXERR_AUTO","subaru.hsc.g",28.1),
			("HSC_r_FLUX_AUTO","HSC_r_FLUXERR_AUTO","subaru.hsc.r",27.8),
			("HSC_i_FLUX_AUTO","HSC_i_FLUXERR_AUTO","subaru.hsc.i",27.6),
			("HSC_z_FLUX_AUTO","HSC_z_FLUXERR_AUTO","subaru.hsc.z",27.2),
			("HSC_y_FLUX_AUTO","HSC_y_FLUXERR_AUTO","subaru.hsc.y",26.5),
			("UVISTA_Y_FLUX_AUTO","UVISTA_Y_FLUXERR_AUTO","vista.vircam.Y",25.3),
			("UVISTA_J_FLUX_AUTO","UVISTA_J_FLUXERR_AUTO","vista.vircam.J",25.2),
			("UVISTA_H_FLUX_AUTO","UVISTA_H_FLUXERR_AUTO","vista.vircam.H",24.9),
			("UVISTA_Ks_FLUX_AUTO","UVISTA_Ks_FLUXERR_AUTO","vista.vircam.Ks",25.3),
			("SC_IB427_FLUX_AUTO","SC_IB427_FLUXERR_AUTO","subaru.suprime.IB427",26.1),
			("SC_IB464_FLUX_AUTO","SC_IB464_FLUXERR_AUTO","subaru.suprime.IB464",25.6),
			("SC_IA484_FLUX_AUTO","SC_IA484_FLUXERR_AUTO","subaru.suprime.IB484",26.5),
			("SC_IB505_FLUX_AUTO","SC_IB505_FLUXERR_AUTO","subaru.suprime.IB505",26.1),
			("SC_IA527_FLUX_AUTO","SC_IA527_FLUXERR_AUTO","subaru.suprime.IB527",26.4),
			("SC_IB574_FLUX_AUTO","SC_IB574_FLUXERR_AUTO","subaru.suprime.IB574",25.8),
			("SC_IA624_FLUX_AUTO","SC_IA624_FLUXERR_AUTO","subaru.suprime.IB624",26.4),
			("SC_IA679_FLUX_AUTO","SC_IA679_FLUXERR_AUTO","subaru.suprime.IB679",25.6),
			("SC_IB709_FLUX_AUTO","SC_IB709_FLUXERR_AUTO","subaru.suprime.IB709",25.9),
			("SC_IA738_FLUX_AUTO","SC_IA738_FLUXERR_AUTO","subaru.suprime.IB738",26.1),
			("SC_IA767_FLUX_AUTO","SC_IA767_FLUXERR_AUTO","subaru.suprime.IB767",25.6),
			("SC_IB827_FLUX_AUTO","SC_IB827_FLUXERR_AUTO","subaru.suprime.IB827",25.6),
			("SC_NB711_FLUX_AUTO","SC_NB711_FLUXERR_AUTO","subaru.suprime.NB711",25.5),
			("SC_NB816_FLUX_AUTO","SC_NB816_FLUXERR_AUTO","subaru.suprime.NB816",25.6),
			("SC_B_FLUX_AUTO","SC_B_FLUXERR_AUTO","SUBARU_B",27.8),
			("SC_gp_FLUX_AUTO","SC_gp_FLUXERR_AUTO","g_prime",26.1),
			("SC_V_FLUX_AUTO","SC_V_FLUXERR_AUTO","subaru.suprime.V",26.8),
			("SC_rp_FLUX_AUTO","SC_rp_FLUXERR_AUTO","subaru.suprime.r",27.1),
			("SC_ip_FLUX_AUTO","SC_ip_FLUXERR_AUTO","subaru.suprime.i",26.7),
			("SC_zp_FLUX_AUTO","SC_zp_FLUXERR_AUTO","SUBARU_z",25.7),
			("SC_zpp_FLUX_AUTO","SC_zpp_FLUXERR_AUTO","subaru.suprime.zpp",26.3),
			("ACS_F814W_FLUX","ACS_F814W_FLUXERR","hst.wfc.F814W",27.8)]

	return (cat,info)


# Extract the Redshift and Save 
def get_redshift(catalog):
	redshift = []
	for ii in range(len(catalog["ID"])):
		zspec = catalog["zspec"][ii]
		zphot = catalog["lp_zBEST"][ii]

		if zspec < 0.:
			redshift.append(zphot)
		else:
			redshift.append(zspec)
	return redshift


def load_linefluxes_for_cigale(data):

	# Now load in the nebular emission line constraints
	spec = fits.open("../data/emline_fits/43158747673038238/1002_lineprops_new.fits")[1].data

	#### BALMER LINES
	# Hbeta Line
	header = header + " line.H-beta line.H-beta_err"
	this = spec["line_ID"] == "Hb_na_1"
	data.append(spec["lineflux_med"][this]*1e-20)
	data.append(spec["lineflux_elow"][this]*1e-20)

	# Hgamma Line
	header = header + " line.H-gamma line.H-gamma_err"
	this = spec["line_ID"] == "Hg_na_1"
	data.append(spec["lineflux_med"][this]*1e-20)
	data.append(spec["lineflux_elow"][this]*1e-20)

	# Hdelta Line
	header = header + " line.H-delta line.H-delta_err"
	this = spec["line_ID"] == "Hd_na_1"
	data.append(spec["lineflux_med"][this]*1e-20)
	data.append(spec["lineflux_elow"][this]*1e-20)

	#### OXYGEN LINES

	# [OIII]5007 Line
	header = header + " line.OIII-500.7 line.OIII-500.7_err"
	this = spec["line_ID"] == "OIII5007c_1"
	data.append(spec["lineflux_med"][this]*1e-20)
	data.append(spec["lineflux_elow"][this]*1e-20)

	# [OIII]4959 Line
	header = header + " line.OIII-495.9 line.OIII-495.9_err"
	this = spec["line_ID"] == "OIII4959c_1"
	data.append(spec["lineflux_med"][this]*1e-20)
	data.append(spec["lineflux_elow"][this]*1e-20)

	# [OII]3729 Line
	header = header + " line.OII-372.9 line.OII-372.9_err"
	this = spec["line_ID"] == "OII3728_1"
	data.append(spec["lineflux_med"][this]*1e-20)
	data.append(spec["lineflux_elow"][this]*1e-20)

	# [OII]3726 Line
	header = header + " line.OII-372.6 line.OII-372.6_err"
	this = spec["line_ID"] == "OII3726_1"
	data.append(spec["lineflux_med"][this]*1e-20)
	data.append(spec["lineflux_elow"][this]*1e-20)

	return data



def run_catalog_conversion(sed_code="cigale"):

	# Make Sure that the Supported SED Codes are iin sed_code
	valid_sed_code = ["cigale","BAGPIPES"]
	if sed_code not in valid_sed_code:
		raise ValueError("%s is not supported in this script. Only the following are supported: %s" % (sed_code,valid_sed_code))

	# Define the Unit for the Catalog
	if sed_code == "cigale":
		fnu_unit = "mJy"
	
	if sed_code == "BAGPIPES":
		fnu_unit = "uJy"

	# Load in the Catalog and Upper Limits
	cat,info = load_catalog_and_upper_limits()
	
	# Bands that will be used in the fitting process
	bands = np.array(info).T.tolist()[0]

	# Main List that will store the data
	data = []
	data.append(cat["ID"])
	
	# Header for the Converted Catalog
	header = "id redshift"

	# Find the assigned redshift. This is generalized as we already have zspec for EELG1002
	data.append(get_redshift(cat))


	# Run for each band
	for bb in range(len(bands)):

		# Append the Header
		header = header + " " + info[bb][2] + " " + info[bb][2] + "_err"

		# If Magnitudes are used, then trigger the conversion
		if "MAG" in bands[bb]:
			fnu,dfnu = util.ab_mag_to_fnu(cat[info[bb][0]],cat[info[bb][1]],unit=fnu_unit)

		# If Fluxes are used, then trigger the conversion
		elif "FLUX" in bands[bb]:

			# Note: COSMOS2020 Fluxes are all ready in uJy
			fnu = cat[info[bb][0]]
			dfnu = cat[info[bb][1]]

			# Convert Fluxes to mJy for Cigale
			if sed_code == "cigale":
				fnu *= 1e-3
				dfnu *= 1e-3
		
		# Localized Non-detections scaled to 5sigma Limits.
		# This essentially takes the error converted to 5 sigma and provides a data point from 0 to the 5 sigma level
		# Hence why it is divided by 2. The negative in dfnu is just a convention for upper limits.
		fix = fnu <= 0.
		fnu[fix] = dfnu[fix]/2.*5.
		dfnu[fix] = -1.*dfnu[fix]/2.*5.

		# Drawn from COSMOS2020 Paper
		limit = pow(10,-0.4*(info[bb][3]+48.6))/3.*5.*1e26 # 5 Sigma Limits

		# Takes into account non-detections in GALEX, Spitzer IRAC , and HST/ACS photometry.
		if any(keyword in bands[bb] for keyword in ["GALEX","SPLASH","ACS"]):
			fix = np.isnan(fnu)
			fnu[fix] = limit/2.
			dfnu[fix] = -1.*limit/2.

		data.append(fnu)
		data.append(dfnu)


	######## ADD IN ANCILLARY DATA ########
	# Include CFHT/Wircam H data (from COSMOS2015)
	fnu,err_fnu = util.ab_mag_to_fnu(mag=23.818542, magerr=0.25989458, unit=fnu_unit)
	header = header + "  cfht.wircam.H  cfht.wircam.H_err"
	data.append(fnu)
	data.append(err_fnu)

	# Include CFHT/Wircam Ks data (from COSMOS2015)
	fnu,err_fnu = util.ab_mag_to_fnu(mag=24.391464, magerr=0.44462332, unit=fnu_unit)
	header = header + "  cfht.wircam.Ks  cfht.wircam.Ks_err"
	data.append(fnu)
	data.append(err_fnu)

	# Include HAP F140W data
	fnu,err_fnu = util.ab_mag_to_fnu(mag=23.89219, magerr=0.08405, unit=fnu_unit)
	header = header + " hst.wfc3.F140W hst.wfc3.F140W_err"
	data.append(fnu)
	data.append(err_fnu)

	# This will include the Line Fluxes for CIGALE SED Fitting
	# Note that the 1D spectra itself is inputted within the Bagpipes SED Fitting Process
	if sed_code == "cigale":
		data = load_linefluxes_for_cigale(data)



	######## WRITE OUT THE DATA ########
	# Convert All into a Numpy Array and Write it out as an ASCII File
	data = np.array(data)
	np.savetxt("../data/catalogs/EELG_OIII_GMOS_%s.txt" % sed_code,np.column_stack(data),header=header,fmt="%s")


def main():
	run_catalog_conversion(sed_code="cigale")
	run_catalog_conversion(sed_code="bagpipes")

if __name__ == "__main__":
	main()

