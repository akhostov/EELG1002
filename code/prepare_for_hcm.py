from astropy.io import fits
import numpy as np 

# Load in the catalog
data = fits.open("../data/emline_fits/1002_lineprops.fits")[1].data

# Hbeta
Hbeta = data["lineflux_med"][data["line_id"] == "Hb_na"]

# [OII]3727
this = data["line_id"] == "[OII]"
o2_3727 = data["lineflux_med"][this]/Hbeta
err_o2_3727 = np.max([data["lineflux_elow"][this],data["lineflux_eupp"][this]])/Hbeta

# [NeIII]3869
this = data["line_id"] == "NeIII3869"
ne3869 = data["lineflux_med"][this]/Hbeta
err_ne3869 = np.max([data["lineflux_elow"][this],data["lineflux_eupp"][this]])/Hbeta

# [OIII]4363
this = data["line_id"] == "OIII4363"
o3_4363 = data["lineflux_med"][this]/Hbeta
err_o3_4363 = np.max([data["lineflux_elow"][this],data["lineflux_eupp"][this]])/Hbeta

# [OIII]4959
this = data["line_id"] == "OIII4959c"
o3_4959 = data["lineflux_med"][this]/Hbeta
err_o3_4959 = np.max([data["lineflux_elow"][this],data["lineflux_eupp"][this]])/Hbeta

# [OIII]5007
this = data["line_id"] == "OIII5007c"
o3_5007 = data["lineflux_med"][this]/Hbeta
err_o3_5007 = np.max([data["lineflux_elow"][this],data["lineflux_eupp"][this]])/Hbeta


np.savetxt("../data/HCM_EELG1002.txt",np.column_stack((["1002"],
                                                       o2_3727,err_o2_3727,
                                                       ne3869,err_ne3869,
                                                       o3_4363,err_o3_4363,
                                                       o3_4959,err_o3_4959,
                                                       o3_5007,err_o3_5007
                                                       )), header = "id OII_3727 eOII_3727 NeIII_3868 eNeIII_3868 OIII_4363 eOIII_4363 OIII_4959 eOIII_4959 OIII_5007 eOIII_5007",fmt="%s")

print("Done")