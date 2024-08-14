import numpy as np 
import bagpipes as pipes
import h5py
from bagpipes.fitting import fitted_model

fname = "/home/aahsps/Research Projects/EELG_OIII_GMOS/bagpipes_runs/with_BPASS/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5"

# Reconstruct the fitted model.
file = h5py.File(fname, "r")

fit_info_str = file.attrs["fit_instructions"]
fit_info_str = fit_info_str.replace("array", "np.array")
fit_info_str = fit_info_str.replace("float", "np.float")
fit_instructions = eval(fit_info_str)

