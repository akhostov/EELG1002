import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pdb 

dname = "../data/Illustris_SFH/"

#fname = "raw/ID_401578_Snapshot_54.txt"
#oname = "final/ID_401578_Snapshot_54_SFH.txt"
fname = "raw/ID_394878_Snapshot_54.txt"
oname = "final/ID_394878_Snapshot_54_SFH.txt"

#fname = "raw/ID_112713_Snapshot_55.txt"
#oname = "final/ID_112713_Snapshot_55_SFH.txt"

# Load in the SFH
snapshot,universe_age,sfr = np.loadtxt(dname+fname,unpack=True)

# Convert Ages into Myr and reverse
universe_age = universe_age*1000. # Myr
if np.min(universe_age) != 0.:
	universe_age = np.append(universe_age,0.)
	sfr = np.append(sfr,0.)

# Reverse the array
universe_age = np.flip(universe_age)
sfr = np.flip(sfr)

# Calculate minor shift
shift = 6738 - universe_age[-1]
universe_age[1:] = universe_age[1:]+shift

# Create Interpolated SFH Function
fx = interp1d(universe_age,sfr,kind="cubic",fill_value=(0,sfr[-1]),bounds_error=False)

# Set Time Grid
time = np.arange(0.,np.max(universe_age),1.) # Time steps of 1 Myr

# Get Final SFH
sfh = fx(time)
sfh[sfh<0] = 0.

plt.plot(time,sfh)
plt.show()
pdb.set_trace()
# Output SFH
np.savetxt(dname+oname,np.column_stack((time,sfh)),header="time sfr")