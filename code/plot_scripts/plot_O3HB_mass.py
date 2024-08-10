import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from astropy.io import fits
import numpy as np 
import pickle
import pdb

import sys
sys.path.insert(0, "../../../My_Modules/color_lib/")
from colors import tableau20

######################################################################
#######						PREPARE THE FIGURE				   #######
######################################################################

fig = plt.figure()
fig.set_size_inches(3,3)
ax = fig.add_subplot(111)

# Define Labels
ax.set_xlabel(r"$\log_{10}$ Stellar Mass (M$_\odot$)",fontsize=8)
ax.set_ylabel(r"[O{\sc iii}]5007\AA/H$\beta$",fontsize=8)


# Define Limits
ax.set_ylim(1.,9.)
ax.set_xlim(6.5,10)
#ax.set_yscale("log")

#ax.set_yticks([1,2,3,4,6,10])
#ax.set_yticks([1,1.5,2,2.5,3,3.5,4,5,6,7,8,9,10],minor=True)
#ax.set_yticklabels(["1","2","3","4","6","10"])

# Plot the Juneau et al. (2011) cuts
xx = np.arange(6.,12.,0.01)
yy_1 = pow(10,0.37/(xx - 10.5)+1.)
yy_2 = pow(10,800.492 - 217.328*xx + 19.6431*xx**2. - 0.591349*xx**3.)
yy_3 = pow(10,594.753 - 167.074*xx + 15.6748*xx**2. - 0.491215*xx**3.)

#Lower Bound
yy_all_1 = np.concatenate((yy_1[xx<=9.9],yy_2[(xx>9.9) & (xx<11.2)]))

#Upper Bound
yy_all_2 = yy_3[(xx>9.9)]

# Plot the MEx selection regions
plt.plot(xx[(xx<11.2)],yy_all_1,color="black",linestyle="--")
plt.plot(xx[(xx>9.9)],yy_all_2,color="black",linestyle="--")

# Fill in Regions
lower_sel = np.concatenate((yy_all_1,yy_3[xx>11.2]))
upper_sel = np.concatenate((yy_1[xx<9.9],yy_all_2))

ax.fill_between(xx,-3.,lower_sel,color=tableau20("Light Blue"),alpha=0.15)
ax.fill_between(xx[(xx>9.9) & (xx<11.2)],yy_2[(xx>9.9) & (xx<11.2)],yy_3[(xx>9.9) & (xx<11.2)],color=tableau20("Light Green"),alpha=0.15)
ax.fill_between(xx,upper_sel,11.,color=tableau20("Light Red"),alpha=0.15)



#### OUR SOURCE
# Load in the Line properties
file = open("../../data/emline_fits/43158747673038238/ratios_and_ISM_props_1002_new.pkl","rb")
data = pickle.load(file)
file.close()

# Load in the SED properties (use CIGALE for now)
sed = fits.open("../../data/cigale_results/sfh_delayed_nodust/results.fits")[1].data
mass = sed["bayes.stellar.m_star"]
err_mass = sed["bayes.stellar.m_star_err"]/(np.log(10.)*mass)
mass = np.log10(mass)
data["pdf_r3"] = pow(10,data["pdf_r3"])
o3hb,o3hb_elow,o3hb_eupp = np.median(data["pdf_r3"]),np.percentile(data["pdf_r3"],16.),np.percentile(data["pdf_r3"],84.)
o3hb_elow = o3hb - o3hb_elow
o3hb_eupp = o3hb_eupp - o3hb

ax.errorbar([mass],[o3hb],xerr=([err_mass]),yerr=([o3hb_elow],[o3hb_eupp]),\
				ls="none",mec=tableau20("Blue"),mfc=tableau20("Light Blue"),ecolor=tableau20("Blue"),\
                marker="*",ms=15,mew=1,capsize=2,capthick=1,elinewidth=0.5)


###### SPECIAL SOURCES
### BOSS-EUVLG1 (Marques-Chaves et al. 2020) # 
#mass = [10.0]
#err_mass = [0.1]
#BOSS_O3HB = [562/61.]
#BOSS_O3HB_err = [562/61. * np.sqrt( (7/562.)**2. + (5./61.)**2. )]
#ax.errorbar(mass,BOSS_O3HB,xerr=(err_mass),yerr=(BOSS_O3HB_err),\
#				ls="none",mec=tableau20("Green"),mfc=tableau20("Light Green"),ecolor=tableau20("Green"),\
#                marker="d",ms=8,mew=1,capsize=2,capthick=1,elinewidth=0.5)
#
#
### Ion2 (de Barros et al. 2016)
#mass = [9.5]
#err_mass = [0.2]
#Ion2_O3Hb = [22.1/1.5]
#Ion2_O3Hb_err = [22.1/1.5 * np.sqrt( (0.8/22.1)**2. + (0.8/1.5)**2. )]
#ax.errorbar(mass,Ion2_O3Hb,xerr=(err_mass),yerr=(Ion2_O3Hb_err),			
#			ls="none",mec=tableau20("Orange"),mfc=tableau20("Light Orange"),ecolor=tableau20("Orange"),\
#            marker="p",ms=8,mew=1,capsize=2,capthick=1,elinewidth=0.5)

#### Low-z Analogs
# Blueberries (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18470038Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_Blueberries/Yang2017_Blueberries.fits")[1].data
Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["Hbeta"] > 0.)]
ax.plot(Ya17["mass"],Ya17["[OIII]5007"]/Ya17["Hbeta"],				
				ls="none",mec="#464196",mfc="none",\
                marker="s",ms=3,mew=0.5)

# Green Peas (Yang et al. 2017) -- # https://ui.adsabs.harvard.edu/abs/2018yCat..18440171Y/abstract
Ya17 = fits.open("../../../Main Catalogs/Yang2017_GreenPeas/Yang2017_GreenPeas.fits")[1].data
Ya17 = Ya17[ (Ya17["[OIII]5007"] > 0.) & (Ya17["Hbeta"] > 0.) ]
ax.plot(Ya17["logM"]-np.log10(1.64),Ya17["[OIII]5007"]/Ya17["Hbeta"],\
				ls="none",mec=tableau20("Green"),mfc="none",\
                marker="p",ms=3,mew=0.5)


# EELG (Amorin et al. 2015) -- # https://ui.adsabs.harvard.edu/abs/2015A%26A...578A.105A/abstract
Am15 = fits.open("../../../Main Catalogs/Amorin2015_EELGs/Amorin2015_EELGs.fits")[1].data
Am15 = Am15[ (Am15["F_Hb_"] > 0.) & (Am15["F_5007_"] > 0.) ]
ax.plot(Am15["logMs"],Am15["F_5007_"]/Am15["F_Hb_"],\
				ls="none",mec=tableau20("Orange"),mfc="none",\
                marker="o",ms=3,mew=0.5)


ax.text(9,8,r"\textit{MEx-AGN}",color=tableau20("Red"),fontsize=6,ha="left",va="center")
ax.text(7.2,3,r"\textit{MEx-SF}",color=tableau20("Blue"),fontsize=6,ha="left",va="center")


# Matthee et al. (2023) 5 < z < 7 #https://ui.adsabs.harvard.edu/abs/2023ApJ...950...67M/abstract
#mass = np.array([7.5,8.2,8.9,9.5])
#mass_elow = np.array([7.5-6.8,0.7/2.,0.7/2.,0.6/2.])
#mass_eupp = np.array([0.7/2.,0.7/2.,0.6/2.,10.2-9.5])
#Ma23_O3HB = np.array([5.3, 6.7, 6.3, 5.4])
#Ma23_O3HB_elow = np.array([0.8, 0.6, 0.7, 0.4])
#Ma23_O3HB_eupp = np.array([0.9, 0.6, 0.9, 0.5])
#
#ax.errorbar(mass,Ma23_O3HB,xerr=(mass_elow,mass_eupp),yerr=(Ma23_O3HB_elow,Ma23_O3HB_eupp),\
#			ls="none",mec="black",mfc=tableau20("Red"),ecolor=tableau20("Grey"),\
#            marker="o",ms=3,mew=0.2,capsize=2,capthick=1,elinewidth=0.5)




handles = [Line2D([],[],ls="none",marker="*",ms=10,mec=tableau20("Blue"),mfc=tableau20("Light Blue"),label=r"\textbf{\textit{EELG1002 (This Work)}}"),
			Line2D([],[],ls="none",marker="s",ms=3,mec="#464196",mfc="none",label=r"Blueberries (Ya+17; $z \sim 0$)"),
			Line2D([],[],ls="none",marker="p",ms=3,mec=tableau20("Green"),mfc="none",label=r"GPs (Ya+17; $z \sim 0.2$)"),
			Line2D([],[],ls="none",marker="o",ms=3,mec=tableau20("Orange"),mfc="none",label=r"EELGs (Am+15; $z \sim 0.1 - 0.9$)")]			

leg1 = ax.legend(handles=handles,loc="lower left",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg1)


handles = [	Line2D([],[],ls="--",lw=1,color="black",label=r"Ju+11")]		

leg3 = ax.legend(handles=handles,loc="upper right",frameon=False,ncol=1,numpoints=1,fontsize=5)
plt.gca().add_artist(leg3)	

plt.savefig("../../plots/O3HB_mass.png",format="png",dpi=300,bbox_inches="tight")

pdb.set_trace()
exit()



