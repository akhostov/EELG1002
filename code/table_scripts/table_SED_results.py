from astropy.io import fits
from astropy import constants as const
from astropy import units as un
import h5py
import pickle
import numpy as np
from scipy.integrate import simps
import asdf

import sys
sys.path.insert(0, '..')
import util


# Load in the Data
cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
bagpipes = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")
emlines = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data
pyneb_stats = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1].data
pysersic = asdf.open('../../data/pysersic_results/pysersic_fit_HST_ACS_F814W.asdf')
with open("../../data/xi_ion_measurements.pkl","rb") as f: uv_measurements = pickle.load(f)


# Load in Spectra for the Sigma Results
spectra = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data
keep = spectra["line_id"] == "Hb_na"
spectra = spectra[keep] # I only will use Hbeta sigma

# Define Sampling Size
nele = 1000

with open("../../paper/tables/Morphology_table.table","w") as table:
   
    cosmo = util.get_cosmology()

    # This is to keep a same-size sampling
    ind_sampling = np.random.randint(low = 0, high = len(pysersic.tree["posterior"]["r_eff"]), size = nele)

    # Half Light Radius (initially in pixels but we convert to kpc for use in the Sigma SFR measurement. However for the table we set it to pc. Thesea re all in proper NOT comoving units.
    pdf_re = pysersic.tree["posterior"]["r_eff"]*0.03*cosmo.kpc_proper_per_arcmin(0.8275).value/60. # Note the 0.03 arcsec per pixel which is the F814W pixel scale
    r_e,r_e_elow,r_e_eupp = util.stats(pdf_re)
    util.write_with_newline(table,r"$r_e$ (pc; proper) & $%0.0f^{+%0.0f}_{-%0.0f}$\\" % (r_e*1000., r_e_eupp*1000., r_e_elow*1000.))

    # Sersic Index
    n_ind,n_ind_elow,n_ind_eupp = util.stats(pysersic.tree["posterior"]["n"])
    util.write_with_newline(table,r"$n$ & $%0.2f^{+%0.2f}_{-%0.2f}$\\" % (n_ind,n_ind_eupp,n_ind_elow))

    ############### Dynamical Mass
    pdf_sigma = spectra["linesigma_int_med"] + np.concatenate( ( -1.*np.abs(np.random.normal(loc=spectra["linesigma_int_med"], scale=spectra["linesigma_int_elow"], size=int(nele/2))),
                                                                     np.abs(np.random.normal(loc=spectra["linesigma_int_med"], scale=spectra["linesigma_int_eupp"], size=int(nele/2))) ) )

    # This is the constant that is dependent on the mass distribution. I assumed a uniform distribution that is centered on 3 but is allowed to spread from 1 to 5 based on Erb et al. (2006)
    pdf_constant = np.random.uniform(low=1,high=5,size=nele)
    pdf_Mdyn = (pdf_constant*(pdf_sigma*1000.*un.m/un.s)**2.*(pdf_re[ind_sampling] * 1000. * un.pc).to(un.m)/const.G).to(un.Msun).value

    Mdyn,Mdyn_elow,Mdyn_eupp = util.stats(pdf_Mdyn)

    util.write_with_newline(table,r"Dynamical Mass (10$^9$ M$_\odot$) & $%0.2f^{+%0.2f}_{-%0.2f}$\\" % (Mdyn/1e9,Mdyn_eupp/1e9,Mdyn_elow/1e9))


# Open and Write All Print Statements to a Table
with open("../../paper/tables/SED_fit_properties.table","w") as table:

    ################# Stellar Mass Measurements
    cigale_mass = cigale["bayes.stellar.m_star"]
    cigale_mass_err = cigale["bayes.stellar.m_star_err"]
    bagpipes_mass = pow(10,bagpipes["median"][10])
    bagpipes_mass_err_low,bagpipes_mass_err_up = bagpipes_mass - pow(10,bagpipes["conf_int"][0][10]), pow(10,bagpipes["conf_int"][1][10]) - bagpipes_mass

    util.write_with_newline(table,r"Stellar Mass ($10^8$ M$_\odot$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_mass/1e8,cigale_mass_err/1e8, bagpipes_mass/1e8, bagpipes_mass_err_up/1e8, bagpipes_mass_err_low/1e8))

    ################# Star Formation Rates
    cigale_SFR = cigale["bayes.sfh.sfr"]; cigale_SFR_err = cigale["bayes.sfh.sfr_err"]
    cigale_SFR10Myrs = cigale["bayes.sfh.sfr10Myrs"]; cigale_SFR10Myrs_err = cigale["bayes.sfh.sfr10Myrs_err"]
    cigale_SFR100Myrs = cigale["bayes.sfh.sfr100Myrs"]; cigale_SFR100Myrs_err = cigale["bayes.sfh.sfr100Myrs_err"]

    # Note that for Bagpipes we actually have to calculate this using the nonparametric SFH to have a comparison to cigale
    # First Load the SFH that was formatted in post-processing
    bagpipes_SFH = fits.open("../../data/SED_results/bagpipes_results/best_fit_SFH_sfh_continuity_spec_BPASS.fits")[1].data

    # Now convert time such that the 0th index is at 0 Gyr (lookback time starting at z = 0.8275)
    bagpipes_SFH["Lookback Time"] = bagpipes_SFH["Lookback Time"][0] - bagpipes_SFH["Lookback Time"] 

    # Fix the error bars as they are 16th and 84th percentiles
    bagpipes_SFH["SFH_1sig_low"] = bagpipes_SFH["SFH_median"] - bagpipes_SFH["SFH_1sig_low"]
    bagpipes_SFH["SFH_1sig_upp"] = bagpipes_SFH["SFH_1sig_upp"] - bagpipes_SFH["SFH_median"]

    # Extract the instantaneous SFR
    bagpipes_SFR = bagpipes_SFH["SFH_median"][0]; bagpipes_SFR_elow = bagpipes_SFH["SFH_1sig_low"][0]; bagpipes_SFR_eupp = bagpipes_SFH["SFH_1sig_upp"][0]

    bagpipes_SFR10Myrs,bagpipes_SFR10Myrs_err_low,bagpipes_SFR10Myrs_err_up = util.calc_SFR(bagpipes_SFH["Lookback Time"],bagpipes_SFH["SFH_median"],
                                                                                            max_time_limit=0.01,
                                                                                            sigma=(bagpipes_SFH["SFH_1sig_low"],bagpipes_SFH["SFH_1sig_upp"]))

    bagpipes_SFR100Myrs,bagpipes_SFR100Myrs_err_low,bagpipes_SFR100Myrs_err_up = util.calc_SFR(bagpipes_SFH["Lookback Time"],bagpipes_SFH["SFH_median"],
                                                                                            max_time_limit=0.10,
                                                                                            sigma=(bagpipes_SFH["SFH_1sig_low"],bagpipes_SFH["SFH_1sig_upp"]))

    # Get the Hbeta Emission Line SFR from the GMOS Spectra
    mask = emlines["line_ID"] == "Hb_na"
    emlines = emlines[mask]

    SFR_Hbeta      = (4.4e-42*2.86)*util.lineFlux_to_Luminosity(lineFlux=emlines["lineflux_med"], redshift=0.8275)
    SFR_Hbeta_elow = (4.4e-42*2.86)*util.lineFlux_to_Luminosity(lineFlux=emlines["lineflux_elow"],redshift=0.8275)
    SFR_Hbeta_eupp = (4.4e-42*2.86)*util.lineFlux_to_Luminosity(lineFlux=emlines["lineflux_eupp"],redshift=0.8275)

    util.write_with_newline(table,r"SFR (1 Myr; M$_\odot$ yr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & $^\star%0.2f^{+%0.2f}_{-%0.2f}$ \\" % (cigale_SFR,cigale_SFR_err,
                                                                                                                                                            bagpipes_SFR,bagpipes_SFR_eupp,bagpipes_SFR_elow,
                                                                                                                                                            SFR_Hbeta,SFR_Hbeta_eupp,SFR_Hbeta_elow))
    util.write_with_newline(table,r"SFR (10 Myr; M$_\odot$ yr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_SFR10Myrs,cigale_SFR10Myrs_err,
                                                                                                                                bagpipes_SFR10Myrs,bagpipes_SFR10Myrs_err_up,bagpipes_SFR10Myrs_err_low))

    util.write_with_newline(table,r"SFR (100 Myr; M$_\odot$ yr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_SFR100Myrs,cigale_SFR100Myrs_err,
                                                                                                                                bagpipes_SFR100Myrs,bagpipes_SFR100Myrs_err_up,bagpipes_SFR100Myrs_err_low))


    ################# Star Formation Rates Surface Densities

    # Cigale
    pdf_cigale_SFR = np.random.normal(loc=cigale_SFR, scale=cigale_SFR_err, size=nele)
    pdf_cigale_SFR10Myrs = np.random.normal(loc=cigale_SFR10Myrs, scale=cigale_SFR10Myrs_err, size=nele)
    pdf_cigale_SFR100Myrs = np.random.normal(loc=cigale_SFR100Myrs, scale=cigale_SFR100Myrs_err, size=nele)

    cigale_Sigma_SFR,\
        cigale_Sigma_SFR_elow,\
        cigale_Sigma_SFR_eupp = util.stats(pdf_cigale_SFR/(2*np.pi*pdf_re[ind_sampling]**2.))
    
    cigale_Sigma_SFR10Myrs,\
        cigale_Sigma_SFR10Myrs_elow,\
        cigale_Sigma_SFR10Myrs_eupp = util.stats(pdf_cigale_SFR10Myrs/(2*np.pi*pdf_re[ind_sampling]**2.))

    cigale_Sigma_SFR100Myrs,\
        cigale_Sigma_SFR100Myrs_elow,\
        cigale_Sigma_SFR100Myrs_eupp = util.stats(pdf_cigale_SFR100Myrs/(2*np.pi*pdf_re[ind_sampling]**2.))

    # Bagpipes
    pdf_bagpipes_SFR1Myrs = bagpipes_SFR + np.concatenate(( -1.*np.abs(np.random.normal(loc=0., scale=bagpipes_SFR_elow, size=int(nele/2))),
                                                                        np.abs(np.random.normal(loc=0., scale=bagpipes_SFR_eupp, size=int(nele/2))) ))
    
    pdf_bagpipes_SFR10Myrs = bagpipes_SFR10Myrs + np.concatenate(( -1.*np.abs(np.random.normal(loc=0., scale=bagpipes_SFR10Myrs_err_low, size=int(nele/2))),
                                                                        np.abs(np.random.normal(loc=0., scale=bagpipes_SFR10Myrs_err_up, size=int(nele/2))) ))

    pdf_bagpipes_SFR100Myrs = bagpipes_SFR100Myrs + np.concatenate(( -1.*np.abs(np.random.normal(loc=0., scale=bagpipes_SFR100Myrs_err_low, size=int(nele/2))),
                                                                         np.abs(np.random.normal(loc=0., scale=bagpipes_SFR100Myrs_err_up, size=int(nele/2))) ))

    bagpipes_Sigma_SFR1Myrs,\
        bagpipes_Sigma_SFR1Myrs_elow,\
        bagpipes_Sigma_SFR1Myrs_eupp = util.stats(pdf_bagpipes_SFR1Myrs/(2*np.pi*pdf_re[ind_sampling]**2.))

    bagpipes_Sigma_SFR10Myrs,\
        bagpipes_Sigma_SFR10Myrs_elow,\
        bagpipes_Sigma_SFR10Myrs_eupp = util.stats(pdf_bagpipes_SFR10Myrs/(2*np.pi*pdf_re[ind_sampling]**2.))

    bagpipes_Sigma_SFR100Myrs,\
        bagpipes_Sigma_SFR100Myrs_elow,\
        bagpipes_Sigma_SFR100Myrs_eupp = util.stats(pdf_bagpipes_SFR100Myrs/(2*np.pi*pdf_re[ind_sampling]**2.))

    # GMOS
    pdf_SFR_Hbeta =  SFR_Hbeta + np.concatenate(( -1.*np.abs(np.random.normal(loc=0., scale=SFR_Hbeta_elow, size=int(nele/2))),
                                                        np.abs(np.random.normal(loc=0., scale=SFR_Hbeta_eupp, size=int(nele/2))) ))

    Hbeta_Sigma_SFR,\
        Hbeta_Sigma_SFR_elow,\
        Hbeta_Sigma_SFR_eupp = util.stats(pdf_SFR_Hbeta/(2*np.pi*pdf_re[ind_sampling]**2.))


    util.write_with_newline(table,r"$\Sigma_\textrm{SFR}$ (1 Myr; M$_\odot$ yr$^{-1}$ kpc$^{-1}$) & $%0.2f^{+%0.2f}_{-%0.2f}$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & $^\star%0.2f^{+%0.2f}_{-%0.2f}$ \\" % (cigale_Sigma_SFR,cigale_Sigma_SFR_eupp,cigale_Sigma_SFR_elow,
                                                                                                                                               bagpipes_Sigma_SFR1Myrs,bagpipes_Sigma_SFR1Myrs_eupp,bagpipes_Sigma_SFR1Myrs_elow,
                                                                                                                                               Hbeta_Sigma_SFR,Hbeta_Sigma_SFR_eupp,Hbeta_Sigma_SFR_elow))
    
    util.write_with_newline(table,r"$\Sigma_\textrm{SFR}$ (10 Myr; M$_\odot$ yr$^{-1}$ kpc$^{-1}$) & $%0.2f^{+%0.2f}_{-%0.2f}$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_Sigma_SFR10Myrs,cigale_Sigma_SFR10Myrs_eupp,cigale_Sigma_SFR10Myrs_elow,
                                                                                                                                bagpipes_Sigma_SFR10Myrs,bagpipes_Sigma_SFR10Myrs_eupp,bagpipes_Sigma_SFR10Myrs_elow))

    util.write_with_newline(table,r"$\Sigma_\textrm{SFR}$ (100 Myr; M$_\odot$ yr$^{-1}$ kpc$^{-1}$) & $%0.2f^{+%0.2f}_{-%0.2f}$& $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_Sigma_SFR100Myrs,cigale_Sigma_SFR100Myrs_eupp,cigale_Sigma_SFR100Myrs_elow,
                                                                                                                                bagpipes_Sigma_SFR100Myrs,bagpipes_Sigma_SFR100Myrs_eupp,bagpipes_Sigma_SFR100Myrs_elow))



    ################# Specific Star Formation Rate
    # Cigale
    cigale_sSFR,cigale_sSFR10Myrs, cigale_sSFR100Myrs = (cigale_SFR,cigale_SFR10Myrs,cigale_SFR100Myrs)/cigale_mass*1e9
    cigale_sSFR_err = cigale_sSFR*np.sqrt( (cigale_SFR_err/cigale_SFR)**2 + (cigale_mass_err/cigale_mass)**2 )
    cigale_sSFR10Myrs_err = cigale_sSFR10Myrs*np.sqrt( (cigale_SFR10Myrs_err/cigale_SFR10Myrs)**2 + (cigale_mass_err/cigale_mass)**2 )
    cigale_sSFR100Myrs_err = cigale_sSFR100Myrs*np.sqrt( (cigale_SFR100Myrs_err/cigale_SFR100Myrs)**2 + (cigale_mass_err/cigale_mass)**2 )

    # Bagpipes
    bagpipes_sSFR1Myrs,bagpipes_sSFR10Myrs, bagpipes_sSFR100Myrs = (bagpipes_SFR,bagpipes_SFR10Myrs,bagpipes_SFR100Myrs)/bagpipes_mass*1e9

    bagpipes_sSFR1Myrs_err_low = bagpipes_sSFR1Myrs*np.sqrt( (bagpipes_SFR_elow/bagpipes_SFR)**2 + (bagpipes_mass_err_low/bagpipes_mass)**2 )
    bagpipes_sSFR1Myrs_err_up = bagpipes_sSFR1Myrs*np.sqrt( (bagpipes_SFR_eupp/bagpipes_SFR)**2 + (bagpipes_mass_err_up/bagpipes_mass)**2 )

    bagpipes_sSFR10Myrs_err_low = bagpipes_sSFR10Myrs*np.sqrt( (bagpipes_SFR10Myrs_err_low/bagpipes_SFR10Myrs)**2 + (bagpipes_mass_err_low/bagpipes_mass)**2 )
    bagpipes_sSFR10Myrs_err_up = bagpipes_sSFR10Myrs*np.sqrt( (bagpipes_SFR10Myrs_err_up/bagpipes_SFR10Myrs)**2 + (bagpipes_mass_err_up/bagpipes_mass)**2 )
    bagpipes_sSFR100Myrs_err_low = bagpipes_sSFR100Myrs*np.sqrt( (bagpipes_SFR100Myrs_err_low/bagpipes_SFR100Myrs)**2 + (bagpipes_mass_err_low/bagpipes_mass)**2 )
    bagpipes_sSFR100Myrs_err_up = bagpipes_sSFR100Myrs*np.sqrt( (bagpipes_SFR100Myrs_err_up/bagpipes_SFR100Myrs)**2 + (bagpipes_mass_err_up/bagpipes_mass)**2 )

    # GMOS
    GMOS_bagpipes = SFR_Hbeta/bagpipes_mass*1e9
    GMOS_bagpipes_err_low = GMOS_bagpipes*np.sqrt( (SFR_Hbeta_elow/SFR_Hbeta)**2 + (bagpipes_mass_err_low/bagpipes_mass)**2 )
    GMOS_bagpipes_err_up = GMOS_bagpipes*np.sqrt( (SFR_Hbeta_eupp/SFR_Hbeta)**2 + (bagpipes_mass_err_up/bagpipes_mass)**2 )

    GMOS_cigale = SFR_Hbeta/cigale_mass*1e9
    GMOS_cigale_err_low = GMOS_cigale * np.sqrt( (SFR_Hbeta_elow/SFR_Hbeta)**2 + (cigale_mass_err/cigale_mass)**2 )
    GMOS_cigale_err_up = GMOS_cigale * np.sqrt( (SFR_Hbeta_eupp/SFR_Hbeta)**2 + (cigale_mass_err/cigale_mass)**2 )

    util.write_with_newline(table,r"sSFR (1 Myr; Gyr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & $(%0.2f^{+%0.2f}_{-%0.2f}; %0.2f^{+%0.2f}_{-%0.2f})$ \\" % (cigale_sSFR,cigale_sSFR_err,
                                                                                                bagpipes_sSFR1Myrs,bagpipes_sSFR1Myrs_err_up,bagpipes_sSFR1Myrs_err_low,
                                                                                                GMOS_cigale,GMOS_cigale_err_up,GMOS_cigale_err_low,
                                                                                                GMOS_bagpipes,GMOS_bagpipes_err_up,GMOS_bagpipes_err_low))
    
    util.write_with_newline(table,r"sSFR (10 Myr; Gyr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_sSFR10Myrs,cigale_sSFR10Myrs_err,
                                                                                                                                bagpipes_sSFR10Myrs,bagpipes_sSFR10Myrs_err_up,bagpipes_sSFR10Myrs_err_low))

    util.write_with_newline(table,r"sSFR (100 Myr; Gyr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_sSFR100Myrs,cigale_sSFR100Myrs_err,
                                                                                                                                bagpipes_sSFR100Myrs,bagpipes_sSFR100Myrs_err_up,bagpipes_sSFR100Myrs_err_low))


    ################# Ionizing Photon Production Efficency    
    cigale_xi_ion = uv_measurements["cigale_xi_ion"][0]
    cigale_xi_ion_err_low,cigale_xi_ion_err_up = uv_measurements["cigale_xi_ion"][1:3]/(np.log(10.)*cigale_xi_ion)
    bagpipes_xi_ion = uv_measurements["bagpipes_xi_ion"][0]
    bagpipes_xi_ion_err_low,bagpipes_xi_ion_err_up = uv_measurements["bagpipes_xi_ion"][1:3]/(np.log(10.)*bagpipes_xi_ion)

    util.write_with_newline(table,r"$\xi_{\textrm{ion}}^{\textrm{H{\sc ii}}}$ (erg$^{-1}$ Hz) & $%0.2f^{+%0.2f}_{-%0.2f}$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (np.log10(cigale_xi_ion),cigale_xi_ion_err_up,cigale_xi_ion_err_low,
                                                                                                                                                                np.log10(bagpipes_xi_ion),bagpipes_xi_ion_err_up,bagpipes_xi_ion_err_low))


    ################# UV Dust-Corrected Luminosity 
    cigale_M_UV = uv_measurements["cigale_M_UV"][0]
    cigale_M_UV_elow,cigale_M_UV_eup = uv_measurements["cigale_M_UV"][1:3]
    bagpipes_M_UV = uv_measurements["bagpipes_M_UV"][0]
    bagpipes_M_UV_err_low,bagpipes_M_UV_err_up = uv_measurements["bagpipes_M_UV"][1:3]

    util.write_with_newline(table,r"$M_{UV} (\textrm{mag})$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_M_UV,cigale_M_UV_eup,cigale_M_UV_elow,
                                                                                                                                                                bagpipes_M_UV,bagpipes_M_UV_err_up,bagpipes_M_UV_err_low))


   ################# [OIII] + Hbeta EW
    cigale_EW_O3HB = uv_measurements["cigale_O3HB_EW"][0]
    cigale_EW_O3HB_elow,cigale_EW_O3HB_eup = uv_measurements["cigale_O3HB_EW"][1:3]
    bagpipes_EW_O3HB = uv_measurements["bagpipes_O3HB_EW"][0]
    bagpipes_EW_O3HB_err_low,bagpipes_EW_O3HB_err_up = uv_measurements["bagpipes_O3HB_EW"][1:3]

    util.write_with_newline(table,r"EW$_0(\textrm{[O{\sc iii}] + H}\beta) (\textrm{\AA})$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_EW_O3HB,cigale_EW_O3HB_eup,cigale_EW_O3HB_elow,
                                                                                                                                                                bagpipes_EW_O3HB,bagpipes_EW_O3HB_err_up,bagpipes_EW_O3HB_err_low))





