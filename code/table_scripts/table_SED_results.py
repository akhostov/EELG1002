from astropy.io import fits
import h5py
import pickle
import numpy as np
from scipy.integrate import simps

import sys
sys.path.insert(0, '..')
import util


# Load in the Data
cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
bagpipes = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5", "r")
emlines = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data
pyneb_stats = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1].data
with open("../../data/xi_ion_measurements.pkl","rb") as f: uv_measurements = pickle.load(f)


# Open and Write All Print Statements to a Table
with open("../../paper/tables/SED_fit_properties.table","w") as table:


    ################# Stellar Mass Measurements
    cigale_mass = cigale["bayes.stellar.m_star"]
    cigale_mass_err = cigale["bayes.stellar.m_star_err"]
    bagpipes_mass = pow(10,bagpipes["median"][10])
    bagpipes_mass_err_low,bagpipes_mass_err_up = bagpipes_mass - pow(10,bagpipes["conf_int"][0][10]), pow(10,bagpipes["conf_int"][1][10]) - bagpipes_mass

    #util.write_with_newline(table,pow(10,np.transpose(bagpipes["samples2d"])[-5]))
    util.write_with_newline(table,r"Stellar Mass ($10^7$ M$_\odot$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_mass/1e7,cigale_mass_err/1e7, bagpipes_mass/1e7, bagpipes_mass_err_up/1e7, bagpipes_mass_err_low/1e7))

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

    util.write_with_newline(table,r"SFR (inst; M$_\odot$ yr$^{-1}$) & $%0.2f\pm%0.2f$ & -- & -- \\" % (cigale_SFR,cigale_SFR_err))
    util.write_with_newline(table,r"SFR (10 Myr; M$_\odot$ yr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & $^\star%0.2f^{+%0.2f}_{-%0.2f}$ \\" % (cigale_SFR10Myrs,cigale_SFR10Myrs_err,
                                                                                                                                bagpipes_SFR10Myrs,bagpipes_SFR10Myrs_err_up,bagpipes_SFR10Myrs_err_low,
                                                                                                                                SFR_Hbeta,SFR_Hbeta_eupp,SFR_Hbeta_elow))

    util.write_with_newline(table,r"SFR (100 Myr; M$_\odot$ yr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_SFR100Myrs,cigale_SFR100Myrs_err,
                                                                                                                                bagpipes_SFR100Myrs,bagpipes_SFR100Myrs_err_up,bagpipes_SFR100Myrs_err_low))


    ################# Specific Star Formation Rate
    # NEED TO CALCULATE

    # Cigale
    cigale_sSFR,cigale_sSFR10Myrs, cigale_sSFR100Myrs = (cigale_SFR,cigale_SFR10Myrs,cigale_SFR100Myrs)/cigale_mass*1e9
    cigale_sSFR_err = cigale_sSFR*np.sqrt( (cigale_SFR_err/cigale_SFR)**2 + (cigale_mass_err/cigale_mass)**2 )
    cigale_sSFR10Myrs_err = cigale_sSFR10Myrs*np.sqrt( (cigale_SFR10Myrs_err/cigale_SFR10Myrs)**2 + (cigale_mass_err/cigale_mass)**2 )
    cigale_sSFR100Myrs_err = cigale_sSFR100Myrs*np.sqrt( (cigale_SFR100Myrs_err/cigale_SFR100Myrs)**2 + (cigale_mass_err/cigale_mass)**2 )

    # Bagpipes
    bagpipes_sSFR10Myrs, bagpipes_sSFR100Myrs = (bagpipes_SFR10Myrs,bagpipes_SFR100Myrs)/bagpipes_mass*1e9
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

    util.write_with_newline(table,r"sSFR (inst; Gyr$^{-1}$) & $%0.2f\pm%0.2f$ & -- & -- \\" % (cigale_sSFR,cigale_sSFR_err))
    util.write_with_newline(table,r"sSFR (10 Myr; Gyr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & $(%0.2f^{+%0.2f}_{-%0.2f}; %0.2f^{+%0.2f}_{-%0.2f})$ \\" % (cigale_sSFR10Myrs,cigale_sSFR10Myrs_err,
                                                                                                                                bagpipes_sSFR10Myrs,bagpipes_sSFR10Myrs_err_up,bagpipes_sSFR10Myrs_err_low,
                                                                                                                                GMOS_cigale,GMOS_cigale_err_up,GMOS_cigale_err_low,
                                                                                                                                GMOS_bagpipes,GMOS_bagpipes_err_up,GMOS_bagpipes_err_low))

    util.write_with_newline(table,r"sSFR (100 Myr; Gyr$^{-1}$) & $%0.2f\pm%0.2f$ & $%0.2f^{+%0.2f}_{-%0.2f}$ & -- \\" % (cigale_sSFR100Myrs,cigale_sSFR100Myrs_err,
                                                                                                                                bagpipes_sSFR100Myrs,bagpipes_sSFR100Myrs_err_up,bagpipes_sSFR100Myrs_err_low))


    """
    ################# Metallicity
    # Get Cigale Stellar Metallicity
    cigale_metal = cigale["bayes.stellar.metallicity"]/0.02; cigale_metal_err = cigale["bayes.stellar.metallicity_err"]/0.02

    # Note Bagpipes does not differentiate gas-phase and stellar metallicity. We use the gas-phase metallicity instead
    util.write_with_newline(table,r"$Z_\star$ ($Z_\odot$) & $%0.3f\pm%0.3f$ & $%0.3f$ (fixed) & -- \\" % (cigale_metal,cigale_metal_err,
                                                                                    pow(10,pyneb_stats["12+log10(O/H)_med"]-8.69))) # The value of 8.69 is from Asplund et al. (2009) for Zsol
    """

    """
    ################# Dust Attenuation
    # Get Cigale
    cigale_EBV = cigale["bayes.attenuation.E_BV_lines"]; cigale_EBV_err = cigale["bayes.attenuation.E_BV_lines_err"]

    # Bagpipes has it in A_V which can be converted to E(B-V) assuming Calzetti et al. (2000) where k(lambda) at V-band is R_V = 4.05
    bagpipes_AV = bagpipes["median"][11]
    bagpipes_AV_elow,bagpipes_AV_eupp = bagpipes_AV - bagpipes["conf_int"][0][11], bagpipes["conf_int"][1][11] - bagpipes_AV

    calzetti_RV = 4.05
    bagpipes_EBV = bagpipes_AV/calzetti_RV
    bagpipes_EBV_elow = bagpipes_AV_elow/calzetti_RV
    bagpipes_EBV_eupp = bagpipes_AV_eupp/calzetti_RV

    util.write_with_newline(table,r"$E(B-V)$ (mag) & $%0.3f\pm%0.2f$ & $%0.3f^{+%0.3f}_{-%0.3f}$ & $%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (cigale_EBV,cigale_EBV_err,
                                                                                                            bagpipes_EBV,bagpipes_EBV_eupp,bagpipes_EBV_elow,
                                                                                                            pyneb_stats["EBV_hghb_med"],pyneb_stats["EBV_hghb_err_up"],pyneb_stats["EBV_hghb_err_low"]))
    """

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





