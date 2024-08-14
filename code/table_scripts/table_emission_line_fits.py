from astropy.io import fits
import numpy as np
import pickle
import h5py

import sys
sys.path.insert(0, '..')
import util


def table_emission_line_fits():

	data = fits.open("../../data/emline_fits/1002_lineprops.fits")[1].data

	names = np.array(["line_ID",
						"linez_med","linez_elow","linez_eupp",
						"lineflux_med","lineflux_elow","lineflux_eupp",
						"lineEW_Cigale_med","lineEW_Cigale_elow","lineEW_Cigale_eupp",
						"lineEW_Bagpipes_med","lineEW_Bagpipes_elow","lineEW_Bagpipes_eupp",
						"linesigma_med","linesigma_elow","linesigma_eupp",
						"linefwhm_med","linefwhm_elow","linefwhm_eupp"])

	# Sort to Table
	sort = np.array(["Hb_na","Hg_na","Hd_na","Hep_na","H9","H8",\
			"OIII5007c","OIII4959c","OIII4363","[OII]","OII3726","OII3728",\
			"NeIII3869","NeIII3968"])

	names = np.array([r"\hbeta",r"H$\gamma$",r"H$\delta$",r"H$\epsilon$",r"H$\zeta+$He{\sc i}3889\AA",r"H$\eta$",\
						r"\oiii5007\AA",r"\oiii4959\AA",r"\oiii4363\AA",r"\oii3726,3729\AA",r"\oii3726\AA",r"\oii3729\AA",\
						r"\neiii3869\AA",r"\neiii3968\AA"])

	keep = np.hstack([np.argwhere(tt == data["line_id"])[0] for tt in sort])
	data = data[keep]

	# Open and Write All Print Statements to a Table
	with open("../../paper/tables/Line_Flux_Props.table","w") as table:

		for ff in range(len(data["line_id"])):

			data['linez_eupp'][ff] = np.sqrt( (data['linez_eupp'][ff])**2. + (2.*(1.+data['linez_med'][ff])/631.)**2. )
			data['linez_elow'][ff] = np.sqrt( (data['linez_elow'][ff])**2. + (2.*(1.+data['linez_med'][ff])/631.)**2. )

			str_lflux = "$%0.2f^{+%0.2f}_{-%0.2f}$" % (data['lineflux_med'][ff]*1e17,data['lineflux_eupp'][ff]*1e17,data['lineflux_elow'][ff]*1e17)
			str_lEW_Cigale   = "$%0.2f^{+%0.2f}_{-%0.2f}$" % (data['lineEW_Cigale_med'][ff],data['lineEW_Cigale_eupp'][ff],data['lineEW_Cigale_elow'][ff])	
			str_lEW_Bagpipes = "$%0.2f^{+%0.2f}_{-%0.2f}$" % (data['lineEW_Bagpipes_med'][ff],data['lineEW_Bagpipes_eupp'][ff],data['lineEW_Bagpipes_elow'][ff])	

			if data["line_id"][ff] in ("Hb_na","OIII5007c","NeIII3869"):
				str_redshift = "$%0.4f^{+%0.4f}_{-%0.4f}$" % (data['linez_med'][ff],data['linez_eupp'][ff],data['linez_elow'][ff])
				str_lfwhm = "$%0.0f^{+%0.0f}_{-%0.0f}$" % (data['linefwhm_med'][ff],data['linefwhm_eupp'][ff],data['linefwhm_elow'][ff])
				str_lsigma = "$%0.0f^{+%0.0f}_{-%0.0f}$" % (data['linesigma_med'][ff],data['linesigma_eupp'][ff],data['linesigma_elow'][ff])
			else:
				str_redshift = "---"
				str_lfwhm = "---"
				str_lsigma = "---"


			util.write_with_newline(table,f"{names[ff]} & {str_redshift} & {str_lflux} & {str_lEW_Cigale} & {str_lEW_Bagpipes} & {str_lfwhm} & {str_lsigma} \\\\")


def table_emission_line_ratios():

	# Open and Write All Print Statements to a Table
	with open("../../paper/tables/ISM_and_Line_properties.table","w") as table:

		# Load in the Pyneb Stats File
		pyneb_stat = fits.open("../../data/emline_fits/1002_pyneb_stats.fits")[1].data

		# Load in the Line ratios pickle file
		with open("../../data/emline_fits/1002_line_ratios.pkl","rb") as file:
			line_ratio = pickle.load(file)

		# Electron Temperature and Density
		util.write_with_newline(table,"\\multicolumn{2}{l}{\\textbf{Electron Temperature \\& Density}} \\\\")
		util.write_with_newline(table,"$T_e(\\textrm{O}^{++})$ & "+r"$%0.0f^{+%0.0f}_{-%0.0f}$ \\" % (pyneb_stat["te_O++_med"],pyneb_stat["te_O++_err_up"],pyneb_stat["te_O++_err_low"]))
		util.write_with_newline(table,"$T_e(\\textrm{O}^{+})$ & "+r"$%0.0f^{+%0.0f}_{-%0.0f}$ \\" % (pyneb_stat["te_O+_med"],pyneb_stat["te_O+_err_up"],pyneb_stat["te_O+_err_low"]))
		util.write_with_newline(table,"$n_e$ & "+r"$%0.0f^{+%0.0f}_{-%0.0f}$ \\" % (pyneb_stat["ne_med"],pyneb_stat["ne_err_up"],pyneb_stat["ne_err_low"]))

		# Oxygen Abundance
		util.write_with_newline(table,"\\multicolumn{2}{l}{\\textbf{Oxygen Abundance}} \\\\")
		util.write_with_newline(table,r"$12+\log_{10}(\rm{O}^{++}/\rm{H})$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (pyneb_stat["12+log10(O++/H)_med"],pyneb_stat["12+log10(O++/H)_err_up"],pyneb_stat["12+log10(O++/H)_err_low"]))
		util.write_with_newline(table,r"$12+\log_{10}(\rm{O}^{+}/\rm{H})$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (pyneb_stat["12+log10(O+/H)_med"],pyneb_stat["12+log10(O+/H)_err_up"],pyneb_stat["12+log10(O+/H)_err_low"]))
		util.write_with_newline(table,r"$12+\log_{10}(\rm{O}/\rm{H})$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (pyneb_stat["12+log10(O/H)_med"],pyneb_stat["12+log10(O/H)_err_up"],pyneb_stat["12+log10(O/H)_err_low"]))

		# Gas Metallicity with 8.69 set by Apslund et al. (2009)
		zgas_med = pow(10,pyneb_stat["12+log10(O/H)_med"] - 8.69)
		zgas_elow = zgas_med - pow(10,pyneb_stat["12+log10(O/H)_med"] - pyneb_stat["12+log10(O/H)_err_low"] - 8.69)
		zgas_eup = pow(10,pyneb_stat["12+log10(O/H)_med"] + pyneb_stat["12+log10(O/H)_err_up"] - 8.69) - zgas_med
		util.write_with_newline(table,r"$Z_{gas} (Z_{\odot})$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (zgas_med,zgas_eup,zgas_elow))

		# Neon Abundance
		util.write_with_newline(table,"\\multicolumn{2}{l}{\\textbf{Neon Abundance}} \\\\")
		util.write_with_newline(table,r"$12+\log_{10}(\rm{Ne}^{++}/\rm{H})$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (pyneb_stat["12+log10(Ne++/H)_med"],pyneb_stat["12+log10(Ne++/H)_err_up"],pyneb_stat["12+log10(Ne++/H)_err_low"]))
		util.write_with_newline(table,r"$12+\log_{10}(\rm{Ne}/\rm{H})$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (pyneb_stat["12+log10(Ne/H)_med"],pyneb_stat["12+log10(Ne/H)_err_up"],pyneb_stat["12+log10(Ne/H)_err_low"]))
		util.write_with_newline(table,r"ICF(Ne$^{++}$) & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (pyneb_stat["ICF_Ne_med"],pyneb_stat["ICF_Ne_err_up"],pyneb_stat["ICF_Ne_err_low"]))
		util.write_with_newline(table,r"$\log_{10}(\rm{Ne}/\rm{O})$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (pyneb_stat["log10(Ne/O)_med"],pyneb_stat["log10(Ne/O)_err_up"],pyneb_stat["log10(Ne/O)_err_low"]))



		# Ionization Parameter
		util.write_with_newline(table,"\\multicolumn{2}{l}{\\textbf{Ionization Parameter} -- $\\boldsymbol{\\log_{10} U}$} \\\\")
		# HCM
		hcm_logU, hcm_logU_err = np.loadtxt("../../data/HCM_EELG1002_hcm-output.dat",unpack=True,dtype=str,usecols=(-2,-1),comments="#")
		hcm_logU = np.double(hcm_logU[-1]); hcm_logU_err = np.double(hcm_logU_err[-1])
		util.write_with_newline(table,r"\texttt{HCM} + \texttt{BPASS} & "+r"$%0.2f\pm%0.2f$ \\" % (hcm_logU,hcm_logU_err))

		# BAGPIPES
		bagpipes = h5py.File("../../data/SED_results/bagpipes_results/pipes/posterior/sfh_continuity_spec_BPASS/1002.h5","r")
		bagpipes_logU_med, bagpipes_logU_err_low, bagpipes_logU_err_up = util.stats(np.transpose(bagpipes['samples2d'])[-3])
		util.write_with_newline(table,r"\texttt{Bagpipes} + \texttt{BPASS} & "+r"$%0.2f^{+%0.2f}_{-%0.2f}$ \\" % (bagpipes_logU_med,bagpipes_logU_err_up,bagpipes_logU_err_low))

		# Cigale
		cigale = fits.open("../../data/SED_results/cigale_results/results.fits")[1].data
		util.write_with_newline(table,r"\texttt{Cigale} + \texttt{BC03} & "+r"$%0.2f\pm%0.2f$ \\" % (cigale["bayes.nebular.logU"],cigale["bayes.nebular.logU_err"]))




		# Balmer Decrement
		util.write_with_newline(table,"\\multicolumn{2}{l}{\\textbf{Balmer Decrement}} \\\\")
		# Hb/Hg
		index = line_ratio["name"].index("Hb/Hg")
		Hbg_measured = "\hbeta/\hgamma & " +r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,Hbg_measured)

		# Hb/Hd
		index = line_ratio["name"].index("Hb/Hd")
		Hbd_measured = "\hbeta/\hdelta & " +r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,Hbd_measured)

		# Hb/Hep
		index = line_ratio["name"].index("Hb/Hep")
		Hbep_measured =  "\hbeta/H$\epsilon$ & " +r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,Hbep_measured)


		util.write_with_newline(table,r"$E(B-V)$ -- (H$\beta$/H$\gamma$) &  $%0.2f^{+%0.2f}_{-%0.2f}$ mag\\" % (pyneb_stat["EBV_hghb_med"], pyneb_stat["EBV_hghb_err_up"], pyneb_stat["EBV_hghb_err_low"])  )
		#util.write_with_newline(table,r"								 &  $%0.2f$ mag ($<2\sigma$) \\" % (pyneb_stat["EBV_hghb_2sig_limit"]) )
		#util.write_with_newline(table,r"$E(B-V)$ -- (H$\beta$/H$\delta$) & $%0.2f^{+%0.2f}_{-%0.2f}$ mag\\" % (pyneb_stat["EBV_hdhb_med"], pyneb_stat["EBV_hdhb_err_up"], pyneb_stat["EBV_hdhb_err_low"])  )




		# Line Ratios
		util.write_with_newline(table,"\\multicolumn{2}{l}{\\textbf{Line ratios}} \\\\")
		# O3HB
		index = line_ratio["name"].index("O3HB")
		o3hb_measured = "$\\textrm{\oiii}_{5007\\textrm{\scriptsize\AA}}/\\textrm{\hbeta}$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,o3hb_measured)

		# O32
		index = line_ratio["name"].index("O32")
		o32_measured = "$\\textrm{\oiii}_{5007\\textrm{\scriptsize\AA}}/\\textrm{\oii}_{3726,3729\\textrm{\scriptsize\AA}}$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,o32_measured)

		# R2
		index = line_ratio["name"].index("R2")
		R2_measured = "$\\textrm{\oii}_{3726,3729\\textrm{\scriptsize\AA}}/\\textrm{\hbeta}$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,R2_measured)

		# R3
		index = line_ratio["name"].index("R3")
		R3_measured = "$\\textrm{\oiii}_{5007,4959\\textrm{\scriptsize\AA}}/\\textrm{\hbeta}$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$\\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,R3_measured)

		# R23
		index = line_ratio["name"].index("R23")
		R23_measured = "$(\\textrm{\oiii}_{5007,4959\\textrm{\scriptsize\AA}}  + \\textrm{\oii}_{3726,3729\\textrm{\scriptsize\AA}})/\\textrm{\hbeta}$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,R23_measured)
		
		# Auroral
		index = line_ratio["name"].index("Auroral")
		Auroral_measured = "$\\textrm{\oiii}_{4363\\textrm{\scriptsize\AA}}/\\textrm{\oiii}_{5007\\textrm{\scriptsize\AA}}$ & " +r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,Auroral_measured)

		# Ne3O2
		index = line_ratio["name"].index("Ne3O2")
		Ne3O2_measured = "$\\textrm{\\neiii}_{3869\\textrm{\scriptsize\AA}}/\\textrm{\oii}_{3726,3729\\textrm{\scriptsize\AA}}$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,Ne3O2_measured)
		
		# Ne3O3
		index = line_ratio["name"].index("Ne3O3")
		Ne3O3_measured = "$\\textrm{\\neiii}_{3869\\textrm{\scriptsize\AA}}/\\textrm{\oiii}5007\\textrm{\AA}$ & "+r"$%0.3f^{+%0.3f}_{-%0.3f}$ \\" % (line_ratio["median"][index],line_ratio["upp_1sigma"][index],line_ratio["low_1sigma"][index])
		util.write_with_newline(table,Ne3O3_measured)

	table.close()



table_emission_line_ratios()
table_emission_line_fits()

