import uproot
import matplotlib
from matplotlib import pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

MASS_BINS = [0,136.7,148.3,165,175,185,195,205,220,240,260,285,325,375,425,475,525,575,650,750,850,950]


#mcfm_file_name ="MCFM_validation_plots_specialSelection_shellSplit.root"  #"MCFM_validation_plots_specialSelection_zoomed.root"
#"MCFM_validation_plots_specialSelection.root" #hww_mcfm_for_validation_Apr10.root" #"hww_mcfm_for_validation_Apr5_specSelection.root" #"MCFM_validation_plots.root" #"hww_mcfm_for_validation_Apr4.root"

#ntuple_file_name = "hww_mcfm_validationMay3_specialSelection_shellSplit_0Jet.root" #"hww_mcfm_validationMay1_specialSelection_shellSplit.root" #"hww_mcfm_validationApr13_specialSelection_shellSplit.root" #hww_mcfm_validationApr10_specialSelection_zoomed.root"

#"hww_mcfm_validationApr10_specialSelection.root" #hww_mcfm_validationApr10.root" #"hww_processed_for_valdiation_Apr5_specSelection.root" #"hww_processed_for_validation_Apr5_specSelection.root"#"hww_processed_for_valdiation_Apr4.root"

plots = ["mww", "mll", "ptll", "MET", "mw", "ele_eta", "ele_pt", "muon_eta", "muon_pt", "mww_offshell", "mll_offshell", "ptll_offshell", "MET_offshell", "mw_offshell", "ele_eta_offshell", "ele_pt_offshell", "muon_eta_offshell", "muon_pt_offshell", "mww_onshell", "mll_onshell", "ptll_onshell", "MET_onshell", "mw_onshell", "ele_eta_onshell", "ele_pt_onshell", "muon_eta_onshell", "muon_pt_onshell"]

#mcfm_file_name = "MCFM_validation_plots_specialSelection_shellSplit_CONT.root"
#ntuple_file_name = "hww_mcfm_validationMay3_specialSelection_shellSplit_CONT.root"
#mcfm_file_name = "hww_2018MC_Validation.root"

#ntuple_file_name = "hww_2018MC_Validation.root"
#ntuple_file_name = "hww_mcfm_validationMay3_specialSelection_shellSplit_CONT.root"
#ntuple_file_name = "hww_mcfm_validationMay9_specialSelection_shellSplit_CONT_v2.root"

#mcfm_file_name = "MCFM_validation_plots_specialSelection_shellSplit.root"
mcfm_file_name = "MCFM_validation_plots_specialSelection_shellSplit_CONT_VBF.root"
mcfm_file_name = "MCFM_validation_plots_specialSelection_shellSplit_SIG_100k_VBF.root"
#ntuple_file_name = "hww_mcfm_validation_v2_SFonly_200.root"
#ntuple_file_name = "hww_mcfm_validation_v2_SFonly_500.root"
#ntuple_file_name = "hww_mcfm_validation_v2_SFonly_unscaled_500.root"
#ntuple_file_name = "hww_mcfm_validationMay10_specialSelection_shellSplit_SIG_v2_noSF.root"
#ntuple_file_name = "hww_validationSept28_specialSelection_shellSplit_CONT.root"
ntuple_file_name = "hww_validationSept28_specialSelection_shellSplit_SIG.root"

#ntuple_file_name = "hww_mcfm_validationMay10_specialSelection_shellSplit_SIG_v2_1p6SF.root"
#plots = ["mww"]

axis_label = ["$m_{WW}$", "$m_{ll}$", "$p_{T}^{ll}$", "MET", "$m_W$", "$\eta_{electron}$", "$p_T^{electron}$", "$\eta_{muon}$", "$p_T^{muon}$", "$m_{WW}$ Offshell", "$m_{ll}$ Offshell", "$p_{T}^{ll}$ Offshell", "MET Offshell", "$m_W$ Offshell", "$\eta_{electron}$ Offshell", "$p_T^{electron}$ Offshell", "$\eta_{muon}$ Offshell", "$p_T^{muon}$ Offshell", "$m_{WW}$ Onshell", "$m_{ll}$ Onshell", "$p_{T}^{ll}$ Onshell", "MET Onshell", "$m_W$ Onshell", "$\eta_{electron}$ Onshell", "$p_T^{electron}$ Onshell", "$\eta_{muon}$ Onshell", "$p_T^{muon}$ Onshell"]


mcfm_file = uproot.open(mcfm_file_name)
ntuple_file = uproot.open(ntuple_file_name)


pdf_pages = PdfPages("./mcfm_comparions_specSelection_shellSplit_SIG_wMassbins_100k_VBF.pdf")


def getRatioHist(num_binned, den_binned, num_binned_err, den_binned_err):

    ratio_binned = np.array([])
    ratio_binned_err = [np.array([]), np.array([])]
    # Iterating through each bin 
    for i in range(0, len(den_binned)):
        # Catching division by 0 error
        if(den_binned[i] == 0 or num_binned[i] == 0):
            ratio_binned = np.append(ratio_binned, [0])
            ratio_binned_err[0] = np.append(ratio_binned_err[0], [0])
            ratio_binned_err[1] = np.append(ratio_binned_err[1], [0])

            continue

        # Filling ratio bins
        ratio = num_binned[i]/den_binned[i]
        ratio_err_up = num_binned[i]/den_binned[i] * (1 + (num_binned_err[i]/num_binned[i] + den_binned_err[i]/den_binned[i]))
        ratio_err_down = num_binned[i]/den_binned[i] * (1 - (num_binned_err[i]/num_binned[i] + den_binned_err[i]/den_binned[i]))

        ratio_binned = np.append(ratio_binned, [ratio])
        ratio_binned_err[0] = np.append(ratio_binned_err[0], [ratio-ratio_err_down])
        ratio_binned_err[1] = np.append(ratio_binned_err[1], [ratio_err_up-ratio])
    return ratio_binned, ratio_binned_err


mcfm_scale = 1
ntuple_scale = 1
for i in range(0,len(plots)):

    mcfm_values, bins = mcfm_file["" + plots[i]].to_numpy()
    ntuple_values, bins = ntuple_file["HWW/" + plots[i]].to_numpy()

    mcfm_scale = 1 / np.sum(mcfm_values)
    ntuple_scale = 1 / np.sum(ntuple_values)
    
    xerr_bins = [(bins[1]-bins[0])/2 for i in range(0,len(bins))]
    yerr_mcfm = [(mcfm_values[i])**(.5) for i in range(0, len(mcfm_values))]
    #yerr_mcfm = mcfm_file["HWW/" + plots[i]].errors()
    yerr_ntuple = ntuple_file["HWW/" + plots[i]].errors()

    ratio_binned, ratio_binned_err = getRatioHist(ntuple_values*ntuple_scale, mcfm_values * mcfm_scale, np.array(yerr_ntuple)*ntuple_scale, np.array(yerr_mcfm) * mcfm_scale)

    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    hep.cms.text("Preliminary Simulation", fontsize=15, ax=ax[0])
    hep.cms.lumitext("13TeV", fontsize=15, ax=ax[0])


    ax[0].errorbar(bins[:-1], mcfm_values * mcfm_scale, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="MCFM") #np.array(yerr_mcfm) * mcfm_scale, label="MCFM")
    ax[0].errorbar(bins[:-1], ntuple_values*ntuple_scale, linestyle="", xerr=xerr_bins[:-1], yerr=0, label="POWHEG+JHUGen(Merged)") #np.array(yerr_ntuple)*ntuple_scale, label="POWHEG+JHUGen(Merged)")
    #hep.histplot(mcfm_values, bins, xerr=True, yerr=yerr_mcfm, density=True, histtype="errorbar", label="MCFM")
    #hep.histplot(ntuple_values, bins, xerr=True, yerr=yerr_ntuple, density=True, histtype="errorbar", label="POWHEG+JHUGen Merged")
    ax[0].set_title("$d\sigma/dE$ vs. " + axis_label[i],loc="left", fontsize=15, pad=20)
    ax[0].set_ylabel("$d\sigma/dE$ (a.u.)")
    #ax[0].set_xlabel(axis_label[i])
    ax[0].set_yscale('log')
    #ax[0].set_xscale('log')
    ax[0].set_xlim([0,1000])
    if(axis_label[i] == "$m_{WW}$"):
        for j in range(0, len(MASS_BINS)):
            ax[0].axvline(x=MASS_BINS[j], color='r', linewidth=.5, linestyle="--")
    ax[0].legend(prop = {"size": 12 })
    for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        item.set_fontsize(15)

    ax[1].errorbar(bins[:-1],ratio_binned,linestyle="", xerr=xerr_bins[:-1], yerr=ratio_binned_err, label="Merged/MCFM")
    ax[1].set_ylabel("Ratio")
    ax[1].set_xlabel(axis_label[i])
    ax[1].axhline(y=1, color='r', linewidth=.5, linestyle='--')
    ax[1].axhline(linewidth=.1)
    ax[1].set_ylim([0,2])
    ax[1].legend(prop = {"size": 12 })
    #ax[1].set_xscale('log')
    ax[1].set_xlim([0,1000])
    for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] + ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        item.set_fontsize(15)

    fig.set_size_inches(6,6)
    pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

pdf_pages.close()


