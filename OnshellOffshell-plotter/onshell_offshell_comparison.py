import uproot
import matplotlib
from matplotlib import pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

ntuple_file_name = "../MCFM_validation/hww_mcfm_validationApr13_specialSelection_shellSplit.root" #hww_mcfm_validationApr10_specialSelection_zoomed.root"


plots = ["mww", "mll", "ptll", "MET", "mw", "ele_eta", "ele_pt", "muon_eta", "muon_pt"]

axis_label = ["$m_{WW}$", "$m_{ll}$", "$p_{T}^{ll}$", "MET", "$m_W$", "$\eta_{electron}$", "$p_T^{electron}$", "$\eta_{muon}$", "$p_T^{muon}$"]

ntuple_file = uproot.open(ntuple_file_name)


pdf_pages = PdfPages("./onshell_offshell_comparions.pdf")

ntuple_scale = 1
for i in range(0,len(plots)):

    ntuple_values, bins = ntuple_file["HWW/" + plots[i]].to_numpy()
    offshell_values, bins = ntuple_file["HWW/" + plots[i] + "_offshell"].to_numpy()
    onshell_values, bins = ntuple_file["HWW/" + plots[i] + "_onshell"].to_numpy()

    ntuple_scale = 1 / np.sum(ntuple_values)
    
    xerr_bins = [(bins[1]-bins[0])/2 for i in range(0,len(bins))]

    fig, ax = plt.subplots()
    hep.cms.text("Preliminary Simulation", fontsize=15)
    hep.cms.lumitext("13TeV", fontsize=15)

    
    ax.hist([bins[:-1], bins[:-1]],bins,weights=[offshell_values*ntuple_scale, onshell_values*ntuple_scale], histtype='barstacked', stacked=True, label=["offshell", "onshell"])
    #ax.errorbar(bins[:-1], mcfm_values * mcfm_scale, linestyle="",  xerr=xerr_bins[:-1], yerr=np.array(yerr_mcfm) * mcfm_scale, label="MCFM")
    #ax.errorbar(bins[:-1], ntuple_values*ntuple_scale, linestyle="--", xerr=xerr_bins[:-1], yerr=0, marker="v", markersize="5", label="POWHEG+JHUGen Merged")
    #hep.histplot(mcfm_values, bins, xerr=True, yerr=yerr_mcfm, density=True, histtype="errorbar", label="MCFM")
    #hep.histplot(ntuple_values, bins, xerr=True, yerr=yerr_ntuple, density=True, histtype="errorbar", label="POWHEG+JHUGen Merged")
    ax.set_title("$d\sigma/dE$ vs. " + axis_label[i],loc="left", fontsize=15, pad=20)
    ax.set_ylabel("$d\sigma/dE$ (a.u.)")
    ax.set_xlabel(axis_label[i])
    ax.set_yscale('log')
    ax.legend(prop = {"size": 12 })
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    fig.set_size_inches(6,6)
    pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

pdf_pages.close()


