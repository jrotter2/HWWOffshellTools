import uproot
import matplotlib
from matplotlib import pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

MASS_BINS = [0,136.7,148.3,165,175,185,195,205,220,240,260,285,325,375,425,475,525,575,650,750,850,950]


ntuple_file_name = "../../merging-procedure/output_preprocessing_Summer20UL18_106x_nAODv9_Full2018v9_VBF.root"
ntuple_file = uproot.open(ntuple_file_name)


pdf_pages = PdfPages("./VBF_S_B_SBI.pdf")


sig_values, bins = ntuple_file["SIG_wgts_combined"].to_numpy()
cont_values, bins = ntuple_file["CONT_wgts_combined"].to_numpy()
sigpluscont_values, bins = ntuple_file["SIGplusCONT_wgts_combined"].to_numpy()
    
xerr_bins = [(bins[1]-bins[0])/2 for i in range(0,len(bins))]

fig, ax = plt.subplots()
hep.cms.text("Preliminary Simulation", fontsize=15, ax=ax)
hep.cms.lumitext("13TeV", fontsize=15, ax=ax)


ax.errorbar(bins[:-1], sig_values, linestyle="", xerr=xerr_bins[:-1], yerr=0, label="HWW")
ax.set_title("$d\sigma/dE$ vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$m_{WW}$")
ax.set_yscale('log')
ax.set_xlim([0,1000])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)


fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots()
hep.cms.text("Preliminary Simulation", fontsize=15, ax=ax)
hep.cms.lumitext("13TeV", fontsize=15, ax=ax)

ax.errorbar(bins[:-1], cont_values, linestyle="", xerr=xerr_bins[:-1], yerr=0, label="qqWW")
ax.set_title("$d\sigma/dE$ vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$m_{WW}$")
ax.set_yscale('log')
ax.set_xlim([0,1000])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)


fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)


pdf_pages.close()
