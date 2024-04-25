"""

Plot Nuisances from Datacards to see the up and down shapes with respect to the nominal and detect any anomolies.

python3 checkNuisance.py -c SR_VBF_OFF_2j -v dnnScore_VBF_OFF -s WW -n CMS_hww_pdf_WW /eos/user/j/jrotter/Latinos_RootFiles/Full2018_2j_v9_W_SYST_FINAL/datacards_2018 /eos/user/j/jrotter/www/2018_Nuisance_Checker_Apr18

#/eos/user/j/jrotter/Latinos_RootFiles/Full2018_2j_v9_W_SYST_FINAL/datacards_2018/SR_VBF_OFF_2j/dnnScore_VBF_OFF/shapes/histos_SR_VBF_OFF_2j.root

"""

import sys
import os
from optparse import OptionParser
import uproot

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import mplhep as hep
plt.style.use(hep.style.CMS)



parser = OptionParser(usage="%prog [options] inputDir outputDir")
parser.add_option("-c","--cut", dest="cut", type="string",
                     help="SR or CR Cut from cuts.py", default="SR_VBF_OFF_2j")
parser.add_option("-v","--var", dest="var", type="string",
                     help="Variable from variables.py", default="dnnScore_VBF_OFF")
parser.add_option("-s","--sample", dest="sample", type="string",
                     help="Sample from sample.py", default="WW")
parser.add_option("-n","--nuisance", dest="nuisance", type="string",
                     help="Nuisance from nuisance.py", default="CMS_hww_pdf_WW")
parser.add_option("-d", action="store_true", dest="verbose")

options, args = parser.parse_args()

if(options.verbose):
    print("##################################################")
    print("################                  ################")
    print("################ NUISANCE CHECKER ################")
    print("################                  ################")
    print("##################################################")


if(options.verbose):
    print("Loading ROOT File at:")
    print(args[0] + "/" + options.cut + "/" + options.var + "/shapes/histos_" + options.cut + ".root")

in_file = uproot.open(args[0] + "/" + options.cut + "/" + options.var + "/shapes/histos_" + options.cut + ".root")


if(options.verbose):
    print("Collecting histograms:")
    print("Nominal: " + "histo_" + options.sample)
    print("Up: " + "histo_" + options.sample + "_" + options.nuisance + "Up")
    print("Down: " + "histo_" + options.sample + "_" + options.nuisance + "Down")

nominal_hist, nominal_bins = in_file["histo_" + options.sample].to_numpy()
up_hist, up_bins  = in_file["histo_" + options.sample + "_" + options.nuisance + "Up"].to_numpy()
down_hist, down_bins = in_file["histo_" + options.sample + "_" + options.nuisance + "Down"].to_numpy()


ratio_up = []
ratio_down = []

for i in range(0, len(nominal_bins)-1):
    if(nominal_hist[i] > 0):
        ratio_up.append((up_hist[i] - nominal_hist[i])/nominal_hist[i] + 1)
        ratio_down.append((down_hist[i] - nominal_hist[i])/nominal_hist[i] + 1)
    else:
        ratio_up.append(1)
        ratio_down.append(1)

if(options.verbose):
    print("Creating Figure...")

fig_name = str(args[1]) + "/" + options.cut + "_" + options.var + "_" + options.sample + "_" + options.nuisance

pdf_pages = PdfPages(fig_name + ".pdf")

fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.subplots_adjust(hspace=.05)
hep.cms.text("Internal", fontsize=15, ax=ax[0])
hep.cms.lumitext("(13TeV)", fontsize=15, ax=ax[0])

hep.histplot(nominal_hist, nominal_bins, histtype="step", label="Nominal", ax=ax[0])
hep.histplot(up_hist, up_bins, histtype="step", label="Up", ax=ax[0])
hep.histplot(down_hist, down_bins, histtype="step", label="Down", ax=ax[0])

hep.histplot(ratio_up, up_bins, histtype="step", label="Up", color="black", ax=ax[1])
hep.histplot(ratio_down, down_bins, histtype="step", label="Down", color="black", ax=ax[1])

ax[0].set_xlim([np.min([nominal_bins[0], up_bins[0], down_bins[0]]),np.max([nominal_bins[-1], up_bins[-1], down_bins[-1]])])
#ax[1].set_xlim([np.min([nominal_bins[0], up_bins[0], down_bins[0]]),np.max([nominal_bins[-1], up_bins[-1], down_bins[-1]])])


ax[1].set_ylim([np.min([np.min(ratio_up), np.min(ratio_down)]) - (1-np.min([np.min(ratio_up), np.min(ratio_down)]))*1.1,np.max([np.max(ratio_up), np.max(ratio_down)]) + (np.max([np.max(ratio_up), np.max(ratio_down)])-1)*1.1])


ax[0].set_ylabel("Events", backgroundcolor="white")
ax[1].set_ylabel("Ratio", backgroundcolor="white")
ax[1].set_xlabel(options.var)

ax[1].axhline(y=1, color='darkgray', linewidth=.5, linestyle='--')

ax[0].legend(prop = {"size": 12 })

for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    item.set_fontsize(15)

for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] + ax[1].get_xticklabels() + ax[1].get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)

if(options.verbose):
   print("Saving Figures:")
   print(fig_name + ".pdf")
   print(fig_name + ".png")

pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

pdf_pages.close()

plt.savefig(fig_name + ".png")





