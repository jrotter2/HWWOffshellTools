"""

python3 makeAllNuisancePlotsInDatacard.py -d /eos/user/j/jrotter/Latinos_RootFiles/Full2018_2j_v9_W_SYST_FINAL/datacards_2018/SR_VBF_OFF_2j/dnnScore_VBF_OFF/datacard.txt /eos/user/j/jrotter/Latinos_RootFiles/Full2018_2j_v9_W_SYST_FINAL/datacards_2018 /eos/user/j/jrotter/www/2018_Nuisance_Checker_Apr18

"""


from tqdm import tqdm

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
parser.add_option("-d","--datacard", dest="datacard_path", type="string",
                     help="Path to datacard.txt", default="./datacard.txt")

options, args = parser.parse_args()

datacard_path = options.datacard_path

datacard_path_pieces = datacard_path.split("/")
variable = datacard_path_pieces[-2]
cut = datacard_path_pieces[-3]

sample_list = []
nuisance_list = []

isNuisancesNext =  False
with open(datacard_path) as my_file:
    for line in my_file:
        if(isNuisancesNext):
             if(len(line.split()) < 2):
                 continue
             if("shape" in line.split()[1] and "pdf" in line.split()[0]):
                 nuisance_list.append(line.split()[0])
                 continue
             else:
                 continue
        elif("process" in line and len(sample_list) == 0):
            sample_list = line.split()[1:] 
            continue
        elif("rate" in line and len(sample_list) != 0):
             isNuisancesNext = True
             continue


def makeNuisancePlot(in_file, outdir, cut, var, sample, nuisance):
    
    isNuisanceInFile = False
    for key in in_file.keys():
        if("histo_" + sample + "_" + nuisance + "Up" in key):
            isNuisanceInFile = True
    if(not isNuisanceInFile):
        return
    nominal_hist, nominal_bins = in_file["histo_" + sample].to_numpy()
    up_hist, up_bins  = in_file["histo_" + sample + "_" + nuisance + "Up"].to_numpy()
    down_hist, down_bins = in_file["histo_" + sample + "_" + nuisance + "Down"].to_numpy()
    
    
    ratio_up = []
    ratio_down = []
    
    for i in range(0, len(nominal_bins)-1):
        if(nominal_hist[i] > 0):
            ratio_up.append((up_hist[i] - nominal_hist[i])/nominal_hist[i] + 1)
            ratio_down.append((down_hist[i] - nominal_hist[i])/nominal_hist[i] + 1)
        else:
            ratio_up.append(1)
            ratio_down.append(1)
    
    fig_name = outdir + "/" + cut + "_" + var + "_" + sample + "_" + nuisance
    
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
    
    
    ax[1].set_ylim([np.min([np.min(ratio_up), np.min(ratio_down)]) - (1-np.min([np.min(ratio_up), np.min(ratio_down)]))*1.1,np.max([np.max(ratio_up), np.max(ratio_down)]) + (np.max([np.max(ratio_up), np.max(ratio_down)])-1)*1.1])
    
    
    ax[0].set_ylabel("Events", backgroundcolor="white")
    ax[1].set_ylabel("Ratio", backgroundcolor="white")
    ax[1].set_xlabel(var)
    
    ax[1].axhline(y=1, color='darkgray', linewidth=.5, linestyle='--')
    
    ax[0].legend(prop = {"size": 12 })
    
    for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        item.set_fontsize(15)
    
    for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] + ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        item.set_fontsize(15)
    
    fig.set_size_inches(6,6)
    
    pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)
    
    pdf_pages.close()
    
    plt.savefig(fig_name + ".png")


print("Making plots for " + str(len(sample_list)) + " samples and " + str(len(nuisance_list)) + " nuisances...")

for sample in sample_list:
    print("Sample: " + sample)
    in_file = uproot.open(args[0] + "/" + cut + "/" + variable + "/shapes/histos_" + cut + ".root")
    for i in tqdm(range(0,len(nuisance_list))):
        makeNuisancePlot(in_file, args[1], cut, variable, sample, nuisance_list[i])
