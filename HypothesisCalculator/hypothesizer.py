import json
import numpy as np
import uproot
import ROOT
#import importlib
#import mplhep as hep
#plt.style.use(hep.style.CMS)
import numpy as np

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

pdf_pages = PdfPages("./hypothesizer_Autumn2018.pdf")

ggH_CONFIG_FILE_PATH = "./ggH_Autumn2018.json" 
ggH_CONFIG_FILE = open(ggH_CONFIG_FILE_PATH, "r")
ggH_CONFIG_FILE_CONTENTS = ggH_CONFIG_FILE.read()
ggH_CONFIG = json.loads(ggH_CONFIG_FILE_CONTENTS)
ggH_sampleMasses = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]

ggWW_CONFIG_FILE_PATH = "./ggWW_Autumn2018.json"
ggWW_CONFIG_FILE = open(ggWW_CONFIG_FILE_PATH, "r")
ggWW_CONFIG_FILE_CONTENTS = ggWW_CONFIG_FILE.read()
ggWW_CONFIG = json.loads(ggWW_CONFIG_FILE_CONTENTS)
ggWW_sampleTypes = ["ENMN", "MNEN"]


ggH_mww = []
ggH_wgt = []
ggH_signal_wgt = []
for sample_mass in ggH_sampleMasses:
    print("Currently on ggH Sample Mass: " + str(sample_mass))
    for f in ggH_CONFIG["HM_" + str(sample_mass)]["files"]:
        rootFile = uproot.open(f)
        pdgIds = rootFile["Events/GenPart_pdgId"].array()
        masses = rootFile["Events/GenPart_mass"].array()
        masses_masked = masses[pdgIds == 25]
        H_mass = masses_masked[:,0]
        baseW = rootFile["Events/baseW"].array()
        XSwgt = np.multiply(rootFile["Events/genWeight"].array(), baseW)
        reWgt = np.multiply(rootFile["Events/p_Gen_CPStoBWPropRewgt"].array(), XSwgt)
        fullWgt = np.multiply(rootFile["Events/HWWOffshell_combineWgt"].array(), reWgt)
        sig_Wgt = np.multiply(rootFile["Events/p_Gen_GG_SIG_kappaTopBot_1_ghz1_1_MCFM"].array(), fullWgt)
#    
        ggH_mww = np.concatenate((H_mass, ggH_mww), axis=0)
        ggH_wgt = np.concatenate((fullWgt, ggH_wgt), axis=0)
        ggH_signal_wgt = np.concatenate((sig_Wgt, ggH_signal_wgt), axis=0)


ggWW_mww = []
ggWW_wgt = []
for sample_type in ggWW_sampleTypes:
    print("Currently on ggWW Sample: " + sample_type)
    for f in ggWW_CONFIG[sample_type]["files"]:
        rootFile = uproot.open(f)
        pdgIds = rootFile["Events/GenPart_pdgId"].array()
        gen_pts = rootFile["Events/GenPart_pt"].array()
        gen_etas = rootFile["Events/GenPart_eta"].array()
        gen_phis = rootFile["Events/GenPart_phi"].array()

        mother_genPartIdx = rootFile["Events/GenPart_genPartIdxMother"].array()

        mother_pdgIds = pdgIds[mother_genPartIdx]

        mothers_mask = abs(mother_pdgIds) == 24
        W_daughters_pdgIds = pdgIds[mothers_mask]
        W_daughters_gen_pts = gen_pts[mothers_mask]
        W_daughters_gen_etas = gen_etas[mothers_mask]
        W_daughters_gen_phis = gen_phis[mothers_mask]

        mWW = []
        for evt_index, evt in enumerate(W_daughters_pdgIds):
            lepton_mass = [0.000511, 0, 0.10566, 0]
            leptons = [ROOT.Math.PtEtaPhiMVector(0,0,0,0.000511), ROOT.Math.PtEtaPhiMVector(0,0,0,0), ROOT.Math.PtEtaPhiMVector(0,0,0,0.10566), ROOT.Math.PtEtaPhiMVector(0,0,0,0)]
            for part_index, pdgId in enumerate(evt):
                if(abs(pdgId) == 24):
                    continue
                current_part_lepton_index = -1
                if(abs(pdgId) == 11):
                    current_part_lepton_index = 0
                elif(abs(pdgId) == 12):
                    current_part_lepton_index = 1
                elif(abs(pdgId) == 13):
                    current_part_lepton_index = 2
                elif(abs(pdgId) == 14):
                    current_part_lepton_index = 3
                leptons[current_part_lepton_index] = ROOT.Math.PtEtaPhiMVector(W_daughters_gen_pts[evt_index][part_index],W_daughters_gen_etas[evt_index][part_index],W_daughters_gen_phis[evt_index][part_index],lepton_mass[current_part_lepton_index])

            WW_vector = leptons[0] + leptons[1] + leptons[2] + leptons[3]
            mWW.append(WW_vector.M())

        baseW = rootFile["Events/baseW"].array()
        XSwgt = np.multiply(rootFile["Events/genWeight"].array(), baseW)

        ggWW_mww = np.concatenate((mWW, ggWW_mww), axis=0)
        ggWW_wgt = np.concatenate((XSwgt, ggWW_wgt), axis=0)

def getRunningAverage(values_binned, window_size):
    new_values_binned = [0 for value in values_binned]
    for i in range(0, len(values_binned)):
        current_average = 0
        current_num_points = 0
        for a in range(0, window_size):
            if(a == 0):
                current_average += values_binned[i]
                current_num_points += 1
            elif(i-a > 0):
                current_average += values_binned[i-a]
                current_num_points += 1
            elif(i+a < len(values_binned)):
                current_average += values_binned[i+a]
                current_num_points += 1
        current_average = current_average / current_num_points
        new_values_binned[i] = current_average
    return new_values_binned
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


ggH_values, bins = np.histogram(ggH_mww, 300, (0,3000), weights=ggH_signal_wgt)
ggH_sig_values, bins = np.histogram(ggH_mww, 300, (0,3000), weights=ggH_wgt)
ggWW_values, bins = np.histogram(ggWW_mww, 300, (0,3000), weights=ggWW_wgt)

xerr_bins = [(bins[1]-bins[0])/2 for i in range(0,len(bins))]
yerr_ggH = [(ggH_values[i])**(.5) for i in range(0, len(ggH_values))]
yerr_sig_ggH = [(ggH_sig_values[i])**(.5) for i in range(0, len(ggH_sig_values))]
#yerr_mcfm = mcfm_file["HWW/" + plots[i]].errors()
yerr_ggWW = [(ggWW_values[i])**(.5) for i in range(0, len(ggWW_values))]


gg_ratio_binned, gg_ratio_binned_err = getRatioHist(ggWW_values, ggH_values, yerr_ggWW, yerr_ggH)


fig, ax = plt.subplots() #plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
ax.errorbar(bins[:-1], ggH_values, linestyle="",  xerr=xerr_bins[:-1], yerr=np.array(yerr_ggH), label="UnWgt")
ax.errorbar(bins[:-1], ggWW_values, linestyle="", xerr=xerr_bins[:-1], yerr=np.array(yerr_ggWW), label="ggWW")
ax.set_title("$d\sigma/dE$ vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$m_{WW}$")
ax.set_yscale('log')
ax.set_xlim([0,3000])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)


fig, ax = plt.subplots() #plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
ax.errorbar(bins[:-1], ggH_sig_values, linestyle="",  xerr=xerr_bins[:-1], yerr=np.array(yerr_sig_ggH), label="Signal")
ax.set_title("$d\sigma/dE$ vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$m_{WW}$")
ax.set_yscale('log')
ax.set_xlim([0,3000])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)



fig, ax = plt.subplots() #plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
ax.errorbar(bins[:-1], gg_ratio_binned, linestyle="",  xerr=xerr_bins[:-1], yerr=np.array(gg_ratio_binned_err), label="ratio")
ax.set_title("RAW/ggWW vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("RAW/ggWW")
ax.set_xlabel("$m_{WW}$")
ax.axhline(y=1, color='r', linewidth=.5, linestyle='--')
ax.set_xlim([0,3000])
ax.set_yscale('log')
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)



fig, ax = plt.subplots() #plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
ax.errorbar(bins[:-1], gg_ratio_binned, linestyle="",  xerr=xerr_bins[:-1], yerr=np.array(gg_ratio_binned_err), label="ratio")
ax.set_title("RAW/ggWW vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("RAW/ggWW")
ax.set_xlabel("$m_{WW}$")
ax.axhline(y=1, color='r', linewidth=.5, linestyle='--')
ax.set_xlim([0,250])
ax.set_yscale('log')
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)



fig, ax = plt.subplots() #plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
ax.errorbar(bins[:-1], gg_ratio_binned, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="ratio")
ax.set_title("RAW/ggWW vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("RAW/ggWW")
ax.set_xlabel("$m_{WW}$")
ax.axhline(y=1, color='r', linewidth=.5, linestyle='--')
ax.set_xlim([0,3000])
ax.set_yscale('log')
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

running_average_binned_3 = getRunningAverage(gg_ratio_binned, 3)
running_average_binned_5 = getRunningAverage(gg_ratio_binned, 5)
running_average_binned_10 = getRunningAverage(gg_ratio_binned, 10)


for i in range(0, 30):
    fig, ax = plt.subplots() #plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    ax.errorbar(bins[:-1], gg_ratio_binned, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="ratio")
    ax.errorbar(bins[:-1], running_average_binned_3, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="$<x>_3$")
    ax.errorbar(bins[:-1], running_average_binned_5, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="$<x>_5$")
    ax.errorbar(bins[:-1], running_average_binned_10, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="$<x>_{10}$")
    ax.set_title("RAW/ggWW vs. $m_{WW}$ : " + str(i) + "/30",loc="left", fontsize=15, pad=20)
    ax.set_ylabel("RAW/ggWW")
    ax.set_xlabel("$m_{WW}$")
    ax.axhline(y=1, color='r', linewidth=.5, linestyle='--')
    ax.set_xlim([100*i,100*(i+1)])
    ax.set_yscale('log')
    ax.legend(prop = {"size": 12 })
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    fig.set_size_inches(6,6)
    pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)



pdf_pages.close()




