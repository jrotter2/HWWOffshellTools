import json
import numpy as np
import uproot
import ROOT
import numpy as np
from scipy.optimize import curve_fit
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.colors as mcolors

import multiprocessing
from multiprocessing import Pool

import gc

pdf_pages = PdfPages("./vizualize_weights.pdf")

CONFIG_FILE_PATH = "../configs/HM_VBF_18_VBF_Summer20UL18_OffshellWgtsIncluded.json"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)
sampleMasses = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]

CONFIG_FILE_PATH_2 = "../configs/HM_ggH_18_ggH_Summer20UL18_OffshellWgtsIncluded.json"
CONFIG_FILE_2 = open(CONFIG_FILE_PATH_2, "r")
CONFIG_FILE_CONTENTS_2 = CONFIG_FILE_2.read()
CONFIG_2 = json.loads(CONFIG_FILE_CONTENTS_2)


def getEventInfo(worker_index, sample_mass, sample_type, return_dict):
    print("Established Worker " + str(worker_index) + " for Sample Mass: " + str(sample_mass))

    files = CONFIG_2["HM_ggH_18_" + str(sample_mass)]["files"]
    if("VBF" in sample_type):
        files = CONFIG["HM_VBF_18_" + str(sample_mass)]["files"]

    for f in files:
        rootFile = uproot.open(f)
        pdgIds = rootFile["Events/GenPart_pdgId"].array()
        masses = rootFile["Events/GenPart_mass"].array()
        masses_masked = masses[pdgIds == 25]
        H_mass = masses_masked[:,0]
        baseW = rootFile["Events/baseW"].array()
        XSwgt = np.multiply(rootFile["Events/genWeight"].array(), baseW)
        reWgt = np.multiply(rootFile["Events/p_Gen_CPStoBWPropRewgt"].array(), XSwgt)
        fullWgt = np.multiply(rootFile["Events/HWWOffshell_combineWgt"].array(), reWgt)

        sig_wgt_name = "p_Gen_GG_SIG_kappaTopBot_1_ghz1_1_MCFM"
        cont_wgt_name = "p_Gen_GG_BKG_MCFM"
        sigpluscont_wgt_name =  "p_Gen_GG_BSI_kappaTopBot_1_ghz1_1_MCFM"
        if("VBF" in sample_type):
            sig_wgt_name = "p_Gen_JJEW_SIG_ghv1_1_MCFM"
            cont_wgt_name = "p_Gen_JJEW_BKG_MCFM"
            sigpluscont_wgt_name = "p_Gen_JJEW_BSI_ghv1_1_MCFM"

        sig_Wgt = np.multiply(rootFile["Events/" + sig_wgt_name].array(), fullWgt)
        cont_Wgt = np.multiply(rootFile["Events/" + cont_wgt_name].array(), fullWgt)
        sigpluscont_Wgt = np.multiply(rootFile["Events/" + sigpluscont_wgt_name].array(), fullWgt)

    return_dict[worker_index] = [H_mass, sig_Wgt, cont_Wgt, sigpluscont_Wgt]


manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []
for i,sample_mass in enumerate(sampleMasses):
    p = multiprocessing.Process(target=getEventInfo, args=(i, sample_mass,"VBF", return_dict))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()
#print(return_dict.values())


mww = []
sonly_wgt = []
bonly_wgt = []
sbi_wgt = []
for worker_vals in return_dict.values():
    mww = np.concatenate((worker_vals[0], mww), axis=0)
    sonly_wgt = np.concatenate((worker_vals[1], sonly_wgt), axis=0)
    bonly_wgt = np.concatenate((worker_vals[2], bonly_wgt), axis=0)
    sbi_wgt = np.concatenate((worker_vals[3], sbi_wgt), axis=0)


manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []
for i,sample_mass in enumerate(sampleMasses):
    p = multiprocessing.Process(target=getEventInfo, args=(i, sample_mass,"ggH" ,return_dict))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()
#print(return_dict.values())

mww_2 = []
sonly_wgt_2 = []
bonly_wgt_2 = []
sbi_wgt_2 = []
for worker_vals in return_dict.values():
    mww_2 = np.concatenate((worker_vals[0], mww_2), axis=0)
    sonly_wgt_2 = np.concatenate((worker_vals[1], sonly_wgt_2), axis=0)
    bonly_wgt_2 = np.concatenate((worker_vals[2], bonly_wgt_2), axis=0)
    sbi_wgt_2 = np.concatenate((worker_vals[3], sbi_wgt_2), axis=0)

pdf_pages = PdfPages("./visualizer_output.pdf")

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0, 0].set_title('SIG')
axes[0, 0].hist2d(mww, sonly_wgt, bins=[100,100], range=[[120,1000],[-.00015,.0003]], norm=mcolors.LogNorm())

axes[1, 0].set_title('BKG')
axes[1, 0].hist2d(mww, bonly_wgt, bins=[100,100], range=[[120,1000],[-.1,.1]], norm=mcolors.LogNorm())

axes[0, 1].set_title('BSI')
axes[0, 1].hist2d(mww, sbi_wgt, bins=[100,100], range=[[120,1000],[-.1,.1]], norm=mcolors.LogNorm())

axes[1, 1].set_title('log(BKG/SIG)')

ratio = np.divide(np.array(bonly_wgt), np.array(sonly_wgt), out=np.zeros_like(np.array(sonly_wgt)), where=np.array(bonly_wgt)!=0)
log_ratio = np.log(ratio, out=np.zeros_like(ratio), where=ratio!=0)

mww_bins = [120 + i*(8.8) for i in range(0, 100)]

ratio_binned = [[] for i in mww_bins]

for i in range(0, len(ratio)):
    if(mww[i] >= 1000):
        continue
    mass_bin = int((mww[i] - 120)/(8.8))
    ratio_binned[mass_bin].append(ratio)

gc.collect()

ratio_averages = [np.average(np.array(ratio_binned[i])) for i in range(len(ratio_binned))]

axes[1, 1].hist2d(mww, log_ratio, bins=[100,100], range=[[120,1000],[-15,20]], norm=mcolors.LogNorm())

fig.tight_layout()

pdf_pages.savefig(fig)

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0, 0].set_title('SIG')
axes[0, 0].hist2d(mww_2, sonly_wgt_2, bins=[100,100], range=[[120,1000],[-.001,.001]], norm=mcolors.LogNorm())

axes[1, 0].set_title('BKG')
axes[1, 0].hist2d(mww_2, bonly_wgt_2, bins=[100,100], range=[[120,1000],[-.1,.1]], norm=mcolors.LogNorm())

axes[0, 1].set_title('BSI')
axes[0, 1].hist2d(mww_2, sbi_wgt_2, bins=[100,100], range=[[120,1000],[-.1,.1]], norm=mcolors.LogNorm())

axes[1, 1].set_title('BKG/SIG')
ratio = np.divide(np.array(bonly_wgt_2), np.array(sonly_wgt_2), out=np.zeros_like(np.array(sonly_wgt_2)), where=np.array(bonly_wgt_2)!=0)
log_ratio = np.log(ratio, out=np.zeros_like(ratio), where=ratio!=0)
axes[1, 1].hist2d(mww_2, log_ratio, bins=[100,50], range=[[120,1000],[-10,10]], norm=mcolors.LogNorm())
axes[1,1].plot(mww_bins, ratio_averages, "r-")

fig.tight_layout()

pdf_pages.savefig(fig)
pdf_pages.close()




