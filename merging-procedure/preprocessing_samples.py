import uproot
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import copy



CONFIG_FILE_PATH = "../configs/dataset_config.txt"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)

SAMPLES = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]

MASS_WINDOW_EDGES = [0,136.7,148.3,165,175,185,195,205,220,240,260,285,325,375,425,475,525,575,650,750,850,950,1250,1750,2250,2750,14000]

SAMPLE_PREFIX = ["HM_", "HM_VBF_new"]
SAMPLE_TYPES = ["ggH", "VBF"]

OUTFILE_NAME = "output_preprocessing.root"
OUTPDF_NAME = "output_preprocessing.pdf"
OUTJSON_NAME = "output_preprocessing.json"

OUTFILE = uproot.recreate("./" + OUTFILE_NAME)

def make_mass_bins(dataset, sample_type):
    SIG_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    CONT_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    SIGplusCONT_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    sum_mass_w = [0 for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    LHE_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    m_ww = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    SIG_wgt_name = "p_Gen_GG_SIG_kappaTopBot_1_ghz1_1_MCFM"
    CONT_wgt_name = "p_Gen_GG_BKG_MCFM"
    SIGplusCONT_wgt_name = "p_Gen_GG_BSI_kappaTopBot_1_ghz1_1_MCFM"
    LHE_wgt_name = "LHEWeight_originalXWGTUP"
    if(sample_type == "VBF"):
        SIG_wgt_name = "p_Gen_JJEW_SIG_ghv1_1_MCFM"
        CONT_wgt_name = "p_Gen_JJEW_BKG_MCFM"
        SIGplusCONT_wgt_name = "p_Gen_JJEW_BSI_ghv1_1_MCFM"

    for f in CONFIG[dataset]["files"]:
        rootFile = uproot.open(f)
        H_mass = rootFile["Events/higgsGenMass"].array()
        reWgt = np.multiply(rootFile["Events/p_Gen_CPStoBWPropRewgt"].array(), rootFile["Events/XSWeight"].array())
        for i in range(0, len(MASS_WINDOW_EDGES)-1):
            H_mask = ((H_mass > MASS_WINDOW_EDGES[i]) & (H_mass < MASS_WINDOW_EDGES[i+1]))
            m_ww[i] = np.concatenate((m_ww[i], H_mass[H_mask]), axis=0)
            SIG_wgts[i] = np.concatenate((SIG_wgts[i], np.multiply(rootFile["Events/" + SIG_wgt_name].array(), reWgt)[H_mask]), axis=0)
            CONT_wgts[i] = np.concatenate((CONT_wgts[i], np.multiply(rootFile["Events/" + CONT_wgt_name].array(), reWgt)[H_mask]), axis=0)
            SIGplusCONT_wgts[i] = np.concatenate((SIGplusCONT_wgts[i], np.multiply(rootFile["Events/" + SIGplusCONT_wgt_name].array(), reWgt)[H_mask]), axis=0)
            LHE_wgts[i] = np.concatenate((LHE_wgts[i], rootFile["Events/" + LHE_wgt_name].array()[H_mask]), axis=0)
    for i in range(0, len(MASS_WINDOW_EDGES)-1):
        sum_mass_w[i] = np.sum(LHE_wgts[i])
    print(len(SIG_wgts[0]), len(CONT_wgts[0]), len(m_ww[0]))
    return SIG_wgts, CONT_wgts, SIGplusCONT_wgts, LHE_wgts, sum_mass_w, m_ww

def remove_large_wgts(SIG_wgts, CONT_wgts, SIGplusCONT_wgts, LHE_wgts, sum_mass_w, m_ww, sample_type):
    quantile_cutoff_SIG = []
    quantile_cutoff_CONT = []
    quantile_cutoff_SIGplusCONT = []
    ratio_lost_SIG = []
    ratio_lost_CONT = []
    ratio_lost_SIGplusCONT = []
    SIG_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    CONT_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    SIGplusCONT_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    m_ww_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]

    quantile_fraction_SIG = .0005
    quantile_fraction_CONT = .0005
    quantile_fraction_SIGplusCONT = .0005
    if(sample_type == "VBF"):
        quantile_fraction_SIG = .0005
        quantile_fraction_CONT = .001
        quantile_fraction_SIGplusCONT = .0005
    for i in range(0, len(SIG_wgts)):
        if(len(SIG_wgts[i]) == 0):
            quantile_cutoff_SIG.append(0)
            quantile_cutoff_CONT.append(0)
            quantile_cutoff_SIGplusCONT.append(0)
            continue
        quantile_cutoff_SIG.append(5*np.quantile(SIG_wgts[i], 1-quantile_fraction_SIG, axis=0))
        quantile_cutoff_CONT.append(5*np.quantile(CONT_wgts[i], 1-quantile_fraction_CONT, axis=0))
        quantile_cutoff_SIGplusCONT.append(5*np.quantile(SIGplusCONT_wgts[i], 1-quantile_fraction_SIGplusCONT, axis=0))
    for i in range(0, len(SIG_wgts)):

        large_wgt_mask = (SIG_wgts[i] < quantile_cutoff_SIG[i]) & (CONT_wgts[i] < quantile_cutoff_CONT[i]) & (SIGplusCONT_wgts[i] < quantile_cutoff_SIGplusCONT[i])
        small_wgt_mask = (SIG_wgts[i] > quantile_cutoff_SIG[i]) | (CONT_wgts[i] > quantile_cutoff_CONT[i]) | (SIGplusCONT_wgts[i] > quantile_cutoff_SIGplusCONT[i])

        SIG_wgts_slimmed[i] = SIG_wgts[i][large_wgt_mask]
        m_ww_slimmed[i] = m_ww[i][large_wgt_mask]
        CONT_wgts_slimmed[i] = CONT_wgts[i][large_wgt_mask]
        SIGplusCONT_wgts_slimmed[i] = SIGplusCONT_wgts[i][large_wgt_mask]
        new_sum_mass_w = sum_mass_w[i] - np.sum(LHE_wgts[i][small_wgt_mask])

        ratio_lost=0
        if(new_sum_mass_w != 0):
            ratio_lost = sum_mass_w[i] / new_sum_mass_w

        SIG_wgts_slimmed[i] = SIG_wgts_slimmed[i] * ratio_lost
        CONT_wgts_slimmed[i] = CONT_wgts_slimmed[i] * ratio_lost
        SIGplusCONT_wgts_slimmed[i] = SIGplusCONT_wgts_slimmed[i] * ratio_lost
        ratio_lost_SIG.append(ratio_lost)
        ratio_lost_CONT.append(ratio_lost)
        ratio_lost_SIGplusCONT.append(ratio_lost)
    #for i in range(0, len(SIG_wgts)):
    #    ratio_lost_SIG.append(len(SIG_wgts_slimmed[i])/len(SIG_wgts[i]))
    #    ratio_lost_CONT.append(len(CONT_wgts_slimmed[i])/len(CONT_wgts[i]))
    #    ratio_lost_SIGplusCONT.append(len(SIGplusCONT_wgts_slimmed[i])/len(SIGplusCONT_wgts[i])) 
    return quantile_cutoff_SIG, quantile_cutoff_CONT, quantile_cutoff_SIGplusCONT, ratio_lost_SIG, SIG_wgts_slimmed, CONT_wgts_slimmed, SIGplusCONT_wgts_slimmed, m_ww_slimmed

def remove_zero_wgts(SIG_wgts, CONT_wgts, SIGplusCONT_wgts):
    ratio_lost_SIG = []
    ratio_lost_CONT = []
    ratio_lost_SIGplusCONT = []
    SIG_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    CONT_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    SIGplusCONT_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    for i in range(0, len(SIG_wgts)):
        large_wgt_mask = (SIG_wgts[i] == 0) & (CONT_wgts[i] == 0) & (SIGplusCONT_wgts[i] == 0)
        SIG_wgts_slimmed[i] = SIG_wgts[i][large_wgt_mask]
        CONT_wgts_slimmed[i] = CONT_wgts[i][large_wgt_mask]
        SIGplusCONT_wgts_slimmed[i] = SIGplusCONT_wgts[i][large_wgt_mask]
    return [ratio_lost_SIG, ratio_lost_CONT, ratio_lost_SIGplusCONT], SIG_wgts_slimmed, CONT_wgts_slimmed, SIGplusCONT_wgts_slimmed

def calculate_nEvents(SIG_wgts, CONT_wgts, SIGplusCONT_wgts):
    nEvents_SIG = [0 for i in range(0, len(SIG_wgts))]
    nEvents_CONT = [0 for i in range(0, len(SIG_wgts))]
    nEvents_SIGplusCONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_SIG = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_CONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_SIGplusCONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_sq_SIG = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_sq_CONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_sq_SIGplusCONT = [0 for i in range(0, len(SIG_wgts))]
    for i in range(0, len(SIG_wgts)):
        sum_wgts_SIG[i] = np.sum(SIG_wgts[i])
        sum_wgts_sq_SIG[i] = np.sum(np.multiply(SIG_wgts[i], SIG_wgts[i]))
        sum_wgts_CONT[i] = np.sum(CONT_wgts[i])
        sum_wgts_sq_CONT[i] = np.sum(np.multiply(CONT_wgts[i], CONT_wgts[i]))
        sum_wgts_SIGplusCONT[i] = np.sum(SIGplusCONT_wgts[i])
        sum_wgts_sq_SIGplusCONT[i] = np.sum(np.multiply(SIGplusCONT_wgts[i], SIGplusCONT_wgts[i]))
    for i in range(0, len(SIG_wgts)):

        if(sum_wgts_sq_SIG[i] == 0):
            nEvents_SIG[i] = 0
        else:
            nEvents_SIG[i] = sum_wgts_SIG[i]*sum_wgts_SIG[i] / sum_wgts_sq_SIG[i]
            if(np.isnan(nEvents_SIG[i])):
                nEvents_SIG[i] = 0

        if(sum_wgts_sq_CONT[i] == 0):
            nEvents_CONT[i] = 0
        else:
            nEvents_CONT[i] = sum_wgts_CONT[i]*sum_wgts_CONT[i] / sum_wgts_sq_CONT[i]
            if(np.isnan(nEvents_CONT[i])):
                nEvents_CONT[i] = 0

        if(sum_wgts_SIGplusCONT[i] == 0):
            nEvents_SIGplusCONT[i]  = 0
        else:
            nEvents_SIGplusCONT[i] = sum_wgts_SIGplusCONT[i]*sum_wgts_SIGplusCONT[i] / sum_wgts_sq_SIGplusCONT[i]
            if(np.isnan(nEvents_SIGplusCONT[i])):
                nEvents_SIGplusCONT[i] = 0

    return nEvents_SIG, nEvents_CONT, nEvents_SIGplusCONT

def normalize_mass_bins(nEvents_by_sample, sample_type):
    combine_wgts_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
    small_cutoff = 0.005
    if(sample_type == "VBF"):
        small_cutoff = 0.01

    for j in range(0, len(MASS_WINDOW_EDGES)-1):
        total = 0
        for i in range(0, len(SAMPLES)):
            total = total + nEvents_by_sample[i][j]
        if(total == 0):
            continue
        for i in range(0, len(SAMPLES)):
            combine_wgts_by_sample[i][j] = nEvents_by_sample[i][j] / total

    for j in range(0, len(MASS_WINDOW_EDGES)-1):
        for i in range(0, len(SAMPLES)):
            if(combine_wgts_by_sample[i][j] < small_cutoff):
                combine_wgts_by_sample[i][j] = 0

    for j in range(0, len(MASS_WINDOW_EDGES)-1):
        total = 0
        for i in range(0, len(SAMPLES)):
            total = total + combine_wgts_by_sample[i][j]
        if(total == 0):
            combine_wgts_by_sample[i][j] = 0
            continue
        for i in range(0, len(SAMPLES)):
            combine_wgts_by_sample[i][j] = combine_wgts_by_sample[i][j] / total

    for j in range(0, len(MASS_WINDOW_EDGES)-1):
        for i in range(0, len(SAMPLES)):
            if(np.isnan(combine_wgts_by_sample[i][j])):
                combine_wgts_by_sample[i][j] = 0
                print("Warning...Potential division by 0...")

    return combine_wgts_by_sample

def scale_factor(combine_wgts_by_sample, wgts_by_sample):
    scale_factor = [1 for i in range(0, len(SAMPLES))]
    running_scale_factor = [1 for i in range(0, len(SAMPLES))]
    for i in range(1, len(SAMPLES)):
        if(SAMPLES[i] <= 200):
            continue
        total_sample = 0
        total_sample_prev = 0
        for j in range(0, len(MASS_WINDOW_EDGES)-1):
            if(combine_wgts_by_sample[i][j] != 0 and combine_wgts_by_sample[i-1][j] != 0):
                total_sample = total_sample+np.sum(wgts_by_sample[i][j])
                total_sample_prev = total_sample_prev+np.sum(wgts_by_sample[i-1][j])
        scale_factor[i] = total_sample_prev / total_sample
    for i in range(1, len(SAMPLES)):
        running_scale_factor[i] = running_scale_factor[i-1] * scale_factor[i]
    return running_scale_factor


def write_to_file(location, data):
    print("Writing to file: " + str(location))
    OUTFILE[location] = data

def make_2d_hist(combine_wgts_SIG, combine_wgts_CONT, combine_wgts_SIGplusCONT, sample_type):
    fig, ax = plt.subplots(1)
    ylabels = [str(SAMPLES[len(SAMPLES)-i-1]) for i in range(0, len(SAMPLES))]
    xlabels = [str(MASS_WINDOW_EDGES[i]) + "-" + str(MASS_WINDOW_EDGES[i+1]) for i in range(0, len(MASS_WINDOW_EDGES)-1)]

    y_axis_flipped_SIG = [combine_wgts_SIG[len(combine_wgts_SIG)-i-1] for i in range(0, len(combine_wgts_SIG))]
    y_axis_flipped_CONT = [combine_wgts_CONT[len(combine_wgts_SIG)-i-1] for i in range(0, len(combine_wgts_CONT))]
    y_axis_flipped_SIGplusCONT = [combine_wgts_SIGplusCONT[len(combine_wgts_SIG)-i-1] for i in range(0, len(combine_wgts_SIGplusCONT))]

    ax = sns.heatmap(y_axis_flipped_SIG, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), xticklabels = xlabels, yticklabels=ylabels, linewidth=.25)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_title("Window Reweighting Factors - H - " + sample_type)
    ax.set_xlabel("Mass Window (GeV)")
    ax.set_ylabel("Sample LHECandMass (GeV)")
    plt.tight_layout()

    fig2, ax2 = plt.subplots(1)
    ax2 = sns.heatmap(y_axis_flipped_CONT,cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), xticklabels = xlabels, yticklabels=ylabels, linewidth=.25)
    ax2.set_title("Window Reweighting Factors - C -" + sample_type)
    ax2.set_xlabel("Mass Window (GeV)")
    ax2.set_ylabel("Sample LHECandMass (GeV)")
    plt.tight_layout()

    fig3, ax3 = plt.subplots(1)
    ax3 = sns.heatmap(y_axis_flipped_SIGplusCONT,cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), xticklabels = xlabels, yticklabels=ylabels, linewidth=.25)
    ax3.set_title("Window Reweighting Factors - H+C -" + sample_type)
    ax3.set_xlabel("Mass Window (GeV)")
    ax3.set_ylabel("Sample LHECandMass (GeV)")
    plt.tight_layout()

    pdfPages = PdfPages("./" + OUTPDF_NAME)
    pdfPages.savefig(fig)
    pdfPages.savefig(fig2)
    pdfPages.savefig(fig3)
    pdfPages.close()

def makeFullCombinedHistogram(m_ww_by_sample, hypothesis_wgt_by_sample, combine_wgt, scalefactor_wgt):
    m_ww_unbinned = []
    wgts_unbinned = []

    for i in range(0, len(SAMPLES)):
        for j in range(0, len(MASS_WINDOW_EDGES)-1):
            for k in range(0, len(m_ww_by_sample[i][j])):
                m_ww_unbinned.append(m_ww_by_sample[i][j][k])
                wgts_unbinned.append(combine_wgt[i][j] * scalefactor_wgt[i] * hypothesis_wgt_by_sample[i][j][k])
    print("X-LEN:" + str(len(m_ww_unbinned)) + " , Y-LEN:" + str(len(wgts_unbinned)))
    return np.histogram(m_ww_unbinned, 150, (0,5000),  weights=wgts_unbinned)

def makeXSecHistogram(m_ww, wgt):
    m_ww_unbinned = []
    wgts_unbinned = []
    for i in range(0, len(MASS_WINDOW_EDGES)-1):
        for j in range(0, len(m_ww[i])):
            m_ww_unbinned.append(m_ww[i][j])
            wgts_unbinned.append(wgt[i][j])
    print("X-LEN:" + str(len(m_ww_unbinned)) + " , Y-LEN:" + str(len(wgts_unbinned)))
    return np.histogram(m_ww_unbinned, 150, (0,5000),  weights=wgts_unbinned)

def makeOutputJSON(SIG_cutoffs_by_sample, CONT_cutoffs_by_sample, SIGplusCONT_cutoffs_by_sample, loss_comp_by_sample, combine_wgts_by_sample, scalefactor_wgt_by_sample, sample_type):

    output_data = {}
    for i in range(0, len(SIG_cutoffs_by_sample)):
        SAMPLE_DATA = {"SIG_CUTOFF": SIG_cutoffs_by_sample[i], "CONT_CUTOFF": CONT_cutoffs_by_sample[i],
                       "SIGplusCONT_CUTOFF": SIGplusCONT_cutoffs_by_sample[i], "LOSS_COMP": loss_comp_by_sample[i],
                       "COMB_WGTS": combine_wgts_by_sample[i], "RENORM_WGT":scalefactor_wgt_by_sample[i]}
        output_data[SAMPLE_PREFIX[sample_index] + str(SAMPLES[i])] = SAMPLE_DATA
    output_json = json.dumps(output_data, indent=4)
    with open(OUTJSON_NAME, "w") as outfile:
        outfile.write(output_json)

def flatten_2d(arr):
    new_list = []
    for outer_list in arr:
        for j in outer_list:
            new_list.append(j)
    return new_list

sample_index = 0 #ggH

m_ww_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
SIG_wgts_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
CONT_wgts_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
SIGplusCONT_wgts_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]

SIG_wgt_cutoffs_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
CONT_wgt_cutoffs_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
SIGplusCONT_wgt_cutoffs_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
loss_comp_by_sample = [[1 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]

nEvents_SIG_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
nEvents_CONT_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
nEvents_SIGplusCONT_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]

for i in range(0, len(SAMPLES)):
     print("Starting Sample: " + SAMPLE_PREFIX[sample_index] + str(SAMPLES[i]) + "...")
     LHE_wgts = [[] for j in range(0, len(MASS_WINDOW_EDGES))]
     sum_mass_w = [0 for j in range(0, len(MASS_WINDOW_EDGES))] 
     SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], LHE_wgts, sum_mass_w, m_ww_by_sample[i] = make_mass_bins(SAMPLE_PREFIX[sample_index] + str(SAMPLES[i]), SAMPLE_TYPES[sample_index])
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIG_wgts_init", makeXSecHistogram(m_ww_by_sample[i], SIG_wgts_by_sample[i]))
     write_to_file("HW_" + str(SAMPLES[i]) + "_CONT_wgts_init", makeXSecHistogram(m_ww_by_sample[i], CONT_wgts_by_sample[i])) 
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIGplusCONT_wgts_init", makeXSecHistogram(m_ww_by_sample[i], SIGplusCONT_wgts_by_sample[i])) 
     SIG_wgt_cutoffs_by_sample[i], CONT_wgt_cutoffs_by_sample[i], SIGplusCONT_wgt_cutoffs_by_sample[i], loss_comp_by_sample[i], SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], m_ww_by_sample[i] = remove_large_wgts(SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], LHE_wgts, sum_mass_w, m_ww_by_sample[i], SAMPLE_TYPES[sample_index])
     #ratio_lossZeroWgts, SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i] =  remove_zero_wgts(SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i])
     print(len(SIG_wgts_by_sample[i]))
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIG_wgts_slimmed", makeXSecHistogram(m_ww_by_sample[i], SIG_wgts_by_sample[i])) 
     write_to_file("HW_" + str(SAMPLES[i]) + "_CONT_wgts_slimmed", makeXSecHistogram(m_ww_by_sample[i], CONT_wgts_by_sample[i]))
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIGplusCONT_wgts_slimmed", makeXSecHistogram(m_ww_by_sample[i], SIGplusCONT_wgts_by_sample[i]))
     nEvents_SIG_by_sample[i], nEvents_CONT_by_sample[i], nEvents_SIGplusCONT_by_sample[i] = calculate_nEvents(SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i])

for i in range(0, len(SAMPLES)):
    print(str(SAMPLES[i]) + "/SIG/nEvents:" + str(nEvents_SIG_by_sample[i]))
    print(str(SAMPLES[i]) + "/CONT/nEvents:" + str(nEvents_SIG_by_sample[i]))
    print(str(SAMPLES[i]) + "/SIGplusCONT/nEvents:" + str(nEvents_SIG_by_sample[i]))

combine_wgts_SIG = normalize_mass_bins(nEvents_SIG_by_sample, SAMPLE_TYPES[sample_index])
combine_wgts_CONT = normalize_mass_bins(nEvents_CONT_by_sample, SAMPLE_TYPES[sample_index])
combine_wgts_SIGplusCONT = normalize_mass_bins(nEvents_SIGplusCONT_by_sample, SAMPLE_TYPES[sample_index])

for i in range(0, len(SAMPLES)):
    print(str(SAMPLES[i]) + "/SIG:" + str(combine_wgts_SIG[i]))
    print(str(SAMPLES[i]) + "/CONT:" + str(combine_wgts_CONT[i]))
    print(str(SAMPLES[i]) + "/SIGplusCONT:" + str(combine_wgts_SIGplusCONT[i]))

make_2d_hist(combine_wgts_SIG, combine_wgts_CONT, combine_wgts_SIGplusCONT, SAMPLE_TYPES[sample_index])

running_scale_factor_SIG = scale_factor(combine_wgts_SIG, SIG_wgts_by_sample)
running_scale_factor_CONT = scale_factor(combine_wgts_SIG, CONT_wgts_by_sample)
running_scale_factor_SIGplusCONT = scale_factor(combine_wgts_SIG, SIGplusCONT_wgts_by_sample)

print("SIG Scale Factor: " + str(running_scale_factor_SIG))
print("CONT Scale Factor: " + str(running_scale_factor_CONT))
print("SIGplusCONT Scale Factor: " + str(running_scale_factor_SIGplusCONT))

write_to_file("SIG_wgts_combined", makeFullCombinedHistogram(m_ww_by_sample, SIG_wgts_by_sample, combine_wgts_SIG, running_scale_factor_SIG))
write_to_file("CONT_wgts_combined", makeFullCombinedHistogram(m_ww_by_sample, CONT_wgts_by_sample, combine_wgts_SIG, running_scale_factor_SIG))
write_to_file("SIGplusCONT_wgts_combined", makeFullCombinedHistogram(m_ww_by_sample, SIGplusCONT_wgts_by_sample, combine_wgts_SIG, running_scale_factor_SIG))

print("Writing output JSON...")
makeOutputJSON(SIG_wgt_cutoffs_by_sample, CONT_wgt_cutoffs_by_sample, SIGplusCONT_wgt_cutoffs_by_sample, loss_comp_by_sample, combine_wgts_SIG, running_scale_factor_SIG, sample_index)

