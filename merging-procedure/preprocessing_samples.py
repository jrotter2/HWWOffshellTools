#JPR 02.28.2023
import uproot
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import copy


# Opening dataset_config
#CONFIG_FILE_PATH = "../configs/dataset_config.txt" #<-- ggH
#CONFIG_FILE_PATH = "../configs/HM_VBF_new_dataset_config.json" #<-- VBF
#CONFIG_FILE_PATH = "../configs/HM_dataset_config_COMBINED_minSel.json"
PRODUCTIONS = ["Summer20UL16_106x_nAODv9_HIPM_Full2016v9", "Summer20UL16_106x_nAODv9_noHIPM_Full2016v9",  "Summer20UL17_106x_nAODv9_Full2017v9", "Summer20UL18_106x_nAODv9_Full2018v9"]
PRODUCTION_INDEX = 2



PRODUCTION = PRODUCTIONS[PRODUCTION_INDEX]
#PRODUCTION = "Autumn2018v7"
# VBF: Summer20UL16_106x_nAODv9_HIPM_Full2016v9 (RUNNING) Summer20UL16_106x_nAODv9_noHIPM_Full2016v9 (RUNNING) Summer20UL17_106x_nAODv9_Full2017v9 (RUNNING) Summer20UL18_106x_nAODv9_Full2018v9 (RUNNING)
# ggH : Summer20UL16_106x_nAODv9_HIPM_Full2016v9(COMPLETE) Summer20UL16_106x_nAODv9_noHIPM_Full2016v9(COMPLETE) Summer20UL17_106x_nAODv9_Full2017v9 Summer20UL18_106x_nAODv9_Full2018v9 (COMPLETE)

SAMPLE_TYPE = "VBF"

#CONFIG_FILE_PATH = "../configs/HM_dataset_config_includeLowerHMassSamples.json" #<-- ggH includes 130, 140, 150
#CONFIG_FILE_PATH = "../configs/HM_VBF_new_dataset_config_VBF_includeLowerHMassSamples.json" #<-- VBF includes 130, 140, 150
CONFIG_FILE_PATH = "../configs/HM_VBF_new_dataset_config_" + PRODUCTION + "_" + SAMPLE_TYPE + ".json"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)

# List of sample LHEHiggsCand masses
#SAMPLES = [125,130,140,150,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]
SAMPLES = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]

# Mass window edges described in AN2021_149
#MASS_WINDOW_EDGES = [0,127.5,135,145,155,165,175,185,195,205,220,240,260,285,325,375,425,475,525,575,650,750,850,950,1250,1750,2250,2750,14000]
MASS_WINDOW_EDGES = [0,136.7,148.3,175,185,195,205,220,240,260,285,325,375,425,475,525,575,650,750,850,950,1250,1750,2250,2750,14000]

# Prefixes for ggH->HM_ and for VBF->HM_VBF_new, ggH->HM_ggH
SAMPLE_PREFIX = ["HM_ggH", "HM_VBF_new_"]

# Sample types, can be expanded to include WH, ZH, etc.
SAMPLE_TYPES = ["ggH", "VBF"]

# Output file names - may be useful to include an option to change these using ArgParser
OUTFILE_NAME = "output_preprocessing_" + PRODUCTION + "_" + SAMPLE_TYPE + ".root"
OUTPDF_NAME = "output_preprocessing_" + PRODUCTION + "_" + SAMPLE_TYPE + ".pdf"
OUTJSON_NAME = "output_preprocessing_" + PRODUCTION + "_" + SAMPLE_TYPE + ".json"
OUTFILE = uproot.recreate("./" + OUTFILE_NAME)


def make_mass_bins(dataset, sample_type):
    """
        Reads all the files in a given dataset and stores the weights for the SIG, CONT, SIGplusCONT hypotheses in different mass windows. 
    """
    SIG_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    CONT_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    SIGplusCONT_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    sum_mass_w = [0 for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    LHE_wgts = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    m_ww = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]

    # ggH hypothesis weight names
    SIG_wgt_name = "p_Gen_GG_SIG_kappaTopBot_1_ghz1_1_MCFM"
    CONT_wgt_name = "p_Gen_GG_BKG_MCFM"
    SIGplusCONT_wgt_name = "p_Gen_GG_BSI_kappaTopBot_1_ghz1_1_MCFM"
    LHE_wgt_name = "LHEWeight_originalXWGTUP"

    # VBF hypothesis weight names
    if(sample_type == "VBF"):
        SIG_wgt_name = "p_Gen_JJEW_SIG_ghv1_1_MCFM"
        CONT_wgt_name = "p_Gen_JJEW_BKG_MCFM"
        SIGplusCONT_wgt_name = "p_Gen_JJEW_BSI_ghv1_1_MCFM"

    # Looping through all files in the dataset 
    for f in CONFIG[dataset]["files"]:
        rootFile = uproot.open(f)
        pdgIds = rootFile["Events/GenPart_pdgId"].array()
        masses = rootFile["Events/GenPart_mass"].array()  
        masses_masked = masses[pdgIds == 25]
        H_mass = masses_masked[:,0]
        #H_mass = rootFile["Events/higgsGenMass"].array()
        #XSwgt = np.absolute(rootFile["Events/XSWeight"].array())
        #XSwgt = rootFile["Events/XSWeight"].array()
        baseW = rootFile["Events/baseW"].array()
        XSwgt = np.multiply(rootFile["Events/genWeight"].array(), baseW)
        reWgt = np.multiply(rootFile["Events/p_Gen_CPStoBWPropRewgt"].array(), XSwgt)
        #reWgt = np.absolute(reWgt)
        # Looping over all mass windows
        for i in range(0, len(MASS_WINDOW_EDGES)-1):
            H_mask = ((H_mass > MASS_WINDOW_EDGES[i]) & (H_mass < MASS_WINDOW_EDGES[i+1]))
            m_ww[i] = np.concatenate((m_ww[i], H_mass[H_mask]), axis=0)
            SIG_wgts[i] = np.concatenate((SIG_wgts[i], np.multiply(rootFile["Events/" + SIG_wgt_name].array(), reWgt)[H_mask]), axis=0)
            CONT_wgts[i] = np.concatenate((CONT_wgts[i], np.multiply(rootFile["Events/" + CONT_wgt_name].array(), reWgt)[H_mask]), axis=0)
            #CONT_wgts[i] = np.concatenate((CONT_wgts[i], np.multiply(rootFile["Events/" + SIG_wgt_name].array(), np.multiply(rootFile["Events/" + CONT_wgt_name].array(), reWgt))[H_mask]), axis=0)
            SIGplusCONT_wgts[i] = np.concatenate((SIGplusCONT_wgts[i], np.multiply(rootFile["Events/" + SIGplusCONT_wgt_name].array(), reWgt)[H_mask]), axis=0)
            LHE_wgts[i] = np.concatenate((LHE_wgts[i], rootFile["Events/" + LHE_wgt_name].array()[H_mask]), axis=0)

    # Looping over all mass windows to find sum of weights in each window for later use in loss compensation
    for i in range(0, len(MASS_WINDOW_EDGES)-1):
        sum_mass_w[i] = np.sum(LHE_wgts[i])

    #print(len(SIG_wgts[0]), len(CONT_wgts[0]), len(m_ww[0]))

    return SIG_wgts, CONT_wgts, SIGplusCONT_wgts, LHE_wgts, sum_mass_w, m_ww


def remove_large_wgts(SIG_wgts, CONT_wgts, SIGplusCONT_wgts, LHE_wgts, sum_mass_w, m_ww, sample_type):
    """
       Removes artificially high weights due to finite number of events being reweighted. Procedure is set by finding the .0005 (.001 for VBF BKG) quantile in each bin and comparing to largest and removing if there is a factor of 5 difference. The loss compensation is then the ratio of original sum of weights in that bin over the new sum of weights with the largest removed.
    """
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
    sum_mass_w_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    LHE_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]

    # ggH quantile fractions 
    quantile_fraction_SIG = .0005
    quantile_fraction_CONT = .001 #tightened PREV .0005
    quantile_fraction_SIGplusCONT = .001 #tighed  PREV .0005

    # VBF quantile fractions 
    if(sample_type == "VBF"):
        quantile_fraction_SIG = .005 #.005
        quantile_fraction_CONT = .001 #.001
        quantile_fraction_SIGplusCONT = .001 #.001

    # Looping over all mass windows
    for i in range(0, len(SIG_wgts)):
        # Catching mass windows with 0 events
        if(len(SIG_wgts[i]) == 0):
            quantile_cutoff_SIG.append(0)
            quantile_cutoff_CONT.append(0)
            quantile_cutoff_SIGplusCONT.append(0)
            continue
        # Calculating cutoff as 5*quantile
        quantile_cutoff_SIG.append(5*np.quantile(np.absolute(SIG_wgts[i]), 1-quantile_fraction_SIG, axis=0))
        quantile_cutoff_CONT.append(5*np.quantile(np.absolute(CONT_wgts[i]), 1-quantile_fraction_CONT, axis=0))
        quantile_cutoff_SIGplusCONT.append(5*np.quantile(np.absolute(SIGplusCONT_wgts[i]), 1-quantile_fraction_SIGplusCONT, axis=0))
    # Looping over all mass windows
    for i in range(0, len(SIG_wgts)):
        # Masking large weights
        large_wgt_mask = (np.absolute(SIG_wgts[i]) < quantile_cutoff_SIG[i]) & (np.absolute(CONT_wgts[i]) < quantile_cutoff_CONT[i]) & (np.absolute(SIGplusCONT_wgts[i]) < quantile_cutoff_SIGplusCONT[i])
        # Masking small weights
        small_wgt_mask = (np.absolute(SIG_wgts[i]) > quantile_cutoff_SIG[i]) | (np.absolute(CONT_wgts[i]) > quantile_cutoff_CONT[i]) | (np.absolute(SIGplusCONT_wgts[i]) > quantile_cutoff_SIGplusCONT[i])

        # Removing large weights
        SIG_wgts_slimmed[i] = SIG_wgts[i][large_wgt_mask]
        m_ww_slimmed[i] = m_ww[i][large_wgt_mask]
        CONT_wgts_slimmed[i] = CONT_wgts[i][large_wgt_mask]
        SIGplusCONT_wgts_slimmed[i] = SIGplusCONT_wgts[i][large_wgt_mask]
        LHE_wgts_slimmed[i] = LHE_wgts[i][large_wgt_mask]
        # Finding new sum of weights
        new_sum_mass_w = sum_mass_w[i] - np.sum(LHE_wgts[i][small_wgt_mask])
        sum_mass_w_slimmed[i] = new_sum_mass_w
        # Calculating loss ratio
        ratio_lost=0
        if(new_sum_mass_w != 0):
            ratio_lost = sum_mass_w[i] / new_sum_mass_w

        if(ratio_lost < 0):
            print("PROBLEM OCCURED TO MAKE LOSS COMP NEGATIVE - largeWeightRemoval!")

        # Correcting by the loss ratio
        SIG_wgts_slimmed[i] = SIG_wgts_slimmed[i] * ratio_lost
        CONT_wgts_slimmed[i] = CONT_wgts_slimmed[i] * ratio_lost
        SIGplusCONT_wgts_slimmed[i] = SIGplusCONT_wgts_slimmed[i] * ratio_lost
        # Storing loss ratio which is the same for all hypotheses
        ratio_lost_SIG.append(ratio_lost)
        ratio_lost_CONT.append(ratio_lost)
        ratio_lost_SIGplusCONT.append(ratio_lost)

    return quantile_cutoff_SIG, quantile_cutoff_CONT, quantile_cutoff_SIGplusCONT, ratio_lost_SIG, SIG_wgts_slimmed, CONT_wgts_slimmed, SIGplusCONT_wgts_slimmed, m_ww_slimmed, LHE_wgts_slimmed, sum_mass_w_slimmed

def remove_zero_wgts(SIG_wgts, CONT_wgts, SIGplusCONT_wgts, m_ww, LHE_wgts, sum_mass_w):
    """
       Removes zero weighted events due to off-diagonal CKM terms for VBF process. The loss compensation is then the ratio of original sum of weights in that bin over the new sum of weights with the zero removed.
    """
    ratio_lost_SIG = []
    SIG_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    CONT_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    SIGplusCONT_wgts_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)]
    m_ww_slimmed = [[] for i in range(0, len(MASS_WINDOW_EDGES)-1)] 
    for i in range(0, len(SIG_wgts)):
        zero_wgt_mask = (SIG_wgts[i] == 0) & (CONT_wgts[i] == 0) & (SIGplusCONT_wgts[i] == 0)
        nonzero_wgt_mask = (SIG_wgts[i] != 0) | (CONT_wgts[i] != 0) | (SIGplusCONT_wgts[i] != 0)
        SIG_wgts_slimmed[i] = SIG_wgts[i][nonzero_wgt_mask]
        CONT_wgts_slimmed[i] = CONT_wgts[i][nonzero_wgt_mask]
        SIGplusCONT_wgts_slimmed[i] = SIGplusCONT_wgts[i][nonzero_wgt_mask]
        m_ww_slimmed[i] = m_ww[i][nonzero_wgt_mask]

        new_sum_mass_w = sum_mass_w[i] - np.sum(LHE_wgts[i][zero_wgt_mask])
        ratio_lost=0
        if(new_sum_mass_w != 0):
            ratio_lost = sum_mass_w[i] / new_sum_mass_w
        ratio_lost_SIG.append(ratio_lost)
        if(ratio_lost < 0):
            print("PROBLEM OCCURED TO MAKE LOSS COMP NEGATIVE - zeroWeightRemoval!")

        SIG_wgts_slimmed[i] = SIG_wgts_slimmed[i] * ratio_lost
        CONT_wgts_slimmed[i] = CONT_wgts_slimmed[i] * ratio_lost
        SIGplusCONT_wgts_slimmed[i] = SIGplusCONT_wgts_slimmed[i] * ratio_lost

    return ratio_lost_SIG, SIG_wgts_slimmed, CONT_wgts_slimmed, SIGplusCONT_wgts_slimmed, m_ww_slimmed

def calculate_nEvents(SIG_wgts, CONT_wgts, SIGplusCONT_wgts):
    """
        Calculates the effective number of events in each bin by takign the ratio of sum of weights squared over the sum of squared weights. 
    """
    nEvents_SIG = [0 for i in range(0, len(SIG_wgts))]
    nEvents_CONT = [0 for i in range(0, len(SIG_wgts))]
    nEvents_SIGplusCONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_SIG = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_CONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_SIGplusCONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_sq_SIG = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_sq_CONT = [0 for i in range(0, len(SIG_wgts))]
    sum_wgts_sq_SIGplusCONT = [0 for i in range(0, len(SIG_wgts))]
    # Looping over all mass windows
    for i in range(0, len(SIG_wgts)):
        sum_wgts_SIG[i] = np.sum(SIG_wgts[i])
        sum_wgts_sq_SIG[i] = np.sum(np.multiply(SIG_wgts[i], SIG_wgts[i]))
        sum_wgts_CONT[i] = np.sum(CONT_wgts[i])
        sum_wgts_sq_CONT[i] = np.sum(np.multiply(CONT_wgts[i], CONT_wgts[i]))
        sum_wgts_SIGplusCONT[i] = np.sum(SIGplusCONT_wgts[i])
        sum_wgts_sq_SIGplusCONT[i] = np.sum(np.multiply(SIGplusCONT_wgts[i], SIGplusCONT_wgts[i]))

    # Looping over all mass windows
    for i in range(0, len(SIG_wgts)):
        # Protecting against division by 0
        if(sum_wgts_sq_SIG[i] == 0):
            nEvents_SIG[i] = 0
        else:
            nEvents_SIG[i] = sum_wgts_SIG[i]*sum_wgts_SIG[i] / sum_wgts_sq_SIG[i]
            if(np.isnan(nEvents_SIG[i])):
                nEvents_SIG[i] = 0
        # Protecting against division by 0
        if(sum_wgts_sq_CONT[i] == 0):
            nEvents_CONT[i] = 0
        else:
            nEvents_CONT[i] = sum_wgts_CONT[i]*sum_wgts_CONT[i] / sum_wgts_sq_CONT[i]
            if(np.isnan(nEvents_CONT[i])):
                nEvents_CONT[i] = 0
        # Protecting against division by 0
        if(sum_wgts_SIGplusCONT[i] == 0):
            nEvents_SIGplusCONT[i]  = 0
        else:
            nEvents_SIGplusCONT[i] = sum_wgts_SIGplusCONT[i]*sum_wgts_SIGplusCONT[i] / sum_wgts_sq_SIGplusCONT[i]
            if(np.isnan(nEvents_SIGplusCONT[i])):
                nEvents_SIGplusCONT[i] = 0

    return nEvents_SIG, nEvents_CONT, nEvents_SIGplusCONT

def normalize_mass_bins(nEvents_by_sample, sample_type):
    """
        Normalizing each mass bin by taking effective number of events for each sample over total effective for all samples. This is done in two iteractions with the second one neglecting small contributions. 
    """
    combine_wgts_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]

    # ggH has a small cutoff of .005
    small_cutoff = 0.005
    # VBF has a small cutoff of .01
    if(sample_type == "VBF"):
        small_cutoff = 0.01

    # Looping over all mass windows
    for j in range(0, len(MASS_WINDOW_EDGES)-1):
        total = 0
        for i in range(0, len(SAMPLES)):
            total = total + nEvents_by_sample[i][j]
        if(total == 0):
            continue
        for i in range(0, len(SAMPLES)):
            combine_wgts_by_sample[i][j] = nEvents_by_sample[i][j] / total

    # Looping over all mass windows
    # Getting rid of any small weight
    for j in range(0, len(MASS_WINDOW_EDGES)-1):
        for i in range(0, len(SAMPLES)):
            if(combine_wgts_by_sample[i][j] < small_cutoff):
                combine_wgts_by_sample[i][j] = 0

    # Looping over all mass windows
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
    """
        Scale factor is computed by the ratio of sum of weights in the previous sample of the sum of weights in current sample in overlapping mass bins. For samples with higgs mass <200 GeV this is set to 1 and a running scale factor is computed for all higgs masses >200GeV. Effectively, these running scale factors are based on higgs mass of 200GeV. 
    """
    scale_factor = [1 for i in range(0, len(SAMPLES))]
    running_scale_factor = [1 for i in range(0, len(SAMPLES))]
    # Looping over all samples
    for i in range(1, len(SAMPLES)):
        if(SAMPLES[i] <= 900):
            continue
        total_sample = 0
        total_sample_prev = 0
        # Looping over all mass windows to find ones in common
        for j in range(0, len(MASS_WINDOW_EDGES)-1):
            if(combine_wgts_by_sample[i][j] != 0 and combine_wgts_by_sample[i-1][j] != 0):
                total_sample = total_sample+np.sum(wgts_by_sample[i][j])
                total_sample_prev = total_sample_prev+np.sum(wgts_by_sample[i-1][j])
        scale_factor[i] = total_sample_prev / total_sample
    # Finding running scale factor for each sample
    for i in range(1, len(SAMPLES)):
        running_scale_factor[i] = running_scale_factor[i-1] * scale_factor[i]
    return running_scale_factor

def write_to_file(location, data):
    """
        Writing data to branch in root file at location.
    """
    print("Writing to file: " + str(location))
    OUTFILE[location] = data

def make_2d_hist(combine_wgts_SIG, combine_wgts_CONT, combine_wgts_SIGplusCONT, sample_type):
    """
        Making 2d histograms of window weights for each sample by mass window. Saves them to a output pdf.
    """
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
    """
        Making combined X-Section histogram for reference. Saves to output root file.
    """
    m_ww_unbinned = []
    wgts_unbinned = []

    for i in range(0, len(SAMPLES)):
        for j in range(0, len(MASS_WINDOW_EDGES)-1):
            for k in range(0, len(m_ww_by_sample[i][j])):
                m_ww_unbinned.append(m_ww_by_sample[i][j][k])
                wgts_unbinned.append(combine_wgt[i][j] * scalefactor_wgt[i] * hypothesis_wgt_by_sample[i][j][k])
    print("X-LEN:" + str(len(m_ww_unbinned)) + " , Y-LEN:" + str(len(wgts_unbinned)))
    return np.histogram(m_ww_unbinned, 100, (0, 1000),  weights=wgts_unbinned)

def makeFullCombinedHistogram_finebin(m_ww_by_sample, hypothesis_wgt_by_sample, combine_wgt, scalefactor_wgt):
    """
        Making combined X-Section histogram for reference. Saves to output root file.
    """
    m_ww_unbinned = []
    wgts_unbinned = []

    for i in range(0, len(SAMPLES)):
        for j in range(0, len(MASS_WINDOW_EDGES)-1):
            for k in range(0, len(m_ww_by_sample[i][j])):
                m_ww_unbinned.append(m_ww_by_sample[i][j][k])
                wgts_unbinned.append(combine_wgt[i][j] * scalefactor_wgt[i] * hypothesis_wgt_by_sample[i][j][k])
    print("X-LEN:" + str(len(m_ww_unbinned)) + " , Y-LEN:" + str(len(wgts_unbinned)))
    return np.histogram(m_ww_unbinned, 1000, (0, 1000),  weights=wgts_unbinned)

def makeXSecHistogram(m_ww, wgt):
    """
        Making sample X-Section histogram for reference. Saves to output root file.
    """
    m_ww_unbinned = []
    wgts_unbinned = []
    for i in range(0, len(MASS_WINDOW_EDGES)-1):
        for j in range(0, len(m_ww[i])):
            m_ww_unbinned.append(m_ww[i][j])
            wgts_unbinned.append(wgt[i][j])
            #if(wgt[i][j] < 0):
                #print("NEGATIVE WGT")
    print("X-LEN:" + str(len(m_ww_unbinned)) + " , Y-LEN:" + str(len(wgts_unbinned)))
    return np.histogram(m_ww_unbinned, 100, (0,1000),  weights=wgts_unbinned)

def makeOutputJSON(SIG_cutoffs_by_sample, CONT_cutoffs_by_sample, SIGplusCONT_cutoffs_by_sample, loss_comp_by_sample, combine_wgts_by_sample, scalefactor_wgt_by_sample, sample_type):
    """
       Writes the output JSON that stores all computed weights by sample. These will be used in the Latinos Framework Module to store in nTuples.
    """
    output_data = {}
    for i in range(0, len(SIG_cutoffs_by_sample)):
        SAMPLE_DATA = {"SIG_CUTOFF": SIG_cutoffs_by_sample[i], "CONT_CUTOFF": CONT_cutoffs_by_sample[i],
                       "SIGplusCONT_CUTOFF": SIGplusCONT_cutoffs_by_sample[i], "LOSS_COMP": loss_comp_by_sample[i],
                       "COMB_WGTS": combine_wgts_by_sample[i], "RENORM_WGT":scalefactor_wgt_by_sample[i]}
        output_data[SAMPLE_PREFIX[sample_index] + str(SAMPLES[i])] = SAMPLE_DATA
    output_json = json.dumps(output_data, indent=4)
    with open(str(SAMPLE_TYPES[sample_type]) + "_" + OUTJSON_NAME, "w") as outfile:
        outfile.write(output_json)

def flatten_2d(arr):
    """
       Flattends a 2D array - Work around for using Python2...
    """
    new_list = []
    for outer_list in arr:
        for j in outer_list:
            new_list.append(j)
    return new_list

# Sample Index
# 0 - ggH
# 1 - VBF
sample_index = 1 #VBF

# Initializing variables by sample in big 3D array...
m_ww_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
SIG_wgts_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
CONT_wgts_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
SIGplusCONT_wgts_by_sample = [[[] for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]

# Initializing variables by sample in moderate 2D array...
SIG_wgt_cutoffs_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
CONT_wgt_cutoffs_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
SIGplusCONT_wgt_cutoffs_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
loss_comp_by_sample = [[1 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]

# Initializing variables by sample in moderate 2D array...
nEvents_SIG_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
nEvents_CONT_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]
nEvents_SIGplusCONT_by_sample = [[0 for j in range(0, len(MASS_WINDOW_EDGES)-1)] for i in range(0, len(SAMPLES))]

# Looping over all samples to get weights, slim large weights, get rid of zero weights, and calculate effective number of events.
for i in range(0, len(SAMPLES)):
     print("Starting Sample: " + SAMPLE_PREFIX[sample_index] + str(SAMPLES[i]) + "...")
     LHE_wgts = [[] for j in range(0, len(MASS_WINDOW_EDGES))]
     sum_mass_w = [0 for j in range(0, len(MASS_WINDOW_EDGES))]
     SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], LHE_wgts, sum_mass_w, m_ww_by_sample[i] = make_mass_bins(SAMPLE_PREFIX[sample_index] + str(SAMPLES[i]), SAMPLE_TYPES[sample_index])
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIG_wgts_init", makeXSecHistogram(m_ww_by_sample[i], SIG_wgts_by_sample[i]))
     write_to_file("HW_" + str(SAMPLES[i]) + "_CONT_wgts_init", makeXSecHistogram(m_ww_by_sample[i], CONT_wgts_by_sample[i])) 
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIGplusCONT_wgts_init", makeXSecHistogram(m_ww_by_sample[i], SIGplusCONT_wgts_by_sample[i])) 
     print("Before Large Weight Removal: " + str([len(sample) for sample in SIG_wgts_by_sample[i]]))
     SIG_wgt_cutoffs_by_sample[i], CONT_wgt_cutoffs_by_sample[i], SIGplusCONT_wgt_cutoffs_by_sample[i], loss_comp_by_sample[i], SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], m_ww_by_sample[i], LHE_wgts, sum_mass_w = remove_large_wgts(SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], LHE_wgts, sum_mass_w, m_ww_by_sample[i], SAMPLE_TYPES[sample_index])

     print("Before Zero Weight Removal: " + str([len(sample) for sample in SIG_wgts_by_sample[i]]))
     ratio_lossZeroWgts, SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], m_ww_by_sample[i] =  remove_zero_wgts(SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i], m_ww_by_sample[i], LHE_wgts, sum_mass_w)
     for j in range(0, len(MASS_WINDOW_EDGES)-1):
         loss_comp_by_sample[i][j] = loss_comp_by_sample[i][j] * ratio_lossZeroWgts[j]
     print("After Bad Weight Removal: " + str([len(sample) for sample in SIG_wgts_by_sample[i]]))
     print(len(SIG_wgts_by_sample[i]))
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIG_wgts_slimmed", makeXSecHistogram(m_ww_by_sample[i], SIG_wgts_by_sample[i])) 
     write_to_file("HW_" + str(SAMPLES[i]) + "_CONT_wgts_slimmed", makeXSecHistogram(m_ww_by_sample[i], CONT_wgts_by_sample[i]))
     write_to_file("HW_" + str(SAMPLES[i]) + "_SIGplusCONT_wgts_slimmed", makeXSecHistogram(m_ww_by_sample[i], SIGplusCONT_wgts_by_sample[i]))
     nEvents_SIG_by_sample[i], nEvents_CONT_by_sample[i], nEvents_SIGplusCONT_by_sample[i] = calculate_nEvents(SIG_wgts_by_sample[i], CONT_wgts_by_sample[i], SIGplusCONT_wgts_by_sample[i])

for i in range(0, len(SAMPLES)):
    print(str(SAMPLES[i]) + "/SIG/nEvents:" + str(nEvents_SIG_by_sample[i]))
    print(str(SAMPLES[i]) + "/CONT/nEvents:" + str(nEvents_SIG_by_sample[i]))
    print(str(SAMPLES[i]) + "/SIGplusCONT/nEvents:" + str(nEvents_SIG_by_sample[i]))

# Normalizing mass bins for all samples
combine_wgts_SIG = normalize_mass_bins(nEvents_SIG_by_sample, SAMPLE_TYPES[sample_index])
combine_wgts_CONT = normalize_mass_bins(nEvents_CONT_by_sample, SAMPLE_TYPES[sample_index])
combine_wgts_SIGplusCONT = normalize_mass_bins(nEvents_SIGplusCONT_by_sample, SAMPLE_TYPES[sample_index])

for i in range(0, len(SAMPLES)):
    print(str(SAMPLES[i]) + "/SIG:" + str(combine_wgts_SIG[i]))
    print(str(SAMPLES[i]) + "/CONT:" + str(combine_wgts_CONT[i]))
    print(str(SAMPLES[i]) + "/SIGplusCONT:" + str(combine_wgts_SIGplusCONT[i]))

# Plotting combine weights
make_2d_hist(combine_wgts_SIG, combine_wgts_CONT, combine_wgts_SIGplusCONT, SAMPLE_TYPES[sample_index])

# Computing scale factors for each sample
running_scale_factor_SIG = scale_factor(combine_wgts_SIG, SIG_wgts_by_sample)
running_scale_factor_CONT = scale_factor(combine_wgts_SIG, CONT_wgts_by_sample)
running_scale_factor_SIGplusCONT = scale_factor(combine_wgts_SIG, SIGplusCONT_wgts_by_sample)

print("SIG Scale Factor: " + str(running_scale_factor_SIG))
print("CONT Scale Factor: " + str(running_scale_factor_CONT))
print("SIGplusCONT Scale Factor: " + str(running_scale_factor_SIGplusCONT))

# Making fully combined histogram
write_to_file("SIG_wgts_combined", makeFullCombinedHistogram(m_ww_by_sample, SIG_wgts_by_sample, combine_wgts_SIG, running_scale_factor_SIG))
write_to_file("CONT_wgts_combined", makeFullCombinedHistogram(m_ww_by_sample, CONT_wgts_by_sample, combine_wgts_SIG, running_scale_factor_SIG))
write_to_file("SIGplusCONT_wgts_combined", makeFullCombinedHistogram(m_ww_by_sample, SIGplusCONT_wgts_by_sample, combine_wgts_SIG, running_scale_factor_SIG))

write_to_file("SIG_wgts_combined_finebins", makeFullCombinedHistogram_finebin(m_ww_by_sample, SIG_wgts_by_sample, combine_wgts_SIG, running_scale_factor_SIG))

# Writing output JSON
print("Writing output JSON...")
makeOutputJSON(SIG_wgt_cutoffs_by_sample, CONT_wgt_cutoffs_by_sample, SIGplusCONT_wgt_cutoffs_by_sample, loss_comp_by_sample, combine_wgts_SIG, running_scale_factor_SIG, sample_index)

