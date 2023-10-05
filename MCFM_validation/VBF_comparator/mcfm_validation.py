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


MCFM_PATH = "/afs/cern.ch/user/t/tcarnaha/public/MCFM_SampleGen/JHUGen_ggH/MCFM-JHUGen/LHEreader/LatinoTreesLHE/HWW_LHE_ggH_rootfiles/"
SIG_FILE_NAME = "HWW_tb_lord_NNPDF30_125_nproc123_H.root"
CONT_FILE_NAME = "ggWWbx_lord_NNPDF30_125_nproc127_C.root"
SIGplusCONT_FILE_NAME = "ggWW4l_lord_NNPDF30__125_nproc126__H+C.root"

MCFM_SIG_FILE = uproot.open(MCFM_PATH + SIG_FILE_NAME)

mww_mcfm = MCFM_SIG_FILE["tree/mww"].array()
wgt_mcfm = MCFM_SIG_FILE["tree/evtWeight"].array()

# Opening dataset_config
CONFIG_FILE_PATH = "../configs/HM_dataset_config_COMBINED.json" #<-- ggH
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)

# List of sample LHEHiggsCand masses
SAMPLES = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]

m_ww = []
hwwOffshell_wgt = []
SIG_wgt = []

full_wgt = []

for mass in SAMPLES:
    print("Starting Sample HM_" + str(mass) + "...")
    for f in CONFIG["HM_" + str(mass)]["files"]:

        SIG_wgt_name = "p_Gen_GG_SIG_kappaTopBot_1_ghz1_1_MCFM"

        rootFile = uproot.open(f)
        m_ww = np.concatenate((m_ww,  rootFile["Events/higgsGenMass"].array()), axis=0)
        hwwOffshell_wgt = np.concatenate((hwwOffshell_wgt, rootFile["Events/HWWOffshell_combineWgt"].array()), axis=0)
        XSwgt = rootFile["Events/XSWeight"].array()
        reWgt = np.multiply(rootFile["Events/p_Gen_CPStoBWPropRewgt"].array(), XSwgt)
        hypothesisWgt = np.multiply(rootFile["Events/" + SIG_wgt_name].array(), reWgt)
        full_wgt = np.concatenate((full_wgt, np.multiply(hypothesisWgt, rootFile["Events/HWWOffshell_combineWgt"].array())), axis=0)
        SIG_wgt = np.concatenate((SIG_wgt, np.multiply(rootFile["Events/" + SIG_wgt_name].array(), reWgt)), axis=0)

OUTFILE = uproot.recreate("./MCFM_validation.root")

mcfm_hist = np.histogram(mww_mcfm, 150, (0,5000),  weights=wgt_mcfm)
hist = np.histogram(m_ww, 150, (0,5000),  weights=full_wgt)

mcfm_hist = np.histogram(mww_mcfm, 150, (0,5000),  weights=wgt_mcfm, density=True)
hist = np.histogram(m_ww, 150, (0,5000),  weights=full_wgt, density=True)

OUTFILE["hww_mcfm"] = mcfm_hist
OUTFILE["hww"] = hist

