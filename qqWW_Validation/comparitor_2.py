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

qqWW_base_path = "/eos/cms/store/mc/RunIISummer20UL18NanoAODv9/WWJJToLNuLNu_EWK_noTop_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/80000/"
qqWW_files = ["9A0B74D3-3E58-D043-BC71-E9C0AF00879A.root",
              "CEDBA430-5C79-7A4D-93AB-24A86C52347F.root"]

qqWW_qcd_base_path = "/eos/cms/store/mc/RunIISummer20UL18NanoAODv9/WWJJToLNuLNu_QCD_noTop_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2560000/"
qqWW_qcd_files = ["100070D0-723A-F647-92C5-8EADEEAACBFD.root",
                  "600B562B-9DE0-D44C-A0BE-556EE47EFEE9.root",
                  "A995588B-BFC5-B74D-BE6E-5DA08E1E6549.root",
                  "B06D6661-65D3-0748-AA88-DF4C5F8D40B3.root"]

qqWW_qcd_files = []



CONFIG_FILE_PATH = "../configs/HM_VBF_18_dataset_config_Summer20UL18_106x_nAODv9_Full2018v9_VBF_newNov28.json"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)
sampleMasses = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]

def getEventInfo(worker_index, sample_mass, sample_type, return_dict):
    print("Established Worker " + str(worker_index) + " for Sample Mass: " + str(sample_mass))
    files = CONFIG["HM_VBF_18_" + str(sample_mass)]["files"]
    H_mass = []
    sig_Wgt = []
    cont_Wgt = []
    sigpluscont_Wgt = []

    mll = []
    ptll = []
    MET = []
    We = []
    Wm = []
    detall = []
    dphill = []
    drll = []

    for f in files:
        rootFile = uproot.open(f)
        pdgIds = rootFile["Events/GenPart_pdgId"].array()
        H_mass = np.concatenate((rootFile["Events/HiggsMass"].array(), H_mass), axis=0)
        baseW = rootFile["Events/baseW"].array()
        XSwgt = np.multiply(rootFile["Events/genWeight"].array(), baseW)
        reWgt = np.multiply(rootFile["Events/p_Gen_CPStoBWPropRewgt"].array(), XSwgt)
        fullWgt = np.multiply(rootFile["Events/HWWOffshell_combineWgt"].array(), reWgt)

        mother_pdgIdx = rootFile["Events/GenPart_genPartIdxMother"].array()
        pts = rootFile["Events/GenPart_pt"].array()
        etas = rootFile["Events/GenPart_eta"].array()
        phis = rootFile["Events/GenPart_phi"].array()

        for evt_index in range(0, len(pdgIds)):
            e_vec = None
            m_vec = None
            ne_vec = None
            nm_vec = None
            nParts = 0
            try:
                nParts = len(pdgIds[evt_index])
            except:
                continue
            for part_index in range(0, nParts):
                partId = pdgIds[evt_index][part_index]
                motherId = pdgIds[evt_index][mother_pdgIdx[evt_index][part_index]]
                if(abs(partId) == 11 and abs(motherId) == 24):
                    e_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], 0.000511)
                elif(abs(partId) == 13 and abs(motherId) == 24):
                    m_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], .106)
                elif(abs(partId) == 12 and abs(motherId) == 24):
                    ne_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], 0)
                elif(abs(partId) == 14 and abs(motherId) == 24):
                    nm_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], 0)

            if(e_vec == None or m_vec == None or ne_vec == None or nm_vec == None):
                mll.append(-1)
                ptll.append(-1)
                MET.append(-1)
                We.append(-1)
                Wm.append(-1)
                detall.append(-1)
                dphill.append(-1)
                drll.append(-1)
            else:
                ll = e_vec + m_vec
                met = ne_vec + nm_vec
                we = e_vec + ne_vec
                wm = m_vec + nm_vec

                if(min(abs(e_vec.Eta() - m_vec.Eta()), abs(m_vec.Eta() - e_vec.Eta())) < 1):
                    mll.append(-1)
                    ptll.append(-1)
                    MET.append(-1)
                    We.append(-1)
                    Wm.append(-1)
                    detall.append(-1)
                    dphill.append(-1)
                    drll.append(-1)
                    continue
                mll.append(ll.M())
                ptll.append(ll.Pt())
                MET.append(met.Et())
                We.append(we.M())
                Wm.append(wm.M())
                detall_var = min(abs(e_vec.Eta() - m_vec.Eta()), abs(m_vec.Eta() - e_vec.Eta()))
                dphill_var = min(abs(e_vec.Phi() - m_vec.Phi()), abs(m_vec.Phi() - e_vec.Phi()))
                while(dphill_var > 3.14159):
                    dphill_var = dphill_var - 3.14159;
                detall.append(detall_var)
                dphill.append(dphill_var)
                drll.append((detall_var**2 + detall_var**2)**(.5))

        sig_wgt_name = "p_Gen_JJEW_SIG_ghv1_1_MCFM"
        cont_wgt_name = "p_Gen_JJEW_BKG_MCFM"
        sigpluscont_wgt_name = "p_Gen_JJEW_BSI_ghv1_1_MCFM"

        sig_Wgt = np.concatenate((np.multiply(rootFile["Events/" + sig_wgt_name].array(), fullWgt), sig_Wgt), axis=0) 
        cont_Wgt = np.concatenate((np.multiply(rootFile["Events/" + cont_wgt_name].array(), fullWgt), cont_Wgt), axis=0)
        sigpluscont_Wgt = np.concatenate((np.multiply(rootFile["Events/" + sigpluscont_wgt_name].array(), fullWgt), sigpluscont_Wgt), axis=0)
        break

    return_dict[worker_index] = [H_mass, sig_Wgt, cont_Wgt, sigpluscont_Wgt, mll, ptll, MET,We,Wm, detall, dphill, drll]


manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []
for i,sample_mass in enumerate(sampleMasses):
    p = multiprocessing.Process(target=getEventInfo, args=(i, sample_mass, "VBF", return_dict))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()
#print(return_dict.values())

mww = []
sonly_wgt = []
bonly_wgt = []
sbi_wgt = []

mll = []
ptll = []
MET = []
We = []
Wm = []
detall = []
dphill = []
drll = []

for worker_vals in return_dict.values():
    mww = np.concatenate((worker_vals[0], mww), axis=0)
    sonly_wgt = np.concatenate((worker_vals[1], sonly_wgt), axis=0)
    bonly_wgt = np.concatenate((worker_vals[2], bonly_wgt), axis=0)
    sbi_wgt = np.concatenate((worker_vals[3], sbi_wgt), axis=0)
    mll = np.concatenate((worker_vals[4], mll), axis=0)
    ptll = np.concatenate((worker_vals[5], ptll), axis=0)
    MET = np.concatenate((worker_vals[6], MET), axis=0)
    We = np.concatenate((worker_vals[7], We), axis=0)
    Wm = np.concatenate((worker_vals[8], Wm), axis=0)
    detall = np.concatenate((worker_vals[9], detall), axis=0)
    dphill = np.concatenate((worker_vals[10], dphill), axis=0)
    drll = np.concatenate((worker_vals[11], drll), axis=0)


cut_off_cont = np.quantile(np.absolute(bonly_wgt[i]), 1-.0001)
print(cut_off_cont)

cont_cutoff_mask = bonly_wgt > cut_off_cont

bonly_wgt = bonly_wgt[cont_cutoff_mask]
mww = mww[cont_cutoff_mask]
mll = np.array(mll)[cont_cutoff_mask]
ptll = np.array(ptll)[cont_cutoff_mask]
MET = np.array(MET)[cont_cutoff_mask]
We = np.array(We)[cont_cutoff_mask]
Wm = np.array(Wm)[cont_cutoff_mask]
detall = np.array(detall)[cont_cutoff_mask]
dphill = np.array(dphill)[cont_cutoff_mask]
drll = np.array(drll)[cont_cutoff_mask]


print("Starting EWK...")

qqWW_mww = []
qqWW_mll = []
qqWW_ptll = []
qqWW_MET = []
qqWW_We = []
qqWW_Wm = []
qqWW_detall = []
qqWW_dphill = []
qqWW_drll = []
qqWW_xSec = 0.09765 ## https://xsdb-temp.app.cern.ch/?searchQuery=DAS=WWJJToLNuLNu_EWK_noTop_TuneCP5_13TeV-madgraph-pythia

for f in qqWW_files:
    rootFile = uproot.open(qqWW_base_path + f)
    pdgIds = rootFile["Events/GenPart_pdgId"].array()
    pts = rootFile["Events/GenPart_pt"].array()
    etas = rootFile["Events/GenPart_eta"].array()
    phis = rootFile["Events/GenPart_phi"].array()
    masses = rootFile["Events/GenPart_mass"].array()
    mother_pdgIdx = rootFile["Events/GenPart_genPartIdxMother"].array()
    for evt_index in range(0, len(pdgIds)):
        W1_p4 = None
        W2_p4 = None
        e_vec = None
        m_vec = None
        ne_vec = None
        nm_vec = None

        for part_index, partId in enumerate(pdgIds[evt_index]):
            if(partId == 24 and W1_p4 == None):
                W1_p4 = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], masses[evt_index][part_index])
            elif(partId == -24 and W2_p4 == None):
                W2_p4 = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], masses[evt_index][part_index])

            motherId = pdgIds[evt_index][mother_pdgIdx[evt_index][part_index]] 
            if(abs(partId) == 11 and abs(motherId) == 24):
                e_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], 0.000511)
            elif(abs(partId) == 13 and abs(motherId) == 24):
                m_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], .106)
            elif(abs(partId) == 12 and abs(motherId) == 24):
                ne_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], 0)
            elif(abs(partId) == 14 and abs(motherId) == 24):
                nm_vec = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], 0)


        WW_p4 = W1_p4 + W2_p4
        if(W1_p4 != None and W2_p4 != None):
            WW_p4 = W1_p4 + W2_p4
            qqWW_mww.append(WW_p4.M())

            if(e_vec == None or m_vec == None or ne_vec == None or nm_vec == None):
                qqWW_mll.append(-1)
                qqWW_ptll.append(-1)
                qqWW_MET.append(-1)
                qqWW_We.append(-1)
                qqWW_Wm.append(-1)
                qqWW_detall.append(-1)
                qqWW_dphill.append(-1)
                qqWW_drll.append(-1)
                continue
            if(min(abs(e_vec.Eta() - m_vec.Eta()), abs(m_vec.Eta() - e_vec.Eta())) < 1):
                qqWW_mll.append(-1)
                qqWW_ptll.append(-1)
                qqWW_MET.append(-1)
                qqWW_We.append(-1)
                qqWW_Wm.append(-1)
                qqWW_detall.append(-1)
                qqWW_dphill.append(-1)
                qqWW_drll.append(-1)
                continue
            ll = e_vec + m_vec
            met = ne_vec + nm_vec
            we = e_vec + ne_vec
            wm = m_vec + nm_vec
            qqWW_mll.append(ll.M())
            qqWW_ptll.append(ll.Pt())
            qqWW_MET.append(met.Et())
            qqWW_We.append(we.M())
            qqWW_Wm.append(wm.M())
            detall_var = min(abs(e_vec.Eta() - m_vec.Eta()), abs(m_vec.Eta() - e_vec.Eta()))
            dphill_var = min(abs(e_vec.Phi() - m_vec.Phi()), abs(m_vec.Phi() - e_vec.Phi()))
            while(dphill_var > 3.14159):
                dphill_var = dphill_var - 3.14159;
            qqWW_detall.append(detall_var)
            qqWW_dphill.append(dphill_var)
            qqWW_drll.append((detall_var**2 + detall_var**2)**(.5))
        else:
            print("SS WW")

nEvents = len(qqWW_mww)
qqWW_wgt = [qqWW_xSec*1000 / nEvents for i in range(0, nEvents)]

print("Starting QCD...")


qqWW_qcd_mww = []
qqWW_qcd_xSec = 2.163  ## https://xsdb-temp.app.cern.ch/?searchQuery=DAS=WWJJToLNuLNu_EWK_noTop_TuneCP5_13TeV-madgraph-pythia

for f in qqWW_qcd_files:
    rootFile = uproot.open(qqWW_qcd_base_path + f)
    pdgIds = rootFile["Events/GenPart_pdgId"].array()
    pts = rootFile["Events/GenPart_pt"].array()
    etas = rootFile["Events/GenPart_eta"].array()
    phis = rootFile["Events/GenPart_phi"].array()
    masses = rootFile["Events/GenPart_mass"].array()
    for evt_index in range(0, len(pdgIds)):
        W1_p4 = None
        W2_p4 = None
        hasHiggs = False
        for part_index, part_pdgId in enumerate(pdgIds[evt_index]):
            if(part_pdgId == 25):
                hasHiggs = True
            if(part_pdgId == 24 and W1_p4 == None):
                W1_p4 = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], masses[evt_index][part_index])
            elif(part_pdgId == -24 and W2_p4 == None):
                W2_p4 = ROOT.Math.PtEtaPhiMVector(pts[evt_index][part_index], etas[evt_index][part_index], phis[evt_index][part_index], masses[evt_index][part_index])
        if(hasHiggs):
            continue

        if(W1_p4 != None and W2_p4 != None):
            WW_p4 = W1_p4 + W2_p4
            qqWW_qcd_mww.append(WW_p4.M())
        else:
            print("SS WW")

nEvents = len(qqWW_qcd_mww)
qqWW_qcd_wgt = [qqWW_qcd_xSec*1000 / nEvents for i in range(0, nEvents)]

"""
qqWW_tot_mww = []
qqWW_tot_wgt = []

for index,mww_entry in enumerate(qqWW_mww):
    qqWW_tot_mww.append(mww_entry)
    qqWW_tot_wgt.append(qqWW_wgt[index])

for index,mww_entry in enumerate(qqWW_qcd_mww):
    qqWW_tot_mww.append(mww_entry)
    qqWW_tot_wgt.append(qqWW_qcd_wgt[index])
"""

pdf_pages = PdfPages("./VBF_comparison_background_newApr9_v2.pdf")



#f = uproot.open("../merging-procedure/output_preprocessing_Summer20UL18_106x_nAODv9_Full2018v9_VBF_newDec3.root")

#hWW_orig_values = f["CONT_wgts_combined"].values
#hWW_orig_values_norm = f["CONT_wgts_combined"].values / (np.sum(hWW_orig_values) * (1000/100))

#hWW_orig_bins = f["CONT_wgts_combined"].bins


qqWW_total_w = np.sum(qqWW_wgt)
hww_total_w = np.sum(bonly_wgt)

norm_factor = qqWW_total_w / hww_total_w

print("Norm Factor : " + str(norm_factor))

bonly_wgt = bonly_wgt * norm_factor



#qqWW_tot_values, bins = np.histogram(qqWW_tot_mww, 100, (0,1000),  weights=qqWW_tot_wgt)
qqWW_ewk_values, bins = np.histogram(qqWW_mww, 100, (0,1000),  weights=qqWW_wgt)
qqWW_ewk_err = np.histogram(qqWW_mww, 100, (0,1000),  weights=np.square(qqWW_wgt))[0]
#qqWW_qcd_values, bins = np.histogram(qqWW_qcd_mww, 100, (0,1000),  weights=qqWW_qcd_wgt)
hWW_values, bins = np.histogram(mww, 100, (0,1000),  weights=bonly_wgt)
hWW_err= np.histogram(mww, 100, (0,1000),  weights=np.square(bonly_wgt))[0]

#qqWW_tot_values_norm, bins = np.histogram(qqWW_tot_mww, 100, (0,1000), density=True,  weights=qqWW_tot_wgt)
qqWW_ewk_values_norm, bins = np.histogram(qqWW_mww, 100, (0,1000), density=True,  weights=qqWW_wgt)

#qqWW_qcd_values_norm, bins = np.histogram(qqWW_qcd_mww, 100, (0,1000), density=True,  weights=qqWW_qcd_wgt)
hWW_values_norm, bins = np.histogram(mww, 100, (0,1000), density=True,  weights=bonly_wgt)


qqWW_mll_values, bins2 = np.histogram(qqWW_mll, 100, (0,500),  weights=qqWW_wgt)
qqWW_mll_err = np.histogram(qqWW_mll, 100, (0,500),  weights=np.square(qqWW_wgt))[0]
mll_values, bins2 = np.histogram(mll, 100, (0,500),  weights=bonly_wgt)
mll_err = np.histogram(mll, 100, (0,500),  weights=np.square(bonly_wgt))[0]

qqWW_ptll_values, bins2 = np.histogram(qqWW_ptll, 100, (0,500),  weights=qqWW_wgt)
qqWW_ptll_err = np.histogram(qqWW_ptll, 100, (0,500),  weights=np.square(qqWW_wgt))[0]
ptll_values, bins2 = np.histogram(ptll, 100, (0,500),  weights=bonly_wgt)
ptll_err= np.histogram(ptll, 100, (0,500),  weights=np.square(bonly_wgt))[0]

qqWW_MET_values, bins2 = np.histogram(qqWW_MET, 100, (0,500),  weights=qqWW_wgt)
qqWW_MET_err = np.histogram(qqWW_MET, 100, (0,500),  weights=np.square(qqWW_wgt))[0]
MET_values, bins2 = np.histogram(MET, 100, (0,500),  weights=bonly_wgt)
MET_err= np.histogram(MET, 100, (0,500),  weights=np.square(bonly_wgt))[0]

qqWW_We_values, bins2 = np.histogram(qqWW_We, 100, (0,500),  weights=qqWW_wgt)
qqWW_We_err = np.histogram(qqWW_We, 100, (0,500),  weights=np.square(qqWW_wgt))[0]
We_values, bins2 = np.histogram(We, 100, (0,500),  weights=bonly_wgt)
We_err = np.histogram(We, 100, (0,500),  weights=np.square(bonly_wgt))[0]

qqWW_Wm_values, bins2 = np.histogram(qqWW_Wm, 100, (0,500),  weights=qqWW_wgt)
qqWW_Wm_err = np.histogram(qqWW_Wm, 100, (0,500),  weights=np.square(qqWW_wgt))[0]
Wm_values, bins2 = np.histogram(Wm, 100, (0,500),  weights=bonly_wgt)
Wm_err= np.histogram(Wm, 100, (0,500),  weights=np.square(bonly_wgt))[0]

qqWW_detall_values, bins_detall = np.histogram(qqWW_detall, 50, (0,5),  weights=qqWW_wgt)
qqWW_detall_err = np.histogram(qqWW_detall, 50, (0,5),  weights=np.square(qqWW_wgt))[0]
detall_values, bins2_detall = np.histogram(detall, 50, (0,5),  weights=bonly_wgt)
detall_err= np.histogram(detall, 50, (0,5),  weights=np.square(bonly_wgt))[0]
xerr_bins_detall = [(bins_detall[1]-bins_detall[0])/2 for i in range(0,len(bins_detall))]

qqWW_dphill_values, bins_dphill = np.histogram(qqWW_dphill, 50, (0,5),  weights=qqWW_wgt)
qqWW_dphill_err = np.histogram(qqWW_dphill, 50, (0,5),  weights=np.square(qqWW_wgt))[0]
dphill_values, bins2_dphill = np.histogram(dphill, 50, (0,5),  weights=bonly_wgt)
dphill_err= np.histogram(dphill, 50, (0,5),  weights=np.square(bonly_wgt))[0]
xerr_bins_dphill = [(bins_dphill[1]-bins_dphill[0])/2 for i in range(0,len(bins_dphill))]

qqWW_drll_values, bins_drll = np.histogram(qqWW_drll, 50, (0,5),  weights=qqWW_wgt)
qqWW_drll_err = np.histogram(qqWW_drll, 50, (0,5),  weights=np.square(qqWW_wgt))[0]
drll_values, bins2_dphill = np.histogram(drll, 50, (0,5),  weights=bonly_wgt)
drll_err= np.histogram(drll, 50, (0,5),  weights=np.square(bonly_wgt))[0]
xerr_bins_drll = [(bins_drll[1]-bins_drll[0])/2 for i in range(0,len(bins_drll))]


xerr_bins = [(bins[1]-bins[0])/2 for i in range(0,len(bins))]
xerr_bins2 = [(bins2[1]-bins2[0])/2 for i in range(0,len(bins2))]


fig, ax = plt.subplots(1)

ax.errorbar(bins[:-1], qqWW_ewk_values, linestyle="",  xerr=xerr_bins[:-1], yerr=qqWW_ewk_err, label="EWK")
#ax.errorbar(bins[:-1], qqWW_qcd_values, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="QCD")
#ax.errorbar(bins[:-1], qqWW_tot_values, linestyle="-.",  xerr=xerr_bins[:-1], yerr=0, label="EWK+QCD")
ax.errorbar(bins[:-1], hWW_values, linestyle="", xerr=xerr_bins[:-1], yerr=0, label="B_Hyp_Wgt")
#ax.errorbar(bins[:-1], hWW_orig_values, linestyle="", xerr=xerr_bins[:-1], yerr=0, label="B_Hyp")
#hep.histplot(mcfm_values, bins, xerr=True, yerr=yerr_mcfm, density=True, histtype="errorbar", label="MCFM")
#hep.histplot(ntuple_values, bins, xerr=True, yerr=yerr_ntuple, density=True, histtype="errorbar", label="POWHEG+JHUGen Merged")
ax.set_title("$d\sigma/dE$ vs. $m_{WW}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$m_{WW}$")
ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlim([0,1000])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)


fig, ax = plt.subplots(1)

ax.errorbar(bins[:-1], qqWW_ewk_values_norm, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="EWK")
#ax.errorbar(bins[:-1], qqWW_qcd_values_norm, linestyle="",  xerr=xerr_bins[:-1], yerr=0, label="QCD")
#ax.errorbar(bins[:-1], qqWW_tot_values_norm, linestyle="-.",  xerr=xerr_bins[:-1], yerr=0, label="EWK+QCD")
ax.errorbar(bins[:-1], hWW_values_norm, linestyle="", xerr=xerr_bins[:-1], yerr=0, label="B_Hyp_Wgt")
#ax.errorbar(bins[:-1], hWW_orig_values_norm, linestyle="", xerr=xerr_bins[:-1], yerr=0, label="B_Hyp")
#hep.histplot(mcfm_values, bins, xerr=True, yerr=yerr_mcfm, density=True, histtype="errorbar", label="MCFM")
#hep.histplot(ntuple_values, bins, xerr=True, yerr=yerr_ntuple, density=True, histtype="errorbar", label="POWHEG+JHUGen Merged")
ax.set_title("$d\sigma/dE$ vs. $m_{WW}$ - Normalized to 1",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$m_{WW}$")
ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlim([0,1000])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)


fig, ax = plt.subplots(1)
ax.errorbar(bins2[:-1], qqWW_mll_values, linestyle="",  xerr=xerr_bins2[:-1], yerr=qqWW_mll_err, label="EWK")
ax.errorbar(bins2[:-1], mll_values, linestyle="", xerr=xerr_bins2[:-1], yerr=mll_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. $m_{ll}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$m_{ll}$")
ax.set_yscale('log')
ax.set_xlim([0,500])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots(1)
ax.errorbar(bins2[:-1], qqWW_ptll_values, linestyle="",  xerr=xerr_bins2[:-1], yerr=qqWW_ptll_err, label="EWK")
ax.errorbar(bins2[:-1], ptll_values, linestyle="", xerr=xerr_bins2[:-1], yerr=ptll_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. $p_{T}^{ll}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$p_{T}^{ll}$")
ax.set_yscale('log')
ax.set_xlim([0,500])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots(1)
ax.errorbar(bins2[:-1], qqWW_MET_values, linestyle="",  xerr=xerr_bins2[:-1], yerr=qqWW_MET_err, label="EWK")
ax.errorbar(bins2[:-1], MET_values, linestyle="", xerr=xerr_bins2[:-1], yerr=MET_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. MET",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("MET")
ax.set_yscale('log')
ax.set_xlim([0,500])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots(1)
ax.errorbar(bins2[:-1], qqWW_We_values, linestyle="",  xerr=xerr_bins2[:-1], yerr=qqWW_We_err, label="EWK")
ax.errorbar(bins2[:-1], We_values, linestyle="", xerr=xerr_bins2[:-1], yerr=We_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. $W_1$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$W_1$")
ax.set_yscale('log')
ax.set_xlim([0,500])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots(1)
ax.errorbar(bins2[:-1], qqWW_Wm_values, linestyle="",  xerr=xerr_bins2[:-1], yerr=qqWW_Wm_err, label="EWK")
ax.errorbar(bins2[:-1], Wm_values, linestyle="", xerr=xerr_bins2[:-1], yerr=Wm_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. $W_2$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$W_2$")
ax.set_yscale('log')
ax.set_xlim([0,500])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots(1)
ax.errorbar(bins_detall[:-1], qqWW_detall_values, linestyle="",  xerr=xerr_bins_detall[:-1], yerr=qqWW_detall_err, label="EWK")
ax.errorbar(bins_detall[:-1], detall_values, linestyle="", xerr=xerr_bins_detall[:-1], yerr=detall_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. $d\eta_{ll}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$d\eta_{ll}$")
ax.set_yscale('log')
ax.set_xlim([1,5])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots(1)
ax.errorbar(bins_dphill[:-1], qqWW_dphill_values, linestyle="",  xerr=xerr_bins_dphill[:-1], yerr=qqWW_dphill_err, label="EWK")
ax.errorbar(bins_dphill[:-1], dphill_values, linestyle="", xerr=xerr_bins_dphill[:-1], yerr=dphill_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. $d\phi_{ll}$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$d\phi_{ll}$")
ax.set_yscale('log')
ax.set_xlim([0,5])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)

fig, ax = plt.subplots(1)
ax.errorbar(bins_drll[:-1], qqWW_drll_values, linestyle="",  xerr=xerr_bins_drll[:-1], yerr=qqWW_drll_err, label="EWK")
ax.errorbar(bins_drll[:-1], drll_values, linestyle="", xerr=xerr_bins_drll[:-1], yerr=drll_err, label="B_Hyp_Wgt")
ax.set_title("$d\sigma/dE$ vs. $dR$",loc="left", fontsize=15, pad=20)
ax.set_ylabel("$d\sigma/dE$ (a.u.)")
ax.set_xlabel("$dR$")
ax.set_yscale('log')
ax.set_xlim([0,5])
ax.legend(prop = {"size": 12 })
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

fig.set_size_inches(6,6)
pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=1)




pdf_pages.close()
