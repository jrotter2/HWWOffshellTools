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
from ROOT import TLorentzVector


MCFM_PATH = "/afs/cern.ch/user/t/tcarnaha/public/MCFM_SampleGen/JHUGen_ggH/MCFM-JHUGen/LHEreader/LatinoTreesLHE/HWW_LHE_ggH_rootfiles/"
SIG_FILE_NAME = "HWW_tb_lord_NNPDF30_125_nproc123_H.root"
CONT_FILE_NAME = "ggWWbx_lord_NNPDF30_125_nproc127_C.root"
SIGplusCONT_FILE_NAME = "ggWW4l_lord_NNPDF30__125_nproc126__H+C.root"

MCFM_SIG_FILE = uproot.open(MCFM_PATH + CONT_FILE_NAME)

wgt_mcfm = MCFM_SIG_FILE["tree/evtWeight"].array()
mww_mcfm = MCFM_SIG_FILE["tree/mww"].array()

pt_l1 = MCFM_SIG_FILE["tree/ptl1"].array()
eta_l1 = MCFM_SIG_FILE["tree/etal1"].array()
phi_l1 = MCFM_SIG_FILE["tree/phil1"].array()
m_l1 = MCFM_SIG_FILE["tree/ml1"].array()
pdgId_l1 = MCFM_SIG_FILE["tree/pdgIdl1"].array()

pt_l2 = MCFM_SIG_FILE["tree/ptl2"].array() 
eta_l2 = MCFM_SIG_FILE["tree/etal2"].array()
phi_l2 = MCFM_SIG_FILE["tree/phil2"].array()
m_l2 = MCFM_SIG_FILE["tree/ml2"].array()
pdgId_l2 = MCFM_SIG_FILE["tree/pdgIdl2"].array()

pt_v1 = MCFM_SIG_FILE["tree/ptv1"].array()
eta_v1 = MCFM_SIG_FILE["tree/etav1"].array()
phi_v1 = MCFM_SIG_FILE["tree/phiv1"].array()
m_v1 = MCFM_SIG_FILE["tree/mv1"].array()
pdgId_v1 = MCFM_SIG_FILE["tree/pdgIdv1"].array()

pt_v2 = MCFM_SIG_FILE["tree/ptv2"].array()
eta_v2 = MCFM_SIG_FILE["tree/etav2"].array()
phi_v2 = MCFM_SIG_FILE["tree/phiv2"].array()
m_v2 = MCFM_SIG_FILE["tree/mv2"].array()
pdgId_v2 = MCFM_SIG_FILE["tree/pdgIdv2"].array()

pt_nu1 = MCFM_SIG_FILE["tree/ptnu1"].array()
eta_nu1 = MCFM_SIG_FILE["tree/etanu1"].array()
phi_nu1 = MCFM_SIG_FILE["tree/phinu1"].array()
m_nu1 = MCFM_SIG_FILE["tree/mnu1"].array()
pdgId_nu1 = MCFM_SIG_FILE["tree/pdgIdnu1"].array()

pt_nu2 = MCFM_SIG_FILE["tree/ptnu2"].array()
eta_nu2 = MCFM_SIG_FILE["tree/etanu2"].array()
phi_nu2 = MCFM_SIG_FILE["tree/phinu2"].array()
m_nu2 = MCFM_SIG_FILE["tree/mnu2"].array()
pdgId_nu2 = MCFM_SIG_FILE["tree/pdgIdnu2"].array()


ele_pt = []
ele_eta = []
muon_pt = []
muon_eta = []
MET = []
mll = []
ptll = []
m_w = []
m_ww = []
wgts = []
m_w_wgts = []
ele_pt_offshell = []
ele_eta_offshell = []
muon_pt_offshell = []
muon_eta_offshell = []
MET_offshell = []
mll_offshell = []
ptll_offshell = []
m_w_offshell = []
m_ww_offshell = []
wgts_offshell = []
m_w_wgts_offshell = []
ele_pt_onshell = []
ele_eta_onshell = []
muon_pt_onshell = []
muon_eta_onshell = []
MET_onshell = []
mll_onshell = []
ptll_onshell = []
m_w_onshell = []
m_ww_onshell = []
wgts_onshell = []
m_w_wgts_onshell = []

for i in range(0, len(wgt_mcfm)):
    ele_vec = TLorentzVector()
    muon_vec = TLorentzVector()
    ele_nu_vec = TLorentzVector()
    muon_nu_vec = TLorentzVector()
    W1_vec = TLorentzVector()
    W2_vec = TLorentzVector()
    W1_vec.SetPtEtaPhiM(pt_v1[i], eta_v1[i], phi_v1[i], m_v1[i])
    W2_vec.SetPtEtaPhiM(pt_v2[i], eta_v2[i], phi_v2[i], m_v2[i])
    ele_c_pt = 0
    muon_c_pt = 0
    if(abs(pdgId_l1[i]) == 11 and abs(pdgId_l2[i]) == 13):
        muon_vec.SetPtEtaPhiM(pt_l2[i], eta_l2[i], phi_l2[i], m_l2[i])
        ele_vec.SetPtEtaPhiM(pt_l1[i], eta_l1[i], phi_l1[i], m_l1[i])
        ele_c_pt=pt_l1[i]
        muon_c_pt=pt_l2[i]

    elif(abs(pdgId_l1[i]) == 13 and abs(pdgId_l2[i]) == 11):
        ele_vec.SetPtEtaPhiM(pt_l2[i], eta_l2[i], phi_l2[i], m_l2[i])
        muon_vec.SetPtEtaPhiM(pt_l1[i], eta_l1[i], phi_l1[i], m_l1[i])
        ele_c_pt=pt_l2[i]
        muon_c_pt=pt_l1[i]
    else:
        continue

    if(abs(pdgId_nu1[i]) == 12 and abs(pdgId_nu2[i]) == 14):
        muon_nu_vec.SetPtEtaPhiM(pt_v2[i], eta_v2[i], phi_v2[i], m_v2[i])
        ele_nu_vec.SetPtEtaPhiM(pt_v1[i], eta_v1[i], phi_v1[i], m_v1[i]) 
    elif(abs(pdgId_nu1[i]) == 14 and abs(pdgId_nu2[i]) == 12):
        ele_nu_vec.SetPtEtaPhiM(pt_v2[i], eta_v2[i], phi_v2[i], m_v2[i])
        muon_nu_vec.SetPtEtaPhiM(pt_v1[i], eta_v1[i], phi_v1[i], m_v1[i])
    else:
        continue
    if(abs(ele_vec.Eta()) > 2.5 or abs(muon_vec.Eta()) > 2.4):
        continue

    if(ele_vec.Pt() < 20 or  muon_vec.Pt() < 20):
        continue

    ll_vec = ele_vec + muon_vec
    WW_vec = W1_vec + W2_vec
    MET_vec = WW_vec - ll_vec
    mll.append(ll_vec.M())
    ptll.append(ll_vec.Pt())
    MET.append(MET_vec.Et())
    m_w.append(m_v1[i])
    m_w.append(m_v2[i])
    m_ww.append(mww_mcfm[i])
    wgts.append(wgt_mcfm[i])
    m_w_wgts.append(wgt_mcfm[i])
    m_w_wgts.append(wgt_mcfm[i])

    muon_pt.append(muon_c_pt)
    muon_eta.append(muon_vec.Eta())
    ele_pt.append(ele_c_pt)
    ele_eta.append(ele_vec.Eta())
    if(mww_mcfm[i] > 2*81):
        mll_offshell.append(ll_vec.M())
        ptll_offshell.append(ll_vec.Pt())
        MET_offshell.append(MET_vec.Et())
        m_w_offshell.append(m_v1[i])
        m_w_offshell.append(m_v2[i])
        m_ww_offshell.append(mww_mcfm[i])
        wgts_offshell.append(wgt_mcfm[i])
        m_w_wgts_offshell.append(wgt_mcfm[i])
        m_w_wgts_offshell.append(wgt_mcfm[i])

        muon_pt_offshell.append(muon_c_pt)
        muon_eta_offshell.append(muon_vec.Eta())
        ele_pt_offshell.append(ele_c_pt)
        ele_eta_offshell.append(ele_vec.Eta())
    else:
        mll_onshell.append(ll_vec.M())
        ptll_onshell.append(ll_vec.Pt())
        MET_onshell.append(MET_vec.Et())
        m_w_onshell.append(m_v1[i])
        m_w_onshell.append(m_v2[i])
        m_ww_onshell.append(mww_mcfm[i])
        wgts_onshell.append(wgt_mcfm[i])
        m_w_wgts_onshell.append(wgt_mcfm[i])
        m_w_wgts_onshell.append(wgt_mcfm[i])

        muon_pt_onshell.append(muon_c_pt)
        muon_eta_onshell.append(muon_vec.Eta())
        ele_pt_onshell.append(ele_c_pt)
        ele_eta_onshell.append(ele_vec.Eta())

OUTFILE = uproot.recreate("./MCFM_validation_plots_specialSelection_shellSplit_CONT.root")

mww_hist = np.histogram(m_ww, 100, (0,1000),  weights=wgts)
mll_hist = np.histogram(mll, 100, (0,1000),  weights=wgts)
ptll_hist = np.histogram(ptll, 100, (0,1000),  weights=wgts)
MET_hist = np.histogram(MET, 100, (0,1000),  weights=wgts)
m_w_hist = np.histogram(m_w, 100, (0,1000),  weights=m_w_wgts)
ele_pt_hist = np.histogram(ele_pt, 100, (0,1000),  weights=wgts)
ele_eta_hist = np.histogram(ele_eta, 100, (-7,7),  weights=wgts)
muon_pt_hist = np.histogram(muon_pt, 100, (0,1000),  weights=wgts)
muon_eta_hist = np.histogram(muon_eta, 100, (-7,7),  weights=wgts)

OUTFILE["mww"] = mww_hist
OUTFILE["mww_wide"] = mww_hist
OUTFILE["mll"] = mll_hist
OUTFILE["ptll"] = ptll_hist
OUTFILE["MET"] = MET_hist 
OUTFILE["mw"] = m_w_hist
OUTFILE["ele_pt"] = ele_pt_hist
OUTFILE["ele_eta"] = ele_eta_hist
OUTFILE["muon_pt"] = muon_pt_hist 
OUTFILE["muon_eta"] = muon_eta_hist

mww_hist_offshell = np.histogram(m_ww_offshell, 100, (0,1000),  weights=wgts_offshell)
mll_hist_offshell = np.histogram(mll_offshell, 100, (0,1000),  weights=wgts_offshell)
ptll_hist_offshell = np.histogram(ptll_offshell, 100, (0,1000),  weights=wgts_offshell)
MET_hist_offshell = np.histogram(MET_offshell, 100, (0,1000),  weights=wgts_offshell)
m_w_hist_offshell = np.histogram(m_w_offshell, 100, (0,1000),  weights=m_w_wgts_offshell)
ele_pt_hist_offshell = np.histogram(ele_pt_offshell, 100, (0,1000),  weights=wgts_offshell)
ele_eta_hist_offshell = np.histogram(ele_eta_offshell, 100, (-7,7),  weights=wgts_offshell)
muon_pt_hist_offshell = np.histogram(muon_pt_offshell, 100, (0,1000),  weights=wgts_offshell)
muon_eta_hist_offshell = np.histogram(muon_eta_offshell, 100, (-7,7),  weights=wgts_offshell)

OUTFILE["mww_offshell"] = mww_hist_offshell
OUTFILE["mll_offshell"] = mll_hist_offshell
OUTFILE["ptll_offshell"] = ptll_hist_offshell
OUTFILE["MET_offshell"] = MET_hist_offshell
OUTFILE["mw_offshell"] = m_w_hist_offshell
OUTFILE["ele_pt_offshell"] = ele_pt_hist_offshell
OUTFILE["ele_eta_offshell"] = ele_eta_hist_offshell
OUTFILE["muon_pt_offshell"] = muon_pt_hist_offshell
OUTFILE["muon_eta_offshell"] = muon_eta_hist_offshell

mww_hist_onshell = np.histogram(m_ww_onshell, 100, (0,1000),  weights=wgts_onshell)
mll_hist_onshell = np.histogram(mll_onshell, 100, (0,1000),  weights=wgts_onshell)
ptll_hist_onshell = np.histogram(ptll_onshell, 100, (0,1000),  weights=wgts_onshell)
MET_hist_onshell = np.histogram(MET_onshell, 100, (0,1000),  weights=wgts_onshell)
m_w_hist_onshell = np.histogram(m_w_onshell, 100, (0,1000),  weights=m_w_wgts_onshell)
ele_pt_hist_onshell = np.histogram(ele_pt_onshell, 100, (0,1000),  weights=wgts_onshell)
ele_eta_hist_onshell = np.histogram(ele_eta_onshell, 100, (-7,7),  weights=wgts_onshell)
muon_pt_hist_onshell = np.histogram(muon_pt_onshell, 100, (0,1000),  weights=wgts_onshell)
muon_eta_hist_onshell = np.histogram(muon_eta_onshell, 100, (-7,7),  weights=wgts_onshell)

OUTFILE["mww_onshell"] = mww_hist_onshell
OUTFILE["mll_onshell"] = mll_hist_onshell
OUTFILE["ptll_onshell"] = ptll_hist_onshell
OUTFILE["MET_onshell"] = MET_hist_onshell
OUTFILE["mw_onshell"] = m_w_hist_onshell
OUTFILE["ele_pt_onshell"] = ele_pt_hist_onshell
OUTFILE["ele_eta_onshell"] = ele_eta_hist_onshell
OUTFILE["muon_pt_onshell"] = muon_pt_hist_onshell
OUTFILE["muon_eta_onshell"] = muon_eta_hist_onshell



