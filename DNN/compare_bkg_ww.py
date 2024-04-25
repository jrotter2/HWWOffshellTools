from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')

import uproot3
import pandas as pd
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import tensorflow as tf
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model

from matplotlib.backends.backend_pdf import PdfPages
import random


#INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v8_six_cats_UL_wBKG.root"
INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_WW_BKG_COMP.root"
MODEL_FILE_NAME = "hww_offshell_dnn_six_cats_newMaxEvents_withPlots_UL_Dec07_NO_QGL_GOOD_DISTS_wBKG_v2.keras"

INPUT_VARS = ["mll","Lepton_phi0","Lepton_phi1","Lepton_eta0","Lepton_eta1","ptll","drll","Lepton_pt0","Lepton_pt1","mth","mjj","jet_eta0","jet_eta1","dphijj","PuppiMET_pt","dphillmet","mcollWW","btag0", "btag1"]

def loadVariables():
    print("Started Loading Variables...")
    ntuple_file = uproot3.open(INFILE_NAME)
    Y_float = ntuple_file["hww_ntuple"]["Y"].array()
    W = ntuple_file["hww_ntuple"]["wgt"].array() #np.array([wgt for i in ntuple_file["hww_ntuple"]["wgt"].array()])
    input_vars = [ntuple_file["hww_ntuple/" + var].array() for var in INPUT_VARS]

    input_vars_mean = [np.mean(input_vars[i]) for i in range(0, len(input_vars))]
    input_vars_std = [np.std(input_vars[i]) for i in range(0, len(input_vars))]

    print(input_vars_mean)
    print(input_vars_std)

    input_vars = [(input_vars[i]-input_vars_mean[i])/input_vars_std[i] for i in range(0, len(input_vars))]

    X = np.transpose(input_vars)

    X_WW = X[Y_float == 5]
    X_VBF_BKG = X[Y_float == 6]

    Y_WW = Y_float[Y_float==5] # [y for y in Y_float[Y_float==5]]#Y_onehot[Y_float == 5]
    Y_VBF_BKG = Y_float[Y_float==6] # [ENCODING_float[5] for y in Y_float[Y_float==6]]#Y_onehot[Y_float == 5]

    W_WW = W[Y_float == 5]
    W_VBF_BKG = W[Y_float == 6]


    X_NEW = np.concatenate((X_WW, X_VBF_BKG), axis=0)
    Y_NEW = np.concatenate((Y_WW, Y_VBF_BKG), axis=0)
    W_NEW = np.concatenate((W_WW, W_VBF_BKG), axis=0)

    return X_NEW, Y_NEW, W_NEW,input_vars_mean,input_vars_std

X, Y, W, means, stds = loadVariables()

model = load_model(MODEL_FILE_NAME)

softmax_outputs = model.predict(np.array(X))

VBF_OFF_score = [[],[]]
VBF_ON_score = [[],[]]
ggH_OFF_score = [[],[]]
ggH_ON_score = [[],[]]
top_score = [[],[]]
WW_score = [[],[]]

wgt_score = [[], []]

for i in range(0, len(softmax_outputs)):
        VBF_OFF_score[int(Y[i]-5)].append(softmax_outputs[i][0])
        VBF_ON_score[int(Y[i]-5)].append(softmax_outputs[i][1])
        ggH_OFF_score[int(Y[i]-5)].append(softmax_outputs[i][2])
        ggH_ON_score[int(Y[i]-5)].append(softmax_outputs[i][3])
        top_score[int(Y[i]-5)].append(softmax_outputs[i][4])
        WW_score[int(Y[i]-5)].append(softmax_outputs[i][5])
        wgt_score[int(Y[i]-5)].append(W[i])

bins = np.histogram(VBF_OFF_score[0], 20, (0,1))[1]

VBF_OFF_score_binned = [np.histogram(VBF_OFF_score[i], 20, (0,1), weights=wgt_score[i])[0] for i in range(0, len(VBF_OFF_score))]
VBF_ON_score_binned = [np.histogram(VBF_ON_score[i], 20, (0,1),  weights=wgt_score[i])[0] for i in range(0, len(VBF_ON_score))]
ggH_OFF_score_binned = [np.histogram(ggH_OFF_score[i], 20, (0,1),  weights=wgt_score[i])[0] for i in range(0, len(ggH_OFF_score))]
ggH_ON_score_binned = [np.histogram(ggH_ON_score[i], 20, (0,1),  weights=wgt_score[i])[0] for i in range(0, len(ggH_ON_score))]
top_score_binned = [np.histogram(top_score[i], 20, (0,1),  weights=wgt_score[i])[0] for i in range(0, len(top_score))]
WW_score_binned = [np.histogram(WW_score[i], 20, (0,1),  weights=wgt_score[i])[0] for i in range(0, len(WW_score))]


def plot_score(score_binned, title, label):
    fig_p, ax_p = plt.subplots(1)
    ax_p.set_title(title)
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[0], label="WW", color="tab:blue")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[1], label="BKG", color="tab:green")
    ax_p.errorbar([-2,-1], [0,1], color="k", label="Test")
    ax_p.set_xlim([0,1])
    ax_p.set_xlabel(label)
    ax_p.set_ylabel("A.U.")
    ax_p.legend()
    ax_p.set_yscale('log')
    plt.tight_layout()
    fig_p.set_size_inches(6,6)
    return fig_p



pdf_pages = PdfPages("./dnn_WW_BKG_comp.pdf")

fig_4 = plot_score(VBF_OFF_score_binned, "DNN VBF Offshell Score", "VBF_OFF score")
pdf_pages.savefig(fig_4)

fig_4 = plot_score(VBF_ON_score_binned, "DNN VBF Onshell Score", "VBF_ON score")
pdf_pages.savefig(fig_4)

fig_4 = plot_score(ggH_OFF_score_binned, "DNN ggH Offshell Score", "ggH_OFF score")
pdf_pages.savefig(fig_4)

fig_4 = plot_score(ggH_ON_score_binned, "DNN ggH Onshell Score", "ggH_ON score")
pdf_pages.savefig(fig_4)

fig_4 = plot_score(top_score_binned, "DNN Top Score", "Top score")
pdf_pages.savefig(fig_4)

fig_4 = plot_score(WW_score_binned, "DNN WW Score", "WW score")
pdf_pages.savefig(fig_4)

pdf_pages.close()






