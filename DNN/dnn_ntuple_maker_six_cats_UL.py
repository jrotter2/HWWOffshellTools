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

import random

MAX_EVENTS_PER_CATEGORY = 1000000

OUTFILE = uproot3.recreate("/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v6_six_cats_UL.root")
# FILE VERSIONING:
# v1 -> {"mll","dphill","detall","ptll","drll","pt1","pt2","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"}


CATEGORIES = ["VBF", "ggH", "top", "WW"] #Titles for categories
ENCODING = {"UL_VBF_OFF" : [1,0,0,0,0,0], "UL_VBF_ON" : [0,1,0,0,0,0], "UL_ggH_OFF" : [0,0,1,0,0,0], "UL_ggH_ON" : [0,0,0,1,0,0], "UL_top": [0,0,0,0,1,0], "UL_WW":[0,0,0,0,0,1]}

CAT_CONFIG_IDS = ["UL_VBF_OFF", "UL_VBF_ON", "UL_ggH_OFF", "UL_ggH_ON", "UL_top", "UL_WW"]


INPUT_VARS = ["mll","dphill","detall","ptll","drll","Lepton_pt0","Lepton_pt1","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW", "qgl0", "qgl1","btag0", "btag1"]

INPUT_VAR_INDEX = {}
for i, var in enumerate(INPUT_VARS):
    INPUT_VAR_INDEX[var] = i

CONFIG_FILE_PATH = "test_dnn_config_v3.json"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)

def convert_onehot(Y_to_convert):
    Y_cat = []
    for i in range(0, len(Y_to_convert)):
        for j in range(0, len(ENCODING["UL_VBF_OFF"])):
            if(Y_to_convert[i][j] == 1):
                Y_cat.append(j)
    return Y_cat

def randomize_test_train(X_to_randomize, Y_to_randomize, W_to_randomize):
    X_train_rand = []
    Y_train_rand = []
    W_train_rand = []
    X_test_rand = []
    Y_test_rand = []
    W_test_rand = []

    for i in range(0,len(X_to_randomize)):
        prob = random.random()
        if(prob > .5):
            X_train_rand.append(X_to_randomize[i])
            Y_train_rand.append(Y_to_randomize[i])
            W_train_rand.append(W_to_randomize[i])
        else:
            X_test_rand.append(X_to_randomize[i])
            Y_test_rand.append(Y_to_randomize[i])
            W_test_rand.append(W_to_randomize[i])


    test_seed = 137
    train_seed = 404
    r_test = random.Random(test_seed)
    r_test.shuffle(X_test_rand)
    r_test = random.Random(test_seed)
    r_test.shuffle(Y_test_rand)
    r_test = random.Random(test_seed)
    r_test.shuffle(W_test_rand)

    r_train = random.Random(train_seed)
    r_train.shuffle(X_train_rand)
    r_train = random.Random(train_seed)
    r_train.shuffle(Y_train_rand)
    r_train = random.Random(train_seed)
    r_train.shuffle(W_train_rand)

    return X_train_rand, Y_train_rand, W_train_rand, X_test_rand, Y_test_rand, W_test_rand

def loadVariables():
    print("Started Loading Variables...")
    X = {}
    W = {}
    for category in CAT_CONFIG_IDS:
        print("Category: " + category + " with " + str(len(CONFIG[category]["subpaths"])) + " Subcategories...")
        X[category] = []
        W[category] = []
        basepath = CONFIG[category]["basepath"]

        max_events_per_subcategory = MAX_EVENTS_PER_CATEGORY / len(CONFIG[category]["subpaths"])
        for subcat_i in range(0,len(CONFIG[category]["subpaths"])):
            weights  = CONFIG[category]["subpaths"][subcat_i]["weights"]
            custom_weights  = CONFIG[category]["subpaths"][subcat_i]["custom_weights"]
            files = CONFIG[category]["subpaths"][subcat_i]["files"]
            current_nEvents_per_subcategory = 0
            print(">Subcategory: " + CONFIG[category]["subpaths"][subcat_i]["name"] + " | Allowed # Events: " + str(max_events_per_subcategory)) 

            for f in files:
                print("    Loading Next File... Currently at Event " + str(current_nEvents_per_subcategory) + " of " + str(max_events_per_subcategory))
                if(max_events_per_subcategory - current_nEvents_per_subcategory <= 0):
                    break
                rootFile = uproot3.open(basepath + f)

                nEvents = rootFile["Events"]._fEntries
                nEvents_for_file = min(nEvents, max_events_per_subcategory - current_nEvents_per_subcategory)           

                full_wgts = np.ones(nEvents_for_file)
                
                for wgt in weights:
                   # if(wgt == "XSWeight"): ## REMOVING SO ALL CATEGORIES ARE EQUALLY TRAINED
                   #     continue
                    full_wgts = np.multiply(rootFile["Events/" + wgt].array()[:nEvents_for_file], full_wgts)

                input_vars = []
                for i, var_name in enumerate(INPUT_VARS):
                    if("btag" in var_name):
                        var = var_name[:-1]
                        var_index =int(var_name[-1:])
                        CleanJet_jetIdx = np.zeros(nEvents_for_file)
                        try:
                            CleanJet_jetIdx = np.array(rootFile["Events/" + "CleanJet_jetIdx"].array()[:nEvents_for_file, var_index])
                        except:
                            CleanJet_jetIdx = np.array([evt[var_index] if len(evt)>var_index else 0 for evt in rootFile["Events/" + "CleanJet_jetIdx"].array()[:nEvents_for_file]])

                        try:
                            Jet_btag = rootFile["Events/Jet_btagDeepFlavB"].array()[:nEvents_for_file]
                            input_vars.append(np.array(Jet_btag[np.indices(CleanJet_jetIdx.shape)[0], CleanJet_jetIdx]))
                        except:
                            print("CleanJet_jetIdx was out-of-bounds for Jet_qgl Collection...")
                            Jet_btag = np.array([evt[CleanJet_jetIdx[evt_index]] if len(evt)>CleanJet_jetIdx[evt_index] else -3 for evt_index,evt in enumerate(rootFile["Events/Jet_btagDeepFlavB"].array()[:nEvents_for_file])])
                            input_vars.append(Jet_btag)
                    elif("qgl" in var_name):
                        var = var_name[:-1]
                        var_index =int(var_name[-1:])
                        CleanJet_jetIdx = np.zeros(nEvents_for_file)
                        try:
                            CleanJet_jetIdx = np.array(rootFile["Events/" + "CleanJet_jetIdx"].array()[:nEvents_for_file, var_index])
                        except:
                            CleanJet_jetIdx = np.array([evt[var_index] if len(evt)>var_index else 0 for evt in rootFile["Events/" + "CleanJet_jetIdx"].array()[:nEvents_for_file]])

                        try:
                            Jet_qgl = rootFile["Events/Jet_qgl"].array()[:nEvents_for_file]
                            input_vars.append(np.array(Jet_qgl[np.indices(CleanJet_jetIdx.shape)[0], CleanJet_jetIdx]))
                        except:
                            print("CleanJet_jetIdx was out-of-bounds for Jet_qgl Collection...")
                            Jet_qgl = np.array([evt[CleanJet_jetIdx[evt_index]] if len(evt)>CleanJet_jetIdx[evt_index] else -3 for evt_index,evt in enumerate(rootFile["Events/Jet_qgl"].array()[:nEvents_for_file])])
                            input_vars.append(Jet_qgl)
                    elif(any(chr.isdigit() for chr in var_name)):
                        var = var_name[:-1]
                        var_index =int(var_name[-1:])
                        input_vars.append(rootFile["Events/" + var].array()[:nEvents_for_file, var_index])
                    else:
                        input_vars.append(rootFile["Events/" + var_name].array()[:nEvents_for_file])

                mll_mask = np.array(input_vars[INPUT_VAR_INDEX["mll"]]) > 12
                ptll_mask = np.array(input_vars[INPUT_VAR_INDEX["ptll"]]) > 30

                ptl1_mask = np.array(input_vars[INPUT_VAR_INDEX["Lepton_pt0"]]) > 25
                ptl2_mask = np.array(input_vars[INPUT_VAR_INDEX["Lepton_pt1"]]) > 20
                ptmiss_mask = np.array(input_vars[INPUT_VAR_INDEX["PuppiMET_pt"]]) > 20

                njet_mask = np.array(rootFile["Events/" + "nCleanJet"].array()[:nEvents_for_file]) >= 2

                CleanJet_pt =  np.array(rootFile["Events/" + "CleanJet_pt"].array()[:nEvents_for_file])

                shell_mask = []
                if("OFFSHELL_HIGGS" in custom_weights):
                    shell_mask = np.array(rootFile["Events/" + "LHECandMass"].array()[:nEvents_for_file]) >= 160
                elif("ONSHELL_HIGGS" in custom_weights):
                    shell_mask = np.array(rootFile["Events/" + "LHECandMass"].array()[:nEvents_for_file]) < 160
                else:
                    shell_mask = np.array([True for i in range(0, nEvents_for_file)])



                evt_mask = mll_mask & ptll_mask & ptl1_mask & ptl2_mask & ptmiss_mask & njet_mask & shell_mask

                x_evt = np.transpose(input_vars)
                nEvents_added = 0
                for i, arr in enumerate(x_evt):
                    if(evt_mask[i]):
                        if(CleanJet_pt[i][0] > 30 and CleanJet_pt[i][1] > 30):
                            X[category].append(arr)
                            nEvents_added += 1
                for i, wgt in enumerate(full_wgts):
                    if(evt_mask[i]):
                        if(CleanJet_pt[i][0] > 30 and CleanJet_pt[i][1] > 30):
                            W[category].append(wgt)
                current_nEvents_per_subcategory = current_nEvents_per_subcategory + nEvents_added # nEvents_for_file
    return X, W

X_mixed, W_mixed = loadVariables()

X_inputs = []
W_inputs = []
Y_inputs = []
Y_inputs_float = []

for i, category in enumerate(CAT_CONFIG_IDS):
    for arr in X_mixed[category]:
        X_inputs.append(arr)
        Y_inputs_float.append(i) #ENCODING[category])
        Y_inputs.append(ENCODING[category])
    for wgt in W_mixed[category]:
        W_inputs.append(wgt)

input_var_arrays = [[] for j in range(0, len(INPUT_VARS))]
branch_dict = {}
branch_data = {}

for i in range(0, len(X_inputs)):
    for j in range(0, len(INPUT_VARS)):
        input_var_arrays[j].append(X_inputs[i][j])

for i in range(0, len(INPUT_VARS)):
    branch_dict[INPUT_VARS[i]] = "float64"
    branch_data[INPUT_VARS[i]] = np.array(input_var_arrays[i])

branch_dict["Y"] = "float64"
branch_dict["wgt"] = "float64"
branch_data["Y"] = np.array(Y_inputs_float)
branch_data["wgt"] = np.array(W_inputs)

OUTFILE["hww_ntuple"] = uproot3.newtree(branch_dict)
OUTFILE["hww_ntuple"].extend(branch_data)
