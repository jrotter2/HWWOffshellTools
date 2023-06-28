import uproot
import pandas
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

MAX_EVENTS_PER_CATEGORY = 500000

CATEGORIES = ["VBF", "ggH", "top", "WW"] #Titles for categories
ENCODING = {"VBF_2018v7" : [1,0,0,0], "ggH_2018v7" : [0,1,0,0], "top_2018v7": [0,0,1,0], "WW_2018v7":[0,0,0,1]}

CAT_CONFIG_IDS = ["VBF_2018v7", "ggH_2018v7", "top_2018v7", "WW_2018v7"]

INPUT_VARS = ["mll","dphill","detall","ptll","drll","pt1","pt2","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"]

CONFIG_FILE_PATH = "dnn_config.json"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)

def convert_onehot(Y_to_convert):
    Y_cat = []
    for i in range(0, len(Y_to_convert)):
        for j in range(0, len(ENCODING["VBF_2018v7"])):
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
        print("Category: " + category)
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
                rootFile = uproot.open(basepath + f)

                nEvents = rootFile["Events"]._fEntries
                nEvents_for_file = min(nEvents, max_events_per_subcategory - current_nEvents_per_subcategory)           

                full_wgts = np.ones(nEvents_for_file)
#                for wgt in weights:
#                    if(wgt == "XSWeight"): ## REMOVING SO ALL CATEGORIES ARE EQUALLY TRAINED
#                        continue
#                    full_wgts = np.multiply(rootFile["Events/" + wgt].array()[:nEvents_for_file], full_wgts)
                ## ADD CUSTOM WEIGHTS               
                input_vars = [rootFile["Events/" + var].array()[:nEvents_for_file] for var in INPUT_VARS]
                x_evt = np.transpose(input_vars)
                for arr in x_evt:
                    X[category].append(arr)
                for wgt in full_wgts:
                    W[category].append(wgt)
                current_nEvents_per_subcategory = current_nEvents_per_subcategory + nEvents_for_file
    return X, W

def baseline_model():
    model = Sequential()
    model.add(Dense(20, input_dim=len(INPUT_VARS), activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X_mixed, W_mixed = loadVariables()

X_inputs = []
W_inputs = []
Y_inputs = []

for category in CAT_CONFIG_IDS:
    for arr in X_mixed[category]:
        X_inputs.append(arr)
        Y_inputs.append(ENCODING[category])
    for wgt in W_mixed[category]:
        W_inputs.append(wgt)

X_train, Y_train, W_train, X_test, Y_test, W_test = randomize_test_train(X_inputs, Y_inputs, W_inputs)

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1000, validation_split=0.2, verbose=1, shuffle=True)
history = estimator.fit(np.array(X_train),np.array(Y_train), sample_weight=np.array(W_train), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30,verbose=1)])

Y_pred = estimator.predict(np.array(X_test))
Y_true_cat = np.array(convert_onehot(Y_test))

cm = sklearn.metrics.confusion_matrix(Y_true_cat, Y_pred)
print(cm)


