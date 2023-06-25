import uproot
import pandas
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

MAX_EVENTS_PER_CATEGORY = 10000

CATEGORIES = ["VBF", "ggH", "top", "WW"] #Titles for categories
CAT_CONFIG_IDS = ["VBF", "ggH", "top", "WW"] #For now are the same as titles but may include versioning later
ENCODING = {"VBF_2018v7" : [1,0,0,0], "ggH_2018v7" : [0,1,0,0], "top_2018v7": [0,0,1,0], "WW_2018v7":[0,0,0,1]}

CAT_CONFIG_IDS = ["VBF_2018v7", "ggH_2018v7", "top_2018v7", "WW_2018v7"]


INPUT_VARS = ["mll","dphill","detall","ptll","drll","pt1","pt2","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"]

CONFIG_FILE_PATH = "dnn_config.json"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)

def loadVariables():
    print("Started Loading Variables...")
    X = {}
    for category in CAT_CONFIG_IDS:
        print("Category: " + category)
        X[category] = []
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
                #for wgt in weights:
                #    full_wgts = np.multiply(rootFile["Events/" + wgt].array()[:nEvents_for_file], full_wgt)

                ## ADD CUSTOM WEIGHTS               
                input_vars = [np.multiply(rootFile["Events/" + var].array()[:nEvents_for_file], full_wgts) for var in INPUT_VARS]
                x_evt = np.transpose(input_vars)
                for arr in x_evt:
                    X[category].append(arr)

                current_nEvents_per_subcategory = current_nEvents_per_subcategory + nEvents_for_file
    return X
def baseline_model():
    model = Sequential()
    model.add(Dense(20, input_dim=len(INPUT_VARS), activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X_mixed = loadVariables()

X = []
Y = []

for category in CAT_CONFIG_IDS:
    for arr in X_mixed[category]:
        X.append(arr)
        Y.append(ENCODING[category])

print("X:" + str(len(X)) + ", " + str(len(X[0])), "Y:" + str(len(Y)) + ", " + str(len(Y[0])))

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1000, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print(results)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

