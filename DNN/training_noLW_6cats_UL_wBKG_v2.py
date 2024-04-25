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

from matplotlib.backends.backend_pdf import PdfPages
import random


TAG = "six_cats_newMaxEvents_withPlots_UL_Dec07_NO_QGL_GOOD_DISTS_wBKG_v2"

#INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v3_six_cats_improvedGGH.root"
#INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v3_six_cats_improvedGGH_improvedW.root"
#INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v3_six_cats_UL.root"
#INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v7_six_cats_UL.root"
#INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v7_six_cats_UL_wBKG.root"
INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v8_six_cats_UL_wBKG.root"
# FILE VERSIONING:
# v1 -> {"mll","dphill","detall","ptll","drll","pt1","pt2","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"}
# v2 -> {"mll","dphill","detall","ptll","drll","Lepton_pt0","Lepton_pt1","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"}
# v4 -> {"mll","dphill","detall","ptll","drll","Lepton_pt0","Lepton_pt1","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW", "qgl0", "qgl1"}
# v5 -> {"mll","dphill","detall","ptll","drll","Lepton_pt0","Lepton_pt1","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW", "qgl0", "qgl1","btag0","btag1"}
# v7 -> BETTER INPUT DISTS

CATEGORIES = ["VBF_OFF", "VBF_ON", "ggH_OFF", "ggH_ON", "top", "WW"] #Titles for categories
ENCODING_float = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,1]]
ENCODING = {"VBF_2018v7_OFF" : [1,0,0,0,0,0],"VBF_2018v7_ON" : [0,1,0,0,0,0], "ggH_2018v7_OFF" : [0,0,1,0,0,0],"ggH_2018v7_ON" : [0,0,0,1,0,0], "top_2018v7": [0,0,0,0,1,0], "WW_2018v7":[0,0,0,0,0,1]}

CAT_CONFIG_IDS = ["VBF_2018v7_OFF","VBF_2018v7_ON", "ggH_2018v7_OFF","ggH_2018v7_ON", "top_2018v7", "WW_2018v7"]

INPUT_VARS = ["mll","Lepton_phi0","Lepton_phi1","Lepton_eta0","Lepton_eta1","ptll","drll","Lepton_pt0","mth","mjj","jet_eta0","jet_eta1","dphijj","PuppiMET_pt","dphillmet","mcollWW","btag0", "btag1"] #Lepton_pt1 

INPUT_VAR_INDEX = {}
for i, var in enumerate(INPUT_VARS):
    INPUT_VAR_INDEX[var] = i

def convert_onehot(Y_to_convert):
    Y_cat = []
    for i in range(0, len(Y_to_convert)):
        for j in range(0, len(ENCODING["VBF_2018v7_OFF"])):
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
    ntuple_file = uproot3.open(INFILE_NAME)
    Y_float = ntuple_file["hww_ntuple"]["Y"].array()
    Y_onehot = [ENCODING_float[int(y)] for y in Y_float]
    W = ntuple_file["hww_ntuple"]["wgt"].array() #np.array([wgt for i in ntuple_file["hww_ntuple"]["wgt"].array()])
    input_vars = [ntuple_file["hww_ntuple/" + var].array() for var in INPUT_VARS]

    #input_vars[14] = input_vars[14]
    #input_vars[15] = input_vars[15]
    #input_vars[16] = input_vars[16]
    #input_vars[17] = input_vars[17]

    input_vars_mean = [np.mean(input_vars[i]) for i in range(0, len(input_vars))]
    input_vars_std = [np.std(input_vars[i]) for i in range(0, len(input_vars))]

    print(input_vars_mean)
    print(input_vars_std)

    input_vars = [(input_vars[i]-input_vars_mean[i])/input_vars_std[i] for i in range(0, len(input_vars))]


    X = np.transpose(input_vars)

    X_VBF_OFF = X[Y_float == 0]
    X_VBF_ON = X[Y_float == 1]
    X_ggH_OFF = X[Y_float == 2]
    X_ggH_ON = X[Y_float == 3]
    X_top = X[Y_float == 4]
    X_WW = X[Y_float == 5]
    X_VBF_BKG = X[Y_float == 6]

    Y_VBF_OFF = [ENCODING_float[int(y)] for y in Y_float[Y_float==0]] #Y_onehot[Y_float == 0]
    Y_VBF_ON = [ENCODING_float[int(y)] for y in Y_float[Y_float==1]]#Y_onehot[Y_float == 1]
    Y_ggH_OFF = [ENCODING_float[int(y)] for y in Y_float[Y_float==2]]#Y_onehot[Y_float == 2]
    Y_ggH_ON = [ENCODING_float[int(y)] for y in Y_float[Y_float==3]]#Y_onehot[Y_float == 3]
    Y_top = [ENCODING_float[int(y)] for y in Y_float[Y_float==4]]#Y_onehot[Y_float == 4]
    Y_WW = [ENCODING_float[int(y)] for y in Y_float[Y_float==5]]#Y_onehot[Y_float == 5]
    Y_VBF_BKG = [ENCODING_float[5] for y in Y_float[Y_float==6]]#Y_onehot[Y_float == 5]

    tot = float(np.sum(W))
    W_VBF_OFF = W[Y_float == 0]
    W_VBF_ON = W[Y_float == 1]
    W_ggH_OFF = W[Y_float == 2]
    W_ggH_ON = W[Y_float == 3]
    W_top = W[Y_float == 4]
    W_WW = W[Y_float == 5]
    W_VBF_BKG = W[Y_float == 6]

    r_test = random.Random(555)
    r_test.shuffle(X_VBF_BKG)
    r_test = random.Random(555)
    r_test.shuffle(Y_VBF_BKG)
    r_test = random.Random(555)
    r_test.shuffle(W_VBF_BKG)

    n_from_VBF = (len(X_VBF_BKG))
    X_WW = np.concatenate((X_WW, X_VBF_BKG[:n_from_VBF]), axis=0)
    Y_WW = np.concatenate((Y_WW, Y_VBF_BKG[:n_from_VBF]), axis=0)
    W_WW = np.concatenate((W_WW, W_VBF_BKG[:n_from_VBF]), axis=0)

    r_test = random.Random(1555)
    r_test.shuffle(X_WW)
    r_test = random.Random(1555)
    r_test.shuffle(W_WW)
    r_test = random.Random(1555)
    r_test.shuffle(Y_WW)

    SF_VBF_OFF = tot/np.sum(W_VBF_OFF)
    SF_VBF_ON = tot/np.sum(W_VBF_ON)
    SF_ggH_OFF = tot/np.sum(W_ggH_OFF)
    SF_ggH_ON = tot/np.sum(W_ggH_ON)
    SF_top = tot/np.sum(W_top)
    SF_WW = tot/np.sum(W_WW)

    MAX_SF = np.amin([SF_VBF_OFF, SF_VBF_ON, SF_ggH_OFF, SF_ggH_ON, SF_top, SF_WW])
    SF_VBF_OFF = float(SF_VBF_OFF / MAX_SF)
    SF_VBF_ON = float(SF_VBF_ON / MAX_SF)
    SF_ggH_OFF = float(SF_ggH_OFF / MAX_SF)
    SF_ggH_ON = float(SF_ggH_ON / MAX_SF)
    SF_top = float (SF_top / MAX_SF)
    SF_WW =float(SF_WW / MAX_SF)

    W_VBF_OFF = W_VBF_OFF * SF_VBF_OFF * 35
    W_VBF_ON = W_VBF_ON * SF_VBF_ON * 40
    W_ggH_OFF = W_ggH_OFF * SF_ggH_OFF * 50
    W_ggH_ON = W_ggH_ON * SF_ggH_ON * 40
    W_top = W_top * SF_top * 40
    W_WW = W_WW * SF_WW * 95

    print("VBF_OFF:"  + str(len(W_VBF_OFF)) + ", wgt'd:"+ str(np.sum(W_VBF_OFF)) + ", SF:" + str(tot/(len(W_VBF_OFF))))
    print("VBF_ON:"  + str(len(W_VBF_ON)) + ", wgt'd:"+ str(np.sum(W_VBF_ON)) + ", SF:"  + str(tot/(len(W_VBF_ON))))
    print("ggH_OFF:"  + str(len(W_ggH_OFF)) + ", wgt'd:"+ str(np.sum(W_ggH_OFF)) + ", SF:"  + str(tot/(len(W_ggH_OFF))))
    print("ggH_ON:"  + str(len(W_ggH_ON)) + ", wgt'd:"+ str(np.sum(W_ggH_ON)) + ", SF:"  + str(tot/(len(W_ggH_ON))))
    print("top:"  + str(len(W_top)) + ", wgt'd:"+ str(np.sum(W_top)) + ", SF:"  + str(tot/(len(W_top))))
    print("WW:"  + str(len(W_WW))  + ", wgt'd:"+ str(np.sum(W_WW)) + ", SF:"  + str(tot/(len(W_WW))))

    max_events = 650000
    X_NEW = np.concatenate((X_VBF_OFF[:min(max_events+250000, len(X_VBF_OFF))], X_VBF_ON[:min(max_events, len(X_VBF_ON))], X_ggH_OFF[:min(max_events+250000, len(X_ggH_OFF))], X_ggH_ON[:min(max_events, len(X_ggH_ON))], X_top[:min(max_events, len(X_top))],X_WW[:min(max_events, len(X_WW))]))
    Y_NEW = np.concatenate((Y_VBF_OFF[:min(max_events+250000, len(X_VBF_OFF))], Y_VBF_ON[:min(max_events, len(X_VBF_ON))], Y_ggH_OFF[:min(max_events+250000, len(X_ggH_OFF))], Y_ggH_ON[:min(max_events, len(X_ggH_ON))], Y_top[:min(max_events, len(X_top))],Y_WW[:min(max_events, len(X_WW))]))
    W_NEW = np.concatenate((W_VBF_OFF[:min(max_events+250000, len(X_VBF_OFF))], W_VBF_ON[:min(max_events, len(X_VBF_ON))], W_ggH_OFF[:min(max_events+250000, len(X_ggH_OFF))], W_ggH_ON[:min(max_events, len(X_ggH_ON))], W_top[:min(max_events, len(X_top))],W_WW[:min(max_events, len(X_WW))]))

    #return X, Y_onehot, W
    return X_NEW, Y_NEW, W_NEW,input_vars_mean,input_vars_std

def baseline_model():
    model = Sequential() # 150, 100, 80, 40, 5
    model.add(Dense(60, input_dim=len(INPUT_VARS), activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='softmax'))
#    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X_inputs, Y_inputs, W_inputs, input_means, input_std = loadVariables()

X_train, Y_train, W_train, X_test, Y_test, W_test = randomize_test_train(X_inputs, Y_inputs, W_inputs)


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=4096, validation_split=0.35, verbose=1, shuffle=True) #500
history = estimator.fit(np.array(X_train),np.array(Y_train), sample_weight=np.array(W_train), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30,verbose=1)])

pdf_pages = PdfPages("./dnn_v2_history_"+TAG+".pdf")
fig, ax = plt.subplots(1)
fig.suptitle("Model Accuracy")
ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylim([0,1.1])
ax.legend(['train', 'test'], loc='upper left')
fig.set_size_inches(6,6)

fig2, ax2 = plt.subplots(1)
fig2.suptitle("Model Loss")
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['train', 'test'], loc='upper left')
fig2.set_size_inches(6,6)

pdf_pages.savefig(fig)
pdf_pages.savefig(fig2)

predictions = estimator.predict(np.array(X_test))

softmax_outputs = estimator.model.predict(np.array(X_test))
softmax_outputs_trained = estimator.model.predict(np.array(X_train))


#print(predictions, predictions[0])
Y_pred = predictions #np.argmax(predictions)
Y_true_cat = np.array(convert_onehot(Y_test))
Y_true_cat_trained = np.array(convert_onehot(Y_train))

for i in range(0, 10):
    print(predictions[i], Y_pred[i], softmax_outputs[i], X_test[i])

cm = sklearn.metrics.confusion_matrix(Y_true_cat, Y_pred)
for i in range(0,len(cm)):
    row_sum = float(np.sum(cm[i]))
    for j in range(0, len(cm[i])):
        cm[i][j] = float(cm[i][j]) / row_sum * 100.0
print(cm)

fig3, ax3 = plt.subplots(1)
ylabels = CATEGORIES
xlabels = CATEGORIES

ax3 = sns.heatmap(cm, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), xticklabels = xlabels, yticklabels=ylabels, linewidth=.25, annot=True)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
ax3.set_title("Confusion Matrix")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("True")
plt.tight_layout()
fig3.set_size_inches(6,6)
pdf_pages.savefig(fig3)


#print(estimator.__dict__)

estimator.model.save('hww_offshell_dnn_' + TAG + '.keras')
estimator.model.save_weights('hww_offshell_weights_'+TAG+'.h5')

import json
weights_list = np.array(estimator.model.get_weights())
output_data = {}

for layer in range(0, int(len(weights_list)/2)):
    n_inputs = len(weights_list[layer*2])
    n_nodes = len(weights_list[layer*2][0])
    layer_data = {}
    transposed_weights = np.transpose(weights_list[layer*2])

    print(str(layer) + " : " + str(n_inputs) + " : " + str(n_nodes))

    for i in range(0, n_nodes):
        node_data = {"weights" : transposed_weights[n_nodes-1-i].tolist(), "bias" : weights_list[layer*2+1].tolist()[n_nodes-1-i]}
        layer_data["Node_" + str(i)] = node_data
    output_data["Layer_" + str(layer)] = layer_data

print("Variable Renorm Factors:")
print(input_means)
print(input_std)
output_data["Var_Means"] = input_means
output_data["Var_Std"] = input_std

with open("hww_offshell_weights_" + TAG + ".json", "w") as json_file:
    json_data = json.dumps(output_data, indent=4)
    json_file.write(json_data)


VBF_OFF_score = [[],[],[],[],[],[]]
VBF_ON_score = [[],[],[],[],[],[]]
ggH_OFF_score = [[],[],[],[],[],[]]
ggH_ON_score = [[],[],[],[],[],[]]
top_score = [[],[],[],[],[],[]]
WW_score = [[],[],[],[],[],[]] 

VBF_OFF_score_trained = [[],[],[],[],[],[]]
VBF_ON_score_trained = [[],[],[],[],[],[]]
ggH_OFF_score_trained = [[],[],[],[],[],[]]
ggH_ON_score_trained = [[],[],[],[],[],[]]
top_score_trained = [[],[],[],[],[],[]]
WW_score_trained = [[],[],[],[],[],[]]

for i in range(0, len(predictions)):
    VBF_OFF_score[Y_true_cat[i]].append(softmax_outputs[i][0])
    VBF_ON_score[Y_true_cat[i]].append(softmax_outputs[i][1])
    ggH_OFF_score[Y_true_cat[i]].append(softmax_outputs[i][2])
    ggH_ON_score[Y_true_cat[i]].append(softmax_outputs[i][3])
    top_score[Y_true_cat[i]].append(softmax_outputs[i][4])
    WW_score[Y_true_cat[i]].append(softmax_outputs[i][5])

for i in range(0, len(softmax_outputs_trained)):
    VBF_OFF_score_trained[Y_true_cat_trained[i]].append(softmax_outputs_trained[i][0])
    VBF_ON_score_trained[Y_true_cat_trained[i]].append(softmax_outputs_trained[i][1])
    ggH_OFF_score_trained[Y_true_cat_trained[i]].append(softmax_outputs_trained[i][2])
    ggH_ON_score_trained[Y_true_cat_trained[i]].append(softmax_outputs_trained[i][3])
    top_score_trained[Y_true_cat_trained[i]].append(softmax_outputs_trained[i][4])
    WW_score_trained[Y_true_cat_trained[i]].append(softmax_outputs_trained[i][5])


bins = np.histogram(VBF_OFF_score[0], 20, (0,1))[1]

VBF_OFF_score_binned = [np.histogram(VBF_OFF_score[i], 20, (0,1), density=True)[0] for i in range(0, len(VBF_OFF_score))]
VBF_ON_score_binned = [np.histogram(VBF_ON_score[i], 20, (0,1), density=True)[0] for i in range(0, len(VBF_ON_score))]
ggH_OFF_score_binned = [np.histogram(ggH_OFF_score[i], 20, (0,1), density=True)[0] for i in range(0, len(ggH_OFF_score))]
ggH_ON_score_binned = [np.histogram(ggH_ON_score[i], 20, (0,1), density=True)[0] for i in range(0, len(ggH_ON_score))]
top_score_binned = [np.histogram(top_score[i], 20, (0,1), density=True)[0] for i in range(0, len(top_score))]
WW_score_binned = [np.histogram(WW_score[i], 20, (0,1), density=True)[0] for i in range(0, len(WW_score))]

VBF_OFF_score_trained_binned = [np.histogram(VBF_OFF_score_trained[i], 20, (0,1), density=True)[0] for i in range(0, len(VBF_OFF_score_trained))]
VBF_ON_score_trained_binned = [np.histogram(VBF_ON_score_trained[i], 20, (0,1), density=True)[0] for i in range(0, len(VBF_ON_score_trained))]
ggH_OFF_score_trained_binned = [np.histogram(ggH_OFF_score_trained[i], 20, (0,1), density=True)[0] for i in range(0, len(ggH_OFF_score_trained))]
ggH_ON_score_trained_binned = [np.histogram(ggH_ON_score_trained[i], 20, (0,1), density=True)[0] for i in range(0, len(ggH_ON_score_trained))]
top_score_trained_binned = [np.histogram(top_score_trained[i], 20, (0,1), density=True)[0] for i in range(0, len(top_score_trained))]
WW_score_trained_binned = [np.histogram(WW_score_trained[i], 20, (0,1), density=True)[0] for i in range(0, len(WW_score_trained))]

VBF_OFF_score_KS = [ks_2samp(VBF_OFF_score_binned[i], VBF_OFF_score_trained_binned[i]) for i in range(0, len(VBF_OFF_score))]
VBF_ON_score_KS = [ks_2samp(VBF_ON_score_binned[i], VBF_ON_score_trained_binned[i]) for i in range(0, len(VBF_ON_score))]
ggH_OFF_score_KS = [ks_2samp(ggH_OFF_score_binned[i], ggH_OFF_score_trained_binned[i]) for i in range(0, len(ggH_OFF_score))]
ggH_ON_score_KS = [ks_2samp(ggH_ON_score_binned[i], ggH_ON_score_trained_binned[i]) for i in range(0, len(ggH_ON_score))]
top_score_KS = [ks_2samp(top_score_binned[i], top_score_trained_binned[i]) for i in range(0, len(top_score))]
WW_score_KS = [ks_2samp(WW_score_binned[i], WW_score_trained_binned[i]) for i in range(0, len(WW_score))]

def plot_score(score_binned, score_trained_binned, KS_stats, title, label):
    fig_p, ax_p = plt.subplots(1)
    ax_p.set_title(title)
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[0], label="VBF_OFF("+str(round(KS_stats[0][0],4))+"," + str(round(KS_stats[0][1],4))+")", color="tab:blue")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[1], label="VBF_ON("+str(round(KS_stats[1][0],4))+"," + str(round(KS_stats[1][1],4))+")", color="tab:green")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[2], label="ggH_OFF("+str(round(KS_stats[2][0],4))+"," + str(round(KS_stats[2][1],4))+")", color="tab:orange")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[3], label="ggH_ON("+str(round(KS_stats[3][0],4))+"," + str(round(KS_stats[3][1],4))+")", color="tab:red")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[4], label="top("+str(round(KS_stats[4][0],4))+"," + str(round(KS_stats[4][1],4))+")", color="tab:purple")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_binned[5], label="WW("+str(round(KS_stats[5][0],4))+"," + str(round(KS_stats[5][1],4))+")", color="tab:cyan")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_trained_binned[0], fmt="-.", color="tab:blue")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_trained_binned[1], fmt="-.", color="tab:green")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_trained_binned[2], fmt="-.", color="tab:orange")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_trained_binned[3], fmt="-.", color="tab:red")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_trained_binned[4], fmt="-.", color="tab:purple")
    ax_p.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], score_trained_binned[5], fmt="-.", color="tab:cyan")
    ax_p.errorbar([-2,-1], [0,1], fmt="-.", color="k", label="Train")
    ax_p.errorbar([-2,-1], [0,1], color="k", label="Test")
    ax_p.set_xlim([0,1])
    ax_p.set_xlabel(label)
    ax_p.set_ylabel("A.U.")
    ax_p.legend()
    ax_p.set_yscale('log')
    plt.tight_layout()
    fig_p.set_size_inches(6,6)
    return fig_p

fig_4 = plot_score(VBF_OFF_score_binned, VBF_OFF_score_trained_binned, VBF_OFF_score_KS,  "DNN VBF Offshell Score", "VBF_OFF score")
pdf_pages.savefig(fig_4)

fig_5 = plot_score(VBF_ON_score_binned, VBF_ON_score_trained_binned,VBF_ON_score_KS, "DNN VBF Onshell Score", "VBF_ON score")
pdf_pages.savefig(fig_5)

fig_6 = plot_score(ggH_OFF_score_binned, ggH_OFF_score_trained_binned,ggH_OFF_score_KS, "DNN ggH Offshell Score", "ggH_OFF score")
pdf_pages.savefig(fig_6)

fig_7 = plot_score(ggH_ON_score_binned, ggH_ON_score_trained_binned,ggH_ON_score_KS, "DNN ggH Onshell Score", "ggH_ON score")
pdf_pages.savefig(fig_7)

fig_8 = plot_score(top_score_binned, top_score_trained_binned,top_score_KS, "DNN Top Score", "top score")
pdf_pages.savefig(fig_8)

fig_9 = plot_score(WW_score_binned, WW_score_trained_binned,WW_score_KS, "DNN WW Score", "WW score")
pdf_pages.savefig(fig_9)

pdf_pages.close()
