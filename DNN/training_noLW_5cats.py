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


#INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v3_six_cats_improvedGGH.root"
INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v3_six_cats_improvedGGH_improvedW.root"
# FILE VERSIONING:
# v1 -> {"mll","dphill","detall","ptll","drll","pt1","pt2","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"}
# v2 -> {"mll","dphill","detall","ptll","drll","Lepton_pt0","Lepton_pt1","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"}


CATEGORIES = ["VBF_OFF", "VBF_ON", "ggH_OFF", "ggH_ON", "BKG"] #Titles for categories
ENCODING_float = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1], [0,0,0,0,1]]
ENCODING = {"VBF_2018v7_OFF" : [1,0,0,0,0],"VBF_2018v7_ON" : [0,1,0,0,0], "ggH_2018v7_OFF" : [0,0,1,0,0],"ggH_2018v7_ON" : [0,0,0,1,0], "top_2018v7": [0,0,0,0,1], "WW_2018v7":[0,0,0,0,1]}

CAT_CONFIG_IDS = ["VBF_2018v7_OFF","VBF_2018v7_ON", "ggH_2018v7_OFF","ggH_2018v7_ON", "top_2018v7", "WW_2018v7"]

INPUT_VARS = ["mll","dphill","detall","ptll","drll","Lepton_pt0","Lepton_pt1","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"]

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
    X = np.transpose(input_vars)

    X_VBF_OFF = X[Y_float == 0]
    X_VBF_ON = X[Y_float == 1]
    X_ggH_OFF = X[Y_float == 2]
    X_ggH_ON = X[Y_float == 3]
    X_top = X[Y_float == 4]
    X_WW = X[Y_float == 5]

    Y_VBF_OFF = [ENCODING_float[int(y)] for y in Y_float[Y_float==0]] #Y_onehot[Y_float == 0]
    Y_VBF_ON = [ENCODING_float[int(y)] for y in Y_float[Y_float==1]]#Y_onehot[Y_float == 1]
    Y_ggH_OFF = [ENCODING_float[int(y)] for y in Y_float[Y_float==2]]#Y_onehot[Y_float == 2]
    Y_ggH_ON = [ENCODING_float[int(y)] for y in Y_float[Y_float==3]]#Y_onehot[Y_float == 3]
    Y_top = [ENCODING_float[int(y)] for y in Y_float[Y_float==4]]#Y_onehot[Y_float == 4]
    Y_WW = [ENCODING_float[int(y)] for y in Y_float[Y_float==5]]#Y_onehot[Y_float == 5]


    tot = float(np.sum(W))
    W_VBF_OFF = W[Y_float == 0]
    W_VBF_ON = W[Y_float == 1]
    W_ggH_OFF = W[Y_float == 2]
    W_ggH_ON = W[Y_float == 3]
    W_top = W[Y_float == 4]
    W_WW = W[Y_float == 5]

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

    W_VBF_OFF = W_VBF_OFF * SF_VBF_OFF
    W_VBF_ON = W_VBF_ON * SF_VBF_ON
    W_ggH_OFF = W_ggH_OFF * SF_ggH_OFF * 2
    W_ggH_ON = W_ggH_ON * SF_ggH_ON
    W_top = W_top * SF_top
    W_WW = W_WW * SF_WW

    print("VBF_OFF:"  + str(len(W_VBF_OFF)) + ", wgt'd:"+ str(np.sum(W_VBF_OFF)) + ", SF:" + str(tot/(len(W_VBF_OFF))))
    print("VBF_ON:"  + str(len(W_VBF_ON)) + ", wgt'd:"+ str(np.sum(W_VBF_ON)) + ", SF:"  + str(tot/(len(W_VBF_ON))))
    print("ggH_OFF:"  + str(len(W_ggH_OFF)) + ", wgt'd:"+ str(np.sum(W_ggH_OFF)) + ", SF:"  + str(tot/(len(W_ggH_OFF))))
    print("ggH_ON:"  + str(len(W_ggH_ON)) + ", wgt'd:"+ str(np.sum(W_ggH_ON)) + ", SF:"  + str(tot/(len(W_ggH_ON))))
    print("top:"  + str(len(W_top)) + ", wgt'd:"+ str(np.sum(W_top)) + ", SF:"  + str(tot/(len(W_top))))
    print("WW:"  + str(len(W_WW))  + ", wgt'd:"+ str(np.sum(W_WW)) + ", SF:"  + str(tot/(len(W_WW))))

    max_events = 100000000
    X_NEW = np.concatenate((X_VBF_OFF[:min(max_events, len(X_VBF_OFF))], X_VBF_ON[:min(max_events, len(X_VBF_ON))], X_ggH_OFF[:min(max_events, len(X_ggH_OFF))], X_ggH_ON[:min(max_events, len(X_ggH_ON))], X_top[:min(max_events, len(X_top))],X_WW[:min(max_events, len(X_WW))]))
    Y_NEW = np.concatenate((Y_VBF_OFF[:min(max_events, len(X_VBF_OFF))], Y_VBF_ON[:min(max_events, len(X_VBF_ON))], Y_ggH_OFF[:min(max_events, len(X_ggH_OFF))], Y_ggH_ON[:min(max_events, len(X_ggH_ON))], Y_top[:min(max_events, len(X_top))],Y_WW[:min(max_events, len(X_WW))]))
    W_NEW = np.concatenate((W_VBF_OFF[:min(max_events, len(X_VBF_OFF))], W_VBF_ON[:min(max_events, len(X_VBF_ON))], W_ggH_OFF[:min(max_events, len(X_ggH_OFF))], W_ggH_ON[:min(max_events, len(X_ggH_ON))], W_top[:min(max_events, len(X_top))],W_WW[:min(max_events, len(X_WW))]))

    #return X, Y_onehot, W
    return X_NEW, Y_NEW, W_NEW

def baseline_model():
    model = Sequential() # 150, 100, 80, 40, 5
    model.add(Dense(150, input_dim=len(INPUT_VARS), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(5, activation='softmax'))
#    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
X_inputs, Y_inputs, W_inputs = loadVariables()

X_train, Y_train, W_train, X_test, Y_test, W_test = randomize_test_train(X_inputs, Y_inputs, W_inputs)


estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=4096, validation_split=0.25, verbose=1, shuffle=True) #500
history = estimator.fit(np.array(X_train),np.array(Y_train), sample_weight=np.array(W_train), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30,verbose=1)])

pdf_pages = PdfPages("./dnn_v2_history_five_cats_newMaxEvents_withPlots_improvedW_FinalOfSept14.pdf")
fig, ax = plt.subplots(1)
fig.suptitle("Model Accuracy")
ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
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
#print(predictions, predictions[0])
Y_pred = predictions #np.argmax(predictions)
Y_true_cat = np.array(convert_onehot(Y_test))

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

ax3 = sns.heatmap(cm, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), xticklabels = xlabels, yticklabels=ylabels, linewidth=.25)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
ax3.set_title("Confusion Matrix")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("True")
plt.tight_layout()
fig3.set_size_inches(6,6)
pdf_pages.savefig(fig3)


#print(estimator.__dict__)

estimator.model.save('hww_offshell_dnn_five_cats_withPlots_improvedW_FinalOfSept14.keras')
estimator.model.save_weights('hww_offshell_weights_five_cats_withPlots_improvedW_FinalOfSept14.h5')

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

with open("hww_offshell_weights_five_cats_improvedW_FinalOfSept14.json", "w") as json_file:
    json_data = json.dumps(output_data, indent=4)
    json_file.write(json_data)


VBF_OFF_score = [[],[],[],[],[]]
VBF_ON_score = [[],[],[],[],[]]
ggH_OFF_score = [[],[],[],[],[]]
ggH_ON_score = [[],[],[],[],[]]
BKG_score = [[],[],[],[],[]]
for i in range(0, len(predictions)):
    VBF_OFF_score[Y_true_cat[i]].append(softmax_outputs[i][0])
    VBF_ON_score[Y_true_cat[i]].append(softmax_outputs[i][1])
    ggH_OFF_score[Y_true_cat[i]].append(softmax_outputs[i][2])
    ggH_ON_score[Y_true_cat[i]].append(softmax_outputs[i][3])
    BKG_score[Y_true_cat[i]].append(softmax_outputs[i][4])

bins = np.histogram(VBF_OFF_score[0], 20, (0,1))[1]

VBF_OFF_score_binned = [np.histogram(VBF_OFF_score[i], 20, (0,1), density=True)[0] for i in range(0, len(VBF_OFF_score))]
VBF_ON_score_binned = [np.histogram(VBF_ON_score[i], 20, (0,1), density=True)[0] for i in range(0, len(VBF_ON_score))]
ggH_OFF_score_binned = [np.histogram(ggH_OFF_score[i], 20, (0,1), density=True)[0] for i in range(0, len(ggH_OFF_score))]
ggH_ON_score_binned = [np.histogram(ggH_ON_score[i], 20, (0,1), density=True)[0] for i in range(0, len(ggH_ON_score))]
BKG_score_binned = [np.histogram(BKG_score[i], 20, (0,1), density=True)[0] for i in range(0, len(BKG_score))]

fig4, ax4 = plt.subplots(1)
ax4.set_title("DNN VBF Offshell Score")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_OFF_score_binned[0], label="VBF_OFF")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_OFF_score_binned[1], label="VBF_ON")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_OFF_score_binned[2], label="ggH_OFF")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_OFF_score_binned[3], label="ggH_ON")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_OFF_score_binned[4], label="BKG")
ax4.set_xlabel("VBF_OFF score")
ax4.set_ylabel("A.U.")
ax4.legend()
ax4.set_yscale('log')
plt.tight_layout()
fig4.set_size_inches(6,6)
pdf_pages.savefig(fig4)

fig5, ax5 = plt.subplots(1)
ax5.set_title("DNN VBF Onshell Score")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_ON_score_binned[0], label="VBF_OFF")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_ON_score_binned[1], label="VBF_ON")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_ON_score_binned[2], label="ggH_OFF")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_ON_score_binned[3], label="ggH_ON")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], VBF_ON_score_binned[4], label="BKG")
ax5.set_xlabel("VBF_ON score")
ax5.set_ylabel("A.U.")
ax5.legend()
ax5.set_yscale('log')
plt.tight_layout()
fig5.set_size_inches(6,6)
pdf_pages.savefig(fig5)

fig6, ax6 = plt.subplots(1)
ax6.set_title("DNN ggH Offshell Score")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_OFF_score_binned[0], label="VBF_OFF")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_OFF_score_binned[1], label="VBF_ON")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_OFF_score_binned[2], label="ggH_OFF")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_OFF_score_binned[3], label="ggH_ON")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_OFF_score_binned[4], label="BKG")
ax6.set_xlabel("ggH_OFF score")
ax6.set_ylabel("A.U.")
ax6.legend()
ax6.set_yscale('log')
plt.tight_layout()
fig5.set_size_inches(6,6)
pdf_pages.savefig(fig6)

fig7, ax7 = plt.subplots(1)
ax7.set_title("DNN ggH Onshell Score")
ax7.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_ON_score_binned[0], label="VBF_OFF")
ax7.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_ON_score_binned[1], label="VBF_ON")
ax7.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_ON_score_binned[2], label="ggH_OFF")
ax7.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_ON_score_binned[3], label="ggH_ON")
ax7.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], ggH_ON_score_binned[4], label="BKG")
ax7.set_xlabel("ggH_ON score")
ax7.set_ylabel("A.U.")
ax7.legend()
ax7.set_yscale('log')
plt.tight_layout()
fig5.set_size_inches(6,6)
pdf_pages.savefig(fig7)

fig8, ax8 = plt.subplots(1)
ax8.set_title("DNN Background Score")
ax8.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_score_binned[0], label="VBF_OFF")
ax8.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_score_binned[1], label="VBF_ON")
ax8.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_score_binned[2], label="ggH_OFF")
ax8.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_score_binned[3], label="ggH_ON")
ax8.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_score_binned[4], label="BKG")
ax8.set_xlabel("BKG score")
ax8.set_ylabel("A.U.")
ax8.legend()
ax8.set_yscale('log')
plt.tight_layout()
fig5.set_size_inches(6,6)
pdf_pages.savefig(fig8)

pdf_pages.close()
