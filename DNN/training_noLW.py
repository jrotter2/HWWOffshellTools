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


INFILE_NAME = "/eos/user/j/jrotter/HWW_DNN_Ntuples/HWW_DNN_Ntuple_v1.root"
# FILE VERSIONING:
# v1 -> {"mll","dphill","detall","ptll","drll","pt1","pt2","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"}


CATEGORIES = ["VBF", "ggH", "top", "WW"] #Titles for categories
ENCODING_float = [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]
ENCODING = {"VBF_2018v7" : [1,0,0,0], "ggH_2018v7" : [0,1,0,0], "top_2018v7": [0,0,1,0], "WW_2018v7":[0,0,0,1]}

CAT_CONFIG_IDS = ["VBF_2018v7", "ggH_2018v7", "top_2018v7", "WW_2018v7"]

INPUT_VARS = ["mll","dphill","detall","ptll","drll","pt1","pt2","mth","mjj","detajj","dphijj","PuppiMET_pt","dphillmet","mcollWW"]

INPUT_VAR_INDEX = {}
for i, var in enumerate(INPUT_VARS):
    INPUT_VAR_INDEX[var] = i

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
    ntuple_file = uproot3.open(INFILE_NAME)
    Y_float = ntuple_file["hww_ntuple"]["Y"].array()
    Y_onehot = [ENCODING_float[int(y)] for y in Y_float]
    W = np.array([1 for i in ntuple_file["hww_ntuple"]["wgt"].array()])
    input_vars = [ntuple_file["hww_ntuple/" + var].array() for var in INPUT_VARS]
    X = np.transpose(input_vars)


    W_VBF = W[Y_float == 0]
    W_ggH = W[Y_float == 1]
    W_WW = W[Y_float == 2]
    W_top = W[Y_float == 3]
    print("VBF:"  + str(np.sum(W_VBF)))
    print("ggH:"  + str(np.sum(W_ggH)))
    print("WW:"  + str(np.sum(W_WW)))
    print("top:"  + str(np.sum(W_top)))

    print(X[0], Y_onehot[0], W[0])
    return X, Y_onehot, W

def baseline_model():
    model = Sequential()
    model.add(Dense(40, input_dim=len(INPUT_VARS), activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(4, activation='softmax'))
#    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
X_inputs, Y_inputs, W_inputs = loadVariables()

X_train, Y_train, W_train, X_test, Y_test, W_test = randomize_test_train(X_inputs, Y_inputs, W_inputs)


# define and fit the base model
#def get_base_model(trainX, trainy, trainW):
#    # define model
#    model = Sequential()
#    model.add(Dense(20, input_dim=len(INPUT_VARS), activation='relu', kernel_initializer='he_uniform'))
#    model.add(Dense(4, activation='softmax'))
    # compile model
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
#    print("Training...")
#    model.fit(trainX, trainy,sample_weight=trainW, epochs=150, batch_size=4096, verbose=1)
#    return model

# evaluate a fit model
#def evaluate_model(model, trainX, testX, trainy, testy):
#    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
#    _, test_acc = model.evaluate(testX, testy, verbose=0)
#    return train_acc, test_acc

# add one new layer and re-train only the new layer
#def add_layer(model, trainX, trainy, trainW):
#    # remember the current output layer
#    output_layer = model.layers[-1]
#    # remove the output layer
#    model.pop()
    # mark all remaining layers as non-trainable
#    for layer in model.layers:
#        layer.trainable = False
    # add a new hidden layer
#    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    # re-add the output layer
#    model.add(output_layer)
    # fit model
#    model.fit(trainX, trainy,sample_weight=trainW, epochs=150, batch_size=4096, verbose=1)

# get the base model

#print("Getting Base Model")
#model = get_base_model(np.array(X_train), np.array(Y_train), np.array(W_train))
# evaluate the base model
#scores = dict()
#train_acc, test_acc = evaluate_model(model, np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test))
#print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
#scores[len(model.layers)] = (train_acc, test_acc)
# add layers and evaluate the updated model
#n_layers = 0
#for i in range(n_layers):
#    print("###### LAYER: " + str(i+1) + " ######")
    # add layer
#    add_layer(model, np.array(X_train), np.array(Y_train), np.array(W_train))
    # evaluate model
#    train_acc, test_acc = evaluate_model(model, np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test))
#    print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
    # store scores for plotting
#    scores[len(model.layers)] = (train_acc, test_acc)
# plot number of added layers vs accuracy


#model = baseline_model()
estimator = KerasClassifier(build_fn=baseline_model, epochs=25, batch_size=4096, validation_split=0.25, verbose=1, shuffle=True)
history = estimator.fit(np.array(X_train),np.array(Y_train), sample_weight=np.array(W_train), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30,verbose=1)])

pdf_pages = PdfPages("./dnn_v1_history.pdf")
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

#print(predictions, predictions[0])
Y_pred = predictions #np.argmax(predictions)
Y_true_cat = np.array(convert_onehot(Y_test))

for i in range(0, 10):
    print(predictions[i], Y_pred[i])

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

pdf_pages.close()

#print(estimator.__dict__)

estimator.model.save('hww_offshell_dnn.keras')
estimator.model.save_weights('hww_offshell_weights.h5')

import json
weights_list = np.array(estimator.model.get_weights())
output_data = {}

for layer in range(0, int(len(weights_list)/2)):
    n_inputs = len(weights_list[layer*2])
    print(str(layer) + " : " + str(n_inputs))
    n_nodes = len(weights_list[layer*2][0])
    layer_data = {}
    transposed_weights = np.transpose(weights_list[layer*2])
    for i in range(0, n_nodes):
        node_data = {"weights" : transposed_weights[i].tolist(), "bias" : weights_list[layer*2+1].tolist()[i]}
        layer_data["Node_" + str(i)] = node_data
    output_data["Layer_" + str(layer)] = layer_data

with open("hww_offshell_weights.json", "w") as json_file:
    json_data = json.dumps(output_data, indent=4)
    json_file.write(json_data)


