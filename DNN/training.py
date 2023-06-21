import uproot
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


CATEGORIES = ["VBF", "ggH", "top", "WW", "DY"] #Titles for categories
CAT_CONFIG_IDS = ["VBF", "ggH", "top", "WW", "DY"] #For now are the same as titles but may include versioning later

CONFIG_FILE_PATH = "dnn_config.json"
CONFIG_FILE = open(CONFIG_FILE_PATH, "r")
CONFIG_FILE_CONTENTS = CONFIG_FILE.read()
CONFIG = json.loads(CONFIG_FILE_CONTENTS)

def loadVariables():
    X = []
    Y = []
    for category in CAT_CONFIG_IDS:
        for f in CONFIG[category]["files"]:
            rootFile = uproot.open(f)

            y_evt = [category == c for c in CAT_CONFIG_IDS] # will produce one-hot encoding
            x_evt = []
            ###
            ### CONSTRUCT AND BUILD INPUT VARIABLES HERE
            ###
            X.append(x_evt)
            Y.append(y_evt)

def baseline_model():
    model = Sequential()
    model.add(Dense(240, input_dim=26, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X, Y = loadVariables()
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

