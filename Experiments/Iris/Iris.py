# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import numpy as np
from math import floor
from random import randint, random
from csv import reader as CSV
import sklearn as skl
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def getDataset():
    # Prepare Data
    data = {"sepal length":[], "sepal width": [], "petal length": [],
            "petal width": [], "species": []}
    with open("C:/Users/abeol/Git/ReverseLearning/DataSets/Iris.tab") as tsv:
        reader = CSV(tsv, dialect="excel-tab")
        for line in reader:
            if reader.line_num not in [1, 2, 3]:
                data["sepal length"].append(line[0])
                data["sepal width"].append(line[1])
                data["petal length"].append(line[2])
                data["petal width"].append(line[3])
                data["species"].append(line[4])
    # Data in training format
    xs = []
    ys = []
    for i in range(len(data["species"])):
        # Inputs
        t = [float(data["sepal length"][i]), float(data["sepal width"][i]), float(data["petal length"][i]),
             float(data["petal width"][i])]
        xs.append(t)
        # Label
        if data["species"][i] == "Iris-setosa":
            ys.append([1.0, 0.0, 0.0])
        elif data["species"][i] == "Iris-versicolor":
            ys.append([0.0, 1.0, 0.0])
        elif data["species"][i] == "Iris-virginica":
            ys.append([0.0, 0.0, 1.0])
    # Final set
    trainingData = [xs, ys]
    return trainingData

def splitData(data, trainsize):
    train = [[], []]
    test = [[], []]
    types = [[], [], []]
    for i in range(len(data[0])):
        if data[1][i][0] == 1.0: types[0].append(i)
        elif data[1][i][1] == 1.0: types[1].append(i)
        elif data[1][i][2] == 1.0: types[2].append(i)
    perType = trainsize / 3
    for i in range(int(floor(perType))):
        train[0].append(data[0][types[0][i]])
        train[0].append(data[0][types[1][i]])
        train[0].append(data[0][types[2][i]])
        train[1].append(data[1][types[0][i]])
        train[1].append(data[1][types[1][i]])
        train[1].append(data[1][types[2][i]])
    m = range(int(floor(perType)), len(types[0]))
    for i in m:
        test[0].append(data[0][types[0][i]])
        test[0].append(data[0][types[1][i]])
        test[0].append(data[0][types[2][i]])
        test[1].append(data[1][types[0][i]])
        test[1].append(data[1][types[1][i]])
        test[1].append(data[1][types[2][i]])
    return (train, test)

full = getDataset()
train, test = splitData(full, 140)

scaler = StandardScaler()
scaler.fit(train[0])
XTrain = train[0]
# XTrain = scaler.transform(train[0])

def getAccuracy(classifier, testset):
    c = classifier
    runner = 0
    for i in range(len(testset[0])):
        y_ = c.predict([testset[0][i]])[0]
        compl = 0
        for j in range(len(y_)):
            if y_[j] == testset[1][i][j]: compl += 1
        if compl == len(y_): runner += 1
    return float(runner) / len(testset[0])

c = MLPClassifier(hidden_layer_sizes = (),
                   activation = "logistic",
                   learning_rate = "constant",
                   learning_rate_init = .001,
                   solver = "lbfgs",
                   verbose = False,
                   max_iter = 10000000,
                   tol = 1e-60)
c.fit(XTrain, train[1])
print(getAccuracy(c, [test[0], test[1]]))
print(getAccuracy(c, [full[0], full[1]]))
# print(getAccuracy(c, [scaler.transform(test[0]), test[1]]))
# print(getAccuracy(c, [scaler.transform(full[0]), full[1]]))

def predictTest(inp, lbl):
    a = inp
    # x = scaler.transform([inp])
    x = [inp]
    print("{0} --> {1}".format(a, x))
    print("LABEL     :: {0}".format(lbl))
    print("PREDICTED :: {0}".format(c.predict(x)[0]))
# predictTest([5.9, 3.0, 5.1, 1.8], [0, 0, 1])
# predictTest([6.0, 2.9, 4.5, 1.5], [0, 1, 0])
# predictTest([5.3, 3.7, 1.5, 0.2], [1, 0, 0])
# predictTest([5.2, 3.8, 1.6, 0.3], [1, 0, 0])