# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import numpy as np
from math import floor
from random import randint, random
from csv import reader as CSV
from Network import Network

def getDataset():
    # Prepare Data
    data = {"sepal length":[], "sepal width": [], "petal length": [],
            "petal width": [], "species": []}
    with open("iris.tab") as tsv:
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

def getAccuracy(classifier, testset):
    c = classifier
    runner = 0
    for i in range(len(testset[0])):
        y_ = c.feed([testset[0][i]])[0]
        compl = 0
        for j in range(len(y_)):
            if y_[j] == testset[1][i][j]: compl += 1
        if compl == len(y_): runner += 1
    return float(runner) / len(testset[0])

# ------------ USE IOA ON TRAINED NETWORK -----------
n = Network([4, 2, 3], shaping="softmax")
n.initWeights(mode="preset", preset=[[[-2.96867418, -2.04050994],
                                      [-3.07419991, -1.34310222],
                                      [5.77198744, 2.33850431],
                                      [4.32180357, 2.56259775]],
                                     [[-4.74904728, -3.10435867, 8.68291569],
                                      [-1.60662615, -3.64023638, 2.82549834]]])
n.initBiases(mode="preset", preset=[[[-6.04968405, -2.69198585]],
                                    [[0.4100841, 2.19560456, -3.45070601]]])

optimal = n.ibp([1.0, 0.0, 0.0], epochs = -1, debug = False, learn_rate = .1,
                error_tolerance = .01, rangeGradientScalar = 1e10,
                restrictions = {0: (4.5, 8.0), 1: (2.0, 5.0), 2: (1.0, 6.5), 3: (0.0, 3.0)})

# Check optimal
print "OPTIMAL INPUTS :: {0}".format(n.eval(optimal))   # [[ 6.42741394  3.89656925  1.88798046  1.17135382 ]]
print "OPTIMAL OUTPUT :: {0}".format(n.feed(optimal))   # [[ 0.99500012  0.00499985  0.                     ]]
print "TARGET OUTPUT  :: {0}".format([1.0, 0.0, 0.0])   # [[1.0          0.          0.                     ]]

# ------------------ TRAIN NETWORK ------------------
# NOTE: The descovered weights and biases were manually entered
#       into the above network after trained here
#
n = Network([4, 2, 3], shaping = "softmax")
n.initWeights(mode = "random", mean = 0.0, stddev = 1.0)
n.initBiases(mode = "random", mean = 0.0, stddev = 1.0)

# n.train(data = full,
#         epochs = 300000,
#         learn_rate = .1)
# ---------------------------------------------------