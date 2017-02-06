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
    with open("Iris.tab") as tsv:
        reader = CSV(tsv, dialect="excel-tab")
        for line in reader:
            if reader.line_num not in [1, 2, 3] and line[4][:3] != "MAX":
                data["sepal length"].append(line[0])
                data["sepal width"].append(line[1])
                data["petal length"].append(line[2])
                data["petal width"].append(line[3])
                data["species"].append(line[4])
    # Data in training format
    xs = []
    ys = []
    for i in range(len(data["species"])):
        if data["species"][i] in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]:
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
# train, test = splitData(full, 140)

def getAccuracy(classifier, testset, err):
    c = classifier
    runner = 0
    for i in range(len(testset[0])):
        y_ = c.feed([testset[0][i]])[0]
        compl = 0
        for j in range(len(y_)):
            if y_[j] >= (testset[1][i][j] - err) and y_[j] <= (testset[1][i][j] + err) :
                compl += 1
        if compl == len(y_): runner += 1
    return float(runner) / len(testset[0])

# Get average feature sets
def getAverage(className, dataset):
    featureSets = [0.0 for q in dataset[0][0]]
    count = 0
    for i in range(len(dataset[1])):
        if dataset[1][i] == className:
            count += 1
            for f in range(len(dataset[0][i])):
                featureSets[f] += dataset[0][i][f]
    if count == 0:
        raise Exception("There are no items of class {0}".format(className))
    return [fs / count for fs in featureSets]

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

# avg_setosa = getAverage([1.0, 0.0, 0.0], full)
# avg_versicolor = getAverage([0.0, 1.0, 0.0], full)
# avg_virginica = getAverage([0.0, 0.0, 1.0], full)
# print "AVERAGE SETOSA {0}".format(avg_setosa)               # [5.006, 3.418, 1.464, 0.244]
# print n.feed(avg_setosa)
# print "AVERAGE VERSICOLOR {0}".format(avg_versicolor)       # [5.936, 2.770, 4.260, 1.326]
# print n.feed(avg_versicolor)
# print "AVERAGE VIRGINICA {0}".format(avg_virginica)         # [6.588, 2.974, 5.552, 2.026]
# print n.feed(avg_virginica)
# print ""
#
# print ("ACCURACY (.3) :: {0}".format(getAccuracy(n, full, .3)))
# print ("ACCURACY (.2) :: {0}".format(getAccuracy(n, full, .2)))
# print ("ACCURACY (.1) :: {0}".format(getAccuracy(n, full, .1)))
# print ""

optimal_setosa = n.ibp([1.0, 0.0, 0.0], epochs = -1, debug = True, learn_rate = .1,
                        error_tolerance = .1, rangeGradientScalar = 1e10, debug_interval = 100,
                        restrictions = {0: (4.5, 8.0), 1: (2.0, 5.0), 2: (1.0, 6.5), 3: (0.0, 3.0)})

optimal_versicolor = n.ibp([0.0, 1.0, 0.0], epochs = -1, debug = True, learn_rate = .1,
                            error_tolerance = .1, rangeGradientScalar = 1e10, debug_interval = 100,
                            restrictions = {0: (4.5, 8.0), 1: (2.0, 5.0), 2: (1.0, 6.5), 3: (0.0, 3.0)})

optimal_virginica = n.ibp([0.0, 0.0, 1.0], epochs = -1, debug = True, learn_rate = .1, error_tolerance = .1,
                          rangeGradientScalar = 1e12, debug_interval = 100, loss_function = "absolute_distance",
                          restrictions = {0: (4.5, 8.0), 1: (2.0, 5.0), 2: (1.0, 6.5), 3: (0.0, 3.0)})

# Check optimal
print "OPTIMAL SETOSA     :: {0}".format(n.eval(optimal_setosa))        # [ 6.427  3.900  1.888  1.171 ]
print "OPTIMAL SETOSA     :: {0}".format(n.feed(optimal_setosa))
print "OPTIMAL VERSICOLOR :: {0}".format(n.eval(optimal_versicolor))    # [ 6.25  3.5   3.75  1.5 ]
print "OPTIMAL VERSICOLOR :: {0}".format(n.feed(optimal_versicolor))
print "TARGET VIRGINICA   :: {0}".format(n.eval(optimal_virginica))
print "TARGET VIRGINICA   :: {0}".format(n.feed(optimal_virginica))

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