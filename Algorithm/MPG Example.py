# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import tensorflow as tf
import numpy as np
from Network import Network
from random import randint, random
from csv import reader as CSV

def getMPGDataset():
    # Prepare Data
    data = {"mpg": [], "cylinders": [], "displacement": [], "horsepower": [], "weight": [],
            "acceleration": [], "model_year": [], "origin": [], "car_name": []}
    with open("../DataSets/auto-mpg.tab") as tsv:
        reader = CSV(tsv, dialect="excel-tab")
        for line in reader:
            if reader.line_num not in [1, 2, 3] and "?" not in line:
                data["mpg"].append(line[0])
                data["cylinders"].append(line[1])
                data["displacement"].append(line[2])
                data["horsepower"].append(line[3])
                data["weight"].append(line[4])
                data["acceleration"].append(line[5])
                data["model_year"].append(line[6])
                data["origin"].append(line[7])
                data["car_name"].append(line[8])
    # Data in training format
    xs = []
    ys = []
    for i in range(len(data["mpg"])):
        # Inputs
        t = [float(data["cylinders"][i]), float(data["displacement"][i]), float(data["horsepower"][i]),
             float(data["weight"][i]), float(data["acceleration"][i]), float(data["model_year"][i])]
        xs.append(t)
        # Label
        ys.append([float(data["mpg"][i])])
    # Final set
    trainingData = [xs, ys]
    return trainingData

mpgData = getMPGDataset()

# INIT AND TRAIN NETWORK
n = Network([6, 1], activation = "sigmoid")
n.initWeights(mode="zeros")
n.initBiases(mode="ones")
def loss(y, y_): return tf.reduce_sum(tf.abs(y - y_))
n.train(data = mpgData,
        epochs = 101,
        learn_rate = .001,
        batch_size = 0,
        loss_function = "custom",
        customLoss = loss,
        debug = False,
        debug_interval = 1)

# Test
for i in range(10):
    print "{0} => {1}".format(mpgData[1][i], n.feed(mpgData[0][i]))

# Input Backprop
# n.ibp("min", epochs = 10000,
#       learn_rate = .1, debug = True, debug_interval= -1,
#       restrictions = {0: (-51, 51), 1: (100, 120), 2: 100},
#       error_tolerance = .01, rangeGradientScalar = 1000000000.0)
