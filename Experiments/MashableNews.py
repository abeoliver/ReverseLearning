# Algorithm Playground
# ReverseLearning

from Algorithm.Network import Network
from random import random, randint
from scipy.special import expit
import numpy as np
import tensorflow as tf

def normalize(vector, span=(0,1)):
    minimum, maximum = (np.min(vector), np.max(vector))
    scaling = (span[1] - span[0]) / (maximum - minimum)
    return ((vector - minimum) * scaling) + span[0]

# Import Dataset
import csv
global f, reader, headers
f = open("../DataSets/OnlineNewsPopularity/OnlineNewsPopularity.csv", 'rb')
reader = csv.reader(f)
# Get column headers and start data at index 1
headers = next(reader)
# Transfer data from reader object into list
data = []
for row in reader:
    data.append(row)

def getByIndex(index):
    for row in reader:
        if reader.line_num == index:
            return row

def inverseSigmoid(t):
    return -1 * np.log((1 / t) - 1)

# Dataset
dataset = ([], [])
for i in data:
    q = normalize([float(a) for a in i[2:]])
    dataset[0].append(q[:58])
    dataset[1].append([q[58]])

n = Network([58, 10, 1], activation = "sigmoid")
n.initBiases(mode = "random")
n.initWeights(mode = "random", mean = 0, stddev = 100)
n.train(dataset, epochs = 100000, learn_rate = .0001, batch_size = 1000,
        debug = True, debug_interval = 100, debug_final_loss = True,
        debug_only_loss = True)
