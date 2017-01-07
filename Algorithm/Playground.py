# Algorithm Playground
# ReverseLearning

from Network import Network
from random import random, randint
from scipy.special import expit
import numpy as np

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
    dataset[0].append([expit(float(a) / 1000) for a in i[2:60]])
    dataset[1].append([expit(int(i[60]))])

n = Network([58, 1], activation = "sigmoid")
n.initBiases(mode = "random")
n.initWeights(mode = "random")
n.train(dataset, epochs = 40001, learn_rate = .00001, batch_size = 100,
        debug = False, debug_interval = 100, debug_final_loss = True)
