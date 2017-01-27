# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import tensorflow as tf
import numpy as np
from Network import Network
from random import randint, random

# Addition dataset
def newSet(size, ins = 2):
    """EXAMPLE"""
    data = []
    labels = []
    for s in range(size):
        newInputs = [random() * randint(-100, 100) for i in range(ins)]
        data.append(newInputs)
        labels.append([sum(newInputs)])
    return (data, labels)

# INIT AND TRAIN NETWORK
n = Network([4, 1])
# n.train(newSet(300, 4), epochs = 5000, learn_rate = .0001,
#         batch_size = 0, debug = False, debug_interval = 5000)
n.initWeights(mode="ones")
n.initBiases(mode="zeros")

# Input Backprop
n.ibp("min", epochs = 10000,
      learn_rate = .1, debug = True, debug_interval= -1,
      restrictions = {0: (-51, 51), 1: (100, 120), 2: 100},
      error_tolerance = .01, rangeGradientScalar = 1000000000.0)
