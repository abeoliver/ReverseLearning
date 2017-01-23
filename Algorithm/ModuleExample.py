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

# QUICK CHECK
# print "\n"
# p = newSet(1, 4)
# a = n.feed(p[0], evaluate = True)
# print p[1]
# print a
# print "Loss :: {0}\n".format(abs(p[1][0][0] - a[0][0]))
#
# print "Weights"
# print n.w
# print "Biases"
# print n.b

# Input Backprop
n.ibp([[1000]], epochs = -1,
      learn_rate = .1, debug = True,
      restrictions = {0: 10, 1: 5},
      debug_interval = 100)