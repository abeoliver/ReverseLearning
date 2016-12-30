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
n = Network([3, 1])
n.initWeights("zeros")
n.train(newSet(300, 3), epochs = 10001, learn_rate = .0001,
        batch_size = 0, debug = False, debug_interval = 5000)

# QUICK CHECK
p = newSet(1, 3)
a = n.eval(n.feed(p[0]))
print p[1]
print a
print "\nQUICK CHECK :: {0}\n".format(abs(p[1][0][0] - a[0][0]))

# Input Backprop
# n.ibp([[20]], epochs = 100000, learn_rate = .0001)