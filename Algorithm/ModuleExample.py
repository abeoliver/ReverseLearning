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
n.train(newSet(300, 4), epochs = 14000, learn_rate = .0001,
        batch_size = 0, debug = False, debug_interval = 5000)
print n.feed([[1.0, 2.0, 3.0, 4.0]])