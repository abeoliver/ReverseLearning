# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import tensorflow as tf
import numpy as np
from Network import Network

from random import random, randint
def newSet(size, inputs = 3):
    data = []
    labels = []
    for s in range(size):
        newInputs = [randint(-100, 100) for i in range(inputs)]
        data.append(newInputs)
        labels.append([sum(newInputs)])
    return (data, labels)

n = Network([3, 1])
n.initWeights(mode="zeros")
n.train(newSet(100, inputs = 3), 20, .01, debug = True, debug_interval = 1)

p = newSet(1)
a = n.feed(p[0])
print p[0]
print a.eval(session = n._session)
print p[1]