# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import tensorflow as tf
import numpy as np
from Network import Network
import json
from random import randint

f = open('../DataSets/WineQuality/winequality-red.json')
data = json.load(f)
f.close()

def newSet(data, size):
    xs = []     # Data
    ys = []     # Labels
    # Make data subset of size 'size'
    columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
               "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
               "pH", "sulphates", "alcohol"]
    for i in range(size):
        index = randint(0, 1598)
        d = data[index]     # Data point being extracted
        ins = []        # Input container to be filled
        # Turn named inputs into un-named input vectors
        for j in columns:
            ins.append(d[j])  # Map column values
        # Add newly separated data to dataset
        xs.append(ins)

        # Create one-hot for label
        id = int(d["quality"])
        f = [0 for i in range(10)]  # One-hot from zero to 10
        f[id] = 1
        ys.append(f)
    return (xs, ys)

n = Network([11, 10, 10])
n.initWeights(mode="zeros")
n.train(newSet(data, 100), epochs = 20001, learn_rate = .01,
        debug = True, debug_interval = 10000)

p = newSet(data, 1)
a = n.feed(p[0]).eval(session = n._session)
print p[0]
print [float(i) for i in p[1][0]]
print [round(i, 1) for i in a[0]]

# Test
