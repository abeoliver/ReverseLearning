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
# n.ibp([[25]], epochs = -1,
#       learn_rate = .1, debug = True,
#       restrictions = {0: 10, 1: 5, 2: (1, 3)},
#       debug_interval = 1)

# s = tf.Session()
#
# def sig(z):
#     return tf.nn.sigmoid(tf.constant(.000001) * z)
#
# def f(x, b, t):
#     q = sig(x)
#     w = tf.mul(q, tf.cast(tf.sub(t, b), tf.float32))
#     return tf.add(w, b)
#
# def b(x, t):
#     return (x - sig(x) * t) / (1 - sig(x))
#
# def t(x, b):
#     return (x - b + (sig(x) * b)) / sig(x)
#
#
# # i = 100.000023
# # q = [[tf.constant(float(i))]]
# # w = [[tf.constant(b(float(i), float(i)).eval(session = s))]]
# # e = [[tf.constant(float(i))]]
#
# x = [[10.0, 5.0, 6.0, 0.0]]
# lp = n._getRestrictionVectors({0: 10, 1: 5, 2: (1, 3)}, x)
# for i in range(len(lp[0])):
#     print "{0} -- {1}".format(lp[0][i].eval(session = s), lp[1][i].eval(session = s))
# print f(x, b(x, x), x).eval(session = s)
# print f(x, lp[0], lp[1]).eval(session = n._session)
