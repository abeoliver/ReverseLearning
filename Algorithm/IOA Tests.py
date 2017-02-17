# Input Optimization Algorithm Tests
# ReverseLearning, 2017

import tensorflow as tf
from IOA import IOA
import pandas

# LINEAR -- f1(x) = 2x + 3
# def f1(x): return tf.add(tf.multiply(x, 2), 3)
# I = IOA(f1, 1)
# final1, digest1 = I.optimize(21.0, epochs = -1, learn_rate = .1, error_tolerance = .2,
#                            restrictions = {}, debug = False, debug_interval = 10,
#                            rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
#                            startPreset = [], returnDigest = True, digestInterval = 1)
# saveDigests(digest1, "Linear-21.ioa")

# QUAD -- f1(x) = x^2 - 8x + 15
def f2(x): return tf.add(tf.subtract(tf.pow(x, 2.0), tf.multiply(8.0, x)), 15.0)
I = IOA(f2, 1)
final1, digest2 = I.optimize("max", epochs = 100, learn_rate = .1, error_tolerance = .2,
                           restrictions = {}, debug = False, debug_interval = 10,
                           rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                           startPreset = [], returnDigest = True, digestInterval = 1)
saveDigests(digest2, "Quad-max.ioa")