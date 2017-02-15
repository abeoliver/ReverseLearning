# Input Optimization Algorithm Tests
# ReverseLearning, 2017

import tensorflow as tf
from IOA import IOA

# LINEAR -- f1(x) = 2x + 3
def f1(x): return tf.add(tf.mul(x, 2), 3)
I = IOA(f1, 1)
I.optimize("min", epochs = 100, learn_rate = .1, error_tolerance = .2,
           restrictions = {}, debug = True, debug_interval = -1,
           rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
           startPreset = [])