# Input Optimization Algorithm Tests
# ReverseLearning, 2017

import tensorflow as tf
from IOA import IOA
import pickle

# LINEAR -- f1(x) = 2x + 3
def f1(x): return tf.add(tf.mul(x, 2), 3)
I = IOA(f1, 1)
final, digest = I.optimize("min", epochs = 100, learn_rate = .1, error_tolerance = .2,
                           restrictions = {}, debug = False, debug_interval = -1,
                           rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                           startPreset = [], returnDigest = True, digestInterval = 10)
with open('linear.pkl', 'w') as f:
    pickle.dump(digest, f)