# Input Optimization Algorithm Tests
# ReverseLearning, 2017

import tensorflow as tf
from IOA import IOA
import pandas

# LINEAR -- f1(x) = 2x + 3
def f1(x): return tf.add(tf.multiply(x, 2), 3)
I = IOA(f1, 1)
final, digest = I.optimize(21.0, epochs = 100, learn_rate = .1, error_tolerance = .2,
                           restrictions = {}, debug = False, debug_interval = -1,
                           rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                           startPreset = [], returnDigest = True, digestInterval = 1)

# Print final digest and answer
print(final)
print()
for d in digest:
    print(d)