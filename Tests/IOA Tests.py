# Input Optimization Algorithm Tests
# ReverseLearning, 2017

# Linear :: f(x) = 2x + 3
#   17.0
#   CALCULATED : f(6.9999) = 16.9999
#   TARGET : f(7) = 17
#   EPOCHS : 35 (Beat Error)

# Positive Quadratic :: f(x) = x^2 - 8x + 15
#   Minimum
#   CALCULATED : f(3.99999) = -1
#   TARGET : f(4.0) = -1
#   EPOCHS : 71 (Low Gradients)

# Negative Quadratic :: f(x) = -x^2 - 3x + 18
#   Maximum
#   CALCULATED : f(-1.49999) = 20.25
#   TARGET : f(-1.5) = 20.25
#   EPOCHS : 67 (Low Gradients)

# Surface #1 :: f(x, y) = x ^ 2 + y ^ 2 - 100
#   Minimum
#   CALCULATED : f(1.9999995, -2.9999995) = 4.547473508864641e-13
#   TARGET : f(2, -3) = 0
#   EPOCHS : 69 (Low Gradients)

import tensorflow as tf
from IOA import IOA, saveDigests, loadDigests
import pandas, sys

def RUN(funcs, mode = ""):
    with open("Results.txt", 'a') as file:
        if mode == "debug":
            for func in funcs:
                func()
        else:
            sys.stdout = file
            for func in funcs:
                func()

# LINEAR -- f(x) = 2x + 3
def linear17():
    title = "Linear Max -- f(x) = 2x + 3"
    def f(x): return tf.add(tf.multiply(x, 2), 3)
    I = IOA(f, 1)
    final, digest = I.optimize(17.0, epochs = -1, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title = title)
    saveDigests(digest, "Records/Linear-17.ioa")
    return digest, final, title, 17.0

# QUAD Positive -- f(x) = x^2 - 8x + 15
def quadraticPositiveMin():
    title = "Quadratic Positive Min -- f(x) = x^2 - 8x + 15"
    def f(x): return tf.add(tf.add(tf.pow(x, 2.0), tf.multiply(-8.0, x)), 15.0)
    I = IOA(f, 1)
    final, digest = I.optimize("min", epochs = -1, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 1e-6,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title = title)
    saveDigests(digest, "Records/Quad-Positive-Min.ioa")
    return digest, final, title, "min"

# QUAD Negative -- f(x) = -x^2 - 3x + 18
def quadraticNegativeMax():
    title = "Quadratic Negative Max -- f(x) = -x^2 - 3x + 18"
    def f(x): return tf.add(tf.add(-tf.pow(x, 2.0), tf.multiply(-3.0, x)), 18.0)
    I = IOA(f, 1)
    final, digest = I.optimize("max", epochs = -1, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 1e-6,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title = title)
    saveDigests(digest, "Records/Quad-Negative-Max.ioa")
    return digest, final, title, "max"

# Surface #1 -- f(x, y) = (x - 2) ^ 2 + (y + 3) ^ 2
def surface1():
    title = "Surface #1 -- f(x, y) = x ^ 2 + y ^ 2 - 100"
    def f(x): return tf.add(tf.pow(tf.add(x[0][0], -2), 2), tf.pow(tf.add(x[0][1], 3), 2))
    I = IOA(f, 2)
    final, digest = I.optimize("min", epochs = -1, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 1e-6,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title = title)
    saveDigests(digest, "Records/Surface-1.ioa")
    return digest, final, title, "min"

# Many Dimensional - f(a, b, c, d, e, f) = (a-2)^2 + (b+4)^2 + c^2 + (d-10)^2 + e^2 + (f+10)^2
def manyVar():
    title = "6D -- f(a, b, c, d, e, f) = (a-2)^2 + (b+4)^2 + c^2 + (d-10)^2 + e^2 + (f+10)^2"
    def f(x):
        p1 = tf.add(tf.pow(tf.add(x[0][0], -2), 2), tf.pow(tf.add(x[0][1], 4), 2))
        p2 = tf.add(tf.pow(x[0][2], 2), tf.pow(tf.add(x[0][3], -10), 2))
        p3 = tf.add(tf.pow(x[0][4], 2), tf.pow(tf.add(x[0][5], 10), 2))
        return tf.add(p1, tf.add(p2, p3))
    I = IOA(f, 6)
    final, digest = I.optimize("min", epochs = 100, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = 1,
                               rangeGradientScalar = 1e11, gradientTolerance = 9e-6,
                               startPreset = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               returnDigest = True, digestInterval = 1,
                               title = title)
    saveDigests(digest, "Records/Surface-1.ioa")
    return digest, final, title, "min"

if __name__ == "__main__":
    funcs = []
    # funcs.append(linear17)
    # funcs.append(quadraticNegativeMax)
    # funcs.append(quadraticPositiveMin)
    # funcs.append(surface1)
    funcs.append(manyVar)
    RUN(funcs, mode = "debug")