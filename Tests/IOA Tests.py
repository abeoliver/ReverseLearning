# Input Optimization Algorithm Tests
# ReverseLearning, 2017

# Linear :: f(x) = 2x + 3
#     Maximum (epochs = 100, learning rate = .1):
#         f(20.00) = 43.00
#         TARGET - Inf

# Positive Quadratic :: f(x) = x^2 - 8x + 15
#     Maximum   (epochs = 100, learning rate = .1):
#         f(-331271840.0) = 1.097410357976105e+17
#         TARGET - Inf
#     Minimum   (epochs = 100, learning rate = .1):
#         f(3.99999) = -1
#         TARGET - f(4.0) = -1
#     17.0      (error tolerance = .15, learning rate = .1):
#         f(-.24673) = 12.0347
#         Target - f(-.243) = 17.0
#         Epochs - 7 (Beat Error)

import tensorflow as tf
from IOA import IOA, saveDigests, loadDigests
import pandas

# LINEAR -- f(x) = 2x + 3
def linearMax():
    def f(x): return tf.add(tf.multiply(x, 2), 3)
    I = IOA(f, 1)
    final, digest = I.optimize("max", epochs = 100, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title = "Linear Max -- f(x) = 2x + 3")
    saveDigests(digest, "Records/Linear-Max.ioa")

# QUAD -- f(x) = x^2 - 8x + 15
def quadraticPositiveMax():
    def f(x): return tf.add(tf.subtract(tf.pow(x, 2.0), tf.multiply(8.0, x)), 15.0)
    I = IOA(f, 1)
    final, digest = I.optimize("max", epochs = 100, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title="Quadratic Positive Max -- f(x) = x^2 - 8x + 15")
    saveDigests(digest, "Records/Quad-Positive-Max.ioa")
def quadraticPositiveMin():
    def f(x): return tf.add(tf.subtract(tf.pow(x, 2.0), tf.multiply(8.0, x)), 15.0)
    I = IOA(f, 1)
    final, digest = I.optimize("min", epochs = 100, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title = "Quadratic Positive Min -- f(x) = x^2 - 8x + 15")
    saveDigests(digest, "Records/Quad-Positive-Min.ioa")
def quadraticPositive17():
    def f(x): return tf.add(tf.subtract(tf.pow(x, 2.0), tf.multiply(8.0, x)), 15.0)
    I = IOA(f, 1)
    final, digest = I.optimize(17.0, epochs = -1, learn_rate = .1, error_tolerance = .15,
                               restrictions = {}, debug = True, debug_interval = -1,
                               rangeGradientScalar = 1e11, gradientTolerance = 5e-7,
                               startPreset = [], returnDigest = True, digestInterval = 1,
                               title = "Quadratic Positive 17 -- f(x) = x^2 - 8x + 15")
    saveDigests(digest, "Records/Quad-Positive-17.ioa")


if __name__ == "__main__":
    # linearMax()
    # quadraticPositiveMax()
    # quadraticPositiveMin()
    quadraticPositive17()