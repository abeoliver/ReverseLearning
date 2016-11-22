# Neural Network Template
# Abraham Oliver, 2016
# ReverseLearning

# Import python3's print as a function
from __future__ import print_function
# Tensorflow and Numpy
import tensorflow as tf
import numpy as np


# CUSTOMIZE NETWORK
# Data import, preparation
"""
def newSet(size):
    ""
    Import data and prepare for use

    ARGUMENTS:
        size - int - number of data in set

    RETURN:
        ([input1, input2, input3, ...], [output_label1, output_label2, output_label3, ...])
        where input(n) is in form [value1, value2, value3, ...]
        and output_label(n) is in form [output_node1, output_node2, output_node3, ...]

    NOTES:
        - DO NOT CHANGE NAME
        - Random import can be removed if not used
    ""
    data = []
    labels = []
    return (data, labels)
"""
from random import random, randint
def newSet(size):
    """EXAMPLE"""
    data = []
    labels = []
    for s in range(size):
        newInputs = [random() * randint(-50, 50) for i in range(3)]
        data.append(newInputs)
        labels.append([sum(newInputs)])
    return (data, labels)
# Number of neurons in each layer
# where LAYERS[0] is the input and LAYERS[-1] is the output
LAYERS = [3, 1]
# HYPERPARAMETERS
LEARN_RATE = .01
EPOCHS = 10000
BATCH_SIZE = 200
# SETTINGS
DEBUG = True
DEBUG_INTERVAL = 2000
# INPUT BACKPROP
INPUT_BACKPROP_EPOCHS = 1000
INPUT_BACKPROP_TARGET = [10.0]


# ======================= DO NOT PROGRAM BELOW LINE =======================
# with exception of loss function or optimizer change

# Start a session
sess = tf.Session()


# Define Model Parameters (CUSTOMIZATION NOT NEEDED)
# Except loss function and optimizer may be customized
# Input
x = tf.placeholder(tf.float32, [None, LAYERS[0]], name="x")
# Weights
w = [tf.Variable(tf.zeros([LAYERS[n], LAYERS[n + 1]]), name="w{0}".format(n)) for n in range(len(LAYERS) - 1)]
# Biases
b = [tf.Variable(tf.ones([LAYERS[n + 1]]), name="b{0}".format(n)) for n in range(len(LAYERS) - 1)]
# Output
def calc(inp, n = 0):
    if n == len(LAYERS) - 2:
        return tf.matmul(inp, w[n]) + b[n]
    return calc(tf.matmul(inp, w[n]) + b[n], n + 1)
y = calc(x)
# Label
y_ = tf.placeholder(tf.float32, [None, LAYERS[-1]], name="y_")
# Loss function
loss = tf.reduce_mean(tf.pow(y_ - y, 2))
# Training step
train_step = tf.train.ProximalGradientDescentOptimizer(LEARN_RATE).minimize(loss)

# Input Backprop
# Optimal input
optimal = tf.Variable(tf.zeros([1, LAYERS[0]]))
# Output to optimize
out = calc(optimal)
# Target label
target = tf.constant(INPUT_BACKPROP_TARGET)
# Loss
IB_loss = tf.pow(tf.reduce_mean(target - out), 2)
# Training
IB_train_step = tf.train.ProximalGradientDescentOptimizer(LEARN_RATE).minimize(IB_loss)


# Train model (CUSTOMIZATION NOT NEEDED)
# Initialize variables
sess.run(tf.initialize_all_variables())
# Status bar
STATUS_INTERVAL = EPOCHS / 10
# Train normal model
print("TRAINING", end="")
for i in range(EPOCHS):
    # Get data
    batch_inps, batch_outs = newSet(BATCH_SIZE)

    # Debug printing
    if i % DEBUG_INTERVAL == 0 and DEBUG:
        with sess.as_default():
            print("Weights ::")
            for i in w:
                print(i.eval())
            print("Biases ::")
            for i in b:
                print(i.eval())
            print("Loss :: {0}\n\n".format(loss.eval(feed_dict={x: batch_inps, y_: batch_outs})))
    # Run train step
    sess.run(train_step, feed_dict={x: batch_inps, y_: batch_outs})

    # Print status bar
    if i % STATUS_INTERVAL == 0 and not DEBUG: print(" * ", end="")
print("\nTRAINING COMPLETE")

# Use trained network
def predict(INPUT):
    """
    Get network prediction

    ARGUMENTS
        INPUT - input vector. FORM: [[x0, x1, x2, ..., x(n-1)]] for n inputs
    """
    return calc(INPUT).eval()


# Perform input backprop
for i in range(INPUT_BACKPROP_EPOCHS): sess.run(IB_train_step)

# ======================= WRITE CODE BELOW =======================

with sess.as_default():
    print(predict([[3.0, 2.0, 1.0]]))
    print(optimal.eval())