# Neural Network Template
# Abraham Oliver, 2016
# ReverseLearning

import tensorflow as tf
import nump as np

# Import python3's print as a function
from __future__ import print_function


# CUSTOMIZE NETWORK
# Data import, preparation
def newSet(size):
    """
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
    """
    data = []
    labels = []
    return (data, labels)
# Number of neurons in each layer
# where LAYERS[0] is the input and LAYERS[-1] is the output
LAYERS = [3, 1]
# HYPERPARAMETERS
LEARN_RATE = .01
EPOCHS = 4000
BATCH_SIZE = 200
# SETTINGS
DEBUG = False
DEBUG_INTERVAL = 2000


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
def predict(INPUT, full=False):
    """
    Get network prediction

    ARGUMENTS
        INPUT - input vector. FORM: [[x0, x1, x2, ..., x(n-1)]] for n inputs
        full - bool - Return full vector output if true and only argmax if false

    EDIT LINE 2 TO CUSTOMIZE PREDICTOR
    """
    if not full:
        return calc(INPUT).eval()[0][0]
    else:
        return calc(INPUT).eval()