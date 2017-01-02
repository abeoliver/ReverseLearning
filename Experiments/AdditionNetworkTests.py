# AdditionNetworkTests.py
# Abraham Oliver, 2016
# ReverseLearning

# Gathers loss and time data from different configurations of networks

# Imports
import tensorflow as tf
import numpy as np
from random import random, randint, uniform
from time import time
import openpyxl

# Number of tests to run
global TESTS
TESTS = 1

# Excel file for data storage
# Workbook to write to
wb = openpyxl.Workbook()
wb.title = "Addition Network Data"
data = wb.create_sheet("Data")

data['A1'] = "Inputs"
data['B1'] = "Epochs"
data['C1'] = "LRate"
data['E1'] = "Loss"
data['F1'] = "Time"

# Run a network trial
def runTrial(ins, epochs, learn_rate):
    # Addition dataset
    def newSet(size, ins):
        """EXAMPLE"""
        data = []
        labels = []
        for s in range(size):
            newInputs = [random() * randint(-100, 100) for i in range(ins)]
            data.append(newInputs)
            labels.append([sum(newInputs)])
        return (data, labels)

    # Start new session for input backprop
    sess = tf.Session()

    # Start timer
    time0 = time()

    # Input
    x = tf.placeholder(tf.float32, [None, ins])
    # Input Weights
    w = tf.ones([ins, 1])
    # Input Biases
    b = tf.zeros([1])
    # Output
    y = tf.matmul(x, w) + b
    # Label
    y_ = tf.placeholder(tf.float32, [None, 1])

    def testingLoss(inputs):
        test_data = newSet(200, inputs)
        final_loss = tf.reduce_mean(tf.abs(y_ - y))
        return final_loss.eval(session = self._session,
                               feed_dict = {x: test_data[0], y_: test_data[1]})

    # Training with quadratic cost and gradient descent with learning rate .01
    loss = tf.pow(tf.reduce_mean(y - y_), 2)
    train_step = tf.train.ProximalGradientDescentOptimizer(LEARN_RATE).minimize(loss)

    # Initialize variables
    sess.run(tf.initialize_all_variables())

    # Train to find three inputs
    for i in range(EPOCHS):
        data = newSet(300, ins)
        sess.run(train_step, feed_dict = {x: data[0], y_: data[1]})

    # print("INPUTS :: {0}".format(optimal.eval()))
    # print("LOSS   :: {0}".format(loss.eval()))
    # End timer
    time1 = time()

    # Get average loss
    returnLoss = testingLoss(ins)
    return (returnLoss, time1 - time0)

# Generate random configurations and run tests
completed = 0
while completed <= TESTS:
    # Configs
    epochs = randint(500, 3000)
    inputs = 4
    learn_rate = uniform(.000001, .00005)
    loss, time = runTrial(inputs, epochs, learn_rate)