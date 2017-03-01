# Linear Regression Model
# Reverse Learning, 2017

# Import dependencies
import tensorflow as tf
from random import randint, uniform
import numpy as np
import IOA

class LinearReg:
    def __init__(self, inps, w = None, b = None):
        self.inps = inps

        if w == None: self.w = [[0.0 for i in range(inps)]]
        else: self.w = w

        if b == None: self.b = [0.0]
        else: self.b = b

    def tensorFeed(self, x):
        return tf.add(tf.reduce_sum(tf.multiply(x, self.w)), self.b)

    def feed(self, x):
        session = tf.Session()
        with session.as_default():
            # Set parameters
            x = tf.constant(x)
            w = tf.constant(self.w)
            b = tf.constant(self.b)
            y = tf.add(tf.reduce_sum(tf.multiply(x, w)), b)
            return y.eval()

    def train(self, data, learn_rate=.001, epochs=1000, debug=False, debug_interval=2000,
              debug_final_loss=False, silence=False, debug_only_loss=False):
        # Tensorflow session
        session = tf.Session()

        # Clean data
        epochs = int(epochs)
        learn_rate = float(learn_rate)

        # Turn of all printing if silence is on (except debug_final_loss)
        if silence: debug = False

        # Parameters
        # Input
        x = tf.placeholder(tf.float32, [None, 3], name="x")

        # Weights
        w = tf.Variable(self.w)

        # Biases
        b = tf.Variable(self.b, dtype = tf.float32)

        # Calculated
        y = tf.add(tf.matmul(x, w, transpose_b = True), b)

        # Labels
        y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

        # Loss
        loss = tf.reduce_sum(tf.pow(tf.subtract(y, y_), 2)) * (1 / (2 * self.inps))

        # Optimizer
        train_step = tf.train.ProximalGradientDescentOptimizer(learn_rate).minimize(loss)

        # Initialize variables
        session.run(tf.global_variables_initializer())

        # Debug
        if not silence and not debug:
            print("TRAINING", end = "")
        STATUS_INTERVAL = epochs / 10

        # Train network
        with session.as_default():
            # Train 'epochs' times
            for i in range(epochs):
                # Get data
                batch_inps = data[0]
                batch_outs = data[1]
                # Debug printing
                if i % debug_interval == 0 and debug and debug_interval != -1:
                    if not debug_only_loss:
                        print("Weights\n{0}".format(w.eval()))
                        print("Biases\n{0}".format(b.eval()))
                    print("Loss :: {0}".format(loss.eval(feed_dict={x: batch_inps, y_: batch_outs})))
                    if not debug_only_loss:
                        print("\n\n")
                # Train
                session.run(train_step, feed_dict={x: batch_inps, y_: batch_outs})
                # Print status bar (debug)
                if not debug and not silence:
                    if i % STATUS_INTERVAL == 0: print(" * ", end="")

            # Debug
            if not silence and not debug:
                print("\nTRAINING COMPLETE")

            # Save weights and biases
            self.w = w.eval()
            self.b = b.eval()

# EXAMPLE
def test():
    # Get inputs
    xs = []
    for i in range(100):
        xs.append([randint(-100, 100) for i in range(3)])
    # Calculate outputs with error
    ys = []
    for x in xs:
        # Calculated expected
        calculated = float(2 * x[0] + 2 * x[1] - 4 * x[2] - 3)
        # Add some error
        withError = calculated + uniform(-1.0, 1.0)
        ys.append([withError])
    #Finalize dataset
    data = [xs, ys]

    # Create and train model
    LR = LinearReg(3)
    LR.train(data, learn_rate = .00001, epochs = 10000, debug = True, debug_interval = -1)

    # Perform input backprop
    i = IOA.IOA(LR.tensorFeed, 3)
    final, digest = i.optimize(100.0, epochs = -1, debug = True, debug_interval = -1,
                               returnDigest = True, learn_rate = .1, error_tolerance = .4)
    print(final)

if __name__ == "__main__":
    test()
