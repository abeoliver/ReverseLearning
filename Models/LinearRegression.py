# Linear Regression Model
# Reverse Learning, 2017

# Import dependencies
import tensorflow as tf
from random import randint

class LinearReg:
    def __init__(self, inps):
        self.inps = inps
        self.w = [2.0 for i in range(inps)]
        self.b = [3.0]

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
        x = tf.placeholder(tf.float32, [None, self.inps], name="x")

        # Weights
        w = tf.Variable(tf.ones([self.inps, 1]))

        # Biases
        b = tf.Variable(1.0)

        # Calculated
        y = tf.add(tf.matmul(x, w), b)

        # Labels
        y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

        # Loss
        loss = tf.reduce_sum(tf.pow(tf.subtract(y, y_), 2)) * (1/(2*self.inps))

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
                batch_inps = [d[0] for d in data]
                batch_outs = [d[1] for d in data]
                # Debug printing
                if i % debug_interval == 0 and debug:
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
    domain = []
    for i in range(100):
        domain.append([randint(-100, 100) for i in range(3)])
    data = [([float(d[0]), float(d[1]), float(d[2])],
             [float(2 * d[0] + 2 * d[1] - 4 * d[2] - 3)])
            for d in domain]
    LR = LinearReg(3)
    LR.train(data, learn_rate = .00001, epochs = 100000, debug = True, debug_interval = 10000)
    print(LR.w)
    print(LR.b)
    print(LR.feed((2.0, 1.0)))

if __name__ == "__main__":
    test()