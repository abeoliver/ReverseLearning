# Network Module
# Abraham Oliver, 2016
# ReverseLearning

# To Do
# TODO Design and implement exceptions

# Import dependencies
import tensorflow as tf
import numpy as np

class Network (object):
    """
    A neural network using tensorflow for training with InputBackprop methods

    METHODS:
        -

    ATTRIBUTES:
        -
    """
    def __init__(self, layers):
        """
        Initialize the network

        Parameters:
            - layers (int[]) : a list of the number of neurons in each layer where layers[0] is the number of neurons
                                and layers[-1] is the number of output neurons

        Notes:
            - Weights are initialized as zeros
            - Biases are initialized as ones
        """
        # TODO add to  __init__ docstring
        # TODO clean __init__ input
        self._session = tf.Session()
        self.layers = layers
        self.w = np.array([0.0])
        self.b = np.array([0.0])
        self.initValues()

    def initValues(self):
        """
        Initializes network weights and biases
        """
        self.initWeights()
        self.initBiases()

    def initWeights(self, mode = "zeros", mean = 0.0, stddev = 1.0, preset = []):
        """
        Initializes weights with either zeros, ones, randoms, or a preset set

        Parameters:
            - mode (string) : the type of initialization (default "zeros")
                                - "zeros" : all zeros
                                - "ones" : all ones
                                - "random" : random values with mean and standard deviation  set
                                - "preset" : a predefined set of values, mode is overridden if present is given
        """
        # TODO add  initWeights options code
        # TODO fix  initWeights docstring
        # TODO clean initWeights input
        if preset != []:
            # Check if compatible shape
            pass
        elif mode == "random":
            # Random
            self.w = [tf.random_normal(mean, stddev, (self.layers[n], self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
        elif mode == "ones":
            # All ones
            self.w = [tf.ones((self.layers[n], self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
        else:
            # Zeros
            self.w = [tf.zeros((self.layers[n], self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
            #self.w = np.array(w)

    def initBiases(self):
        """"""
        # TODO write initBiases docstring
        # TODO add initBiases options
        self.b = [tf.ones([self.layers[n + 1]]) for n in range(len(self.layers) - 1)]
        #self.b = np.array(b)

    def clean(self, input_vector):
        """Clean input"""
        # All entries must be floats
        if type(input_vector) == list or type(input_vector) == np.ndarray:
            for i in range(len(input_vector)):
                if type(input_vector[i]) == int:
                    input_vector[i] = float(input_vector[i])
                elif type(input_vector[i]) == list or type(input_vector[i]) == np.ndarray or \
                                type(input_vector[i]) == tf.Variable or type(input_vector[i]) == tf.constant:
                    for j in range(len(input_vector[i])):
                        if type(input_vector[i][j]) == int: input_vector[i][j] = float(input_vector[i][j])
        # Input must be a list, array, or tensor
        ityp = type(input_vector)
        if ityp != list and ityp != np.ndarray and ityp != tf.Variable and ityp != tf.constant:
            if (ityp == int or ityp == float) and autocorrect:
                input_vector = [[float(input_vector)]]
            else:
                raise TypeError  # TODO Error to replace
        # Input must be a list, array, or tensor of lists, arrays, or tensors TODO Error to replace
        ityp2 = type(input_vector[0])
        if ityp2 != list and ityp2 != np.ndarray and ityp2 != tf.Variable and ityp2 != tf.constant:
            if ityp2 == int or ityp2 == float:
                input_vector = [input_vector]
            else:
                raise TypeError  # TODO Error to replace

        # Finally, clean returned input
        return input_vector

    def feed(self, input_vector, autocorrect = True):
        """
        Feed-forward input_vector through network

        Parameters:
            - input_vector (Tensor, Nd-array,  list) : input_vector to feed through
            - autocorrect (bool) : if true, function attempts to fix errors and run (if it can't fix, error),
                                    if false, function throws exception
        """
        # Recursive calculation function
        def calc(inp, n=0):
            """Recursive function for feeding through layers"""
            if n == len(self.layers) - 2:
                return tf.matmul(inp, self.w[n]) + self.b[n]
            return calc(tf.matmul(inp, self.w[n]) + self.b[n], n + 1)

        # Clean input_vector
        self.clean(input_vector)

        # Begin and return recursively calculated output
        # TODO same shaping function for feed and train
        return calc(input_vector)

    def train(self, data, epochs, learn_rate, loss = "mean_squared",
              optimizer = "proximal_gradient", debug = False, debug_interval = 2000):
        # TODO Finish basic train function
        # TODO Break train into smaller functions
        # TODO Clean data function
        # Clean data
        epochs = int(epochs)
        learn_rate = float(learn_rate)
        # Parameters
        # Input
        x = tf.placeholder(tf.float32, [None, self.layers[0]], name = "x")
        # Weights
        w = [tf.Variable(self.w[i], name = "w") for i in range(len(self.w))]
        # Biases
        b = [tf.Variable(self.b[i], name = "b") for i in range(len(self.b))]
        self._session.run(tf.initialize_all_variables())
        # Predicted output
        def calc(inp, n=0):
            """Recursive function for feeding through layers"""
            if n == len(self.layers) - 2:
                return tf.matmul(inp, w[n], name = "mul{0}".format(n)) + b[n]
            return calc(tf.matmul(inp, w[n], name = "mul{0}".format(n)) + b[n], n + 1)
        # TODO shaping functions
        y = calc(x)
        # Labels
        y_ = tf.placeholder(tf.float32, [None, self.layers[-1]], name = "y_")

        # Loss function TODO Add more loss functions
        # loss = tf.reduce_mean(tf.pow(y_ - y, 2))
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        # Optimizer TODO Add more optimizers
        train_step = tf.train.ProximalGradientDescentOptimizer(learn_rate).minimize(loss)

        # TODO Minibatches
        # Initialize variables
        self._session.run(tf.initialize_all_variables())
        print "TRAINING",
        STATUS_INTERVAL = epochs / 10
        with self._session .as_default():
            for i in range(epochs):
                # Get data
                batch_inps, batch_outs = data
                # Debug printing
                if i % debug_interval == 0 and debug:
                    print("Weights ::")
                    for i in w:
                        print(i.eval())
                    print("Biases ::")
                    for i in b:
                        print(i.eval())
                    print("Loss :: {0}".format(loss.eval(feed_dict={x: batch_inps, y_: batch_outs})))
                    #print("OUT :: {0}".format(y.eval(feed_dict = {x: batch_inps, y_:batch_outs})))
                    print("\n\n")
                self._session.run(train_step, feed_dict = {x: batch_inps, y_:batch_outs})
                # Print status bar
                if i % STATUS_INTERVAL == 0 and not debug: print" * ",
            print("\nTRAINING COMPLETE")
            # Save weights and biases
            # Turn variables into ndarrays
            save_w = [i.eval() for i in w]
            save_b = [i.eval() for i in b]
            self.w = np.array(save_w)
            self.b = np.array(save_b)

    def ibp(self, target, epochs = 1000, learn_rate = .01):
        """Applies the Input Backprop Algorithm and returns an input with
        a target output

        TODO:
            Add more options
        """
        # Clean taget
        target = self.clean(target)
        # Define paramaters
        # Input
        optimal = tf.Variable(tf.zeros([1, self.layers[0]]))
        # Input Weights
        w = [tf.constant(i) for i in self.w]
        # Input Biases
        b = [tf.constant(i) for i in self.b]
        # Output
        def calc(inp, n=0):
            """Recursive function for feeding through layers"""
            if n == len(self.layers) - 2:
                return tf.matmul(inp, self.w[n]) + self.b[n]
            return calc(tf.matmul(inp, self.w[n]) + self.b[n], n + 1)
        out = calc(optimal)
        # Label
        lbl = tf.constant(target)

        # Training with quadratic cost and gradient descent with learning rate .01
        loss = tf.reduce_sum(tf.abs(lbl - out))
        train_step = tf.train.ProximalGradientDescentOptimizer(learn_rate).minimize(loss)

        # Initialize
        self._session.run(tf.initialize_all_variables())
        # Train to find three inputs
        for i in range(epochs):
            self._session.run(train_step)

        print("OPTIMAL INPUT       :: {0}".format(optimal.eval(session = self._session)))
        print("CALCULATED OUT      :: {0}".format(calc(optimal.eval(session = self._session)).eval(session = self._session)))
        print("TARGET OUT          :: {0}".format(target))
        print("TARGET vs CALC LOSS :: {0}".format(loss.eval(session = self._session)))
