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
            self.w = [np.random.normal(mean, stddev, (self.layers[n], self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
        elif mode == "ones":
            # All ones
            pass
        else:
            # Zeros
            self.w = [np.zeros((self.layers[n], self.layers[n + 1]), dtype = np.float32)
                      for n in range(len(self.layers) - 1)]

    def initBiases(self):
        """"""
        # TODO write initBiases docstring
        # TODO add initBiases options
        self.b = [np.ones([self.layers[n + 1]]) for n in range(len(self.layers) - 1)]

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
        # All entries must be floats
        if type(input_vector) == list or type(input_vector) == np.ndarray:
            for i in range(len(input_vector)):
                if type(input_vector[i]) == int: input_vector[i] = float(input_vector[i])
                elif type(input_vector[i]) == list or type(input_vector[i]) == np.ndarray or \
                    type(input_vector[i]) == tf.Variable or type(input_vector[i]) == tf.constant:
                    for j in range(len(input_vector[i])):
                        if type(input_vector[i][j]) == int: input_vector[i][j] = float(input_vector[i][j])
        # Input must be a list, array, or tensor
        ityp = type(input_vector)
        if ityp != list and ityp != np.ndarray and ityp != tf.Variable and ityp != tf.constant:
            if (ityp == int or ityp == float) and autocorrect: input_vector = [[float(input_vector)]]
            else: raise TypeError # TODO Error to replace
        # Input must be a list, array, or tensor of lists, arrays, or tensors TODO Error to replace
        ityp2 = type(input_vector[0])
        if ityp2 != list and ityp2 != np.ndarray and ityp2 != tf.Variable and ityp2 != tf.constant:
            if ityp2 == int or ityp2 == float: input_vector = [input_vector]
            else: raise TypeError # TODO Error to replace

        # Begin and return recursively calculated output
        return calc(input_vector)

    def train(self, data, epochs, learn_rate, miniBatches = 0, loss = "mean_squared",
              optimizer = "proximal_gradient", debug = False, debug_interval = 2000):
        # TODO Finish basic train function
        # Loss function TODO Add more loss functions
        loss = tf.reduce_mean(tf.pow(y_ - y, 2))

        # Optimizer TODO Add more optimizers

        # Initialize variables
        sess.run(tf.initialize_all_variables())

        print "TRAINING",

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