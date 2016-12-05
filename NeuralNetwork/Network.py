# Network Module
# Abraham Oliver, 2016
# ReverseLearning

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
        # TODO add to docstring
        self._session = tf.Session()
        self.layers = layers
        self.w = np.array([0])
        self.b = np.array([0])
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
        # TODO add options code
        # TODO fix docstring
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
            self.w = [np.zeros((self.layers[n], self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]

    def initBiases(self):
        """"""
        # TODO write docstring
        # TODO add options
        self.b = [np.ones([self.layers[n + 1]]) for n in range(len(self.layers) - 1)]

    def feed(self, input):
        """
        Feed-forward input through network

        Parameters:
            - input (Tensor, Nd-array,  list) : input to feed through
        """
        # TODO clean input
        # TODO DEBUG
        def calc(inp, n = 0):
            """Recursive function for feeding through layers"""
            if n == len(self.layers) - 2:
                return tf.matmul(inp, self.w[n]) + self.b[n]
            return calc(tf.matmul(inp, self.w[n]) + self.b[n], n + 1)
        return calc(input)