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
        # TODO self.w and self.b needed
        self.layers = layers
        self.initValues()

    def initValues(self):
        """
        Initializes network weights and biases
        """
        self.initWeights()
        # TODO biases

    def initWeights(self, mode = "zeros", mean = 0, stddev = 1, preset = []):
        """
        Initializes weights with either zeros, ones, randoms, or a preset set

        Parameters:
            - mode (string) : the type of initialization (default "zeros")
                                - "zeros" : all zeros
                                - "ones" : all ones
                                - "random" : random values with mean and standard deviation  set
                                - "preset" : a predefined set of values, mode is overridden if present is given
        """
        # TODO actually write this function
        if preset != []:
            # Check if compatible shape
            pass
        elif mode == "random":
            # Random
            pass
        elif mode == "ones":
            # All ones
            pass
        else:
            # Zeros
            pass

    def feed(self, input):
        """
        Feed-forward input through network

        Parameters:
            - input (Tensor, Nd-array,  list) : input to feed through
        """
        # TODO clean input
        # TODO test this
        def calc(inp, n = 0):
            """Recursive function for feeding through layers"""
            if n == len(self.layers) - 2:
                return tf.matmul(inp, self.w[n]) + self.b[n]
            return calc(tf.matmul(inp, self.w[n]) + self.b[n], n + 1)
        return calc(input)