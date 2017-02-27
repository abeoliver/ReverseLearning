# Network Module
# Abraham Oliver, 2016
# ReverseLearning

# Import dependencies
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as TENSOR
import numpy as np
from random import randint
from time import time

# TODO Add support for custom activation, shaping, and loss functions

class Network (object):
    """
    A neural network using tensorflow for training with InputBackprop methods

    METHODS:
        - initValues    : initiate default weights and biases
        - intiWeights   : initiate weights with given paramaters
        - initBiases    : initiate biases with given paramaters
        - _clean         : _clean an input and fix it if wrong format
        - feed          : feed an input into the network
        - train         : train the network with given data
        - ibp           : perform the Input Backpropagation algorithm
        - eval          : evaluate a tensor with network session
        - storeParams   : store weights and biases as ndarrays instead of tensors

    ATTRIBUTES:
        - w             : the weights
        - b             : the biases
        - layers        : number of neurons in each layer
        - _session      : tensorflow session for network graph
    """
    def __init__(self, layers, activation = "none", shaping = "none",
                 customActivation = None, customShaping = None):
        """
        Initialize the network with default weights and biases, a given design, and a tensorflow session

        Parameters:
            layers (int[])  : a list of the number of neurons in each layer where layers[0] is the number of neurons
                                and layers[-1] is the number of output neurons
            activation  : activation function to use
                            - "none"            : no activiation function
                            - "sigmoid"         : sigmoid function
                            - "custom"          : apply a custom activation function
            shaping         : shaping function to use
                            - "none"            : no shaping function (DEFAULT)
                            - "softmax"         : the tensorflow softmax function
                            - "custom"          : apply a custom shaping function
        """
        # Clean input
        # Must be list or tuple
        if type(layers) != list and type(layers) != tuple:
            raise TypeError("Layers must be a tuple or a list but instead is a {0}".format(type(layers)))
        # Must not be empty
        if len(layers) == 0: raise ValueError("There must be more than zero layers")
        # Each layer must be an integer above zero
        for layerIndex in range(len(layers)):
            if type(layers[layerIndex]) != int: raise TypeError("Layer {0} must be an integer".format(layerIndex))
            if layers[layerIndex] <= 0: raise ValueError("Layer {0} must be above zero".format(layerIndex))

        # Clean shaping input
        if shaping in ["none", "softmax", "custom"]:
            self.shaping = shaping
            if customShaping != None:
                self.customShaping = customShaping
                self.shaping = "custom"
        else:
            raise TypeError("'{0}' is not a valid shaping function".format(shaping))

        # Clean activation input
        if activation in ["none", "sigmoid", "custom"]:
            self.activation = activation
            if customActivation != None:
                self.customActivation = customActivation
                self.activation = "custom"
        else:
            raise TypeError("'{0}' is not a valid activation function".format(activation))

        # Initialize network
        # Create a tensorflow session
        self._session = tf.Session()
        # Set layers
        self.layers = layers
        # Placeholders for weights and biases (will be set with initValues)
        self.w = np.array([0.0])
        self.b = np.array([0.0])
        # Initiate weights and biases with default parameters
        self.initValues()

    def initValues(self):
        """ Initializes network weights and biases with default parameters """
        self.initWeights()
        self.initBiases()

    def initWeights(self, mode = "zeros", mean = 0.0, stddev = 1.0, preset = []):
        """
        Initializes weights with either zeros, ones, randoms, or a preset set

        Arguments:
            - mode (string) : the type of initialization
                            - "zeros" : all zeros (DEFAULT)
                            - "ones" : all ones
                            - "random" : random values with mean and standard deviation  set
                            - "preset" : a predefined set of values, other modes
                                        overridden if present is given
        Keyword Arguments:
            - mean      : if using random generation, the mean
            - stddev    : if using random generation, the standard deviation
            - preset    : if using present values, the preset values

        Note:
            Preset must be of form [ [ w1,1  | [w2,1
                                       w1,2] |  w2,2] ]
            Which is equivalent to [[]]
        """
        # Clean input
        # Change mode to all lowercase
        mode = mode.lower()

        # Mean must be a number
        if type(mean) not in [int, float]: raise TypeError("Mean must be a number")
        # Cast mean to float
        mean = float(mean)

        # Sttdev must be a number
        if type(stddev) not in [int, float]: raise TypeError("Standard deviation must be a number")
        # Cast stddev to float
        stddev = float(stddev)

        if preset != []:
            # Preset must be a list, array, tuple, or tensor
            preset = self._clean(preset)
            w = []
            for i in preset:
                q = []
                for j in i:
                    q.append(np.float32(j))
                w.append(np.array(q))
            self.w = w
            return None
        elif mode in ["preset", "presets", "pre", "p"]:
            # If Mode is set to preset but no preset is given, raise error
            # Implied that preset wasn't given because the first if didn't trigger
            raise ValueError("If preset mode is set, a valid preset argument must be given")
        elif mode in ["random", "randoms", "rand", "r"]:
            # Create list of random weights tensors of correct shape for each layer
            self.w = [tf.random_normal([self.layers[n], self.layers[n + 1]], float(mean), float(stddev))
                      for n in range(len(self.layers) - 1)]
        elif mode in ["ones", "one", "o"]:
            # Create list of one-tensors of correct shape for each layer
            self.w = [tf.ones((self.layers[n], self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
        elif mode in ["zero", "zeros", "z"]:
            # Create list of zero-tensors of correct shape for each layer
            self.w = [tf.zeros((self.layers[n], self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
        else:
            raise ValueError("A valid mode must be given from random, ones, zeros, or preset")

        # Save weights
        self.w = [i.eval(session=self._session) for i in self.w]

    def initBiases(self, mode = "ones", mean = 0.0, stddev = 1.0, preset = []):
        """
        Initializes biases with either zeros, ones, randoms, or a preset set

        Arguments:
            - mode (string) : the type of initialization
                            - "zeros" : all zeros (DEFAULT)
                            - "ones" : all ones
                            - "random" : random values with mean and standard deviation  set
                            - "preset" : a predefined set of values, other modes
                                        overridden if present is given
        Keyword Arguments:
            - mean      : if using random generation, the mean
            - stddev    : if using random generation, the standard deviation
            - preset    : if using present values, the preset values
        """
        # Clean input
        # Change mode to all lowercase
        mode = mode.lower()

        # Mean must be a number
        if type(mean) not in [int, float]: raise TypeError("Mean must be a number")
        # Cast mean to float
        mean = float(mean)

        # Sttdev must be a number
        if type(stddev) not in [int, float]: raise TypeError("Standard deviation must be a number")
        # Cast stddev to float
        stddev = float(stddev)

        if preset != []:
            # Preset must be a list, array, tuple, or tensor
            # TODO Clean preset input
            # Preset must be a list, array, tuple, or tensor
            preset = self._clean(preset)
            b = []
            for i in preset:
                q = []
                for j in i:
                    q.append(np.float32(j))
                b.append(np.array(q))
            self.b = b
            return None
        elif mode in ["preset", "presets", "pre", "p"]:
            # If Mode is set to preset but no preset is given, raise error
            # Implied that preset wasn't given because the first if didn't trigger
            raise ValueError("If preset mode is set, a valid preset argument must be given")
        elif mode in ["random", "randoms", "rand", "r"]:
            # Create list of random tensors of correct shape for each layer
            self.b = [tf.random_normal([1, self.layers[n + 1]], float(mean), float(stddev))
                      for n in range(len(self.layers) - 1)]
        elif mode in ["ones", "one", "o"]:
            # Create list of one-tensors of correct shape for each layer
            self.b = [tf.ones((1, self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
        elif mode in ["zero", "zeros", "z"]:
            # Create list of zero-tensors of correct shape for each layer
            self.b = [tf.zeros((1, self.layers[n + 1]))
                      for n in range(len(self.layers) - 1)]
        else:
            raise ValueError("A valid mode must be given from random, ones, zeros, or preset")

        # Save biases
        self.b = [i.eval(session = self._session) for i in self.b]

    def _clean(self, input_vector):
        """Clean input for network functions"""
        ityp = type(input_vector)
        # All entries must be floats
        if ityp in [list, tuple, np.ndarray]:
            for i in range(len(input_vector)):
                if type(input_vector[i]) == int:
                    input_vector[i] = float(input_vector[i])
                elif type(input_vector[i]) == list or type(input_vector[i]) == np.ndarray or \
                                type(input_vector[i]) == tf.Variable or type(input_vector[i]) == tf.constant:
                    for j in range(len(input_vector[i])):
                        if type(input_vector[i][j]) == int: input_vector[i][j] = float(input_vector[i][j])
        # Input must be a list, array, or tensor
        if ityp not in [list, tuple, np.ndarray, tf.Variable, tf.constant]:
            if (ityp == int or ityp == float):
                input_vector = [[float(input_vector)]]
            else:
                raise TypeError("Input must be a list, array, or tensor of ints, floats, lists, arrays, or tensors")
        # Input must be a list, array, or tensor of lists, arrays, or tensors
        ityp2 = type(input_vector[0])
        if ityp2 not in [list, tuple, np.ndarray, tf.Variable, tf.constant]:
            if ityp2 in [int, float, np.int32, np.float32, np.int64, np.float64]:
                input_vector = [input_vector]
            else:
                raise TypeError("Input must be a list, array, or tensor of ints, floats, lists, arrays, or tensors")

        # Finally, _clean returned input
        return input_vector

    def feed(self, input_vector, evaluate = True):
        """
        Feed-forward input_vector through network

        Parameters:
            - input_vector (Tensor, Nd-array,  list) : input_vector to feed through
            - evaluate (bool): evaluate output tensor. Yes or no?
        """
        # Predicted output
        def _calc(inp, n=0):
            """Recursive function for feeding through layers"""
            # End recursion
            if n == len(self.layers) - 2:
                # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
                calculated = tf.matmul(inp, self.w[n], name="mul{0}".format(n)) + self.b[n]
                # Apply activation if set
                if self.activation == "sigmoid":
                    return tf.sigmoid(calculated)
                else:
                    return calculated
            # Continue recursion
            calculated = tf.matmul(inp, self.w[n], name="mul{0}".format(n)) + self.b[n]
            # Apply activation if set
            if self.activation == "sigmoid":
                return _calc(tf.sigmoid(calculated), n + 1)
            elif self.activation == "custom":
                return _calc(self.customActivation(calculated))
            else:
                return _calc(calculated, n + 1)

        # Clean input_vector
        # input_vector = self._clean(input_vector)

        # Shape output
        def calc(inp):
            if self.shaping == "softmax":
                y = tf.nn.softmax(_calc(inp))
            elif self.shaping == "custom":
                y = self.customShaping(_calc(inp))
            else:
                y = _calc(inp)
            return y

        # Calculate
        y = calc(input_vector)

        # Begin and return recursively calculated output
        if evaluate:
            return self.eval(y)
        else:
            return calc(y)

    def train(self, data, learn_rate = .001, epochs = 1000, batch_size = 0,
              loss_function = "mean_squared", debug = False, debug_interval = 2000,
              debug_final_loss = False, silence = False, debug_only_loss = False,
              customLoss = None):
        """
        Train the network using given data

        Parameters:
            data        : inputs and labels in format [inputs, labels]
            epochs      : number of epochs to run (DEFAULT 1000)
            batch_size  : size of each minibatch (leave 0 for full dataset)
            loss_function : loss function to use
                            - "mean_squared"    : average of the squares of the errors (DEFAULT)
                            - "cross_entropy"   : cross entropy function
                            - "custom"          : custom loss function
            debug       : on / off debug mode
            debug_interval : number of epochs between debugs
            debug_final_loss : print the final accuracy of the network (debug does not
                                        have to be enabled)
            debug_only_loss : don't print weights and biases on debug (debug must be enabled)
            silence     : print NOTHING (debug_final_loss is exception)
        """
        # TODO Clean data for training
        # TODO Clean training parameters
        # Clean data
        epochs = int(epochs)
        learn_rate = float(learn_rate)

        # Turn of all printing if silence is on (except debug_final_loss)
        if silence: debug = False
        
        # Parameters
        # Input
        x = tf.placeholder(tf.float32, [None, self.layers[0]], name = "x")
        
        # Weights
        w = [tf.Variable(self.w[i], name = "w") for i in range(len(self.w))]
        
        # Biases
        b = [tf.Variable(self.b[i], name = "b") for i in range(len(self.b))]
        
        # Predicted output
        def _calc(inp, n=0):
            """Recursive function for feeding through layers"""
            # End recursion
            if n == len(self.layers) - 2:
                # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
                calculated = tf.matmul(inp, w[n], name = "mul{0}".format(n)) + b[n]
                # Apply activation if set
                if self.activation == "sigmoid":
                    return tf.sigmoid(calculated)
                else:
                    return calculated
            # Continue recursion
            calculated = tf.matmul(inp, w[n], name = "mul{0}".format(n)) + b[n]
            # Apply activation if set
            if self.activation == "sigmoid":
                return _calc(tf.sigmoid(calculated), n + 1)
            elif self.activation == "custom":
                return _calc(self.customActivation(calculated))
            else:
                return _calc(calculated, n + 1)

        def calc(inp):
            # Shape output
            if self.shaping == "softmax":
                y = tf.nn.softmax(_calc(inp))
            elif self.shaping == "custom":
                y = self.customShaping(_calc(inp))
            else:
                y = _calc(inp)
            return y

        # Get calculated
        y = calc(x)
        
        # Labels
        if self.activation == "sigmoid":
            y_ = tf.sigmoid(tf.placeholder(tf.float32, [None, self.layers[-1]], name = "y_"))
        elif self.activation == "custom":
            y_ = self.customActivation(tf.placeholder(tf.float32, [None, self.layers[-1]], name = "y_"))
        else:
            y_ = tf.placeholder(tf.float32, [None, self.layers[-1]], name = "y_")

        # Loss function TODO Add more loss functions
        if loss_function == "cross_entropy":
            loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        if loss_function == "custom" or customLoss != None:
            loss = customLoss(y, y_)
        else:
            loss = tf.reduce_mean(tf.pow(y_ - y, 2))

        # Optimizer
        train_step = tf.train.ProximalGradientDescentOptimizer(learn_rate).minimize(loss)

        # Minibatch creation
        def newBatch(bSize = batch_size):
            # If batch size is zero, use full dataset
            if bSize == 0:
                return data
            else:
                # Randomly choose inputs and corresponding labels for batches
                ins = []
                lbls = []
                for i in range(bSize):
                    r = randint(0, len(data[0]) - 1)
                    ins.append(data[0][r])
                    lbls.append(data[1][r])
                return [ins, lbls]

        # Initialize variables
        self._session.run(tf.global_variables_initializer())

        # Debug
        if not silence and not debug:
            print("TRAINING", end = "")
        STATUS_INTERVAL = epochs / 10

        # Train network
        with self._session.as_default():
            # Train 'epochs' times
            for i in range(epochs):
                # Get data
                batch_inps, batch_outs = newBatch()
                # Debug printing
                if i % debug_interval == 0 and debug:
                    if not debug_only_loss:
                        print("Weights ::")
                        for j in w:
                            print(j.eval())
                        print("Biases ::")
                        for j in b:
                            print(j.eval())
                    print("Loss :: {0}".format(loss.eval(feed_dict={x: batch_inps, y_: batch_outs})))
                    if not debug_only_loss:
                        print("\n\n")
                # Train
                self._session.run(train_step, feed_dict = {x: batch_inps, y_:batch_outs})
                # Print status bar (debug)
                if not debug and not silence:
                    if i % STATUS_INTERVAL == 0: print(" * "),

            # Debug
            if not silence and not debug:
                print("\nTRAINING COMPLETE")

            if debug_final_loss:
                batch = newBatch(0)
                print("Final Loss :: {0}".format(loss.eval(feed_dict={x: batch[0], y_: batch[1]})))

            # Save weights and biases
            self.w = [i.eval() for i in w]
            self.b = [i.eval() for i in b]

    def eval(self, tensor, feed_dict= {}):
        """
        Evaluate a tensor using network tensorflow session

        Parameters:
            - tensor    : tensor to be evaluated
            - feed_dict : Feed dictionary for placeholders (only if tensor is a placeholder)

        Return:
            Evaluated tensor as a numpy array
        """
        if type(tensor) == list:
            final = []
            for i in tensor:
                final.append(i.eval(session = self._session, feed_dict = feed_dict))
            return final
        elif type(tensor) == np.ndarray:
            return tensor
        elif type(tensor) == TENSOR:
            return tensor.eval(session = self._session, feed_dict = feed_dict)
        else:
            raise TypeError("Not a tensor")
