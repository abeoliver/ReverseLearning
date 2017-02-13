# Input Optimization Algorithm
# ReverseLearning, 2017

# TODO Test ranging
# TODO Random start

import tensorflow as tf
from Network import Network
import numpy as np
from time import time

# Suppress warnings
from warnings import filterwarnings
filterwarnings("ignore")

class IOA:
    def __init__(self, model, ins):
        self.model = model
        self.ins = ins

    def clean(self, inp):
        """Cleans an input"""
        return inp

    def optimize(self, target, epochs = 1000, learn_rate = .01, debug = False,
                 loss_function="absolute_distance", restrictions = {}, debug_interval = -1,
                 error_tolerance = None, rangeGradientScalar = 10e10, gradientTolerance = 0.0,
                 evaluate = True):
        """
        Applies the Input Backprop Algorithm and returns an input with
        a target output

        Parameters:
            target          : target value
                            - "max" for maximum value
                            - "min" for minimum value
            epochs          : number of epochs to run (DEFAULT 1000)
                            - (-1) runs until error is below learning rate
            loss_function   : loss function to use (DEFAULT absolute distance)
                            - "absolute_distance" : absolute difference between label and output
                            - "cross_entropy"   : cross entropy function
                            - "quadratic_distance" : absolute distance squared
            debug       : on / off debug mode
            restrictions : a dictionary of range and type restrictions for the optimal
                         - Format: {index0 : restriction0, ..., indexN : restrictionN}
                         - For constant value: restrictionX = value
                         - For range: restricionX = (lower, upper)
            debug_interval : number of epochs between each debug statement
                            - Use a negative number to only print ending statement
            error_tolerance : the largest acceptable error before auto-breaking
                            (default is learn_rate)
            rangeGradientScalar : scalar for the gradients of the range-restricted vars
        """

        # Start a tensorflow session
        sess = tf.Session()

        # Reason for breaking the loop (zero grad, finished epocs, etc.)
        breakReason = None

        # Clean inputs
        # Ensure thar epochs is an integer
        epochs = int(epochs)
        # Ensure that learning rate is a float
        learn_rate = float(learn_rate)
        # If the target us a string (ie. max or min), make sure all lowercase and valid
        # Otherwise clean input
        if type(target) == str:
            target = target.lower()
            if target not in ["max", "min"]: raise ValueError("'{0}' is not a valid target".format(target))
        else:
            try: target = self.clean(target)
            except: raise ValueError("'{0}' is not a valid target".format(target))
        # If the error tolerance wasn't set, set it to the learning rate
        if error_tolerance == None: error_tolerance = learn_rate

        # Get and format the range-restricted restrictions
        rangeRestrictedIndexes = []
        for i in restrictions.keys():
            if type(restrictions[i]) in [list, tuple]:
                rangeRestrictedIndexes.append(i)

        # - DEFINE PARAMETERS -
        # Input
        # Start with all variables at 0
        startOptimal = [[tf.Variable(0.0) for i in range(self.ins)]]

        # Apply constant restrictions to startOptimal and collect restricted vars
        rangeRestrictedVars = []
        for k in restrictions.keys():
            if type(restrictions[k]) in [float, int]:
                # Apply constant
                startOptimal[0][k] = tf.constant(float(restrictions[k]))
            elif type(restrictions[k]) in [list, tuple]:
                rangeRestrictedVars.append(startOptimal[0][k])


        # Get the range-restriction vectors for startOptimal
        rangeRestrictedVectors = self._getRestrictionVectors(restrictions, startOptimal)

        # Tensor of the lowest gradients for use when checking if max / min / best is found
        lowGrad = self._getLowGrad(startOptimal, gradientTolerance)

        # Finalize optimal
        optimal = self._applyRestrictionVector(startOptimal, rangeRestrictedVectors)

        # Calculate output from the model (restrictions applied)
        out = self.model(optimal)

        # Target label
        # If the target is max or min, don't set label
        if target in ["max", "min"]: label = None
        else: label = tf.constant(target)

        # Loss function
        loss = self._getLossFunction(loss_function, target, label, out)

        # Get variables (exclude constants)
        vlist = self._getVarList(startOptimal)
        # Create an optimizer of the given learning rate
        optimizer = tf.train.ProximalGradientDescentOptimizer(learn_rate)
        # Get the gradients from the loss function for each variable
        gradients = optimizer.compute_gradients(loss, var_list = vlist)

        # Raise range-restricted variables
        newGrads = [self._raiseGrad(g, rangeGradientScalar)
                    if g[1] in rangeRestrictedVars else g
                    for g in gradients]

        # Gradient application
        applyNewGrads = optimizer.apply_gradients(newGrads)

        # Get the absolute error
        if target in ["max", "min"]:
            absoluteError = tf.constant(0.0)
        else:
            absoluteError = tf.abs(tf.sub(label, out))

        # Initialize the computation graph
        sess.run(tf.initialize_all_variables())

        # - TRAIN -
        # A counter for counting epochs
        counter = 0
        # If debug is on, print intial debug report
        if debug and debug_interval > 0:
            self._printDebugStatus(sess, epochs = counter, startOptimal = startOptimal, optimal = optimal)

        # The main traing loop
        while True:
            # Start timer (for DEBUG profiling)
            time0 = time()

            # Break if error is 0 or within learning rate of zero
            absoluteErrorEvaluated = absoluteError.eval(session = sess)
            # If absolute error is a single number, put it in a list
            if type(absoluteErrorEvaluated) not in [list, tuple, np.ndarray]:
                absoluteErrorEvaluated = [absoluteErrorEvaluated]
            if sum(absoluteErrorEvaluated) <= error_tolerance \
                    and target not in ["max", "min"]:
                breakReason = "Beat Error"
                break

            # Break if gradients are all zero
            gradCheck = self._checkGradients(newGrads, lowGrad, sess)
            if gradCheck:
                breakReason = gradCheck
                break

            # Break if epochs limit reached
            if counter >= epochs and epochs != -1:
                breakReason = "Epoch Limit Reached"
                break

            # Apply training step to find optimal
            sess.run(applyNewGrads)

            # Increment counter
            counter += 1

            # Debug printing
            if counter % debug_interval == 0 and debug and debug_interval > 0:
                # Dont show error for max and min
                if target == "max" or target == "min": absErrorDebug = None
                else: absErrorDebug = absoluteErrorEvaluated
                self._printDebugStatus(sess, epochs = counter, startOptimal = startOptimal,
                                       optimal = optimal, absoluteError = absErrorDebug,
                                       timer = time() - time0, gradients = newGrads)

        # Print final digest
        if debug:
            print("\nOPTIMAL INPUT       :: {0}".format([i.eval(session = sess) for i in optimal[0]]))
            print("CALCULATED OUT      :: {0}".format(self.model(optimal).eval(session = sess)))
            if label != None:
                print("TARGET OUT          :: {0}".format(label.eval(session = sess)))
            print("ERROR               :: {0}".format(absoluteError.eval(session = sess)))
            print("EPOCHS              :: {0} ({1})".format(counter, breakReason))

        # If evaluation is requested, returned evaluated
        # Don't evaluate if not
        if evaluate: return optimal
        else: return optimal

    def feed(self, input_vector):
        return self.model(input_vector)

    def _getRestrictionVectors(self, restrictions, vars):
        rVector = [[], []]

        for i in range(len(vars[0])):
            if i in restrictions.keys():
                if type(restrictions[i]) in [list, tuple]:
                    rVector[0].append(tf.cast(restrictions[i][0], tf.float32))
                    rVector[1].append(tf.cast(restrictions[i][1], tf.float32))
            else:
                rVector[0].append(None)
                rVector[1].append(None)
        return rVector

    def _applyRestrictionVector(self, inputs, restrictVector):
        # Restriction reshaping function
        def restrict(x, b, t):
            q = tf.nn.sigmoid(x)
            w = tf.mul(q, tf.cast(tf.sub(t, b), tf.float32))
            return tf.add(w, b, name = "restricted")

        optimal = [[]]
        for i in range(len(inputs[0])):
            if restrictVector[0][i] != None:
                optimal[0].append(restrict(inputs[0][i], restrictVector[0][i], restrictVector[1][i]))
            else:
                optimal[0].append(inputs[0][i])

        return optimal

    def _raiseGrad(self, grad, scaler):
        """
        Scale a gradient
        Parameter:
            grad : (gradient, variable)
            scaler : scaler to raise gradient by
        """
        return (tf.mul(grad[0], scaler), grad[1])

    def _getLossFunction(self, requested, target, label, out):
        # Use variations / throw errors for max and min
        if requested == "cross_entropy":
            if target == "max":
                raise ValueError("Target 'max' and 'cross_entropy' loss are not compatible")
            elif target == "min":
                raise ValueError("Target 'min' and 'cross_entropy' loss are not compatible")
            else:
                return tf.reduce_mean(-tf.reduce_sum(label * tf.log(out), reduction_indices=[1]))
        elif requested == "quadratic":
            if target == "max":
                raise ValueError("Target 'max' and 'quadratic' loss are not compatible")
            elif target == "min":
                raise ValueError("Target 'min' and 'quadratic' loss are not compatible")
            else:
                return tf.reduce_sum(tf.pow(label - out, 2))
        else:
            if target == "max":
                return tf.mul(-1.0, out)
            elif target == "min":
                return out
            else:
                return tf.abs(label - out)

    def _getVarList(self, vector):
        # Start list
        vlist = []
        # Iterate through the given vector
        for i in vector:
            # If it's another list, iterate through one more dimension
            if type(i) in [list, tuple]:
                for j in i:
                    # If it's a variable, add it to the list
                    if type(j) == tf.Variable:
                        vlist.append(j)
            else:
                # If it's a variable, add it to the list
                if type(i) == tf.Variable:
                    vlist.append(i)
        return vlist

    def _printDebugStatus(self, session, epochs = None, startOptimal = None,
                          optimal = None, absoluteError = None,
                          timer = None, gradients = None):
        """Prints the debug information during training"""
        if epochs != None:
            print "@ Epoch {0}".format(epochs)
        if startOptimal != None:
            # Evaluate optimal
            op = []
            for i in startOptimal[0]:
                op.append(i.eval(session = session))
            if len(op) == 1:
                print "Value        :: {0}".format(op[0])
            else:
                print "Value        :: {0}".format(op)
        if optimal != None:
            q = []
            for i in optimal[0]:
                q.append(i.eval(session = session))
            if len(q) == 1:
                print "Restricted   :: {0}".format(q[0])
            else:
                print "Restricted   :: {0}".format(q)
            fed = self.model(q).eval(session = session)
            if type(fed) in [list, tuple, np.array]:
                if len(fed) == 1:
                    print "Evaluated    :: {0}".format(fed[0])
            else:
                print "Evaluated    :: {0}".format(fed)
        if absoluteError != None:
            if type(absoluteError) in [list, tuple]:
                print "Error        :: {0}".format(absoluteError[0])
                if len(absoluteError) != 1:
                    # If the error is only one number then total error is not needed
                    print "Total Error  :: {0}".format( sum(absoluteError))
            else:
                print "Error        :: {0}".format(absoluteError)
        if timer != None:
            print "Time         :: {0}".format(timer)
        if gradients != None:
            print "Gradients    :: {0}".format([g[0].eval(session = session) for g in gradients])
        print ""

    def _checkGradients(self, gradients, checkAgainst, sess):
        # Get gradients
        grads = [abs(p[0]) for p in gradients]
        # Check for infinite or nan grads
        for g in grads:
            if sess.run(tf.is_nan(g)): return "NaN Gradients"
            elif sess.run(tf.is_inf(g)): return "Inf Gradients"
        # Check for low grads
        areSame = sess.run(tf.less_equal(grads, checkAgainst))
        # Check each truth value, add to counter if true
        low = 0
        for g in areSame:
            if g: low += 1
        # If all are the same, return true, else false
        if low == len(grads): return "Low Gradients"
        else: return False

    def _getLowGrad(self, optimal, gradTolerance):
        """Creates a list of the lowest allowed gradient for every variable in optimal for lowest-grad checking"""
        zs = []
        for i in optimal[0]:
            if type(i) == tf.Variable:
                zs.append(tf.constant(gradTolerance))
        return zs

class Models:
    def f1(self, x):
        return tf.reduce_sum(x)
    def f2(self, x):
        return tf.add(tf.add(-tf.square(x), tf.mul(4.0, x)), 8.0)
    def f3(self, x):
        """Not differentiable for x <= 0"""
        return tf.pow(x, tf.div(1.0, 2.0))

def test():
    # Example model
    a = Models()

    # Input Optimization
    I = IOA(a.f2, 1)
    I.optimize("max", epochs = -1, learn_rate = .1, error_tolerance = .2,
                 restrictions = {}, debug = True, debug_interval = 100,
                 rangeGradientScalar = 1e11, gradientTolerance = 5e-7)

if __name__ == "__main__":
    test()