# Input Optimization Algorithm
# ReverseLearning, 2017

import tensorflow as tf
from Network import Network
import numpy as np

class InputOptimizer:
    def __init__(self, model, ins, outs):
        self.model = model
        self.ins = ins
        self.outs = outs
        self.session = tf.Session()

    def clean(self, inp):
        """Cleans an input"""
        return inp

    def optimize(self, target, epochs = 1000, learn_rate = .01, debug = False,
            loss_function="absolute_distance", shaping="none", activation="none",
            restrictions = {}, debug_interval = -1, error_tolerance = None,
            rangeGradientScalar = 100000000.0, evaluate = True):
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
            activation  : activation function to use
                            - "none"            : no activiation function
                            - "sigmoid"         : sigmoid function
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
        # If the error tolerance wasn't, set it to the learning rate
        if error_tolerance == None: error_tolerance = learn_rate

        # Get and format the range-restricted restrictions
        rangeRestrictedIndexes = []
        for i in restrictions.keys():
            if type(restrictions[i]) in [list, tuple]:
                rangeRestrictedIndexes.append(i)

        # - DEFINE PARAMETERS -
        # Input
        # Start with all variables at 0
        optimal = [[tf.Variable(0.0) for i in range(self.layers[0])]]
        # Apply constant restrictions to optimal
        for k in restrictions.keys():
            if type(restrictions[k]) in [float, int]:
                optimal[0][k] = tf.constant(float(restrictions[k]))

        # Get the range-restricted variables
        rangeRestrictedVars = []
        for index in rangeRestrictedIndexes:
            rangeRestrictedVars.append(optimal[0][index])

        # For checking if all gradients are zero
        zeroGrad = [tf.constant(0.0) for v in optimal[0]]

        # Calculate output from the model
        out = self.model.feed(optimal)

        # Target label
        # If the target is max or min, don't set label
        if target in ["max", "min"]: label = None
        else: label = tf.constant(target)

        # Loss function
        loss = self._getLossFunction(loss_function, target)

        # Get variables (exclude constants)
        vlist = self._getVarList(optimal)
        # Create an optimizer of the given learning rate
        optimizer = tf.train.ProximalGradientDescentOptimizer(learn_rate)
        # Get the gradients from the loss function for each variable
        gradients = optimizer.compute_gradients(loss, var_list = vlist)

        # Raise range-restricted variables
        newGrads = [raiseGrad(gradients[g])
                    if g[1] in rangeRestrictedVars else g
                    for g in gradients]

        # Gradient application
        applyNewGrads = optimizer.apply_gradients(newGrads)

        # Get the absolute error (for DEBUG)
        if target in ["max", "min"]:
            absoluteError = tf.constant(0.0)
        else:
            absoluteError = tf.abs(tf.sub(label, out))

        # Initialize the computation graph
        self._session.run(tf.initialize_all_variables())

        # - TRAIN -
        # A counter for counting epochs
        counter = 0
        # If debug is on, print intial debug report
        if debug and debug_interval > 0:
            self._printDebugStatus(counter, optimal, restrictions,
                                   absoluteError = "NOT APPLICABLE",
                                   timer = "NOT APPLICABLE")
        # The main traing loop
        while True:
            # Start timer (for DEBUG profiling)
            time0 = time()

            # Break if error is 0 or within learning rate of zero
            absoluteErrorEvaluated = absoluteError.eval(session = self._session)[0]
            if sum(absoluteErrorEvaluated) <= error_tolerance \
                    and target not in ["max", "min"]:
                breakReason = "Beat Error"
                break

            # Break if gradients are all zero
            if self._checkGradients(newGrads, zeroGrad):
                breakReason = "Zero Gradients"
                break

            # Break if epochs limit reached
            if counter >= epochs and epochs != -1:
                breakReason = "Epoch Limit Reached"
                break

            # Apply training step to find optimal
            self._session.run(applyNewGrads)

            # Increment counter
            counter += 1

            # Debug printing
            if counter % debug_interval == 0 and debug and debug_interval > 0:
                self._printDebugStatus(counter, optimal, restrictions,
                                       absoluteErrorEvaluated,
                                       time() - time0)


        # Finalize restricted output
        # Get restricion vectors
        rv = self._getRestrictionVectors(restrictions, optimal)
        # Apply restriction vectors
        final = self._applyRestrictionVector(optimal, rv).eval(session = self._session)

        # Print final digest
        if debug:
            print("\nOPTIMAL INPUT       :: {0}".format(final[0]))
            print("CALCULATED OUT      :: {0}".format(calc(optimal).eval(session = self._session)[0]))
            print("TARGET OUT          :: {0}".format(lbl.eval(session = self._session)[0]))
            print("ERROR               :: {0}".format(absoluteError.eval(session = self._session)[0]))
            print("EPOCHS              :: {0} ({1})".format(counter, breakReason))

        # If evaluation is requested, returned evaluated
        # Don't evaluate if not
        if evaluate: return final
        else: return optimal

    def feed(self, input_vector, evaluate = True):
        return self.model.feed(input_vector)

    def eval(self, input_vector):
        final = []
        for i in input_vector:
            if type(i) in [list, tuple, np.ndarray]:
                temp = []
                for j in i:
                    if type(j) in [tf.constant, tf.Variable]:
                        temp.append(j.eval(self.session))
                    else:
                        temp.append(j)
                final.append(temp)
            elif type(i) in [tf.constant, tf.Variable]:
                final.append(j.eval(self.session))
            else:
                final.append(i)
        return final

    def _getRestrictionVectors(self, restrictions, vars):
        rVector = [[], []]

        # Get bottom of a range to negate function
        def b(x):
            return (x - sig(x) * x) / (1 - sig(x))

        # Slightly modified sigmoid function
        def sig(z):
            return tf.nn.sigmoid(tf.constant(.000001) * z)

        for i in range(len(vars[0])):
            if i in restrictions.keys():
                if type(restrictions[i]) in [list, tuple]:
                    rVector[0].append(tf.cast(restrictions[i][0], tf.float32))
                    rVector[1].append(tf.cast(restrictions[i][1], tf.float32))
                else:
                    rVector[0].append(tf.cast(b(restrictions[i]), tf.float32))
                    rVector[1].append(tf.cast(restrictions[i], tf.float32))
            else:
                rVector[0].append(tf.cast(b(vars[0][i]), tf.float32))
                rVector[1].append(tf.cast(vars[0][i], tf.float32))
        return rVector

    def _applyRestrictionVector(self, inputs, restrictVector):
        # Slightly modified sigmoid function
        def sig(z):
            return tf.nn.sigmoid(tf.constant(.000001) * z)
        # Restriction reshaping function
        def f(x, b, t):
            q = sig(x)
            w = tf.mul(q, tf.cast(tf.sub(t, b), tf.float32))
            return tf.add(w, b)

        return f(inputs, restrictVector[0], restrictVector[1])

    def _raiseGrad(self, grad):
        """
        Scale a gradient
        Parameter:
            grad : (gradient, variable)
        """
        return (tf.mul(grad[0], rangeGradientScalar), grad[1])

    def _getLossFunction(self, requested, target):
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
                return tf.reduce_sum(tf.pow(lbl - out, 2))
        else:
            if target == "max":
                return tf.mul(-1.0, out)
            elif target == "min":
                return out
            else:
                return tf.abs(lbl - out)

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

    def _printDebugStatus(self, epoch, optimal, restrictions, absoluteError, timer):
        """Prints the debug information during training"""
        # Combine optimal (constants and variables)
        op = []
        for i in optimal[0]:
            if type(i) == tf.constant:
                op.append(i)
            else:
                # Evaluate if a variables
                op.append(i.eval(session=self._session))

        # Get restricion vectors
        rv = self._getRestrictionVectors(restrictions, optimal)
        # Apply restriction vectors
        q = self._applyRestrictionVector(optimal, rv).eval(session=self._session)

        # Begin printing
        print "@ Epoch {0}".format(counter)
        print "Value        :: {0}".format(op)
        print "Restricted   :: {0}".format(q)
        print "Evaluated    :: {0}".format(self.model.feed(q))
        print "Error        :: {0}".format(absoluteError)
        print "Total Error  :: {0}".format(sum(absoluteError))
        print "Time         :: {0}".format(timer)
        print ""

    def _checkGradients(self, gradients, checkAgainst):
        # Check equality
        grads = [p[0] for p in gradients]
        areSame = self._session.run(tf.equal(gs, checkAgainst))
        # Check each truth value, add to counter if true
        same = 0
        for g in areSame:
            if g: same += 1
        # If all are the same, return true, else false
        if same == len(grads): return True
        else: return False

class LinearReg:
    def __init__(self):
        pass

    def feed(self, input_vector):
        return tf.reduce_sum(input_vector)

class PolyReg:
    def feed(self, input_vector):
        return input_vector

def test():
    lr = LinearReg()

    # Input Optimization
    I = InputOptimizer(lr, 1, 1)
    print I.eval(I.optimize("a"))

test()