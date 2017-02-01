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

    def optimize(self, target, restrictions = {}, learn_rate = 1.0):
        # Range restriction list
        rangeRestricted = []
        for i in restrictions.keys():
            if type(restrictions[i]) in [list, tuple]:
                rangeRestricted.append(i)

        # Optimal input
        optimal = [[tf.Variable(0.0) for i in range(self.ins)]]
        # Apply constant restrictions
        for k in restrictions.keys():
            if type(restrictions[k]) in [float, int]:
                optimal[k] = tf.constant(float(restrictions[k]))

        # Label
        y_ = [[tf.constant(target[0])]]
        # Prediction
        y = self.model.feed(optimal)
        # Loss
        loss = tf.abs(y_ - y)

        # Optimizer
        # Get variables
        vlist = []
        for i in optimal[0]:
            if type(i) == tf.Variable:
                vlist.append(i)
        # Make optimizer
        trainer = tf.train.ProximalGradientDescentOptimizer(learn_rate)
        # Collect gradients
        gradients = trainer.compute_gradients(loss, var_list=vlist)
        # Increase gradients (only if range restricted)
        newGrads = [self._raiseGrad(gradients[g][0], gradients[g][1])
                    if g in rangeRestricted
                    else (gradients[g][0], gradients[g][1])
                    for g in range(len(gradients))]
        # Gradient application
        applyGrads = trainer.apply_gradients(newGrads)

        # Initialize
        self.session.run(tf.initialize_all_variables())

        # Train for optimal
        error = 1000.0
        while error > 1:
            self.session.run(applyGrads)
            error = loss.eval(session = self.session)
        return optimal

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
        pass

    def _applyRestrictionVector(self, inputs, restrictVector):
        pass

    def _raiseGrad(self, grad, var):
        return (tf.mul(grad, rangeGradientScalar), var)


class LinearReg:
    def __init__(self):
        pass

    def feed(self, input_vector):
        return tf.reduce_sum(input_vector)

class PolyReg:
    def feed(self, input_vector):
        return input_vector

def test():
    lr = PolyReg()

    # Input Optimization
    I = InputOptimizer(lr, 1, 1)
    # print I.feed([0, 1, 2, 3, 4])
    print I.eval(I.optimize([25.0]))

test()