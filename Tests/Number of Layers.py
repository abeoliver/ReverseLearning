# Functionality of IBP Algorithm on Multi-layer networks
# ReverseLearning

from Algorithm.Network import Network

"""
Test IBP Algorithm on multi-layer networks

The feed-forward networks are 100% accurate because there is a solution for the
AdditionNetwork Dataset. All weights are 1's and the biases are 0's (because all
the inputs multiplied by one, added to zero, and then summed together is the same
as summing all the inputs in the first place)

All configurations have 1 output
Configurations vary in learning rate, epochs, number of inputs, and size of target value
and number of layers
"""


# Trial: 1
# Inputs: 2
# Learning Rate: .001
# Epochs: 10000
# Target: 4.0
# FEED FORWARD
n = Network([2, 1])
n.initWeights(mode="ones")
n.initBiases(mode="zeros")
# IBP
n.ibp(target = 4.0, epochs = 10000, learn_rate = .001, debug = True)
"""
OUTPUT
------
"""