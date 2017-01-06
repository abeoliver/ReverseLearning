# General Functionality of IBP Algorithm
# ReverseLearning

from Algorithm.Network import Network

"""
Test IBP Algorithm on multiple network configurations

The feed-forward networks are 100% accurate because there is a solution for the
AdditionNetwork Dataset. All weights are 1's and the biases are 0's (because all
the inputs multiplied by one, added to zero, and then summed together is the same
as summing all the inputs in the first place)

All configurations have 1 output
Configurations vary in learning rate, epochs, number of inputs, and size of target value
"""

def AdditionDataSet(size):
    """
    data = multiple floats between -100 and 100
    label = the sum of the floats
    """
    data = []
    labels = []
    for s in range(size):
        newInputs = [random() * randint(-100, 100) for i in range(ins)]
        data.append(newInputs)
        labels.append([sum(newInputs)])
    return (data, labels)

# Trial: 1
# Inputs: 2
# Learning Rate: .001
# Epochs: 10000
# Target: 4.0
n = Network([2, 1])
n.initWeights(mode = "ones")
n.initBiases(mode = "zeros")
print n.w
print n.b
