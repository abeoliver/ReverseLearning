# General Functionality of IBP Algorithm
# ReverseLearning

from Algorithm.Network import Network

"""
Test IBP Algorithm with multiple configurations

The feed-forward networks are 100% accurate because there is a solution for the
AdditionNetwork Dataset. All weights are 1's and the biases are 0's (because all
the inputs multiplied by one, added to zero, and then summed together is the same
as summing all the inputs in the first place)

All configurations have 1 output
Configurations vary in learning rate, epochs, number of inputs, and size of target value
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
OPTIMAL INPUT       :: [[ 2.00003743  2.00003743]]
CALCULATED OUT      :: [[ 4.00007486]]
TARGET OUT          :: [[4.0]]
TARGET vs CALC LOSS :: 7.48634338379e-05
"""


# Trial: 2
# Inputs: 100
# Learning Rate: .001
# Epochs: 10000
# Target: 4.0
# FEED FORWARD
n = Network([100, 1])
n.initWeights(mode="ones")
n.initBiases(mode="zeros")
# IBP
n.ibp(target = 4.0, epochs = 10000, learn_rate = .001, debug = True)
"""
OUTPUT
------
OPTIMAL INPUT       :: [[ 0.03999999  0.03999999  0.03999999  0.03999999  0.03999999  0.03999999
                          0.03999999  0.03999999  0.03999999  0.03999999  0.03999999  0.03999999
                          ...
                          0.03999999  0.03999999  0.03999999  0.03999999]]
CALCULATED OUT      :: [[ 3.99999666]]
TARGET OUT          :: [[4.0]]
TARGET vs CALC LOSS :: 3.33786010742e-06
"""


# Trial: 3
# Inputs: 10
# Learning Rate: .001
# Epochs: 100000
# Target: 1000.0
# FEED FORWARD
n = Network([10, 1])
n.initWeights(mode="ones")
n.initBiases(mode="zeros")
# IBP
n.ibp(target = 1000.0, epochs = 100000, learn_rate = .001, debug = True)
"""
OUTPUT
------
OPTIMAL INPUT       :: [[ 99.95669556  99.95669556  99.95669556  99.95669556  99.95669556
                          99.95669556  99.95669556  99.95669556  99.95669556  99.95669556]]
CALCULATED OUT      :: [[ 999.56689453]]
TARGET OUT          :: [[1000.0]]
TARGET vs CALC LOSS :: 0.43310546875
"""