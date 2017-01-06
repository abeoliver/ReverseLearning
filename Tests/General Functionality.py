# General Functionality of IBP Algorithm
# ReverseLearning

from Algorithm.Network import Network

"""
Test IBP Algorithm with multiple configurations

The feed-forward networks are 100% accurate because there is a solution for the
AdditionNetwork Dataset. All weights are 1's and the biases are 0's (because all
the inputs multiplied by one, added to zero, and then summed together is the same
as summing all the inputs in the first place)

All configurations have 1 output, 100000 epochs, .001 learning rate, and a 4.0 target
Configurations vary in number of inputs
"""


# Trial: 1
# Inputs: 2
# FEED FORWARD
n = Network([2, 1])
n.initWeights(mode="ones")
n.initBiases(mode="zeros")
# IBP
n.ibp(target = 4.0, epochs = 100000, learn_rate = .001, debug = True)
"""
OUTPUT
------
OPTIMAL INPUT       :: [[ 2.00003743  2.00003743]]
CALCULATED OUT      :: [[ 4.00007486]]
TARGET OUT          :: [[4.0]]
TARGET vs CALC LOSS :: [[  7.48634338e-05]]
"""

# Trial: 2
# Inputs: 50
# FEED FORWARD
n = Network([50, 1])
n.initWeights(mode="ones")
n.initBiases(mode="zeros")
# IBP
n.ibp(target = 4.0, epochs = 100000, learn_rate = .001, debug = True)
"""
OUTPUT
------
OPTIMAL INPUT       :: [[ 0.07999998  0.07999998  0.07999998  0.07999998  0.07999998  0.07999998
                          ...
                          0.07999998  0.07999998  0.07999998  0.07999998  0.07999998  0.07999998
                          0.07999998  0.07999998  0.07999998  0.07999998  0.07999998  0.07999998
                          0.07999998  0.07999998]]
CALCULATED OUT      :: [[ 3.9999969]]
TARGET OUT          :: [[4.0]]
TARGET vs CALC LOSS :: [[  3.09944153e-06]]
"""


# Trial: 3
# Inputs: 100
# FEED FORWARD
n = Network([100, 1])
n.initWeights(mode="ones")
n.initBiases(mode="zeros")
# IBP
n.ibp(target = 4.0, epochs = 100000, learn_rate = .001, debug = True)
"""
OUTPUT
------
OPTIMAL INPUT       :: [[ 0.03999999  0.03999999  0.03999999  0.03999999  0.03999999  0.03999999
                          ...
                          0.03999999  0.03999999  0.03999999  0.03999999  0.03999999  0.03999999
                          0.03999999  0.03999999  0.03999999  0.03999999  0.03999999  0.03999999
                          0.03999999  0.03999999  0.03999999  0.03999999]]
CALCULATED OUT      :: [[ 3.99999666]]
TARGET OUT          :: [[4.0]]
TARGET vs CALC LOSS :: [[  3.33786011e-06]]
"""


# Trial: 4
# Inputs: 1000
# FEED FORWARD
n = Network([1000, 1])
n.initWeights(mode="ones")
n.initBiases(mode="zeros")
# IBP
n.ibp(target = 4.0, epochs = 100000, learn_rate = .001, debug = True)
"""
OUTPUT
------
OPTIMAL INPUT       :: [[ 0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004
                          ...
                          0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004
                          0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004
                          0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004]]
CALCULATED OUT      :: [[ 3.99996281]]
TARGET OUT          :: [[4.0]]
TARGET vs CALC LOSS :: [[  3.71932983e-05]]
"""