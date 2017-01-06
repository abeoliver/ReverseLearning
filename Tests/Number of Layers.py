# Functionality of IBP Algorithm on Multi-layer networks
# ReverseLearning

from Algorithm.Network import Network
from random import randint, random

"""
Test IBP Algorithm on multi-layer networks

Feed-forward networks are trained until error is less than .1

All configurations have 1 output, 100000 epochs, .001 learning rate, 1000 training example,
5 inputs, and a 10.0 target
Configurations vary in number of hidden neurons or number of layers
"""

# Addition Dataset
def newSet(size, ins = 2):
    """EXAMPLE"""
    data = []
    labels = []
    for s in range(size):
        newInputs = [random() * randint(-100, 100) for i in range(ins)]
        data.append(newInputs)
        labels.append([sum(newInputs)])
    return (data, labels)


# Trial: 1
# Hidden Structure: [10]
# Target: 10.0
# FEED FORWARD
print "---1---"
n = Network([5, 10, 1])
n.train(newSet(1000, 5), epochs = 9000, learn_rate = .0001, debug_final_loss = True)
# IBP
n.ibp(target = 10.0, epochs = 10000, learn_rate = .001, debug = True)
print ""
"""
OUTPUT
------
CALCULATED OUT      :: [[ 9.99597836]]
TARGET OUT          :: [[10.0]]
TARGET vs CALC LOSS :: [[ 0.00402164]]
"""


# Trial: 2
# Hidden Structure: [100]
# Target: 10.0
# FEED FORWARD
print "---2---"
n = Network([5, 100, 1])
n.train(newSet(1000, 5), epochs = 20000, learn_rate = .0001, debug_final_loss = True)
# IBP
n.ibp(target = 10.0, epochs = 10000, learn_rate = .001, debug = True)
print ""
"""
OUTPUT
------
CALCULATED OUT      :: [[ 9.9976511]]
TARGET OUT          :: [[10.0]]
TARGET vs CALC LOSS :: [[ 0.0023489]]
"""


# Trial: 3
# Hidden Structure: [10, 10]
# Target: 10.0
# FEED FORWARD
print "---3---"
n = Network([5, 10, 10, 1])
n.train(newSet(1000, 5), epochs = 30000, learn_rate = .0001, debug_final_loss = True)
# IBP
n.ibp(target = 10.0, epochs = 10000, learn_rate = .001, debug = True)
print ""
"""
OUTPUT
------
CALCULATED OUT      :: [[ 10.00484276]]
TARGET OUT          :: [[10.0]]
TARGET vs CALC LOSS :: [[ 0.00484276]]
"""