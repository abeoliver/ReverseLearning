# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import tensorflow as tf
from Network import Network

n = Network([1,2,3,4])
print n.layers
print n.w
print n.b
print n.feed([2.0])