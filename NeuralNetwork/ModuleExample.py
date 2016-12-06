# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import tensorflow as tf
from Network import Network

n = Network([1, 1])
n.initWeights(mode="zeros")
print n.feed([1.0]).eval(session=n._session)