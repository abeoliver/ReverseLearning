# mnist_outs_to_ins.py
# MNIST Labels to Image ANN
# Abraham Oliver, 2016

# General structure from tensorflow.org
# Matplotlib help from http://tneal.org/post/tensorflow-ipython/TensorFlowMNIST/

# Import python3's print as a function
from __future__ import print_function
# Import Tensorflow
import tensorflow as tf
# Import numpy
import numpy as np

# Import tools for image visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def showImage(img):
    """Display image of recovered digit"""
    tmp = img.reshape((28,28))
    plt.imshow(tmp, cmap = cm.Greys)
    plt.show()

# Initiate dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../DataSets/MNIST/", one_hot=True)

# Design model variables
# Input
x = tf.placeholder(tf.float32, [None, 10])
# Weights
W = tf.Variable(tf.zeros([10, 784]))
# Biases
b = tf.Variable(tf.zeros([784]))
# Predicted Output
y = tf.nn.softmax(tf.matmul(x, W) + b)
# Correct Outputs
y_ = tf.placeholder(tf.float32, [None, 784])

# Training with cross-entropy cost and gradient descent with learning rate .5
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.ProximalGradientDescentOptimizer(0.5).minimize(loss)

# Start computation graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Train over data
print("Training......")
for i in range(2000):
    # Labels are the inputs and images are the outputs
    batch_ys, batch_xs = mnist.train.next_batch(200)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

def calcAccuracy(margin = .005):
    """Calculate the accuracy of the model with test data"""
    diff = tf.abs(tf.sub(y, y_))
    corrects = tf.cast(tf.less_equal(diff, margin), tf.float32)
    accuracy = tf.reduce_mean(corrects)
    print(sess.run(accuracy, feed_dict={x: mnist.test.labels, y_: mnist.test.images}))
print(calcAccuracy())

def compute(INPUT):
    """Shows the network's output image from a label vector input"""
    i = np.array([INPUT])
    z = y.eval(feed_dict = {x: i}, session = sess)
    showImage(z[0])

# Show image from each digit
for i in range(3):
    # Make label vector for digit
    lv = np.zeros(10)
    lv[i] = 1.0
    compute(lv)