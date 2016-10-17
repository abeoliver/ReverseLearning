from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def showImage(img):
    tmp = img.reshape((28,28))
    plt.imshow(tmp, cmap = cm.Greys)
    plt.show()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([10, 784]))
b = tf.Variable(tf.zeros([784]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 784])

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.ProximalGradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1):
    batch_ys, batch_xs = mnist.train.next_batch(1)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

diff = tf.abs(tf.sub(y, y_))
corrects = tf.cast(tf.less_equal(diff, .001), tf.float32)
accuracy = tf.reduce_mean(corrects)
print(sess.run(accuracy, feed_dict={x: mnist.test.labels, y_: mnist.test.images}))