"""Classification of distorted MNIST using CNN with ReLU activation function and Adam optimizer"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tf_utils import weight_variable, bias_variable, dense_to_one_hot

# %% Load data
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')

X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)

# %% Graph representation of our network

# %% Placeholders for 40x40 resolution
x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 10])

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#First convolutional layer
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])

# %% Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph.  If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant.  Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
x_image = tf.reshape(x, [-1, 40, 40, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer
W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely connected layer
W_fc1 = weight_variable([3 * 3 * 16, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


# %% We'll now train in minibatches and report accuracy, loss:
iter_per_epoch = 100
n_epochs = 500
train_size = 10000

indices = np.linspace(0, 10000 - 1, iter_per_epoch)
indices = indices.astype('int')

f = open('cluttered_mnist_cnn.dat','w')

for epoch_i in range(n_epochs):
  for iter_i in range(iter_per_epoch - 1):
    batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
    if iter_i % 10 == 0:
      loss = sess.run(cross_entropy,feed_dict={x: batch_xs,y: batch_ys,keep_prob: 1.0})
      print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})
  print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,feed_dict={x: X_valid,y: Y_valid,keep_prob: 1.0})))
  f.write('%f\n' %sess.run(accuracy,feed_dict={x: X_valid,y: Y_valid,keep_prob: 1.0}))

f.close()

# for i in range(20000):
#   batch = mnist_cluttered.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#       x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# Test trained model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval({x: mnist_cluttered.test.images, y_: mnist_cluttered.test.labels}))