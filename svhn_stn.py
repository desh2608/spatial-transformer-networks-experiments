import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io as sio
import cv2
from spatial_transformer import transformer

import svhnInput

data=svhnInput.read_data_sets()

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1e-3)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1e-3)
  return tf.Variable(initial)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 32*32*3])
y_ = tf.placeholder("float", shape=[None, 10])

x_tensor = tf.reshape(x, [-1, 32, 32, 3])

W_conv_loc1 = weight_variable([5,5,3,32])
b_conv_loc1 = bias_variable([32])

h_conv_loc1 = tf.nn.relu(conv2d(x_tensor,W_conv_loc1) + b_conv_loc1)

h_pool_loc1 = max_pool_2x2(h_conv_loc1)

W_conv_loc2 = weight_variable([5,5,32,32])
b_conv_loc2 = bias_variable([32])

h_conv_loc2 = tf.nn.relu(conv2d(h_pool_loc1,W_conv_loc2) +  b_conv_loc1)

h_pool_loc2 = max_pool_2x2(h_conv_loc2)

h_pool2_loc2_flat = tf.reshape(h_pool_loc2, [-1, 8*8*32])

W_fc_loc1 = weight_variable([8*8*32, 32])
b_fc_loc1 = bias_variable([32])

W_fc_loc2 = weight_variable([32, 6])
# Use identity transformation as starting point
initial = np.array([[1., 0, 0], [0, 1., 0]])
initial = initial.astype('float32')
initial = initial.flatten()
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

# %% Define the two layer localisation network
h_fc_loc1 = tf.nn.tanh(tf.matmul(h_pool2_loc2_flat, W_fc_loc1) + b_fc_loc1)
# %% We can add dropout for regularizing and to reduce overfitting like so:
keep_prob_loc = tf.placeholder(tf.float32)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob_loc)
# %% Second layer
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

# %% We'll create a spatial transformer module to identify discriminative
# %% patches
out_size = (32, 32)
h_trans = transformer(x_tensor, h_fc_loc2, out_size)

W_conv1 = weight_variable([5, 5, 3, 48])
b_conv1 = bias_variable([48])

h_conv1 = tf.nn.relu(
    conv2d(h_trans,W_conv1) +
    b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 48, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(
    conv2d(h_pool1,W_conv2) +
    b_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(
    conv2d(h_conv2,W_conv3) +
    b_conv3)

h_pool3 = max_pool_2x2(h_conv3)

h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*128])

W_fc1 = weight_variable([16 * 16 *128,1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc3 = weight_variable([1024, 10])
b_fc3 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

f = open('svhn_stn.dat','w')

for i in range(5000):
  batch = data.train.next_batch(100)
  batch_test = data.test.next_batch(1000)
  if i%10 == 0:
    test_accuracy = accuracy.eval(feed_dict={
        x:batch_test[0], y_: batch_test[1], keep_prob: 1.0, keep_prob_loc: 1.0})
    print "step %d, testing accuracy %g"%(i, test_accuracy)
    f.write("%f\n"%test_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, keep_prob_loc: 1.0})

f.close()