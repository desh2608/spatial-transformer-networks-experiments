"""Classification of egocentric objects using CNN with ReLU activation function and Adam optimizer"""


import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from sklearn.cross_validation import train_test_split

# Load data and shuffle
ego_data = np.load('./ego_data.npz')
X = ego_data['images']
Y = ego_data['labels']
p = np.random.permutation(len(X))
X = X[p]
Y = Y[p]

# Create training, validation and testing partitions
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_test, X_valid, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=7)
Y_valid = dense_to_one_hot(y_val, n_classes=7)
Y_test = dense_to_one_hot(y_test, n_classes=7)

# %% Graph representation of our network

# %% Placeholders for 81x144x3 resolution
x = tf.placeholder(tf.float32, [None, 81, 144, 3])
y = tf.placeholder(tf.float32, [None, 7])

# %% We'll setup the first convolutional layer
# Weight matrix is [height x width x input_channels x output_channels]

filter_size = 3
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 3, n_filters_1])

# %% Bias is [output_channels]
b_conv1 = bias_variable([n_filters_1])

# %% Now we can build a graph which does the first layer of convolution:
# we define our stride as batch x height x width x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.

h_conv1 = tf.nn.relu(
  tf.nn.conv2d(input=x,
         filter=W_conv1,
         strides=[1, 2, 2, 1],
         padding='SAME') +
  b_conv1)

# %% And just like the first layer, add additional layers to create
# a deep net
n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
  tf.nn.conv2d(input=h_conv1,
         filter=W_conv2,
         strides=[1, 2, 2, 1],
         padding='SAME') +
  b_conv2)

# %% We'll now reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 21 * 36 * n_filters_2])

# %% Create a fully-connected layer:
n_fc = 512
W_fc1 = weight_variable([21 * 36 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
# %% We can add dropout for regularizing and to reduce overfitting like so:
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# %% And finally our softmax layer:
W_fc2 = weight_variable([n_fc, 7])
b_fc2 = bias_variable([7])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# %% Define loss/eval/training functions
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer(1e-4)
optimizer = opt.minimize(cross_entropy)
grads = opt.compute_gradients(cross_entropy, [b_fc2])

# %% Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())


# %% We'll now train in minibatches and report accuracy, loss:
train_size = len(X_train)
batch_size = 100
num_batches = train_size/batch_size
n_epochs = 500


indices = np.linspace(0, train_size - 1, num_batches)
indices = indices.astype('int')

f = open("result2.dat","w")
for epoch_i in range(n_epochs):
  for batch_i in range(num_batches-1):
    batch_xs = X_train[indices[batch_i]:indices[batch_i+1]]
    batch_ys = Y_train[indices[batch_i]:indices[batch_i+1]]
    loss = sess.run(cross_entropy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
    print('Iteration: ' + str(batch_i) + ' Loss: ' + str(loss))

    sess.run(optimizer, feed_dict={
      x: batch_xs, y: batch_ys, keep_prob: 0.8})

  print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
                           feed_dict={
                             x: X_valid,
                             y: Y_valid,
                             keep_prob: 1.0
                           })))
  f.write("%d,%f\n"%(epoch_i,sess.run(accuracy,feed_dict={x: X_valid,y: Y_valid,keep_prob: 1.0})))

f.close()

