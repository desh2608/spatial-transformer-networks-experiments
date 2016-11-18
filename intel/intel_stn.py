import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from scipy import misc
import os
from glob import glob
from random import randint

def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

RESULT_DIR = "./results/"
checkpoint_file = "./model.ckpt"
restore = False

# Load data and shuffle
trainx01 = np.load('./Data/train/01.npy')
trainx02 = np.load('./Data/train/02.npy')
trainx05 = np.load('./Data/train/05.npy')
trainx07 = np.load('./Data/train/07.npy')
trainx09 = np.load('./Data/train/09.npy')
trainx03 = np.load('./Data/test/03.npy')
trainx04 = np.load('./Data/test/04.npy')
trainx06 = np.load('./Data/test/06.npy')
trainx08 = np.load('./Data/test/08.npy')
trainy01 = np.load('./labels/01.npy')
trainy02 = np.load('./labels/02.npy')
trainy03 = np.load('./labels/03.npy')
trainy04 = np.load('./labels/04.npy')
trainy05 = np.load('./labels/05.npy')
trainy06 = np.load('./labels/06.npy')
trainy07 = np.load('./labels/07.npy')
trainy08 = np.load('./labels/08.npy')
trainy09 = np.load('./labels/09.npy')
X_train = np.concatenate((trainx01,trainx02,trainx03,trainx04,trainx05,trainx06,trainx07,trainx08,trainx09))
y_train = np.concatenate((trainy01,trainy02,trainy03,trainy04,trainy05,trainy06,trainy07,trainy08,trainy09))
X_test = np.load('./Data/test/10.npy')
y_test = np.load('./labels/10.npy')

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=42)
Y_test = dense_to_one_hot(y_test, n_classes=42)

# %% Graph representation of our network

# %% Placeholders for 81x144x3 resolution
x = tf.placeholder(tf.float32, [None, 76, 102, 3])
y = tf.placeholder(tf.float32, [None, 42])


# %% We'll setup the two-layer localisation network to figure out the
# %% parameters for an affine transformation of the input
# %% Create variables for fully connected layer
x_flat = tf.reshape(x,[-1,76*102*3])
W_fc_loc1 = tf.Variable(tf.random_normal([76*102*3,32], mean=0.0, stddev=0.01),name='W_fc_loc1')
b_fc_loc1 = tf.Variable(tf.random_normal([32], mean=0.0, stddev=0.1),name='b_fc_loc1')

W_fc_loc2 = tf.Variable(tf.random_normal([32, 6], mean=0.0, stddev=0.01),name='W_fc_loc2')
# Use identity transformation as starting point
initial = np.array([[1., 0, 0], [0, 1., 0]])
initial = initial.astype('float32')
initial = initial.flatten()
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

# %% Define the two layer localisation network
h_fc_loc1 = tf.nn.tanh(tf.matmul(x_flat, W_fc_loc1) + b_fc_loc1)
# %% We can add dropout for regularizing and to reduce overfitting like so:
keep_prob = tf.placeholder(tf.float32)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
# %% Second layer
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

# %% We'll create a spatial transformer module to identify discriminative
# %% patches
out_size = (76, 102)
h_trans = transformer(x, h_fc_loc2, out_size)

#print h_trans.get_shape()
# %% We'll setup the first convolutional layer
# Weight matrix is [height x width x input_channels x output_channels]

filter_size = 3
n_filters_1 = 16
W_conv1 = tf.Variable(tf.random_normal([filter_size, filter_size, 3, n_filters_1], mean=0.0, stddev=0.01),name='W_conv1')

# %% Bias is [output_channels]
b_conv1 = tf.Variable(tf.random_normal([n_filters_1], mean=0.0, stddev=0.1),name='b_fc_loc1')

# %% Now we can build a graph which does the first layer of convolution:
# we define our stride as batch x height x width x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.

h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=h_trans,
				 filter=W_conv1,
				 strides=[1, 2, 2, 1],
				 padding='SAME') +
	b_conv1)

# %% And just like the first layer, add additional layers to create
# a deep net
n_filters_2 = 16
W_conv2 = tf.Variable(tf.random_normal([filter_size, filter_size, n_filters_1, n_filters_2], mean=0.0, stddev=0.01),name='W_conv2')
b_conv2 = tf.Variable(tf.random_normal([n_filters_2], mean=0.0, stddev=0.1),name='b_conv2')
h_conv2 = tf.nn.relu(
	tf.nn.conv2d(input=h_conv1,
				 filter=W_conv2,
				 strides=[1, 2, 2, 1],
				 padding='SAME') +
	b_conv2)

# %% We'll now reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 19 * 26 * n_filters_2])

# %% Create a fully-connected layer:
n_fc = 512
W_fc1 = tf.Variable(tf.random_normal([19 * 26 * n_filters_2, n_fc], mean=0.0, stddev=0.005),name='W_fc1')
b_fc1 = tf.Variable(tf.random_normal([n_fc], mean=0.0, stddev=0.1),name='b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# %% And finally our softmax layer:
W_fc2 = tf.Variable(tf.random_normal([n_fc, 42], mean=0.0, stddev=0.005),name='W_fc2')
b_fc2 = tf.Variable(tf.random_normal([42], mean=0.0, stddev=0.1),name='b_fc2')
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# %% Define loss/eval/training functions
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer(1e-4)
optimizer = opt.minimize(cross_entropy)
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

# %% Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
sess = tf.Session()

if restore:
	print("Loading variables from '%s'." % checkpoint_file)
	saver.restore(sess,checkpoint_file)
else:
	sess.run(tf.initialize_all_variables())


# %% We'll now train in minibatches and report accuracy, loss:
train_size = len(X_train)
batch_size = 1000
num_batches = train_size/batch_size
n_epochs = 100


indices = np.linspace(0, train_size - 1, num_batches)
indices = indices.astype('int')

f = open("result1.dat","a")
for epoch_i in range(n_epochs):
	for batch_i in range(num_batches-1):
		batch_xs = X_train[indices[batch_i]:indices[batch_i+1]]
		batch_ys = Y_train[indices[batch_i]:indices[batch_i+1]]
		loss = sess.run(cross_entropy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
		print('Iteration: ' + str(batch_i) + ' Loss: ' + str(loss))

		sess.run(optimizer, feed_dict={
			x: batch_xs, y: batch_ys, keep_prob: 0.8})

	p = np.random.permutation(len(X_test))
	acc = sess.run(accuracy,feed_dict={x: X_test[p][0:100],y: Y_test[p][0:100],keep_prob: 1.0})
	print('Accuracy (%d): ' % epoch_i + str(acc))
	f.write("%d,%f\n"%(epoch_i,acc))

	# if epoch_i%20==0:
	# 	i = randint(0,len(X_test)-1)
	# 	input_img = X_test[i]
	# 	input_lbl = Y_test[i]
	# 	x_in = [input_img]
	# 	y_in = [input_lbl]
	# 	res = sess.run([accuracy,h_trans],feed_dict={x:x_in,y:y_in,keep_prob:1.0})
	# 	accu = res[0]
	# 	x_out = res[1]
	# 	out_img = x_out[0]
	# 	pred = int(accu)
	# 	if not os.path.exists(RESULT_DIR):
	# 		os.makedirs(RESULT_DIR)
	# 	misc.imsave(RESULT_DIR+"04_epoch"+str(epoch_i)+"_"+"in.png",input_img)
	# 	misc.imsave(RESULT_DIR+"04_epoch"+str(epoch_i)+"_"+"out"+"_"+str(pred)+".png",out_img)

f.close()

if checkpoint_file is not None:
	print ("Saving variables to '%s'." % checkpoint_file)
	saver.save(sess,checkpoint_file)
