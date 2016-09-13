import numpy as np
import matplotlib.pyplot as plt

N = 50
x = range(1,501)
y1 = []
y2 = []
y3 = []

# with open('./results/cluttered_mnist_cnn_3x3.dat','r') as f:
# 	for line in f:
# 		line = line.strip()
# 		y1.append(float(line))

# with open('./results/cluttered_mnist_cnn_5x5.dat','r') as f:
# 	for line in f:
# 		line = line.strip()
# 		y2.append(float(line))

# with open('./results/cluttered_mnist_stn.dat','r') as f:
# 	for line in f:
# 		line = line.strip().split(',')
# 		y3.append(float(line[1]))

with open('./results/simple_mnist_cnn_3x3.dat','r') as f:
	for line in f:
		line = line.strip()
		y1.append(float(line))

with open('./results/cluttered_mnist_cnn_3x3.dat','r') as f:
	for line in f:
		line = line.strip()
		y2.append(float(line))

plt.plot(x, y1, label='Simple MNIST',color='green')
plt.plot(x, y2, label='Distorted MNIST',color='magenta')

plt.title('Simple vs Distorted MNIST using simple CNN')
plt.ylabel('Training accuracy')
plt.xlabel('Number of epochs')
plt.legend(loc=4)
plt.show()