import numpy as np
import matplotlib.pyplot as plt

N = 50
x = range(1,501)
y1 = []
y2 = []
y3 = []

with open('./results/cluttered_mnist_cnn_3x3.dat','r') as f:
	for line in f:
		line = line.strip()
		y1.append(float(line))

with open('./results/cluttered_mnist_cnn_5x5.dat','r') as f:
	for line in f:
		line = line.strip()
		y2.append(float(line))

with open('./results/cluttered_mnist_stn.dat','r') as f:
	for line in f:
		line = line.strip().split(',')
		y3.append(float(line[1]))

plt.plot(x, y1, label='CNN-Simple',color='blue')
plt.plot(x, y2, label='CNN-Complex',color='red')
plt.plot(x, y3, label='CNN-STN',color='green')

plt.title('Performance of CNN variants on distorted MNIST')
plt.ylabel('Training accuracy')
plt.xlabel('Number of epochs')
plt.legend(loc=4)
plt.show()