#!/usr/local/bin/python3

# To be used with random_sparsity.rs

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('sp2.csv', delimiter=',')
avgs = np.average(data[:, 1:], axis=1)
markers = np.arange(len(data))
plt.bar(markers-0.3, avgs, width=0.2, label='2 Dimensional Matrix', color='r')

data = np.genfromtxt('sp4.csv', delimiter=',')
avgs = np.average(data[:, 1:], axis=1)
plt.bar(markers-0.1, avgs, width=0.2, label='4 Dimensional Matrix', color='g')

data = np.genfromtxt('sp6.csv', delimiter=',')
avgs = np.average(data[:, 1:], axis=1)
plt.bar(markers+0.1, avgs, width=0.2, label='6 Dimensional Matrix', color='b')

data = np.genfromtxt('sp8.csv', delimiter=',')
avgs = np.average(data[:, 1:], axis=1)
plt.bar(markers+0.3, avgs, width=0.2, label='8 Dimensional Matrix', color='y')



plt.xticks(markers, data[:, 0])
plt.legend()
plt.ylabel("Log Bytes")
plt.xlabel("Sparsity(%)")
plt.yscale('log')
plt.show()

