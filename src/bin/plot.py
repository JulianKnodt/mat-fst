import matplotlib.pyplot as plt
import numpy as np
data = np.genfromtxt('sparsity.csv', delimiter=',')
x = data[:, 0]
markers = np.arange(len(x))

# FST
plt.bar(markers+0.1, data[:, 1], color='r', align='center', label="FST size", width=0.2)
# CSR representation
plt.bar(markers-0.1, data[:, 2], color='b', align='center', label="CSR size", width=0.2)
plt.xticks(markers, x)
plt.yscale('log')
plt.legend()
plt.xlabel('Sparsity(%)')
plt.ylabel('Bytes used by representation(Log Bytes)')
plt.title("Sparsity's affect on compression of representation")

plt.show()
