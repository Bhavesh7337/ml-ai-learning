import collections
import numpy as np
from numpy import random
import matplotlib.pyplot as plt 

array_out = random.randint(6, size = (10000))

unq, counts = np.unique(array_out, return_counts=True)

counter = collections.Counter(array_out)

plt.plot(unq, counts)
plt.show()
print(array_out)
print(counter)
print(unq)
print(counts)