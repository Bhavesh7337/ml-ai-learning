import numpy as np
from numpy import random


arr = np.random.randint(100, size=(400))

print(arr)

print("Mean:", arr.mean())
print("Sum:", arr.sum())
print("Max:", arr.max())
print("Min:", arr.min())
print("Std Dev:", arr.std())


vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
print("Dot Product:", np.dot(vec1, vec2))
print("Element-wise Sum:", vec1 + vec2)

arr = np.arange(10)
print(np.random.choice(arr, size=5, replace=False))

