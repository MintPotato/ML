import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from mega_ns import Model, Layer

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
images, numbers = mnist['data'], mnist['target']

numbers = np.array(list(map(int, numbers)))
# images = images.reshape(70000, 784, 1)
numbers = numbers.reshape(70000, 1)
print(numbers.shape, numbers, numbers[0])


model = Model(images, numbers, 0.1)
model.add_layer(64, 'tanh')
model.add_layer(128, 'tanh')
model.add_layer(1, 'sigmoid')


model.learn(100)

print(model.compute(images[0]))
print(numbers[0])
