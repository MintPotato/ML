import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_openml
from back_prop_model import Model as Model
from back_prop_6 import Model as Model1
from time import time


# def draw_pictures(draw_list, shape):
#     fig, axs = plt.subplots(*shape)
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             digit = draw_list[shape[0] * i + j]
#             digit = digit.reshape(28, 28)
#             axs[i, j].imshow(digit)
#             axs[i, j].axis('off')
#             axs[i, j].set_title(f'{bool(np.round(model.compute(draw_list[i * shape[0] + j])))}')
#
#     plt.show()
#
#
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# images, numbers = mnist['data'], mnist['target']
#
# images = images / 255
# numbers1 = np.array(list(map(lambda x: 1 if x == '3' else 0, numbers)))
# numbers1 = numbers1.reshape(70000, 1)

with open('booba.npy', 'rb') as f:
    images = np.load(f)
    numbers1 = np.load(f)

# model = Model(images[:1000], numbers1[:1000], 0.1)
# model.add_layer(16, 'tanh')
# model.add_layer(64, 'tanh')
# model.add_layer(1, 'sigmoid')
#
# model.learn(50)
# model.show_errors()
# print(model.average_errors[-1])
# draw_pictures(images[-25:], (5, 5))


model1 = Model1(784)
model1.add_layer(16, 'tanh')
model1.add_layer(64, 'tanh')
model1.add_layer(1, 'sigmoid')

model1.learn(images[:1000], numbers1[:1000], 0.1, 50)
model1.show_errors()


model1.compute(images[0])

