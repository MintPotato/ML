import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
images, numbers = mnist['data'], mnist['target']


def find_number(num):
    count = 0
    for el in range(len(numbers)):
        if numbers[el] == str(num):
            show_number(el)
            count += 1
        if count == 5:
            break


def show_number(num):
    digit = images[num]
    digit = digit.reshape(28, 28)
    plt.imshow(digit, cmap=mpl.cm.binary)
    plt.axis('off')
    plt.show()


find_number(3)
