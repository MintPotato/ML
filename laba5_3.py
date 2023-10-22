import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
images, numbers = mnist['data'], mnist['target']


def find_number(find):
    count = 0
    numbers_to_show = []
    for num in range(len(numbers)):
        if numbers[num] == str(find):
            numbers_to_show += [images[num]]
            count += 1
        if count == 5:
            break
    show_number(numbers_to_show)


def show_number(numbers):
    numbers.append(np.zeros(784))
    print(numbers)
    print(len(numbers))
    fig, axs = plt.subplots(3, 2)
    for i in range(3):
        for j in range(2):
            digit = numbers[2 * i + j]
            print(3 * i + j)
            digit = digit.reshape(28, 28)
            axs[i, j].imshow(digit)
            axs[i, j].axis('off')
    plt.show()


find_number(3)
