from itertools import product
import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(7)


class Perceptron:
    def __init__(self, x_data, y_data, w):
        self.x_data = x_data
        self.y_data = y_data
        self.w = w
        self.learn_rate = 0.1

    def learn(self):
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        sol_found = False
        index_list = [i for i in range(len(self.x_data))]
        while not sol_found:
            sol_found = True
            random.shuffle(index_list)

            for i in index_list:
                x = self.x_data[i]
                y = self.y_data[i]
                output = self.compute_output(x)

                if output != y:
                    sol_found = False
                    for j in range(len(self.w)):
                        self.w[j] += y * self.learn_rate * x[j]
                    self.draw_line(plt)

        self.draw_line(plt, 'g')
        self.draw_dots()
        plt.show()

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        self.draw_line(plt, 'g')
        self.draw_dots()
        plt.show()

    def compute_output(self, x):
        z = np.dot(x, self.w)
        return np.sign(z)

    def draw_line(self, plot, color=''):
        x_coord = [x / 10 for x in range(-15, 16)]
        y_coord = [(-self.w[2] * x) / self.w[1] - self.w[0] / self.w[1] for x in x_coord]
        plot.plot(x_coord, y_coord, color)

    def draw_dots(self):
        x_coord = [-1, 1]
        y_coord = [-1, 1]

        for x in x_coord:
            for y in y_coord:
                if self.compute_output([1, x, y]) > 0:
                    plt.plot(x, y, marker='+', color='r')
                else:
                    plt.plot(x, y, marker='_', color='b')


class XOR(Perceptron):
    def __init__(self, x_data, y_data, w):
        super().__init__(x_data, y_data, w)

    def compute_output(self, x):
        in_1 = p_or.compute_output(x)
        in_2 = p_nand.compute_output(x)
        out = np.sign(np.dot([1, in_1, in_2], self.w))
        return out


class NOT(Perceptron):
    def __init__(self, x_data, y_data, w):
        super().__init__(x_data, y_data, w)

    def draw_line(self, plot, color=''):
        x_coord = [x / 10 for x in range(-15, 16)]
        y_coord = [-self.w[0] / self.w[1] for _ in x_coord]
        plot.plot(x_coord, y_coord, color)

    def draw_dots(self):
        y_coord = [-1, 1]

        for y in y_coord:
            if self.compute_output([1, y]) > 0:
                plt.plot(0, y, marker='+', color='r')
            else:
                plt.plot(0, y, marker='_', color='b')


x_train = list(product([-1, 1], repeat=2))
x_train = [[1, *x] for x in x_train]

p_or = Perceptron(x_train, [-1, 1, 1, 1], [0.1, -0.5, 0.3])
p_or.learn()

p_and = Perceptron(x_train, [-1, -1, -1, 1], [0.1, -0.5, 0.3])
p_and.learn()

p_not = NOT([[1, -1], [1, 1]], [1, -1], [0.1, -0.5])
p_not.learn()

p_nand = Perceptron(x_train, [1, 1, 1, -1], [0.1, -0.5, 0.3])
p_nand.learn()

p_xor = XOR(x_train, [-1, 1, 1, -1], [0, -0.5, 0.3])
p_xor.learn()

x_axis = list(product([-1, 1], repeat=4))
x_axis = [[1, *x] for x in x_axis]

print('x1 x2 x3 x4 f')
for el in x_axis:
    xor_output = p_xor.compute_output([1, el[1], el[3]])
    or_output = p_or.compute_output([1, el[2], el[4]])
    and_output = p_and.compute_output([1, xor_output, or_output])
    not_output = int(p_not.compute_output([1, and_output]))
    print(f'{not_output}, '.replace('-1', '0'), end=' ')
    # print(f'{el[1]}  {el[2]}  {el[3]}  {el[4]}  {not_output}'.replace('-1', '0'))