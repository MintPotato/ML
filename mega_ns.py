import numpy as np
from itertools import product
import matplotlib.pyplot as plt

np.random.seed(256)


class Layer:
    sigmoid = lambda x: (1 / (1 + np.exp(-x)))
    tanh = lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    sigmoid_der = lambda x: x * (1.0 - x)
    tanh_der = lambda x: 1.0 - x ** 2

    def __init__(self, knots: int, func, prev_knots: int):
        self.knots: list[np.array] = [i for i in range(knots)]
        self.outputs: list[float] = [0 for _ in range(knots)]
        self.local_errors = None
        self.prev_knots: int = prev_knots

        self.set_weights()

        if func == 'sigmoid':
            self.func = __class__.sigmoid
            self.der = __class__.sigmoid_der
        elif func == 'tanh':
            self.func = __class__.tanh
            self.der = __class__.tanh_der
        else:
            exit('хуйня а не функция')

    def set_weights(self):
        for knot in self.knots:
            self.knots[knot] = np.array([1] + [np.random.uniform(-1.0, 1.0) for _ in range(self.prev_knots)])

    def compute_outputs(self, x):
        for knot in range(len(self.knots)):
            self.outputs[knot] = self.func(np.dot(self.knots[knot], x))

    def adjust_weights(self, learn_rate, prev_outputs):
        self.local_errors = self.local_errors * [self.der(x) for x in self.outputs]
        for knot in range(len(self.knots)):
            self.knots[knot] -= learn_rate * self.local_errors[knot] * np.array(prev_outputs)

    def compute_next_errors(self):
        output = list()

        if len(self.knots) != 1:
            weights = list(zip(*self.knots))
        else:
            weights = list(zip(*self.knots, np.zeros(len(self.knots[0]))))

        for w in weights:
            output += [sum(w * self.local_errors)]
        return output[1:]


class Model:
    def __init__(self, inputs: list[tuple[int]], output: list[int], learn_rate: float):
        self.layers: list[Layer] = []
        self.inputs: list[tuple[int]] = inputs
        self.n_of_inputs: int = len(inputs[0])
        self.true_outputs: list[int] = output
        self.index_list: list[int] = [i for i in range(len(inputs))]
        self.learn_rate: float = learn_rate
        self.average_errors = []

    def add_layer(self, n, func):
        self.layers += [Layer(n, func, self.n_of_inputs)]
        self.n_of_inputs = len(self.layers[-1].knots)

    def learn(self, steps=1024):
        for i in range(steps):
            np.random.shuffle(self.index_list)
            middle_error = 0
            for inp in self.index_list:
                self.forward_pass(inp)
                middle_error += self.true_outputs[inp] - self.layers[-1].outputs[0]
                self.backward_pass(inp)
            self.average_errors += [np.abs((middle_error / len(self.index_list)))]

    def forward_pass(self, inp):
        neuron_input = (1,) + self.inputs[inp]
        for layer in self.layers:
            layer.compute_outputs(neuron_input)
            neuron_input = [1.0] + layer.outputs

    def backward_pass(self, inp):
        next_error = -(self.true_outputs[inp] - self.layers[-1].outputs[0])
        knots_outputs = [[1] + list(self.inputs[inp])] + [[1] + layer.outputs for layer in self.layers]
        for layer in range(len(self.layers)):
            self.layers[-1 - layer].local_errors = np.array(next_error)
            self.layers[-1 - layer].adjust_weights(self.learn_rate, knots_outputs[-2 - layer])
            next_error = self.layers[-1 - layer].compute_next_errors()

    def compute(self, x: list[int]):
        outputs = [1] + x
        for layer in self.layers:
            layer.compute_outputs(outputs)
            outputs = [1] + layer.outputs
        return np.round(outputs[-1], 2)

    def show_errors(self):
        plt.xlabel('Количество итераций обучения')
        plt.ylabel('Средняя ошибка на итерацию')
        x = [i for i in range(len(self.average_errors))]
        plt.plot(x, self.average_errors)
        plt.show()


x_train = list(product([-1, 1], repeat=4))
y_train = [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1]

m = Model(x_train, y_train, 0.1)
m.add_layer(4, 'tanh')
m.add_layer(8, 'tanh')
m.add_layer(1, 'sigmoid')
m.learn()
m.show_errors()
count = 0
for x in x_train:
    count += 1
    print(f'{count}: Выход нейронной сети {m.compute(list(x))} - Искомый выход {y_train[count - 1]}')

x_train = list(product([-1, 1], repeat=2))
y_train = [0, 1, 1, 1]
m1 = Model(x_train, y_train, 0.1)
m1.add_layer(4, 'tanh')
m1.add_layer(8, 'tanh')
m1.add_layer(1, 'sigmoid')
m1.learn()
m1.show_errors()
count = 0
for x in x_train:
    count += 1
    print(f'{count}: Выход нейронной сети {m1.compute(list(x))} - Искомый выход {y_train[count - 1]}')

'''
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

np.random.seed(256)


class Layer:
    sigmoid = lambda x: (1 / (1 + np.exp(-x)))
    tanh = lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    sigmoid_der = lambda x: x * (1.0 - x)
    tanh_der = lambda x: 1.0 - x ** 2

    def __init__(self, knots: int, func, prev_knots: int):
        self.knots: list[np.array] = [i for i in range(knots)]
        self.outputs: list[float] = [0 for _ in range(knots)]
        self.local_errors = None
        self.prev_knots: int = prev_knots

        self.set_weights()

        if func == 'sigmoid':
            self.func = __class__.sigmoid
            self.der = __class__.sigmoid_der
        elif func == 'tanh':
            self.func = __class__.tanh
            self.der = __class__.tanh_der
        else:
            exit('не функция')

    def set_weights(self):
        for knot in self.knots:
            self.knots[knot] = np.array([1] + [np.random.uniform(-1.0, 1.0) for _ in range(self.prev_knots)])

    def compute_outputs(self, x):
        for knot in range(len(self.knots)):
            self.outputs[knot] = self.func(np.dot(self.knots[knot], x))

    def adjust_weights(self, learn_rate, prev_outputs):
        self.local_errors = self.local_errors * [self.der(x) for x in self.outputs]
        for knot in range(len(self.knots)):
            self.knots[knot] -= learn_rate * self.local_errors[knot] * np.array(prev_outputs)

    def compute_next_errors(self):
        output = list()

        if len(self.knots) != 1:
            weights = list(zip(*self.knots))
        else:
            weights = list(zip(*self.knots, np.zeros(len(self.knots[0]))))

        for w in weights:
            output += [sum(w * self.local_errors)]
        return output[1:]


class Model:
    def __init__(self, inputs: list[tuple[int]], output: list[list[int]], learn_rate: float):
        self.layers: list[Layer] = []
        self.inputs: list[tuple[int]] = inputs
        self.n_of_inputs: int = len(inputs[0])
        self.true_outputs: list[int] = output
        self.index_list: list[int] = [i for i in range(len(inputs))]
        self.learn_rate: float = learn_rate
        self.average_errors = []

    def add_layer(self, n, func):
        self.layers += [Layer(n, func, self.n_of_inputs)]
        self.n_of_inputs = len(self.layers[-1].knots)

    def learn(self, steps=1024):
        for i in range(steps):
            np.random.shuffle(self.index_list)
            middle_error = 0
            for inp in self.index_list:
                self.forward_pass(inp)
                middle_error += self.true_outputs[inp] - self.layers[-1].outputs[0]
                self.backward_pass(inp)
            self.average_errors += [np.abs((middle_error / len(self.index_list)))]

    def forward_pass(self, inp):
        neuron_input = (1,) + self.inputs[inp]
        for layer in self.layers:
            layer.compute_outputs(neuron_input)
            neuron_input = [1.0] + layer.outputs

    def backward_pass(self, inp):
        next_error = []
        for out in range(len(self.layers[-1].outputs)):
            next_error += [-(self.true_outputs[inp][out] - self.layers[-1].outputs[out])]
        knots_outputs = [[1] + list(self.inputs[inp])] + [[1] + layer.outputs for layer in self.layers]
        for layer in range(len(self.layers)):
            self.layers[-1 - layer].local_errors = np.array(next_error)
            self.layers[-1 - layer].adjust_weights(self.learn_rate, knots_outputs[-2 - layer])
            next_error = self.layers[-1 - layer].compute_next_errors()

    def compute(self, x: list[int]):
        outputs = [1] + x
        for layer in self.layers:
            layer.compute_outputs(outputs)
            outputs = [1] + layer.outputs
        return np.round(outputs[1:], 2)

    def show_errors(self):
        plt.xlabel('Количество итераций обучения')
        plt.ylabel('Средняя ошибка на итерацию')
        x = [i for i in range(len(self.average_errors))]
        plt.plot(x, self.average_errors)
        plt.show()


x_train = list(product([-1, 1], repeat=4))
y_train = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]

m = Model(x_train, y_train, 0.1)
m.add_layer(4, 'tanh')
m.add_layer(8, 'tanh')
m.add_layer(3, 'sigmoid')
m.learn(512)
m.show_errors()
count = 0
for x in x_train:
    count += 1
    print(f'{count}: Выход нейронной сети {m.compute(list(x))} - Искомый выход {y_train[count - 1]}')
'''
