import numpy as np
import matplotlib.pyplot as plt
from time import time

np.random.seed(128)


class Layer:
    funcs = {'sigmoid': (lambda x: (1 / (1 + np.exp(-x))), lambda x: x * (1.0 - x)),
             'tanh': (lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1), lambda x: 1.0 - x ** 2)}

    def __init__(self, knots: int, func, prev_knots_n):
        self.knots: list[np.array] = [i for i in range(knots)]
        self.outputs: list[float] = [0 for _ in range(knots)]
        self.local_errors = None
        self.prev_knots_n: int = prev_knots_n
        self.func = func

        self.set_weights()

    def set_weights(self):
        for knot in self.knots:
            self.knots[knot] = np.array([1] + [np.random.uniform(-1, 1) for _ in range(self.prev_knots_n)])

    def compute_outputs(self, x):
        for knot in range(len(self.knots)):
            self.outputs[knot] = __class__.funcs[self.func][0](np.dot(self.knots[knot], x))

    def adjust_weights(self, learn_rate, prev_outputs):
        self.local_errors = self.local_errors * [__class__.funcs[self.func][1](x) for x in self.outputs]
        for knot in range(len(self.knots)):
            self.knots[knot] -= learn_rate * self.local_errors[knot] * np.array(prev_outputs)

    def compute_next_errors(self):
        output = []

        if len(self.knots) != 1:
            weights = list(zip(*self.knots))
        else:
            weights = list(zip(*self.knots, np.zeros((len(self.knots[0])))))

        for w in weights:
            output += [sum(w * self.local_errors)]
        return output[1:]


class Model:
    def __init__(self, inputs: np.array, output: np.array, learn_rate: float):
        self.layers: list[Layer] = []
        self.inputs: np.array = inputs
        self.true_outputs: np.array = output
        self.n_of_inputs: int = len(inputs[0])
        self.index_list: list[int] = [i for i in range(len(inputs))]
        self.learn_rate: float = learn_rate
        self.average_errors = []

    def add_layer(self, n, func):
        self.layers += [Layer(n, func, self.n_of_inputs)]
        self.n_of_inputs = len(self.layers[-1].knots)

    def learn(self, epoches=10):
        start_time = time()
        for epoch in range(epoches):
            print(f'Now epoch {epoch} / time since start: {time() - start_time}')
            np.random.shuffle(self.index_list)
            avrg_error = 0
            for inp in self.index_list:
                self.forward_pass(inp)
                avrg_error += (abs(sum(self.true_outputs[inp]) - sum(self.layers[-1].outputs))) / len(self.true_outputs[inp])
                self.backward_pass(inp)
            self.average_errors += [np.abs(avrg_error / len(self.index_list))]
        self.save_weights()

    def forward_pass(self, inp):
        neuron_input = [1] + list(self.inputs[inp])
        for layer in self.layers:
            layer.compute_outputs(neuron_input)
            neuron_input = [1.0] + layer.outputs

    def backward_pass(self, inp):
        next_error = []

        for out in range(len(self.layers[-1].outputs)):
            next_error += [-(self.true_outputs[inp][out] - self.layers[-1].outputs[out])]

        layers_outputs = [[1] + list(self.inputs[inp])] + [[1] + layer.outputs for layer in self.layers] # list()
        for layer in range(len(self.layers) - 1, -1, -1):
            self.layers[layer].local_errors = np.array(next_error)
            self.layers[layer].adjust_weights(self.learn_rate, layers_outputs[layer])
            next_error = self.layers[layer].compute_next_errors()

    def compute(self, x: np.array):
        outputs = [1] + list(x)
        for layer in self.layers:
            layer.compute_outputs(outputs)
            outputs = [1] + layer.outputs

        return np.round(outputs[1:], 2)

    def save_weights(self):
        with open('weights.npy', 'wb') as f:
            for layer in self.layers:
                array = [knot for knot in layer.knots]
                np.save(f, np.array(array))
            np.save(f, np.array(self.average_errors))

    def read_weights(self):
        with open('weights.npy', 'rb') as f:
            for layer in self.layers:
                weights = np.load(f)
                for knot in range(len(weights)):
                    layer.knots[knot] = weights[knot]

    def show_errors(self):
        plt.xlabel('Количество итераций обучения')
        plt.ylabel('Средняя ошибка на итерацию')
        x = [i for i in range(len(self.average_errors))]
        plt.plot(x, self.average_errors)
        plt.show()


if __name__ == '__main__':
    a = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    out = [[0], [1], [1], [1]]

    m = Model(a, out, 0.1)
    m.add_layer(16, 'tanh')
    m.add_layer(64, 'tanh')
    m.add_layer(1, 'sigmoid')

    m.learn(100)
    print(m.compute(a[0]))
    m.show_errors()
