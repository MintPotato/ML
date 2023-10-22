import numpy as np
import matplotlib.pyplot as plt
from time import time


class Layer:
    funcs = {'sigmoid': (lambda x: (1 / (1 + np.exp(-x))), lambda x: x * (1.0 - x)),
             'tanh': (lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1), lambda x: 1.0 - x ** 2)}

    def __init__(self, knots: int, func, prev_knots_n: int):
        self.knots: list[np.array] = [i for i in range(knots)]
        self.outputs: np.array = np.zeros(knots)
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

    def adjust_weights(self, learn_rate: float, prev_outputs: np.array):
        self.local_errors = self.local_errors * [__class__.funcs[self.func][1](x) for x in self.outputs]
        for knot in range(len(self.knots)):
            self.knots[knot] -= learn_rate * self.local_errors[knot] * prev_outputs

    def compute_next_errors(self):
        output = []

        if len(self.knots) != 1:
            weights = tuple(zip(*self.knots))
        else:
            weights = tuple(zip(*self.knots, np.zeros((len(self.knots[0])))))

        for w in weights:
            output += [sum(w * self.local_errors)]
        return output[1:]


class Model:
    def __init__(self, number_of_inputs: int):
        self.layers: list[Layer] = []
        self.inputs: np.array = None
        self.true_outputs: np.array = None
        self.n_of_inputs: int = number_of_inputs
        self.learn_rate: float = 0

        self.train_errors: list = []
        self.test_errors: list = []

    def add_layer(self, n, func):
        self.layers += [Layer(n, func, self.n_of_inputs)]
        self.n_of_inputs = len(self.layers[-1].knots)

    def learn_w_tests(self, x_train: np.array, y_train: np.array,
                      x_test: np.array, y_test: np.array, learn_rate: float, epoches: int):
        self.learn_rate = learn_rate
        self.inputs = x_train
        self.true_outputs = y_train
        index_list: list[int] = [i for i in range(len(x_train))]

        start_time = time()

        for epoch in range(epoches):
            print(f'Now epoch {epoch} | Time: {time() - start_time}')

            np.random.shuffle(index_list)
            avrg_error = 0
            for inp in index_list:
                self.forward_pass(inp)
                avrg_error += np.abs((self.true_outputs[inp] - self.layers[-1].outputs) / len(self.true_outputs[inp]))
                self.backward_pass(inp)
            self.train_errors += [avrg_error / len(index_list)]

            avrg_error = 0
            for test in range(len(x_test)):
                pass
                output = self.compute(x_test[test])
                avrg_error += np.abs((y_test[test] - output) / len(y_test[test]))
            self.test_errors += [avrg_error / len(x_test)]

        self.save_weights()
        self.save_errors()

    def learn(self, x_train: np.array, y_train: np.array, learn_rate: float, epoches: int):
        self.learn_rate = learn_rate
        self.inputs = x_train
        self.true_outputs = y_train
        index_list: list[int] = [i for i in range(len(x_train))]

        start_time = time()

        for epoch in range(epoches):
            # print(f'Now epoch {epoch} | Time: {time() - start_time}')

            np.random.shuffle(index_list)
            avrg_error = 0
            for inp in index_list:
                self.forward_pass(inp)
                avrg_error += sum(np.abs(self.true_outputs[inp] - self.layers[-1].outputs)) / len(self.true_outputs[inp])
                self.backward_pass(inp)

            self.train_errors += [avrg_error / len(index_list)]

        print(f'Полное время: {time() - start_time}')
        self.save_weights()
        self.save_errors()

    def forward_pass(self, inp):
        st = time()
        neuron_input = np.append([1], self.inputs[inp])
        for layer in self.layers:
            layer.compute_outputs(neuron_input)
            neuron_input = np.append([1], layer.outputs)
        print(time() - st, end=' ')

    def backward_pass(self, inp):
        st = time()
        next_error = []

        for out in range(len(self.layers[-1].outputs)):
            next_error += [-(self.true_outputs[inp][out] - self.layers[-1].outputs[out])]

        layers_outputs = [np.append([1], self.inputs[inp])] + [np.append([1], layer.outputs) for layer in self.layers]

        for layer in range(len(self.layers) - 1, -1, -1):
            self.layers[layer].local_errors = np.array(next_error)
            self.layers[layer].adjust_weights(self.learn_rate, layers_outputs[layer])
            next_error = self.layers[layer].compute_next_errors()

        print(time() - st)

    def compute(self, x: np.array):
        st = time()
        outputs = np.append([1], x)
        for layer in self.layers:
            layer.compute_outputs(outputs)
            outputs = np.append([1], layer.outputs)
        print(time() - st)
        return np.round(outputs[1:], 2)

    def save_weights(self):
        with open('weights.npy', 'wb') as f:
            for layer in self.layers:
                array = [knot for knot in layer.knots]
                np.save(f, np.array(array))

    def read_weights(self):
        with open('weights.npy', 'rb') as f:
            for layer in self.layers:
                weights = np.load(f)
                for knot in range(len(weights)):
                    layer.knots[knot] = weights[knot]

    def save_errors(self):
        with open('errors.npy', 'wb') as f:
            np.save(f, np.array(self.train_errors))
            np.save(f, np.array(self.test_errors))

    def show_errors(self):
        with open('errors.npy', 'rb') as f:
            train_errors = np.load(f)
            print(f'{np.round(train_errors[-1] * 100, 2)}%')
            test_errors = np.load(f)

        plt.title('Ошибка тренировки')
        plt.xlabel('Количество эпох')
        plt.ylabel('Ошибка выхода')
        plt.plot([i for i in range(len(train_errors))], train_errors)
        plt.show()

        if len(test_errors) != 0:
            plt.title('Ошибка теста')
            plt.xlabel('Количество эпох')
            plt.ylabel('Ошибка выхода')
            plt.plot([i for i in range(len(train_errors))], train_errors)
            plt.show()


if __name__ == '__main__':
    np.random.seed(128)
    m = Model(2)
    m.add_layer(16, 'tanh')
    m.add_layer(64, 'tanh')
    m.add_layer(2, 'sigmoid')

    train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    out = np.array([[0, 1], [1, 0], [1, 0], [1, 0]])
    m.learn(train, out, 0.1, 500)

    m.show_errors()
