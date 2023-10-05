import random
import matplotlib.pyplot as plt

x1 = (1.8, 0.5, -0.6, -1.8)
x2 = (-1.7, -0.8, 0.7, 1.8)

x_train = [0] * 16
y_train = [-1] * 16

for i in range(4):
    for j in range(4):
        x_train[i * 4 + j] = (1.0, x1[i], x2[j])

y_train[7], y_train[10], y_train[11], y_train[13], y_train[14], y_train[15] = 1, 1, 1, 1, 1, 1

w = [0.1, -0.5, 0.3]

random.seed(7)
learn_rate = 0.1
index_list = [i for i in range(16)]


def show_learn(w):
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])


def compute_output(x, w):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]
    if z > 0:
        return 1
    else:
        return -1


def learning():
    sol_found = False
    plt.xticks(x2)
    plt.yticks(x1)
    plt.xlim(-1.8, 1.8)
    plt.ylim(-1.8, 1.8)
    plt.xlabel('x2')
    plt.ylabel('x1')
    while not sol_found:
        sol_found = True

        random.shuffle(index_list)
        for i in index_list:
            x = x_train[i]
            y = y_train[i]
            output = compute_output(x, w)

            if output != y:
                for j in range(len(w)):
                    w[j] += y * learn_rate * x[j]
                sol_found = False
                show_learn(w)
                draw_line(plt)
    draw_line(plt, 'g')
    draw_dots(plt)
    plt.show()


def draw_line(plot, color=''):
    x_coord = [x / 10 for x in range(-20, 20)]
    y_coord = [(-w[2] * x) / w[1] - w[0] / w[1] for x in x_coord]
    plot.plt(x_coord, y_coord, color)


def draw_dots(plot):
    x_coord = [x / 10 for x in range(-20, 20)]
    y_coord = [(-w[2] * x) / w[1] - w[0] / w[1] for x in x_coord]

    for i in range(0, 41, 3):
        for j in range(-20, 20, 3):
            if j / 10 > y_coord[i]:
                plot.plt(x_coord[i], j / 10, marker='_', color='blue')
            else:
                plot.plt(x_coord[i], j / 10, marker='+', color='red')


show_learn(w)
learning()

plt.xticks(x2)
plt.yticks(x1)
plt.xlim(-1.8, 1.8)
plt.ylim(-1.8, 1.8)
plt.xlabel('x2')
plt.ylabel('x1')


plt.show()
