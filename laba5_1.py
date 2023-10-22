import idx2numpy

TRAIN_IMAGES_FILE = 'C:/Users/Andre/OneDrive/Рабочий стол/data/data/mnist/train-images.idx3-ubyte'
TRAIN_LABELS_FILE = 'C:/Users/Andre/OneDrive/Рабочий стол/data/data/mnist/train-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(TRAIN_IMAGES_FILE)
train_label = idx2numpy.convert_from_file(TRAIN_LABELS_FILE)


def number_to_console(el):
    for line in train_images[el]:
        for num in line:
            if num > 0:
                print('*', end='')
            else:
                print(' ', end='')
        print()

count = 0
for el in range(len(train_label)):
    if train_label[el] == 3:
        number_to_console(el)
        count += 1

    if count == 5:
        break
