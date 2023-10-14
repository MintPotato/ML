import os
import numpy as np
import sklearn
# Указываем куда сохранять рисунки
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# для стабильного запуска примера более одного раза
np.random.seed(42)

# импортируем библиотеку позволяющую нам брать данные с сайта openml, что мы и делаем импортируя словарь с цифрами из MNIST
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys() # ключи из полученного словаря возврающаются куда-то

# передаем матрицу признаков и список со значениями соответствующими своим матрицам признаков
X, y = mnist["data"], mnist["target"]
X.shape # возвращает размерность массива матрицы признаков

print(X.shape)
#возвращает размерность массива со значениями
y.shape
print(y.shape)

# размерность изображений в
28 * 28

# импортируем библиотеки matplotlib для работы с изображениями
import matplotlib as mpl
import matplotlib.pyplot as plt

# выбираем матрицу признаков (матрица содержащая цвет каждого пикселя изображения) и задаем ей размерность 28х28, далее выводим ее через plt с цветокоррекциец оттенков черного
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

# сохраняем выбранное изображение в файл
save_fig("some_digit_plot")
plt.show()

