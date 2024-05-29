import random
import numpy as np
from keras.datasets import mnist

# Загрузка обучающей и тестовой выборки
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Выравнивание матрицы пикселей для каждого изображения и нормализация данных
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Некоторые константы, вытекающие из параметров выборки
X_vector_len = 28*28
y_vector_len = 10

# Параметры обучения
learning_rate = 0.01
offset = 0.03  # Смещение синаптических весов
e_threshold = 3e-6  # Порог останова


# Функция для склонения слов после числительных
def postfix(v, pf):
    if v % 10 in (0, 5, 6, 7, 8, 9) or v % 100 in (11, 12, 13, 14):
        p = pf[2]
    elif v % 10 == 1:
        p = pf[0]
    else:
        p = pf[1]
    return f'{v} {p}'


# Функция активации (сигмоида) и её производная
def activation_function(x):
    return 1 / (1 + np.exp(-x))


def activation_derivative(x):
    return activation_function(x) * (1 - activation_function(x))


# Обучающий алгоритм
def train(X_train, y_train):
    # Инициализация весов
    weights = np.random.rand(X_vector_len, y_vector_len) * offset * 2 - offset
    bias = np.random.rand(y_vector_len) * offset * 2 - offset

    epochs = 1
    while True:
        # Получение векторов X и D
        s = random.randrange(len(X_train))
        sample = X_train[s]
        desired = np.zeros(y_vector_len, dtype=np.float64)
        desired[y_train[s]] = 1

        net = np.dot(sample, weights) + bias  # Взвешенная сумма
        y_pred = activation_function(net)  # Вычисление выходов
        e_vec = desired - y_pred  # Ошибка для каждого выходного нейрона
        e = np.sum(e_vec ** 2) / 2  # Ошибка ε для текущего обучающего вектора

        if e < e_threshold:  # Проверка критерия останова
            break

        # Коррекция весов
        delta = e_vec * activation_derivative(net)
        weights += learning_rate * np.outer(sample, delta)
        bias += learning_rate * delta

        epochs += 1
    return weights, bias, epochs


# Тестирование готового перцептрона
def test(X_test, y_test, weights, bias):
    correct_predictions = 0
    for j, text in enumerate(y_test):
        sample = X_test[j]
        net = np.dot(sample, weights) + bias
        print(f"Тест {j+1}: Символ {text}, предсказание: {net.argmax()} ({net})")
        if text == net.argmax():
            correct_predictions += 1
    return correct_predictions / len(X_test)


if __name__ == "__main__":
    # Обучение перцептрона
    weights, bias, epochs = train(X_train, y_train)

    # Проверка тестовой выборки и оценка точности
    accuracy = test(X_test, y_test, weights, bias)

    print(f"Точность модели на тестовой выборке: {accuracy * 100:.2f}%")
    print(f"Модель обучена за {postfix(epochs, ['эпоху', 'эпохи', 'эпох'])}")
