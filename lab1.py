import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os

# Параметры обучающей и тестовой выборки
symbols = ["D", "E", "F", "K"]
train_fonts = [ImageFont.truetype(font, size=19) for font in ["times.ttf", "comic.ttf", "BLOODSUC.ttf", "haunted.ttf"]]
test_fonts = [ImageFont.truetype(font, size=19) for font in ["Byedaft.otf", "Alice.ttf"]]
image_size = (32, 32)

# Некоторые константы, вытекающие из параметров
X_vector_len = image_size[0] * image_size[1]
X_train_shape = (len(symbols)*len(train_fonts), X_vector_len)
y_train_shape = (len(symbols)*len(train_fonts), len(symbols))
X_test_shape = (len(symbols)*len(test_fonts), X_vector_len)
y_test_shape = (len(symbols)*len(test_fonts), len(symbols))

# Параметры обучения
learning_rate = 0.1
offset = 0.03  # Смещение синаптических весов
theta = 1  # Порог активации нейрона


# Функция для склонения слов после числительных
def postfix(v, pf):
    if v % 10 in (0, 5, 6, 7, 8, 9) or v % 100 in (11, 12, 13, 14):
        p = pf[2]
    elif v % 10 == 1:
        p = pf[0]
    else:
        p = pf[1]
    return f'{v} {p}'


# Функция активации нейрона
def activation_function(net):
    return 1 if net >= theta else 0
activation_function = np.vectorize(activation_function)


# Обучающий алгоритм
def train(X_train, y_train):
    # Инициализация весов
    weights = np.random.rand(X_vector_len, len(symbols)) * offset * 2 - offset
    bias = np.random.rand(len(symbols)) * offset * 2 - offset
    # Параметры цикла
    errors = np.ones(y_train_shape, dtype=np.float64)
    epochs = 0
    while errors.any():
        for s, sample in enumerate(X_train):
            desired = y_train[s]
            net = np.dot(sample, weights) + bias  # Взвешенная сумма
            y_pred = activation_function(net)  # Фукнция активации для вычисления выходов

            # Коррекция весов и смещения
            e = desired - y_pred
            errors[s] = e
            for j in range(len(symbols)):
                for i in range(X_vector_len):
                    weights[i, j] += learning_rate * float(e[j]) * float(sample[i])
                bias[j] += learning_rate * float(e[j])
        epochs += 1
    return weights, bias, epochs


# Тестирование готового перцептрона
def test(X_test, weights, bias):
    correct_predictions = 0
    for j, text in enumerate(symbols):
        for i, font in enumerate(test_fonts):
            s = j*len(test_fonts)+i
            sample = X_test[s]
            net = np.dot(sample, weights) + bias
            print(f"Тест {s+1}: Символ {text} шрифтом {font.getname()[0]}. Предсказание: {symbols[net.argmax()]} ({net})")
            if text == symbols[net.argmax()]:
                correct_predictions += 1
    return correct_predictions / len(X_test)


# Генерация обучающей и тестовой выборки
def generate_data(save_samples=False):
    if save_samples and "samples" not in os.listdir():
        os.mkdir("samples")

    X_train = np.zeros(X_train_shape, dtype=np.float64)
    y_train = np.zeros(y_train_shape, dtype=np.float64)
    sample_idx = 0
    for j, text in enumerate(symbols):
        for font in train_fonts:
            image = Image.new("RGB", image_size)
            draw = ImageDraw.Draw(image)
            draw.text((15, 15), text, font=font, anchor="mm")
            if save_samples:
                image.save(f"samples/{text}_{font.getname()[0]}.png")
            X_train[sample_idx] = np.append(np.asarray(image, dtype=np.float64)[:,:,0]/255, [])
            y_train[sample_idx] = np.array([1 if i == j else 0 for i in range(len(symbols))], dtype=np.float64)
            sample_idx += 1

    X_test = np.zeros(X_test_shape, dtype=np.float64)
    sample_idx = 0
    for j, text in enumerate(symbols):
        for font in test_fonts:
            image = Image.new("RGB", image_size)
            draw = ImageDraw.Draw(image)
            draw.text((15, 15), text, font=font, anchor="mm")
            if save_samples:
                image.save(f"samples/{text}_{font.getname()[0]}.png")
            X_test[sample_idx] = np.append(np.asarray(image, dtype=np.float64)[:,:,0]/255, [])
            sample_idx += 1

    return X_train, y_train, X_test


if __name__ == "__main__":
    # Генерация данных
    X_train, y_train, X_test = generate_data()

    # Обучение перцептрона
    weights, bias, epochs = train(X_train, y_train)
    print(f"Модель обучена за {postfix(epochs, ['эпоху', 'эпохи', 'эпох'])}")

    # Проверка тестовой выборки и оценка точности
    accuracy = test(X_test, weights, bias)
    print(f"Точность модели на тестовой выборке: {accuracy * 100:.2f}%")
