import numpy as np


# Задание №1: Создаем функцию для генерации последовательности случайных чисел методом Линейного конгруэнтного генератора (LCG)
def random_sequence(n, a=22695477, b=1, m=2 ** 32):
    x0 = 1
    result = []
    for _ in range(n):
        x0 = (a * x0 + b) % m
        result.append(x0)
    return result


print(f"Задание 1: {random_sequence(5)}\n")

# Задание №2: Генерируем последовательности случайных чисел
A = 0
B = 10
N_values = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]

for N in N_values:
    generated_sequence = random_sequence(N, a=22695477, b=1, m=2 ** 32)
    uniform_sequence = [A + (x / 2 ** 32) * (B - A) for x in generated_sequence]
    print(f"Длина последовательности N={N}")

    # Задание №3: Рассчитываем математическое ожидание и дисперсию для каждой последовательности
    mean = np.mean(uniform_sequence)
    variance = np.var(uniform_sequence)

    print(f"Математическое ожидание: {mean}")
    print(f"Дисперсия: {variance}")

    # Задание №4: Определяем период последовательности (количество чисел до начала повторений)
    sequence_set = set(uniform_sequence)
    period = len(uniform_sequence) - len(sequence_set)
    print(f"Период последовательности: {period}")
    print("\n")

# Задание №5,6:
import matplotlib.pyplot as plt
import numpy as np


# Функция для вычисления относительных частот
def calculate_relative_frequencies(sequence, left, right, num_intervals):
    interval_width = (right - left) / num_intervals
    intervals = [(left + i * interval_width, left + (i + 1) * interval_width) for i in range(num_intervals)]
    counts = [0] * num_intervals
    for value in sequence:
        for i, (start, end) in enumerate(intervals):
            if start <= value < end:
                counts[i] += 1
                break
    relative_freqs = [count / len(sequence) for count in counts]
    return relative_freqs


# Функция для вычисления критерия Пирсона
def chi_squared_test(observed, expected):
    return sum((observed - expected) ** 2 / expected)


# Функция для анализа последовательности
def analyze_sequence(sequence, num_intervals):
    A = min(sequence)
    B = max(sequence)

    # Вычисление относительных частот
    relative_freqs = calculate_relative_frequencies(sequence, A, B, num_intervals)

    # Создание интервалов для гистограммы
    interval_width = (B - A) / num_intervals
    intervals = [A + i * interval_width for i in range(num_intervals + 1)]

    # Вычисление ожидаемых частот (равномерное распределение)
    expected_freq = len(sequence) / num_intervals
    expected = np.array([expected_freq] * num_intervals)

    # Вычисление критерия Пирсона
    chi_square = chi_squared_test(np.array(relative_freqs) * len(sequence), expected)

    # Построение гистограммы
    plt.bar(intervals[:-1], relative_freqs, width=interval_width, align='edge')
    plt.xlabel('Интервалы')
    plt.ylabel('Относительные частоты')
    plt.title(f'Гистограмма относительных частот (n = {len(sequence)})')
    plt.show()

    return chi_square


# Создание случайной последовательности
# np.random.seed(0)
# random_sequence = np.random.uniform(0, 10, 1000)

import random
# Создание последовательности с использованием встроенного генератора случайных чисел в Python (библиотека random)
random.seed(0)  # Инициализация генератора случайных чисел
random_sequence = [random.uniform(0, 10) for _ in range(1000)]  # Генерация 1000 случайных чисел


# Проведение анализа и вывод результата
chi_squared_value = analyze_sequence(random_sequence, 10)
print(f"Значение критерия Пирсона: {chi_squared_value:.4f}")
