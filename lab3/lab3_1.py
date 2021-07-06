
import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    return np.arccos(x) + x


points1 = np.array([-0.4, -0.1, 0.2, 0.5])
points2 = np.array([-0.4, 0, 0.2, 0.5])


def Lagrange(points, values=None, function=None):
    if values is None and function is None:
        raise ValueError("function needs function values or function itself")

    if values is None:
        values = function(points)

    if len(values) != len(points):
        raise ValueError("length is wrong")

    values = values[:]
    points = points[:]
    n = len(values)
    return lambda x: np.sum(
        [values[i] * np.prod([1 if j == i else ((x - points[j]) / (points[i] - points[j])) for j in range(0, n)]) for i
         in range(0, n)], axis = 0)

def divided_diff(points, function):
    if (len(points) == 1):
        return function(points[0])

    if (len(points) == 2):
        return (function(points[0]) - function(points[1])) / (points[0] - points[1])

    n = len(points)
    return (divided_diff(points[:n-1], function) - divided_diff(points[1:], function)) / (points[0] - points[n - 1])

def Newton(points, function):
    points = points[:]
    n = len(points)
    cur_arr = []
    divided_sums_counted = []
    for i in range(0, n):
        cur_arr.append(points[i])
        divided_sums_counted.append(divided_diff(cur_arr, function))

    def result(x):
        cur_prod = 1
        res_sum = 0
        for i in range(0, n):
            res_sum += divided_sums_counted[i] * cur_prod
            cur_prod *= (x - points[i])
        return res_sum

    return result



interval = np.arange(-1, 1, 0.1)


plt.figure()
plt.subplot(221)
plt.plot(interval, [Lagrange(points1, function=f)(x) for x in interval])
plt.plot(interval, f(interval))
plt.title("Lagrange 1")

plt.subplot(222)
plt.plot(interval, [Lagrange(points2, function=f)(x) for x in interval])
plt.plot(interval, f(interval))
plt.title("Lagrange 2")

plt.subplot(223)
plt.plot(interval, [Newton(points1, function=f)(x) for x in interval])
plt.plot(interval, f(interval))
plt.title("Newton 1")

plt.subplot(224)
plt.plot(interval, [Newton(points2, function=f)(x) for x in interval])
plt.plot(interval, f(interval))
plt.title("Newton 2")
plt.show()

