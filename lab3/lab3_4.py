import numpy as np
import math
import matplotlib.pyplot as plt

x = np.array([-1.0, 0, 1.0, 2.0, 3.0])
y = np.array([1.3562, 1.5708, 1.7854, 2.4636, 3.3218])


# x = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
# y = np.array([1.0, 1.1052, 1.2214, 1.3499, 1.4918], dtype=np.float64)

def get_polynom(x,y):
    x = np.copy(x)
    y = np.copy(y)
    def polynom(arg):
        i = np.searchsorted(x, arg)
        if (arg == x[0]):
            i = 1
        i -= 1
        return y[i] \
               + (arg - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) \
               + (arg - x[i]) * (arg - x[i + 1]) * ((y[i + 2] - y[i + 1])/(x[i + 2] - x[i + 1]) - (y[i + 1] - y[i])/(x[i + 1] - x[i])) / (x[i + 2] - x[i])
    return polynom



def get_polynom_derivative(x,y):
    x = np.copy(x)
    y = np.copy(y)

    def polynom(arg):
        i = np.searchsorted(x, arg)
        if (arg == x[0]):
            i = 1
        i -= 1
        return (y[i + 1] - y[i])/(x[i + 1] - x[i]) + \
               ((y[i + 2] - y[i + 1])/(x[i + 2] - x[i + 1]) - (y[i + 1] - y[i])/(x[i + 1] - x[i])) * (2 * arg - x[i] - x[i + 1]) / (x[i + 2] - x[i])

    return polynom

def get_polynom_derivative2(x,y):
    x = np.copy(x)
    y = np.copy(y)

    def polynom(arg):
        i = np.searchsorted(x, arg)
        if (arg == x[0]):
            i = 1
        i -= 1
        return 2 * ((y[i + 2] - y[i + 1])/(x[i + 2] - x[i + 1]) - (y[i + 1] - y[i])/(x[i + 1] - x[i])) / (x[i + 2] - x[i])

    return polynom

def approx_polynom(x,y):
    x = np.copy(x)
    y = np.copy(y)
    return get_polynom(x,y), get_polynom_derivative(x,y), get_polynom_derivative2(x,y)

f, fd, fd2 = approx_polynom(x,y)

print("f(x) = {}, f'(x) = {}, f''(x) = {}".format(f(0.2), fd(0.2), fd2(0.2)))