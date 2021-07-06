import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt


def newton(fun, der, x, epsilon=0.001):
    iter_info = [[0, x, fun(x)]]
    iter_number = 1
    while True:
        new_x = x - fun(x) / der(x)
        iter_info.append([iter_number, new_x, fun(new_x)])
        if np.abs(new_x - x) < epsilon:
            return iter_info
        x = new_x
        iter_number = iter_number + 1

def simple_iteration_prep(fi_der, x0, x1, step = 0.001):
    interval = np.arange(x0, x1, step)
    return max(np.abs(fi_der(interval)))

def simple_iteration(fi, x, q, fun, epsilon = 0.001):
    iter_info = [[0, x, fi(x)]]
    iter_number = 1
    while True:
        new_x = fi(x)
        iter_info.append([iter_number, new_x, fun(x)])
        if (q / (1 - q)) * np.abs(new_x - x) < epsilon:
            return iter_info
        x = new_x
        iter_number = iter_number + 1


def secant(fun, x0, x1, epsilon=0.001):
    iter_info = [[0, x0, fun(x0)], [1, x1, fun(x1)]]
    iter_number = 2
    while True:
        new_x0, new_x1 = x1, x1 - (fun(x1) * (x1 - x0))/(fun(x1) - f(x0))
        iter_info.append([iter_number, new_x1, fun(new_x1)])
        if np.abs(new_x1 - new_x0) < epsilon:
            return iter_info
        x0, x1 = new_x0, new_x1
        iter_number = iter_number + 1


def f(x):
    return np.tan(x) - 5 * np.power(x,2) + 1


# функция фи, полученная заменой уравнения f(x) = 0 эквивалентным x = fi(x)
def fi(x):
    return np.sqrt((np.tan(x) + 1) / 5)


def fi_der(x):
    return 1 / (10 * np.sqrt((np.tan(x) + 1) / 5) * np.cos(x)**2)


def get_derivative(fun, epsilon=0.0001):
    return lambda x: (fun(x + epsilon) - fun(x)) / epsilon


t1 = np.arange(-1, 1, 0.0001)

der = get_derivative(f)
der2 = get_derivative(der)

plt.plot(t1, f(t1), label="f(x)")
plt.plot(t1, der(t1), label="f'(x)")
plt.plot(t1, der2(t1), label="f''(x)")
plt.legend()
plt.grid()
plt.show()

t2 = np.arange(-np.pi / 4, 1, 0.01)

plt.plot(t2, fi(t2), label="phi(x)")
plt.plot(t2, get_derivative(fi)(t2), label="phi'(x)")
#plt.plot(t2, fi_der(t2))
plt.xticks(np.arange(-1, 1, 0.1))
plt.axis([-1, 1, -2, 2])
plt.legend()
plt.grid()
plt.show()

first_root_interval = (-0.75, 0)
second_root_interval = (0.5, 0.8)

epsilon = 0.001

newton_roots = [newton(f, get_derivative(f, epsilon), -0.5, epsilon), newton(f, get_derivative(f, epsilon), 0.75, epsilon)]
secant_roots = [secant(f, -0.5, -0.5 + 0.01, epsilon), secant(f, 0.75, 0.75 - 0.01, epsilon)]

q = simple_iteration_prep(fi_der, *second_root_interval)
simple_iteration_root = simple_iteration(fi, -0.3, q, f, epsilon)

print("Newton roots")
for info in newton_roots:
    print(tabulate(info, headers=['iteration', 'x', 'f(x)']))

print("Secant roots")
for info in secant_roots:
    print(tabulate(info, headers=['iteration', 'x', 'f(x)']))

print("Simple iterations")
print("Chosen interval: [{},{}]".format(*second_root_interval))
print("q: {}".format(q))
print(tabulate(simple_iteration_root, headers=['iteration', 'x', 'f(x)']))

# print("newton root 1 x: {}, f(x): {}".format(newton_roots[0], f(newton_roots[0])))
# print("newton root 2 x: {}, f(x): {}".format(newton_roots[1], f(newton_roots[1])))
# print("secant root 1 x: {}, f(x): {}".format(secant_roots[0], f(secant_roots[0])))
# print("secant root 2 x: {}, f(x): {}".format(secant_roots[1], f(secant_roots[1])))
# print("simple iteration root x: {}, f(x): {}".format(simple_iteration_root, f(simple_iteration_root)))

