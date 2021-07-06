import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
    return np.sqrt(x) / (4 + 3 * x)

def rect_method(x, f):
    return np.sum([f((x[i + 1] + x[i]) / 2) * (x[i + 1] - x[i]) for i in range(0,len(x) - 1)])

def trapeze_method(x, f):
    y = f(x)
    return np.sum([(y[i + 1] + y[i]) * (x[i + 1] - x[i]) for i in range(0, len(x) - 1)]) / 2.

def simpson_method(x, f):
    y = f(x)
    return np.sum([(f(x[i]) + 4 * f((x[i] + x[i + 1]) / 2) + f(x[i + 1])) * (x[i + 1] - x[i]) for i in range(0, len(x) - 1)]) / 6.

rect_method.runge_coef = 2
trapeze_method.runge_coef = 2
simpson_method.runge_coef = 4

def integrate(method, function, x0, xk, h):
    res = method(np.arange(x0, xk, h), f)
    return {"result": res, "h": h}



def print_results(name, result_list, err):
    print("method: {}, results: {}, errors: {}".format(name,result_list, err))


# analitic_solution = 0.5312191613

def test_method(method, function, x0, xk, h):
    result_list = [integrate(method, function, x0, xk, h_cur) for h_cur in h]
    k = result_list[0]["h"] / result_list[1]["h"]
    err = np.abs(result_list[0]["result"] - result_list[1]["result"]) / (np.power(k, method.runge_coef) - 1)
    # err_analitic = np.abs(result_list[0]["result"] - analitic_solution) / (np.power(k, method.runge_coef) - 1)
    return result_list, err

x0 = 1
xk = 5
h = [1.0, 0.5, 0.1, 0.01]

print("h: {}".format(h))
print_results("rectangle method", *test_method(rect_method, f, x0, xk, h))
print_results("trapeze method", *test_method(trapeze_method, f, x0, xk, h))
print_results("simpson method", *test_method(simpson_method, f, x0, xk, h))