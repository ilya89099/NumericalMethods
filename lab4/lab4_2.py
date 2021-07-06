import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt

class RungeKutta:
    def __init__(self, functions, x0, start_conditions):
        self.functions = functions
        self.x0 = x0
        self.start_conditions = start_conditions
        self.last_points = None
        self.last_values = None
        self.label = "Runge Kutta 4-th"
        self.p = 4

    def integrate(self, x_end, h):
        points = np.array([self.x0])
        values = np.array([self.start_conditions])
        while points[-1] <= x_end:
            cur_x = points[-1]
            cur_condition = values[-1]
            k1 = np.array([h * self.functions[i](cur_x, *cur_condition) for i in range(0, len(cur_condition))])
            k2 = np.array([h * self.functions[i](cur_x + h / 2, *(cur_condition + (k1 / 2))) for i in range(0, len(cur_condition))])
            k3 = np.array([h * self.functions[i](cur_x + h / 2, *(cur_condition + (k2 / 2))) for i in range(0, len(cur_condition))])
            k4 = np.array([h * self.functions[i](cur_x + h, *(cur_condition + k3)) for i in range(0, len(cur_condition))])
            new_x = cur_x + h
            new_condition = cur_condition + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            num = k2 - k3
            den = k1 - k2
            # if np.min(np.abs(den)) == 0 or np.mean(np.abs(num / den)) >= 0.1:
            #     h /= 2
            # elif np.mean(np.abs(num / den)) < 0.01:
            #     h *= 2

            points = np.append(points, [new_x], axis=0)
            values = np.append(values, [new_condition], axis=0)

        self.last_points = np.array(points)
        self.last_values = np.array(values)
        return np.array(points), np.array(values)

def sweep_method(matrix, d):
    P = np.zeros((len(d),))
    Q = np.zeros((len(d),))
    n = len(d)
    M = lambda i, j, inc: matrix[i+inc, j+inc]
    a = lambda i: M(0, -1, i)
    b = lambda i: M(0, 0, i)
    c = lambda i: M(0, 1, i)
    P[0] = -c(0) / b(0)
    Q[0] = d[0] / b(0)
    for i in range(1, n - 1):
        P[i] = -c(i) / (b(i) + a(i) * P[i - 1])
        Q[i] = (d[i] - a(i) * Q[i - 1]) / (b(i) + a(i) * P[i - 1])
    P[n - 1] = 0
    Q[n - 1] = (d[n - 1] - a(n - 1) * Q[n - 2]) / (b(n - 1) + a(n - 1) * P[n - 2])
    x = np.zeros(n)
    x[n - 1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x



def increase_matrix_size(matrix, begin):
    shape = matrix.shape
    shape = tuple([i + 1 for i in shape])
    temp = np.zeros(shape)
    if begin:
        temp[1:, 1:] = matrix
    else:
        temp[:-1,:-1] = matrix
    return temp

def increase_vec_size(vec, begin):
    size = vec.shape[0] + 1
    temp = np.zeros(size)
    if begin:
        temp[1:] = vec
    else:
        temp[:-1] = vec
    return temp


def solution(x):
    return np.sin(x) + 2 - np.sin(x) * np.log((1 + np.sin(x)) / (1 - np.sin(x)))

def f(x):
    return 0

def p(x):
    return np.tan(x)

def q(x):
    return 2



# уравнение xy'' - 2(x + 1)y + 2y = 0
# так как x0 = 0, в точке x0 уравнение принимает вид
# -y' + 2y = 0, что легко приводится к y = e^2x => y(0) = 1

# cond1 = {"order":1, "value":1}
# cond2 = {"order":2, "beta":-2, "value":-4}
# cond2 = {"order":1, "value": 2 + np.exp(2)}

h = 0.01
x0 = 0
x1 = np.pi / 6
y0 = 2
y1 = 2.5 - 0.5 * np.log(3)
cond1 = {"order": 1, "value": y0}
cond2 = {"order": 1, "value": y1}


def finite_differences(p, q, f, h, x0, x1, cond1, cond2):
    points = np.arange(x0 + h, x1, h)
    a = lambda x: (1 / np.power(h, 2) - p(x) / (2 * h))
    b = lambda x: -2 / np.power(h, 2) + q(x)
    c = lambda x: (1 / np.power(h, 2) + p(x) / (2 * h))
    n = len(points)
    matrix = np.zeros((n, n))
    d = np.zeros((n,))
    for i in range(0, n):
        d[i] = f(points[i])

        if i - 1 >= 0:
            matrix[i,i - 1] = a(points[i])

        matrix[i,i] = b(points[i])

        if i + 1 < len(points):
            matrix[i, i + 1] = c(points[i])

    if cond1["order"] == 1:
        d[0] -= cond1["value"] * a(points[0])
    else:
        matrix = increase_matrix_size(matrix, True)
        d = increase_vec_size(d, True)
        matrix[1,0] = a(points[0])
        matrix[0,0] = -(2 / (h * (2 - p(x0) * h))) + (q(x0) * h) / (2 - p(x0) * h) + cond1["alpha"]
        matrix[0,1] = 2 / (h * (2 - p(x0) * h))
        d[0] = cond1["value"] + (h * f(x0)) / (2 - p(x0) * h)
        points = np.insert(points, 0, x0)

    if cond2["order"] == 1:
        d[-1] -= cond2["value"] * c(points[-1])
    else:
        matrix = increase_matrix_size(matrix, False)
        d = increase_vec_size(d, False)
        matrix[-2, -1] = c(points[-1])
        matrix[-1, -2] = - 2 / (h * (2 + p(x1) * h))
        matrix[-1, -1] = (2 / (h * (2 + p(x1) * h))) - (q(x1) * h) / (2 + p(x1) * h) + cond2["beta"]
        d[-1] = cond2["value"] - (h * f(x1)) / (2 + p(x1) * h)
        points = np.insert(points, len(points), x1)

    y = sweep_method(matrix, d)
    return points, y

def f1(x,y1,y2):
    return y2

def f2(x,y1,y2):
    return np.tan(y2) - 2 * y1

def secant(fun, x0, x1, epsilon=0.001):
    fx0 = None

    while True:
        fx1 = fun(x1)
        if fx0 is None:
            fx0 = fun(x0)
        new_x0, new_x1 = x1, x1 - (fx1 * (x1 - x0))/(fx1 - fx0)
        if np.abs(new_x1 - new_x0) < epsilon:
            return new_x1
        x0, x1 = new_x0, new_x1
        fx0 = fx1


def shooting_method(functions, x0, x1, y0, y1, tan1, h, epsilon):
    def solve(x):
        solver = RungeKutta(functions, x0, [y0, x])
        points, values = solver.integrate(x1, h)
        return values[-1, 0] - y1

    result_tan = secant(solve, tan1, tan1 + h / 2, epsilon)
    p, y = RungeKutta(functions, x0, [y0, result_tan]).integrate(x1, h)
    return p, y[:,0]


def fd_error():
    hp, hy = finite_differences(p, q, f, h, x0, x1, cond1, cond2)
    h2p, h2y = finite_differences(p, q, f, h / 2, x0, x1, cond1, cond2)
    error = [np.abs(hy[i] - h2y[2*i]) for i in range(0, len(hp))]
    error = np.array(error) / (np.power(2,2) - 1)
    return error

def sm_error():
    hp, hy = shooting_method([f1, f2], x0, x1, y0, y1, 0.5, h, 0.001)
    h2p, h2y = shooting_method([f1, f2], x0, x1, y0, y1, 0.5, h / 2, 0.001)
    error = [np.abs(hy[i] - h2y[2*i]) for i in range(0, min([len(hp), len(h2p) // 2]))]
    error = np.array(error) / (np.power(2, 2) - 1)
    return error




fd_points, fd_y = finite_differences(p, q, f, h, x0, x1, cond1, cond2)
precise = solution(fd_points)
sm_points, sm_y = shooting_method([f1, f2], x0, x1, y0, y1, 0.5, h, 0.001)
plt.plot(fd_points, solution(fd_points), label="precise solution")
plt.plot(fd_points, fd_y, label="finite differences solution")
plt.plot(sm_points, sm_y, label="shooting solution")
print("finite differences error {}".format(fd_error()))
print()
print("shooting method error {}".format(sm_error()))
print()
print("finide differences precise error {}".format(np.abs(solution(fd_points) - fd_y)))
print()
print("shooting method precise error {}".format(np.abs(solution(sm_points) - sm_points)))



plt.legend()
plt.show()
