import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt


def precise_solution(x):
    return np.power(np.abs(x), 1.5) - np.sqrt(2)


def f1(x, y1, y2):
    return y2


def f2(x, y1, y2):
    return (0.5 * y2 - 0.75 * y1) / (x * (1 - x))


class Euler:
    def __init__(self, functions, x0, start_conditions):
        self.functions = functions
        self.x0 = x0
        self.start_conditions = start_conditions
        self.label = "Euler"
        self.last_points = None
        self.last_values = None
        self.p = 1

    def integrate(self, x_end, h):
        points = [self.x0]
        values = [self.start_conditions]  # type: list
        while points[-1] <= x_end:
            cur_x = points[-1]
            cur_condition = values[-1]
            new_x = cur_x + h
            new_condition = [cur_condition[i] + h * self.functions[i](cur_x, *cur_condition) for i in range(0, len(cur_condition))]
            points.append(new_x)
            values.append(new_condition)
        self.last_points = np.array(points)
        self.last_values = np.array(values)
        return np.array(points), np.array(values)

class EulerKoshi:
    def __init__(self, functions, x0, start_conditions):
        self.functions = functions
        self.x0 = x0
        self.start_conditions = start_conditions
        self.label = "Euler Koshi"
        self.last_points = None
        self.last_values = None
        self.p = 2

    def integrate(self, x_end, h):
        points = [self.x0]
        values = [self.start_conditions]  # type: list
        while points[-1] <= x_end:
            cur_x = points[-1]
            cur_condition = values[-1]
            new_x = cur_x + h
            new_condition = [cur_condition[i] + h * self.functions[i](cur_x, *cur_condition) for i in range(0, len(cur_condition))]
            new_condition_corrected = [cur_condition[i] + h * (self.functions[i](cur_x, *cur_condition) + self.functions[i](new_x, *new_condition)) / 2 for i in range(0, len(cur_condition))]
            points.append(new_x)
            values.append(new_condition_corrected)
        self.last_points = np.array(points)
        self.last_values = np.array(values)
        return np.array(points), np.array(values)


class EulerAdvanced:
    def __init__(self, functions, x0, start_conditions):
        self.functions = functions
        self.x0 = x0
        self.start_conditions = start_conditions
        self.last_points = None
        self.last_values = None
        self.label = "Euler advanced"
        self.p = 2

    def integrate(self, x_end, h):
        points = [self.x0]
        values = [self.start_conditions]  # type: list
        while points[-1] <= x_end:
            cur_x = points[-1]
            cur_condition = values[-1]
            new_x = cur_x + h
            half_condition = [cur_condition[i] + (h / 2) * self.functions[i](cur_x, *cur_condition) for i in range(0, len(cur_condition))]
            new_condition = [cur_condition[i] + h * self.functions[i](cur_x + h / 2, *half_condition) for i in range(0, len(cur_condition))]
            points.append(new_x)
            values.append(new_condition)
        self.last_points = np.array(points)
        self.last_values = np.array(values)
        return np.array(points), np.array(values)


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


class Adams:
    def __init__(self, functions, x0, start_conditions):
        self.functions = functions
        self.x0 = x0
        self.start_conditions = start_conditions
        self.last_points = None
        self.last_values = None
        self.label = "Adams 4-th"
        self.p = 4

    def integrate(self, x_end, h):
        points = np.array(self.x0)
        values = np.array(self.start_conditions)
        while points[-1] <= x_end:
            cur_points = points[-4:]
            cur_values = values[-4:]
            f = np.array([[self.functions[i](cur_points[j], *cur_values[j]) for i in range(0, len(self.functions))] for j in range(0,len(cur_points))])
            new_values = np.array([cur_values[-1,i] + (h/24)*(55 * f[-1,i] - 59 * f[-2,i] + 37 * f[-3,i] - 9 * f[-4,i]) for i in range(0, len(self.functions))])
            new_x = points[-1] + h
            points = np.append(points, [new_x], axis=0)
            values = np.append(values, [new_values], axis=0)
        self.last_points = points
        self.last_values = values
        return points, values

fig, ax = plt.subplots(2)

ax[0].set_ylim([1, 4])
ax[1].set_ylim([1, 4])

def plot_solved(solver):
    for i in range(0, solver.last_values.shape[-1]):
       ax[i].plot(solver.last_points, solver.last_values[:, i], label=solver.label)

x0 = 2
x1 = 3
start_conditions = [np.sqrt(2), 1.5 * np.sqrt(2)]
h = 0.1

x_interval = np.arange(x0, x1 + h/2, h)
ax[0].plot(x_interval, precise_solution(x_interval), label="precise solution")

solvers = [Euler, EulerKoshi, RungeKutta]

adams_start_x = None
adams_start_y = None

solver_instances = []

for solver_class in solvers:
    solver = solver_class([f1, f2], x0, start_conditions)
    points, values = solver.integrate(x1, h)
    if (solver.__class__.__name__ == "RungeKutta"):
        adams_start_x = points[:3]
        adams_start_y = values[:3]
    solver_instances.append(solver)
    plot_solved(solver)



x0 = np.append([x0], adams_start_x, axis=0)
start_conditions = np.append([start_conditions], adams_start_y, axis=0)

adams = Adams([f1,f2], x0, start_conditions)
adams.integrate(x1,h)

solver_instances.append(adams)

plot_solved(adams)

ax[0].legend()
plt.show()

for solver in solver_instances:
    points1, values1 = solver.integrate(x1, h / 2)
    points1 = points1[::2]
    values1 = values1[::2]

    points2, values2 = solver.integrate(x1, h)

    if (values1.shape[0] > values2.shape[0]):
        values1 = np.resize(values1, values2.shape)
    else:
        values2 = np.resize(values2, values1.shape)

    error = np.linalg.norm(np.abs(values1[:, 0] - values2[:, 0]) / (np.power(2, solver.p) - 1))
    print("{} error: {}".format(solver.label, error))

# def new_f(x, y):
#     return x**2 - y
#
# rk = RungeKutta([new_f], 1, [1])
# rk.integrate(3, 0.5)
# plt.plot(rk.last_points, rk.last_values[:, 0], label=rk.label)
#
#
# eul = Euler([new_f], 1, [1])
# eul.integrate(3, 0.5)
# plt.plot(eul.last_points, eul.last_values[:, 0], label=eul.label)
# plt.show()
#

#
# euler_solver = Euler([f1, f2], x0, start_conditions)
# euler_koshi_solver = EulerKoshi([f1,f2], x0, start_conditions)
# euler_advanced_solver = EulerAdvanced([f1,f2], x0, start_conditions)
# runge_kutta_solver = RungeKutta([f1,f2], x0, start_conditions)
#
# euler_solver.integrate(x1, h)
# euler_koshi_solver.integrate(x1, h)
# euler_advanced_solver.integrate(x1, h)
# runge_kutta_solver.integrate(x1, h)
#
#
# plot_solved(euler_solver)
# plot_solved(euler_koshi_solver)
# plot_solved(euler_advanced_solver)
# plot_solved(runge_kutta_solver)
#
