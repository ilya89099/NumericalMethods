import numpy as np
import math
import matplotlib.pyplot as plt


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


def make_spline(a,b,c,d, points):
    a = np.copy(a)
    b = np.copy(b)
    c = np.copy(c)
    d = np.copy(d)
    points = np.copy(points)

    def result(x):
        i = np.searchsorted(points, x)
        if i == 0 and x == points[0]:
            i += 1
        if i == 0 or i == len(points):
            raise ValueError("This point {} doesnt belong to spline".format(x))
        i -= 1
        return a[i] + b[i] * (x - points[i]) + c[i] * np.power((x - points[i]),2) + d[i] * np.power((x - points[i]), 3)

    return result

def find_coefs(points, values):
    #points, values - n + 1
    n = len(points) - 1
    h = [points[i] - points[i - 1] for i in range(1, n + 1)]
    A = np.zeros((n-1,n-1), dtype=np.float32)

    r = np.array([3 * (((values[i + 1] - values[i]) / h[i]) - ((values[i] - values[i - 1]) / h[i - 1])) for i in range(1, n)])

    last = n - 2
    A[0,0] = 2 * (h[0] + h[1])
    A[0,1] = h[1]
    A[last, last - 1] = h[n - 2]
    A[last, last] = 2 * (h[n - 2] + h[n - 1])
    for i in range(1, last):
        A[i, i - 1] = h[i]
        A[i, i] = 2 * (h[i] + h[i + 1])
        A[i, i + 1] = h[i]

    # c = np.linalg.solve(A, r)
    c = sweep_method(A, r)
    c = np.insert(c,0,0)
    a = [values[i] for i in range(0, n)]
    b = [(values[i] - values[i - 1]) / h[i - 1] - (1/3) * h[i - 1] * (c[i] + 2 * c[i - 1]) for i in range(1, n)]
    b.append((values[-1] - values[-2]) / h[-1] - (2/3) * h[-1] * c[-1])
    d = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(0, n - 1)]
    d.append(-(c[-1] / (3 * h[-1])))
    return a,b,c,d

points = [-0.4, -0.1, 0.2, 0.5, 0.8]
values = [1.5823, 1.5710, 1.5694, 1.5472, 1.4435]


a,b,c,d = find_coefs(points, values)
for a0, b0, c0, d0 in zip(a,b,c,d):
    print("{}x^3 + {}x^2 + {}x + {}".format(a0, b0, c0, d0))

spline_function = make_spline(*find_coefs(points, values), points)

interval = np.linspace(-0.4, 0.8, 120, endpoint=False)
plt.plot(points, values, "ro")
plt.plot(interval, [spline_function(x) for x in interval])
plt.show()