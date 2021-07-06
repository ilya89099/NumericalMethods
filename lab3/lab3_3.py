import numpy as np
import math
import matplotlib.pyplot as plt


def lu_decomposition(other):
    n, m = other.shape
    L = np.zeros(other.shape)
    U = np.copy(other)
    P = np.eye(n)
    odd = False
    for i in range(0, n):
        index = i
        for j in range(i, n):
            if np.abs(U[j, i]) > np.abs(U[index, i]):
                index = j

        if i != index:
            L[[i, index]] = L[[index, i]]
            U[[i, index]] = U[[index, i]]
            P[[i, index]] = P[[index, i]]
            odd = not odd

        L[i, i] = 1
        for j in range(i + 1, n):
            L[j][i] = U[j][i] / U[i][i]
            U[j][i] = 0

        for j in range(i + 1, n):
            for k in range(i + 1, n):
                U[j][k] = U[j][k] - U[i][k] * L[j][i]

    return L, U, P, odd


def solve_eq(A, b):
    L, U, P, odd = lu_decomposition(A)
    n, m = A.shape
    b = P.dot(b)
    z = np.empty(n, dtype=float)
    z[0] = b[0]
    for i in range(1, n):
        sum = 0
        for j in range(0, i):
            sum += L[i, j] * z[j]
        z[i] = b[i] - sum

    x = np.empty(n, dtype=float)
    x[n - 1] = z[n - 1] / U[n - 1, n - 1]
    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += U[i, j] * x[j]
        x[i] = (z[i] - sum) / U[i, i]

    return x

#метод наименьших квадратов
x = np.array([-0.7, -0.4, -0.1, 0.2, 0.5, 0.8])
y = np.array([1.6462, 1.5823, 1.571, 1.5694, 1.5472, 1.4435])

# x = [0.0, 1.7, 3.4, 5.1, 6.8, 8.5]
# y = [0.4713, 1.0114, 1.5515, 2.0916, 2.6317, 3.1718]

def count_mse(approx, x, y):
    return np.sum(np.square(approx(x) - y))

def LSM(x, y, polynom_power):
    n = polynom_power + 1
    m = len(x)
    A = np.zeros((n,n), dtype=np.float32)

    A[-1] = np.array([np.sum([np.power(x_cur,i) for x_cur in x]) for i in range(n - 1, -1, -1)])
    for i in range(n - 2, -1, -1):
        A[i,1:n] = A[i + 1, 0:n-1]
        A[i, 0] = np.sum([np.power(x_cur,2*n - i - 2) for x_cur in x])

    b = np.array([np.sum([np.power(x[j], i) * y[j] for j in range(0, m)]) for i in range(0, n)], dtype=np.float32)
    b = np.flip(b)

    coefs = solve_eq(A, b)

    return lambda arg: np.sum([coefs[i] * np.power(arg, len(coefs) - i - 1) for i in range(0, len(coefs))], axis=0)

linear_approximator = LSM(x,y,1)
square_approximator = LSM(x,y,2)
cubic_approximator = LSM(x,y,3)

interval = np.arange(-1, 1, 0.01)


plt.plot(x, y, 'ro', label="points")
plt.plot(interval, linear_approximator(interval), label="linear approximator")
plt.plot(interval, square_approximator(interval), label="square approximator")
plt.plot(interval, cubic_approximator(interval), label="cubic approximator")

print("Mean squared errors:\nlinear {}\nsquare {}\ncubic {}".format(count_mse(linear_approximator, x, y),
                                                                         count_mse(square_approximator, x, y),
                                                                         count_mse(cubic_approximator, x, y)))


plt.legend()
plt.show()
