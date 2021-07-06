import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.linalg import lu

def lu_decomposition(other):
        n, m = other.shape
        L = np.zeros(other.shape)
        U = np.copy(other)
        P = np.eye(n)
        odd = False
        for i in range(0,n):
            index = i
            for j in range(i,n):
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
                for k in  range(i + 1, n):
                    U[j][k] = U[j][k] - U[i][k] * L[j][i]

        return L, U, P, odd

def solve_eq(A, b):
    L,U,P,odd = lu_decomposition(A)
    n, m = A.shape
    b = P.dot(b)
    z = np.empty(n, dtype=float)
    z[0] = b[0]
    for i in range(1, n):
        sum = 0
        for j in range(0, i):
            sum += L[i,j] * z[j]
        z[i] = b[i] - sum

    x = np.empty(n, dtype=float)
    x[n - 1] = z[n - 1] / U[n - 1, n - 1]
    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += U[i, j] * x[j]
        x[i] = (z[i] - sum) / U[i,i]

    return x


def f1(x):
    return np.power(x[0], 2) - 2 * np.log10(x[1]) - 1


def f2(x):
    return np.power(x[0], 2) - 2 * x[0] * x[1] + 2


# def df1x1(x):
#     return 2 * x[0]
#
#
# def df1x2(x):
#     return 2/(x[1] * np.log(10))
#
#
# def df2x1(x):
#     return 2*x[0] - 2*x[1]
#
#
# def df2x2(x):
#     return -2*x[0]

def construct_equiv_function(f, var, lbd, coef):
    return lambda x: (coef * x[var] - lbd * f(x)) / coef

def f1_equiv(x):
    return np.sqrt(2*np.log10(x[1]) + 1)

def f2_equiv(x):
    return (np.power(x[0], 2) + 2)/(2 * x[0])

def df1edx1(x):
    return 0

def df1edx2(x):
    return 1/(np.sqrt(2 * np.log10(x[1]) + 1) * x[1] * np.log(10))

def df2edx1(x):
    return 0.5 - 1/(np.power(x[0], 2))

def df2edx2(x):
    return 0

# def df1edx1(x):
#     return df1x1(x) + 1
#
# def df1edx2(x):
#     return df1x2(x)
#
# def df2edx1(x):
#     return df2x1(x)
#
# def df2edx2(x):
#     return df2x2(x) + 1

def derivative(fun, var_num, epsilon=0.001):
    def res(x):
        eps_vector = [(0 if i != var_num else epsilon) for i in range(0, len(x))]
        return (fun(x + eps_vector) - fun(x)) / epsilon
    return res

def simple_iterations(functions, x, q, epsilon=0.001):
    x = np.array(x, dtype=float)
    while True:
        new_x = np.array([f(x) for f in functions], dtype=float)
        if (q / (1 - q)) * np.linalg.norm(new_x - x) < epsilon:
            return new_x
        x = new_x

def newton(functions, x, epsilon=0.001):
    jacobi_matrix = [[derivative(functions[i], j, epsilon) for j in range(0, len(x))] for i in range(0, len(functions))]
    while True:
        der_values = [[der(x) for der in ders] for ders in jacobi_matrix]
        b = [-f(x) for f in functions]
        # new_x = x + np.linalg.solve(np.array(der_values, dtype=float), np.array(b, dtype=float))
        dx = solve_eq(np.array(der_values, dtype=float), np.array(b, dtype=float))
        new_x = x + dx
        if (np.linalg.norm(x - new_x) < epsilon):
            return new_x
        x = new_x


def make_mesh(interval):
    indices = np.zeros((len(interval),), dtype=np.int64)
    dims = [len(part) for part in interval]
    result_points = []
    for k in range(0, np.int64(np.prod([len(x) for x in interval]))):
        result_points.append([interval[i, indices[i]] for i in range(0, len(indices))])
        for i in range(0, len(indices)):
            indices[i] += 1
            if indices[i] == dims[i]:
                indices[i] = 0
            else:
                break
    return result_points



def test_function(f, nvar, start, end, step=0.01, ders=None):
    if ders is None:
        ders = [derivative(f, i) for i in range(0, nvar)]

    interval = np.array([np.arange(x0, x1, step) for x0, x1 in zip(start, end)])
    mesh = make_mesh(interval)
    a = [[np.abs(der(x)) for x in mesh] for der in ders]
    b = np.sum(a, axis=0)
    return max(b)


start_x = np.array([1.1, 1.4])

g_wide = 0.2
x1_int = [1.14 - g_wide / 2, 1.14 + g_wide / 2]
x2_int = [1.4 - g_wide / 2, 1.4 + g_wide / 2]


q = max([test_function(f1_equiv, 2, x1_int, x2_int, 0.001, [df1edx1,df1edx2]),
         test_function(f2_equiv, 2, x1_int, x2_int, 0.001, [df2edx1,df2edx2])])
print("Q in simple iteration method: ", q)


epsilon = 0.001

n_root = newton([f1, f2], start_x, epsilon)
si_root = simple_iterations([f1_equiv, f2_equiv], start_x, q, epsilon)
print(n_root, f1(n_root), f2(n_root))
print(si_root, f1(si_root), f2(si_root))

