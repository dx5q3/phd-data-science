import math
import numpy as np

def avg(a):
    return sum(a) / len(a)

def SLR(x, y):
    n = len(x)
    b1 = n * sum([a * b for a, b in zip(x, y)]) - sum(x) * sum(y)
    b2 = n * sum(pow(a, 2) for a in x) - math.pow(sum(x), 2)
    b = b1 / b2
    a = avg(y) - b * avg(x)
    return a, b

def MLR(x, y):  # find multiple linear regression coefficients
    a = [1] * len(x[0])

    x1 = np.insert(
        x, 0, a, axis=0
    )  # adding [1, 1, 1, ...] on top of the given dataset (due to assumption the y-intercept )

    x2 = np.matmul(x1, np.transpose(x1))  # A = (XX^T)^-1XY^T
    x3 = np.linalg.inv(x2)
    x4 = np.matmul(x3, x1)
    y1 = np.transpose(y)

    f = np.matmul(x4, y1)

    f = np.asarray(f).reshape(-1)
    return f


# calculate (predict) result based on experimental data and multiple linear regression coefficients
def FX(x, f):
    x = np.asarray(x).reshape(-1)

    y = f[0]

    for i in range(len(x)):  # y = a + b * x1 + c * x2 ...
        y += x[i] * f[i + 1]

    return y

def FFX(x, f):
    y = np.array([])

    for i in x:
        y = np.append(y, FX(i, f))
    
    return y

    

def MSE(y1, y2):  # mean squared error
    s = 0

    for y1i, y2i in zip(y1, y2):  # y = 1/n * E (y - y`)^2
        s += math.pow(y1i - y2i, 2)

    return (1 / len(y1)) * s


def norm(x):
    x_max = max(x)
    x_min = min(x)
    x_norm = []
    for i in x:
        x_norm.append(((i - x_min) / (x_max - x_min)).item())

    return x_norm