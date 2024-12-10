import math
import numpy as np
from functions import MLR, FX
np.set_printoptions(precision=3, suppress=True)

x_edu = [
    [1, 5, 12, 23, 34, 53, 66, 69, 78],  # x1
    [88, 77, 66, 56, 43, 34, 31, 23, 22],  # x2
    [11, 32, 34, 45, 48, 65, 77, 88, 96],  # x3
]
y_edu = [2, 4, 8, 12, 17, 32, 54, 65, 77]
x_ctl = [33, 50, 54]


n = len(x_edu)  # number of factors
m = len(x_edu[0])  # number of observations

R = np.corrcoef(x_edu)

chi_sq = -1 * (m - 1 - (1 / 6) * (2 * n + 5)) * math.log(np.linalg.det(R))

freedom = 0.5 * n * (n - 1)
alpha = 0.5

print("")
print(f"   chi-square: {chi_sq:.2f}")
print(f"freedon level: {int(freedom)}")
print(f"  alpha level: {alpha}")
print("")


D = np.linalg.inv(R)

F0 = abs(D[0][0] - 1) * ((m - n) / (n - 1))
F1 = abs(D[1][1] - 1) * ((m - n) / (n - 1))
F2 = abs(D[2][2] - 1) * ((m - n) / (n - 1))

print(f"\nF0 = {F0:.2f}\tF1 = {F1:.2f}\tF2 = {F2:.2f}")
print(f"\n\tf1 = m - n = {m - n}\n\tf2 = n - 1 = {n - 1}")


p01 = -1 * (R[0][1] / math.sqrt(R[0][0] / R[1][1]))
p02 = -1 * (R[0][2] / math.sqrt(R[0][0] / R[2][2]))
p12 = -1 * (R[1][2] / math.sqrt(R[1][1] / R[2][2]))

print(f"\np01 = {p01:.2f}\tp02 = {p02:.2f}\tp23 = {p12:.2f}")

t01 = (p01 * math.sqrt(m - n)) / (math.sqrt(1 - math.pow(p01, 2)))
t02 = (p02 * math.sqrt(m - n)) / (math.sqrt(1 - math.pow(p02, 2)))
t12 = (p12 * math.sqrt(m - n)) / (math.sqrt(1 - math.pow(p12, 2)))

print(f"t01 = {t01:.2f}\tt02 = {t02:.2f}\tt12 = {t12:.2f}")

del x_edu[1]
del x_ctl[1]

f = MLR(x_edu, y_edu)
y_ctl = FX(x_ctl, f)

formula = f"F(x) = {f[0]:.2f} {"+" if f[1] > 0 else "-"} {abs(f[1]):.2f}x0 {"+" if f[2] > 0 else "-"} {abs(f[2]):.2f}x2"

print(f"\n  Final formula:   {formula}")
print(f"           y_ctl = {y_ctl}")
print("")
