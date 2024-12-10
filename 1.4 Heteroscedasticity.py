import math
import numpy as np
from functions import MLR, avg, FFX

x_edu = [
    [1, 5, 12, 23, 34, 53, 66, 69, 78],
    [88, 77, 66, 56, 43, 34, 31, 23, 22],
    [11, 32, 34, 45, 48, 65, 77, 88, 96],
]
y_edu = [2, 4, 8, 12, 17, 32, 54, 65, 77]

n = len(x_edu[0])
c = (4/15) * n
qty = math.floor((n - c) / 2)

x_edu_1 = [x_edu[0][:qty], x_edu[1][:qty], x_edu[2][:qty]]
x_edu_2 = [x_edu[0][-1 * qty:], x_edu[1][-1 * qty:], x_edu[2][-1 * qty:]]

y_edu_1 = y_edu[:qty]
y_edu_2 = y_edu[-1 * qty:]


f1 = MLR(x_edu_1, y_edu_1)
f2 = MLR(x_edu_2, y_edu_2)

y_test_1 = FFX(np.transpose(x_edu_1), f1)
y_test_2 = FFX(np.transpose(x_edu_2), f2)

s1 = 0
s2 = 0

for i in range(qty):
    s1 += math.pow(y_edu_1[i] - avg(y_edu_1), 2)
    s2 += math.pow(y_edu_2[i] - avg(y_edu_2), 2)

print(f"\nS1 = {s1:.2f}, S2 = {s2:.2f}")

R = s2/s1

print(f"R = {R:.2f}\n")