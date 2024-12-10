import numpy as np
from functions import *

np.set_printoptions(precision=2, linewidth=200, suppress=True)

if __name__ == "__main__":
    x_edu = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [9, 8, 7, 6, 5, 4, 3, 1, 2, 4],
        [2, 3, 7, 1, 5, 7, 8, 9, 2, 1],
    ]
    y_edu = [-235, -173, -93, -41, 69, 195, 347, 544, 724, 955]

    x_ctl = [6,6,6]

    f = MLR(x_edu, y_edu)
    y_ctl = FX(x_ctl, f)

    print("\n * multiple linear regression:")
    print(f"\n   > f = {f}")

    formula = f"F(x) = {f[0]:.2f}"
    for i in range(1, len(f)):
        formula += f" {"+" if f[i] > 0 else "-"} {abs(f[i]):.2f}x{i-1}"
    
    print(f"\n  Final formula:   {formula}")
    print(f"           y_ctl = {y_ctl}")
    print("")
