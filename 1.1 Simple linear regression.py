from functions import *

def FX(a, b, x):
    return a + b * x


if __name__ == "__main__":
    x_edu = [24.32, 28.34, 34.56, 39.45, 44.76, 50.32, 55.34, 60.43, 65.87, 88.98]
    y_edu = [76.33, 70.34, 65.82, 60.23, 54.99, 50.22, 45.74, 40.34, 34.84, 30.23]
    x_test = 43.34

    a, b = SLR(x_edu, y_edu)
    y = FX(a, b, x_test)

    print("\n * simple linear regression:")
    print(f"\n   > a = {a}, b = {b}")
    print(
        f"\n  Final formula:   F(x) = {a:.2f} {"+" if b > 0 else "-"} {abs(b):.2f}x\n"
    )
