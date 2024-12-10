import math
from functions import *  # ignore the error

data = [
    #   x1     x2     x3     x4     x5     x6    x7       x8     x9    x10       y
    [71.20, 54.70, 128.0, 38995, 10.43, 412.8, 3436, 101.600, 21.00, 17.90, 287.00],
    [71.60, 55.56, 120.1, 13636, 14.92, 452.7, 3899, 130.408, 19.00, 18.26, 282.56],
    [74.25, 55.49, 114.4, 12905, 12.11, 410.2, 4644, 101.306, 18.00, 18.71, 281.05],
    [74.25, 56.12, 113.9, 13271, 18.67, 458.6, 2051, 141.993, 26.00, 19.69, 283.68],
    [78.38, 61.78, 116.8, 26785, 56.83, 518.6, 2562, 385.409, 36.00, 20.19, 299.65],
    [82.20, 64.22, 115.3, 30437, 80.62, 555.9, 2855, 482.030, 15.30, 20.42, 307.13],
    [84.28, 65.32, 120.1, 42156, 18.65, 458.6, 2855, 141.932, 16.25, 20.89, 316.85],
    [86.08, 68.48, 121.0, 12936, 34.30, 483.1, 2891, 222.110, 17.64, 20.90, 324.33],
    [87.94, 71.80, 125.8, 23894, 41.60, 494.6, 3512, 274.297, 19.23, 21.48, 336.90],
    [89.43, 74.24, 127.8, 25355, 34.31, 483.2, 1251, 222.196, 19.74, 21.53, 343.80],
    [90.93, 75.87, 129.3, 31929, 8.994, 443.4, 3402, 117.285, 20.36, 22.35, 349.66],
    [86.29, 76.69, 128.7, 39058, 71.66, 541.9, 3878, 459.166, 21.18, 22.83, 346.86],
    [86.05, 76.90, 128.9, 13332, 34.20, 483.0, 4720, 221.451, 21.28, 22.71, 347.53],
    [84.28, 78.41, 128.7, 24289, 23.87, 466.8, 1141, 162.880, 38.00, 23.94, 349.32],
    [86.27, 79.18, 130.5, 46600, 78.43, 552.5, 1798, 477.721, 22.41, 24.15, 355.32],
    [88.87, 80.24, 133.3, 20302, 37.43, 488.1, 2346, 243.742, 23.26, 25.34, 364.23],
    [88.92, 83.99, 136.6, 27973, 64.67, 530.9, 2967, 430.234, 39.00, 25.79, 372.98],
    [92.15, 84.20, 140.5, 34547, 54.29, 514.6, 3332, 368.392, 23.88, 38.00, 380.75],
    [90.86, 85.90, 143.1, 42979, 60.34, 524.1, 3259, 407.016, 24.32, 34.00, 384.39],
    [90.43, 87.80, 140.7, 17411, 90.85, 572.0, 3896, 494.201, 38.00, 26.56, 385.32],
    [91.36, 91.30, 146.5, 32021, 17.01, 456.0, 1290, 136.536, 25.66, 44.00, 397.13],
    [96.89, 93.70, 145.3, 11963, 22.91, 465.2, 4577, 158.583, 26.39, 26.39, 404.30],
    [97.05, 96.70, 143.1, 19633, 40.07, 492.2, 2130, 262.944, 26.68, 26.44, 406.12],
    [104.6, 98.90, 147.9, 38759, 28.25, 473.6, 3299, 185.061, 26.45, 26.83, 421.71],
]




def box_count_1d(arr, e_discr):
    n_discr = int(1 / e_discr)

    cnt = [0] * n_discr
    for a in arr:
        if a == 1:
            a -= e_discr / 2

        cnt[int(a / e_discr)] += 1

    return n_discr - cnt.count(0)


def box_count_2d(arr_x, arr_y, e_discr):
    n_discr = int(1 / e_discr)
    cnt = [[0 for x in range(n_discr)] for y in range(n_discr)]
    for i in range(len(arr_x)):
        if arr_x[i] == 1:
            arr_x[i] -= e_discr / 2
        if arr_y[i] == 1:
            arr_y[i] -= e_discr / 2
        cnt[int(arr_x[i] / e_discr)][int(arr_y[i] / e_discr)] += 1

    N0 = 0
    for c in cnt:
        N0 += c.count(0)

    return int(math.pow(n_discr, 2) - N0)


if __name__ == "__main__":

    data_edu = np.transpose(data[:20])
    data_ctl = np.transpose(data[20:])

    data_edu_x = data_edu[:-1]
    data_edu_y = data_edu[-1]

    data_ctl_x = data_ctl[:-1]
    data_ctl_y = data_ctl[-1]

    data_edu_norm_x = []
    data_edu_norm_y = norm(data_edu_y)

    for x in data_edu_x:
        data_edu_norm_x.append(norm(x))

    e_discr = 0.001

    N_y = box_count_1d(data_edu_norm_y, e_discr)
    N_x = []
    N_xy = []

    for x in data_edu_norm_x:
        N_x.append(box_count_1d(x, e_discr))
        N_xy.append(box_count_2d(x, data_edu_norm_y, e_discr))

    cross_entropy = []

    print("\n * cross-entropy:")
    for i in range(len(N_x)):
        entr = math.log2((N_x[i] * N_y) / N_xy[i])
        cross_entropy.append(entr)
        print(f"\tX{i}<>Y: {round(entr,3)}")

    rsl = [f"{str(i)}" for i in range(len(data_edu_norm_x))]

    tmp_f = cross_entropy
    n = 3

    rmv = []
    for i in range(n):
        ind_min = np.argmin(tmp_f)
        rmv.append(rsl[ind_min])
        
        data_edu_x = np.delete(data_edu_x, (ind_min), axis=0)
        data_ctl_x = np.delete(data_ctl_x, (ind_min), axis=0)

        del tmp_f[ind_min]
        del rsl[ind_min]
    
    rmv.sort()

    print(f"\n* removing {n} factors with least enthropy: {", ".join(rmv)}")

    f = MLR(data_edu_x, data_edu_y)


    formula = f"F(x) = {f[0]:.4f}"
    for i, j in zip(rsl, range(1, len(f))):
        formula += f" {"+" if f[j] > 0 else "-"} {abs(f[j]):.4f}x{i}"


    tmp_y = FFX(np.transpose(data_ctl_x), f)
    e = MSE(data_ctl_y, tmp_y)

    print(f"\n  Final formula:   {formula}")
    print(f"            MSE:   {e:.3f}")
    print("")

