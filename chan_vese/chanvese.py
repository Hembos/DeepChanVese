from numba.pycc import CC
from math import sqrt

import numpy as np

cc = CC('chanvese_module')

@cc.export('exec_chan_vese', 'f8[:](f8[:], i4, i4, f8[:], i4)')
def exec_chan_vese(image, width, height, init_mask, iter_num):
    max_iter = iter_num
    phi = np.zeros(width * height)
    F = [float(0)] * width * height
    phi_s = [float(0)] * width * height
    dphidt = [float(0)] * width * height
    curvature = [float(0)] * width * height
    eps = 0.00001
    sussman_dt = 0.5
    dt_init = 0.45
    lam1 = 0.5
    lam2 = 0.5
    alpha = 0.2

    for i in range(height):
        for j in range(width):
            phi[i * width + j] = 1 if init_mask[i * width + j] else -1

    for iter in range(max_iter):
        mean_neg = float(0)
        mean_pos = float(0)

        c_neg = float(0)
        c_pos = float(0)
        for i in range(height):
            for j in range(width):
                if phi[i * width + j] < 0:
                    mean_neg += image[i * width + j]
                    c_neg += 1
                else:
                    mean_pos += image[i * width + j]
                    c_pos += 1

        mean_neg = mean_neg / (c_neg + eps)
        mean_pos = mean_pos / (c_pos + eps)

        max_F = float(0)
        for i in range(height):
            for j in range(width):
                if phi[i * width + j] < 1.2 and phi[i * width + j] > -1.2:
                    F[i * width + j] = lam2 * (image[i * width + j] - mean_neg) ** 2 - lam1 * (image[i * width + j] - mean_pos) ** 2
                    max_F = max(max_F, abs(F[i * width + j]))

        for y in range(height):
            for x in range(width):
                if phi[y * width + x] < 1.2 and phi[y * width + x] > -1.2:
                    xm1 = 0 if x - 1 < 0 else x - 1
                    ym1 = 0 if y - 1 < 0 else y - 1
                    xp1 = width - 1 if x + 1 >= width else x + 1
                    yp1 = height - 1 if y + 1 >= height else y + 1

                    phi_x = -phi[y * width + xm1] + phi[y * width + xp1]
                    phi_y = -phi[ym1*width + x] + phi[yp1*width + x]
                    phi_xx = phi[y * width + xm1] + phi[y * width + xp1] - 2 * phi[y*width + x]
                    phi_yy = phi[ym1 * width + x] + phi[yp1 * width + x] - 2 * phi[y * width + x]
                    phi_xy = 0.25*(-phi[ym1*width + xm1] - phi[yp1*width + xp1] + phi[ym1*width + xp1] + phi[yp1*width + xm1])

                    curvature[y*width + x] = phi_x*phi_x * phi_yy + phi_y*phi_y * phi_xx - 2 * phi_x * phi_y * phi_xy
                    curvature[y*width + x] = curvature[y*width + x] / (phi_x*phi_x + phi_y*phi_y + eps)
                else:
                    curvature[y*width + x] = 0

        max_dphidt = float(0)
        for y in range(height):
            for x in range(width):
                if phi[y * width + x] < 1.2 and phi[y * width + x] > -1.2:
                    dphidt[y*width + x] = F[y*width + x] / max_F + alpha * curvature[y*width + x]
                    max_dphidt = max(max_dphidt, abs(dphidt[y * width + x]))

        dt = dt_init / (max_dphidt + eps)

        for y in range(height):
            for x in range(width):
                if phi[y * width + x] < 1.2 and phi[y * width + x] > -1.2:
                    phi[y * width + x] += dt * dphidt[y * width + x]

        for y in range(height):
            for x in range(width):
                l_x = x - 1 if x - 1 > 0 else width - 1
                r_x = x + 1 if x + 1 < width else 0
                u_y = y - 1 if y - 1 > 0 else height - 1
                d_y = y + 1 if y + 1 < height else 0

                if phi[y * width + x] > 0:
                    a_p = max(phi[y * width + x] - phi[y * width + l_x], 0)
                    b_n = min(phi[y * width + r_x] - phi[y * width + x], 0)
                    c_p = max(phi[y*width + x] - phi[d_y*width + x], 0)
                    d_n = min(phi[u_y*width + x] - phi[y*width + x], 0)

                    d_phi = sqrt(max(a_p*a_p, b_n*b_n) + max(c_p*c_p, d_n*d_n)) - 1
                    sussman_sign = phi[y*width + x] / sqrt(phi[y*width + x] * phi[y*width + x] + 1)
                    phi_s[y*width + x] = phi[y*width + x] - sussman_dt * sussman_sign * d_phi
                elif phi[y * width + x] < 0:
                    a_n = min(phi[y * width + x] - phi[y * width + l_x], 0)
                    b_p = max(phi[y * width + r_x] - phi[y * width + x], 0)
                    c_n = min(phi[y*width + x] - phi[d_y*width + x], 0)
                    d_p = max(phi[u_y*width + x] - phi[y*width + x], 0)

                    d_phi = sqrt(max(a_n*a_n, b_p*b_p) + max(c_n*c_n, d_p*d_p)) - 1
                    sussman_sign = phi[y*width + x] / sqrt(phi[y*width + x] * phi[y*width + x] + 1)
                    phi_s[y*width + x] = phi[y*width + x] - sussman_dt * sussman_sign * d_phi
                else:
                    phi_s[y*width + x] = 0

        for y in range(height):
            for x in range(width):
                phi[y*width + x] = phi_s[y*width + x]

    return phi

if __name__=="__main__":
    cc.compile()