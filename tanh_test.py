import numpy as np
import matplotlib.pyplot as plt

SHIFT = 12
MULT1732 = np.int32((17/32)*2**(SHIFT))

def tanh_approx(val):
    if val < -15 * (2 ** (SHIFT - 3)):  # in kleiner -1,875
        output_local = -2 ** SHIFT  # output = -1
    elif val > 15 * (2 ** (SHIFT - 3)):  # in größer 1,875
        output_local = 2 ** SHIFT  # output = 1
    elif val < -9 * (2 ** (SHIFT - 3)):  # in kleiner -1,125
        output_local = val * (2 ** (-2)) - MULT1732  # output = x/4 - 17/32
    elif val > 9 * (2 ** (SHIFT - 3)):  # in größer 1,125
        output_local = val * (2 ** (-2)) + MULT1732  # output = x/4 + 17/32
    elif val < -(2 ** (SHIFT - 1)):  # in kleiner -0,5
        output_local = val * (2 ** (-1)) - (2 ** (SHIFT-2))  # output = in/2 - 0,25
    elif val > (2 ** (SHIFT - 1)):  # in größer 0,5
        output_local = val * (2 ** (-1)) + (2 ** (SHIFT-2))  # output = in/2 + 0,25
    else:
        output_local = val  # In der Mitte: ouput = in
    return np.int32(output_local)

if __name__ == '__main__':
    n = 12
    x = np.linspace(-3,3, 1000)
    x_fix = (x*(2**n)).astype(np.int32)

    fix_arr = [tanh_approx(x_fix[k])*(2**-n) for k in range(0,1000)]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x,np.tanh(x) )
    ax.plot(x,fix_arr)
    plt.show()