import numpy as np
import matplotlib.pyplot as plt

x0 = 0.5

SHIFT      = 12
fix_exp_x0 = int(np.exp(x0)*(2**(SHIFT)))
fix_x0     = int(x0*(2.0**SHIFT))
fix_half   = int(0.5*(2.0**SHIFT))
fix_one_oversix =  int((1.0/6.0)*(2.0**SHIFT))

def exp_approx_fix(x):
    a = np.int32(x-fix_x0)
    b = np.int32(a*(x-fix_x0)*(2.0**(-SHIFT)))
    c = np.int32(b*(x-fix_x0)*(2.0**(-SHIFT)))

    factor_a = fix_exp_x0
    factor_b = np.int32(fix_half*fix_exp_x0*(2.0**(-SHIFT)))
    factor_c = np.int32(fix_one_oversix*fix_exp_x0*(2.0**(-SHIFT)))

    sm = fix_exp_x0
    sm = sm + np.int32(factor_a*a*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_b*b*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_c*c*(2.0**(-SHIFT)))

    return sm


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_approx_fix(x):
    x_tan = np.int32(x*fix_half*( 2.0**(-SHIFT) ))
    rslt = np.zeros(10000)

    for k in range(10000):
        rslt[k] = fix_half + np.int32(fix_half*tanh_approx(x_tan[k])*(2**-SHIFT))

    return rslt

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


def tanh_fix_approx(x):
    a = exp_approx_fix(x)
    b = exp_approx_fix(-x)

    return (a-b)//(a+b)

if __name__ == '__main__':
    x = np.linspace(0, 1, 10000)
    x_sig = np.linspace(-100, 100, 10000)

    #print(np.max(np.abs(np.exp(x0) + np.exp(x0) * (x - x0) + np.exp(x0) * (x - x0) ** 2 / 2 + np.exp(x0) * (x - x0) ** 3 / 6 - np.exp(x))))
    #print(np.max(np.abs(np.exp(x0) + np.exp(x0) * (x - x0) + np.exp(x0) * (x - x0) ** 2 / 2 - np.exp(x))))
    #print(np.max(np.abs(np.exp(x0) + np.exp(x0) * (x - x0) - np.exp(x))))
    #print(np.sum(sftmax))

    fix_e = exp_approx_fix(np.array(x*(2.0**SHIFT)).astype(np.int32))
    fix_sigmoid  = np.int32( sigmoid_approx_fix(np.array(x_sig*(2.0**SHIFT) ) ) )
    fix_tanh  = np.array([np.int32( tanh_approx(np.array(x_sig[k]*( 2.0**SHIFT ) ) ) ) for k in range(10000)])
    fix_tanh_2  = np.int32( tanh_fix_approx(np.array(x_sig*( 2.0**SHIFT ) ) ) )


    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].plot(x, np.exp(x))
    ax[0].plot(x, fix_e*(2.0**(-SHIFT)))
    ax[1].plot(x_sig, sigmoid(x_sig))
    ax[1].plot(x_sig, fix_sigmoid*(2.0**(-SHIFT)))
    ax[2].plot(x_sig, np.tanh(x_sig))
    ax[2].plot(x_sig, fix_tanh*( 2.0**(-SHIFT) ) )
    ax[2].plot(x_sig, fix_tanh_2*( 2.0**(-SHIFT) ) )
    ax[0].legend(['numpy', 'approx'])
    plt.show()