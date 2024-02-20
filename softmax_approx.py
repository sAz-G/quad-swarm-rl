import numpy as np
import matplotlib.pyplot as plt
import copy

plt.rcParams.update({'font.size': 17})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif"
})

x0 = 0.5

SHIFT      = 12


def exp_approx_fix(x,shft, dpt=None):
    fix_exp_x0 = int(np.exp(x0) * (2 ** (shft)))
    fix_x0 = int(x0 * (2.0 ** shft))
    fix_half = int(0.5 * (2.0 ** shft))
    fix_one_oversix = int((1.0 / 6.0) * (2.0 ** shft))
    fix_one_over24 = int((1.0 / 24) * (2.0 ** shft))
    fix_one_over120 = int((1.0 / 120) * (2.0 ** shft))
    fix_one_over720 = int((1.0 / 720) * (2.0 ** shft))
    fix_one_over1540 = int((1.0 / 1540) * (2.0 ** shft))

    a = np.int32(x-fix_x0)
    b = np.int32(a*(x-fix_x0)*(2.0**(-shft)))
    c = np.int32(b*(x-fix_x0)*(2.0**(-shft)))
    d = np.int32(c*(x-fix_x0)*(2.0**(-shft)))
    e = np.int32(d*(x-fix_x0)*(2.0**(-shft)))
    f = np.int32(e*(x-fix_x0)*(2.0**(-shft)))
    g = np.int32(f*(x-fix_x0)*(2.0**(-shft)))

    factor_a = fix_exp_x0
    factor_b = np.int32(fix_half*fix_exp_x0*(2.0**(-shft)))
    factor_c = np.int32(fix_one_oversix*fix_exp_x0*(2.0**(-shft)))
    factor_d = np.int32(fix_one_over24*fix_exp_x0*(2.0**(-shft)))
    factor_e = np.int32(fix_one_over120*fix_exp_x0*(2.0**(-shft)))
    factor_f = np.int32(fix_one_over720*fix_exp_x0*(2.0**(-shft)))
    factor_g = np.int32(fix_one_over1540*fix_exp_x0*(2.0**(-shft)))

    sm = fix_exp_x0

    if dpt == 1:
        sm = sm + np.int32(factor_a * a * (2.0 ** (-shft)))
    elif dpt == 2:
        sm = sm + np.int32(factor_a * a * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_b * b * (2.0 ** (-shft)))
    elif dpt == 3:
        sm = sm + np.int32(factor_a * a * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_b * b * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_c * c * (2.0 ** (-shft)))
    elif dpt == 4:
        sm = sm + np.int32(factor_a * a * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_b * b * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_c * c * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_d * d * (2.0 ** (-shft)))
    elif dpt == 5:
        sm = sm + np.int32(factor_a * a * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_b * b * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_c * c * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_d * d * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_e * e * (2.0 ** (-shft)))
    elif dpt == 6:
        sm = sm + np.int32(factor_a * a * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_b * b * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_c * c * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_d * d * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_e * e * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_f * f * (2.0 ** (-shft)))
    else:
        sm = sm + np.int32(factor_a * a * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_b * b * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_c * c * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_d * d * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_e * e * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_f * f * (2.0 ** (-shft)))
        sm = sm + np.int32(factor_g * g * (2.0 ** (-shft)))



    return sm


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_approx_fix(x, shft):
    fix_half = int(0.5 * (2.0 ** shft))
    x_tan = np.int32(x*fix_half*( 2.0**(-shft) ))

    rslt = fix_half + np.int32(fix_half*tanh_fix_approx_johan_heinrich(x_tan, shft)*(2**-shft))
    return rslt

MULT1732 = np.int32((17/32)*2**(SHIFT))


def tanh_approx(val, SHIFT):
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

    return np.int32(((a-b)/(a+b))*2**SHIFT)


def tanh_fix_approx_johan_heinrich(x, shft, depth=None):

    x_arg = copy.deepcopy(x)

    #x_arg[np.abs(x) >= 10*(2**shft)] = 10*(2**shft)

    x13_shift = np.int32(13*(2**shft))
    x11_shift = np.int32(11*(2**shft))
    x9_shift = np.int32(9*(2**shft))
    x7_shift = np.int32(7*(2**shft))
    x5_shift = np.int32(5*(2**shft))
    x3_shift = np.int32(3*(2**shft))
    x1_shift = np.int32(1*(2**shft))


    xx = x_arg*x_arg*(2**-shft)

    if depth == 1:
        x1 = x1_shift + (xx / (x3_shift + (xx / (x5_shift + xx) ) * (2 ** shft))) * (2 ** shft)
        x_final = (x / x1) * (2 ** shft)
    elif depth == 2:
        x1 = x1_shift + (xx / (x3_shift + (xx / (x5_shift + (xx / (x7_shift + xx)) * (2 ** shft) ) ) * (2 ** shft) ) )*(2 ** shft)
        x_final = (x / x1) * (2 ** shft)
    elif depth == 3:
        x1 = x1_shift + (xx / (x3_shift + (xx / (x5_shift + (xx / (x7_shift + xx ) ) * (2 ** shft)))*(2 ** shft))) * (2 ** shft)
        x_final = (x / x1) * (2 ** shft)
    elif depth == 4:
        x1 = x1_shift + (xx / (x3_shift + (xx / (x5_shift + (xx / (x7_shift + (xx / (x9_shift +xx ) ) * (2 ** shft))) * (2 ** shft))) * (2 ** shft))) * (2 ** shft)
        x_final = (x / x1) * (2 ** shft)
    elif depth == 5:
        x1 = x1_shift + (xx / (x3_shift + (xx / (x5_shift + (xx / (x7_shift + (xx / (x9_shift + (xx / (x11_shift + xx ) ) * (2 ** shft))) * (2 ** shft))) * (2 ** shft))) * (2 ** shft))) * (2 ** shft)
        x_final = (x / x1) * (2 ** shft)
    else:
        x1 = x1_shift + (xx / (x3_shift + (xx / (x5_shift + (xx / (x7_shift + (xx / (x9_shift + (xx / (x11_shift + (xx / (x13_shift + xx)) * (2 ** shft))) * (2 ** shft))) * (2 ** shft))) * (2 ** shft))) * (2 ** shft))) * (2 ** shft)
        x_final = (x / x1) * (2 ** shft)

    return x_final

def softmax_fix_approx(x, SHIFT):
    a = exp_approx_fix(x, SHIFT)
    b = np.sum(exp_approx_fix(x, SHIFT))
    return np.int32((a/b)*(2**SHIFT))

def softmax_local(x):
    return np.exp(x)/np.sum(np.exp(x))

if __name__ == '__main__':
    x = np.linspace(0, 1, 10000)

    x_sig_min, x_sig_max = -25,25
    x_sig = np.linspace(x_sig_min, x_sig_max, 10000)
    x_sft = np.linspace(0, 1, 6)
    x_sft_1 = np.linspace(0, 1, 20)
    x_sft_2 = np.linspace(0, 1, 30)
    x_sft_3 = np.linspace(0, 1, 6)

    #print(np.max(np.abs(np.exp(x0) + np.exp(x0) * (x - x0) + np.exp(x0) * (x - x0) ** 2 / 2 + np.exp(x0) * (x - x0) ** 3 / 6 - np.exp(x))))
    #print(np.max(np.abs(np.exp(x0) + np.exp(x0) * (x - x0) + np.exp(x0) * (x - x0) ** 2 / 2 - np.exp(x))))
    #print(np.max(np.abs(np.exp(x0) + np.exp(x0) * (x - x0) - np.exp(x))))
    #print(np.sum(sftmax))

    sftmax_ideal  = softmax_local(x_sft)
    sftmax_approx = softmax_fix_approx(np.int32(x_sft*(2**SHIFT)), SHIFT)
    sftmax_approx_1 = softmax_fix_approx(np.int32(x_sft*(2**10)), 10)
    sftmax_approx_2 = softmax_fix_approx(np.int32(x_sft*(2**8)), 8)
    sftmax_approx_3 = softmax_fix_approx(np.int32(x_sft*(2**6)), 6)
    sftmax_approx_4 = softmax_fix_approx(np.int32(x_sft*(2**4)), 4)

    print(np.max(np.abs(sftmax_ideal - sftmax_approx*(2**-SHIFT))))
    print(np.max(np.abs(sftmax_ideal - sftmax_approx_1*(2**-10))))
    print(np.max(np.abs(sftmax_ideal - sftmax_approx_2*(2**-8))))
    print(np.max(np.abs(sftmax_ideal - sftmax_approx_3*(2**-6))))
    print(np.max(np.abs(sftmax_ideal - sftmax_approx_4*(2**-4))))
    #fix_e_12 = exp_approx_fix(np.array(x*(2.0**SHIFT)).astype(np.int32), SHIFT)
    #fix_e_10 = exp_approx_fix(np.array(x*(2.0**10)).astype(np.int32), 10)
    #fix_e_8 = exp_approx_fix(np.array(x*(2.0**8)).astype(np.int32), 8)
    #fix_e_6 = exp_approx_fix(np.array(x*(2.0**6)).astype(np.int32), 6)
    #fix_e_4 = exp_approx_fix(np.array(x*(2.0**4)).astype(np.int32), 4)


    #fix_sigmoid  = np.int32( sigmoid_approx_fix(np.array(x_sig*(2.0**SHIFT) ) ,SHIFT) )
    # fix_tanh_david  = np.array([np.int32( tanh_approx(np.array(x_sig[k]*( 2.0**SHIFT ) ), SHIFT ) ) for k in range(10000)])
    #
    # fix_tanh_6  = np.int32( tanh_fix_approx_johan_heinrich(np.array(np.int32(x_sig*( 2.0**12 ) ) ) , 12) )
    # fix_tanh_6[x_sig >= 5] = 1*(2**SHIFT)
    # fix_tanh_6[x_sig <= -5] = -1*(2**SHIFT)
    # fix_tanh_5  = np.array([np.int32( tanh_fix_approx_johan_heinrich(np.array(x_sig[k]*( 2.0**10 ) ), 10 ) ) for k in range(10000)])
    # fix_tanh_5[x_sig >= 5] = 1 * (2 ** 10)
    # fix_tanh_5[x_sig <= -5] = -1 * (2 ** 10)
    # fix_tanh_4  = np.int32( tanh_fix_approx_johan_heinrich(np.array(np.int32(x_sig*( 2.0**8 ) ) ) , 8) )
    # fix_tanh_4[x_sig >= 5] = 1 * (2 ** 8)
    # fix_tanh_4[x_sig <= -5] = -1 * (2 ** 8)
    # fix_tanh_3  = np.int32( tanh_fix_approx_johan_heinrich(np.array(np.int32(x_sig*( 2.0**6 ) ) ) , 6 ) )
    # fix_tanh_3[x_sig >= 5] = 1 * (2 ** 6)
    # fix_tanh_3[x_sig <= -5] = -1 * (2 ** 6)
    # fix_tanh_2  = np.int32( tanh_fix_approx_johan_heinrich(np.array(np.int32(x_sig*( 2.0**4 ) ) ) , 4 ) )
    # fix_tanh_2[x_sig >= 5] = 1 * (2 ** 4)
    # fix_tanh_2[x_sig <= -5] = -1 * (2 ** 4)
    # fix_tanh_1  = np.int32( tanh_fix_approx_johan_heinrich(np.array(np.int32(x_sig*( 2.0**SHIFT ) ) ) , 2 ) )
    # fix_tanh_1[x_sig >= 5] = 1 * (2 ** SHIFT)
    # fix_tanh_1[x_sig <= -5] = -1 * (2 ** SHIFT)

    # print(np.max(np.abs(fix_tanh_david*(2**-SHIFT) - np.tanh(x_sig))) )
    # print(np.max(np.abs(fix_tanh_6*(2**-12) - np.tanh(x_sig))) )
    # print(np.max(np.abs(fix_tanh_5*(2**-10) - np.tanh(x_sig))) )
    # print(np.max(np.abs(fix_tanh_4*(2**-8) - np.tanh(x_sig))) )
    # print(np.max(np.abs(fix_tanh_3*(2**-6) - np.tanh(x_sig))) )
    # print(np.max(np.abs(fix_tanh_2*(2**-4) - np.tanh(x_sig))) )
    # print(np.max(np.abs(fix_tanh_1*(2**-SHIFT) - np.tanh(x_sig))) )

    # exp_fix   = exp_approx_fix(np.int32(x_sft*(2**SHIFT))  , SHIFT)
    # exp_fix_1 = exp_approx_fix(np.int32(x_sft*(2**SHIFT)), SHIFT,6)
    # exp_fix_2 = exp_approx_fix(np.int32(x_sft*(2**SHIFT)), SHIFT,5)
    # exp_fix_3 = exp_approx_fix(np.int32(x_sft*(2**SHIFT)), SHIFT,4)
    # exp_fix_4 = exp_approx_fix(np.int32(x_sft*(2**SHIFT)), SHIFT,3)
    # exp_fix_5 = exp_approx_fix(np.int32(x_sft*(2**SHIFT)), SHIFT,2)
    # exp_fix_6 = exp_approx_fix(np.int32(x_sft*(2**SHIFT)), SHIFT,1)
    #
    # print(np.max( np.abs( exp_fix*(2**-SHIFT)   -  np.exp(x_sft) ) ) )
    # print(np.max(np.abs(exp_fix_1*(2**-SHIFT) -  np.exp(x_sft))))
    # print(np.max(np.abs(exp_fix_2*(2**-SHIFT) -  np.exp(x_sft))))
    # print(np.max(np.abs(exp_fix_3*(2**-SHIFT) -  np.exp(x_sft))))
    # print(np.max(np.abs(exp_fix_4*(2**-SHIFT) -  np.exp(x_sft))))
    # print(np.max(np.abs(exp_fix_5*(2**-SHIFT) -  np.exp(x_sft))))
    # print(np.max(np.abs(exp_fix_6*(2**-SHIFT) -  np.exp(x_sft))))
    #exp_fix_6 = exp_approx_fix(x_sft*(2**SHIFT), SHIFT)


    #sig_fun = sigmoid_approx_fix(np.int32(x_sig*(2**SHIFT)), SHIFT)
    #sig_fun[x_sig >= 10] = 1*(2**SHIFT)
    #sig_fun[x_sig <= -10] = 0
    #print(np.max(np.abs(sig_fun*(2**-SHIFT) - sigmoid(x_sig))))

    #sftmax = np.int32(softmax_fix_approx(np.array(np.int32(x_sft*( 2.0**SHIFT ) ) ) ) )

    #print(fix_tanh_3)
    # lw = 2
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    #
    # #ax.plot(x_sig, sigmoid(x_sig), linewidth=2)
    # #ax.plot(x_sig, sig_fun*(2**-SHIFT)  , '--',  linewidth=2)
    #
    # ax.plot(x_sft, np.exp(x_sft) , '--', linewidth=4)
    # ax.plot(x_sft, exp_fix * (2 ** -12), linewidth=lw)
    # ax.plot(x_sft, exp_fix_1*(2**-10), linewidth=lw)
    # ax.plot(x_sft, exp_fix_2*(2**-8), linewidth=lw)
    # ax.plot(x_sft, exp_fix_3*(2**-6), linewidth=lw)
    # ax.plot(x_sft, exp_fix_4*(2**-4), linewidth=lw)
    #ax.plot(x_sft, exp_fix_5*(2**-SHIFT), '--', linewidth=lw)
    #ax.plot(x_sft, exp_fix_6*(2**-SHIFT), '--', linewidth=lw)
    #ax.plot(x_sft, exp_fix*(2**-SHIFT), '--', linewidth=lw)

    #ax.plot(x_sft, sftmax_ideal, '--o')
    #ax.plot(x_sft_3, sftmax_approx_3*(2**-SHIFT),  '-o' ,color='m' )
    #ax.plot(x_sft, sftmax_approx*(2**-SHIFT),  '-o' ,color='k' )
    #ax.plot(x_sft_1, sftmax_approx_1*(2**-SHIFT),  '-o' ,color='b' )
    #ax.plot(x_sft_2, sftmax_approx_2*(2**-SHIFT),  '-o' ,color='g' )

    # ax.plot(x_sig, np.tanh(x_sig), '--',linewidth=5)
    # ax.plot(x_sig, fix_tanh*(2**-12))
    # ax.plot(x_sig, fix_tanh_david*(2**-12), 'k')


    #ax.plot(x_sig, fix_tanh_2*(2**-SHIFT))
    #ax.plot(x_sig, fix_tanh_3*(2**-SHIFT))
    #ax.plot(x_sig, fix_tanh_4*(2**-SHIFT))
    #ax.plot(x_sig, fix_tanh_5*(2**-SHIFT))
    #ax.plot( x, np.exp( x ) ,linewidth=lw*2)
    # ax.plot( x, fix_e_12*(2.0**( -SHIFT ) ) ,linewidth=lw)
    # ax.plot( x, fix_e_10*(2.0**( -10 ) ) ,linewidth=lw)
    # ax.plot( x, fix_e_8*(2.0**( -8 ) ) ,linewidth=lw)
    # ax.plot( x, fix_e_6*(2.0**( -6 ) ) ,linewidth=lw)
    # ax.plot( x, fix_e_4*(2.0**( -4 ) ) ,linewidth=lw)
    # plt.xlabel('$x$', fontsize=16)
    # plt.ylabel('$e^x$',rotation=0, fontsize=16, loc='center')
    #
    # ax.legend(["ideal", "12", "10", "8", "6", "4"])
    # ax[1].plot(x_sig, sigmoid(x_sig))
    # ax[1].plot(x_sig, fix_sigmoid*( 2.0**( -SHIFT ) ) )
    # ax[2].plot(x_sig, np.tanh(x_sig))
    # ax[2].plot(x_sig, fix_tanh*( 2.0**( -SHIFT ) ) )
    # #ax[2].plot(x_sig, fix_tanh_2*( 2.0**( -SHIFT ) ) )
    # ax[2].plot(x_sig, (fix_tanh_3*( 2.0**( -SHIFT ) )))
    # ax[3].plot(x_sft, (sftmax*( 2.0**( -SHIFT ) )), '--o')
    # ax[3].plot(x_sft, softmax_local(x_sft), 'o')
    # ax[0].legend(['numpy', 'approx'])
    # ax[2].set_xlim([-16.5,16.5])
    # ax[2].set_ylim([-1.5,1.5])
    # plt.show()