import numpy as np
import torch
import matplotlib.pyplot as plt
#plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 17})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif"
})


def sigmoid_approx_fix(x, SHIFT):
    fix_half = np.int32(0.5 * (2.0 ** SHIFT))
    x_tan = np.int32(x*fix_half*( 2.0**(-SHIFT) ))

    rslt = fix_half + np.int32(fix_half*tanh_fix_approx_johan_heinrich(x_tan,n)*(2**-SHIFT))

    return rslt


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def exp_approx_fix(x, SHIFT):
    x0 = 0.5

    fix_exp_x0 = int(np.exp(x0) * (2 ** (SHIFT)))
    fix_x0 = int(x0 * (2.0 ** SHIFT))
    fix_half = int(0.5 * (2.0 ** SHIFT))
    fix_one_oversix = int((1.0 / 6.0) * (2.0 ** SHIFT))
    fix_one_over24 = int((1.0 / 24) * (2.0 ** SHIFT))
    fix_one_over120 = int((1.0 / 120) * (2.0 ** SHIFT))
    fix_one_over720 = int((1.0 / 720) * (2.0 ** SHIFT))
    fix_one_over1540 = int((1.0 / 1540) * (2.0 ** SHIFT))

    a = np.int32(x-fix_x0)
    b = np.int32(a*(x-fix_x0)*(2.0**(-SHIFT)))
    c = np.int32(b*(x-fix_x0)*(2.0**(-SHIFT)))
    d = np.int32(c*(x-fix_x0)*(2.0**(-SHIFT)))
    e = np.int32(d*(x-fix_x0)*(2.0**(-SHIFT)))
    f = np.int32(e*(x-fix_x0)*(2.0**(-SHIFT)))
    g = np.int32(f*(x-fix_x0)*(2.0**(-SHIFT)))

    factor_a = fix_exp_x0
    factor_b = np.int32(fix_half*fix_exp_x0*(2.0**(-SHIFT)))
    factor_c = np.int32(fix_one_oversix*fix_exp_x0*(2.0**(-SHIFT)))
    factor_d = np.int32(fix_one_over24*fix_exp_x0*(2.0**(-SHIFT)))
    factor_e = np.int32(fix_one_over120*fix_exp_x0*(2.0**(-SHIFT)))
    factor_f = np.int32(fix_one_over720*fix_exp_x0*(2.0**(-SHIFT)))
    factor_g = np.int32(fix_one_over1540*fix_exp_x0*(2.0**(-SHIFT)))

    sm = fix_exp_x0
    sm = sm + np.int32(factor_a*a*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_b*b*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_c*c*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_d*d*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_e*e*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_f*f*(2.0**(-SHIFT)))
    sm = sm + np.int32(factor_g*g*(2.0**(-SHIFT)))

    return sm


def tanh_fix_approx_johan_heinrich(x, SHIFT):
    x13_shift = np.int32(13*(2**SHIFT))
    x11_shift = np.int32(11*(2**SHIFT))
    x9_shift = np.int32(9*(2**SHIFT))
    x7_shift = np.int32(7*(2**SHIFT))
    x5_shift = np.int32(5*(2**SHIFT))
    x3_shift = np.int32(3*(2**SHIFT))
    x1_shift = np.int32(1*(2**SHIFT))


    xx = x*x*(2**-SHIFT)
    x13 = x13_shift + xx
    x11 = x11_shift + (xx/(x13_shift + xx))*(2**SHIFT)
    x9  = x9_shift + (xx/(x11_shift + (xx/(x13_shift + xx))*(2**SHIFT)))*(2**SHIFT)
    x7 = x7_shift + (xx/(x9_shift + (xx/(x11_shift + (xx/(x13_shift + xx))*(2**SHIFT)))*(2**SHIFT)))*(2**SHIFT)
    x5 = x5_shift+ (xx/(x7_shift + (xx/(x9_shift + (xx/(x11_shift + (xx/(x13_shift + xx))*(2**SHIFT)))*(2**SHIFT)))*(2**SHIFT)))*(2**SHIFT)
    x3 = x3_shift + (xx/(x5_shift+ (xx/(x7_shift + (xx/(x9_shift + (xx/(x11_shift + (xx/(x13_shift + xx))*(2**SHIFT)))*(2**SHIFT)))*(2**SHIFT)))*(2**SHIFT)))*(2**SHIFT)
    x1 = x1_shift + (xx/(x3_shift + (xx/(x5_shift+ (xx/(x7_shift + (xx/(x9_shift + ( xx/(x11_shift + (xx/(x13_shift + xx))*(2**SHIFT) ) )*( 2**SHIFT ) ) )*( 2**SHIFT ) ) )*( 2**SHIFT ) ) )*( 2**SHIFT ) ) )*(2**SHIFT)

    x_final = (x/x1)*(2**SHIFT)

    return np.int32(x_final)


def softmax_fix_approx(x,SHIFT):
    a = exp_approx_fix(x, SHIFT)
    b = np.sum(exp_approx_fix(x, SHIFT))
    return np.int32((a/b)*(2**SHIFT))


def softmax_local(x):
    return np.exp(x)/np.sum(np.exp(x))



if __name__ == '__main__':
    TEST = False
    np.random.seed(0)
    model = torch.load("/home/saz/Desktop/qsrl_test/train_dir/attention_16_16_tanh_sigmoid/checkpoint_p0/best_000498336_510296064_reward_2.310.pth", map_location='cpu')

    modelKeys = model['model'].keys()

    actor  = {}
    z = 12

    nerr_arr = np.array([None]*z)
    for k in modelKeys:
        if 'actor_encoder' in k.split('.'):
            #if 'neighbor_value_mlp' in k.split('.'):
            #    continue
            actor[k] = model['model'][k].detach().cpu().clone().numpy()
            actor[k] = np.array(actor[k], dtype=float)

    actor['action_parameterization.distribution_linear.weight'] = model['model'][
        'action_parameterization.distribution_linear.weight'].detach().cpu().clone().numpy()
    actor['action_parameterization.distribution_linear.bias'] = model['model'][
        'action_parameterization.distribution_linear.bias'].detach().cpu().clone().numpy()

    actor['action_parameterization.distribution_linear.weight'] = actor[
        'action_parameterization.distribution_linear.weight'].astype(dtype=float)
    actor['action_parameterization.distribution_linear.bias'] = actor[
        'action_parameterization.distribution_linear.bias'].astype(dtype=float)

    nerr_arr = np.zeros(z)*10**6
    iter = 13
    kk = 0
    cc = np.zeros((z, iter))*10**6

    if TEST:
        print(actor.keys())
        print(model['model'].keys())
        for v in actor.values():
            print(v.shape)
        #exit()

    neighb_neu = 16
    self_neu = 16

    while kk < iter:
        kk = kk+1
        for n in range(z,z+1):

            self_obs  = np.zeros(18) #  np.random.rand(18)

            self_obs[0] =  np.random.rand()*20 - 10
            self_obs[1] =  np.random.rand()*20 - 10
            self_obs[2] =  np.random.rand()*20 - 10

            self_obs[3] = np.random.rand() * 3 - 1.5
            self_obs[4] = np.random.rand() * 3 - 1.5
            self_obs[5] = np.random.rand() * 3 - 1.5

            self_obs[6]  = np.random.rand()
            self_obs[7]  = np.random.rand()
            self_obs[8]  = np.random.rand()
            self_obs[9]  = np.random.rand()
            self_obs[10] = np.random.rand()
            self_obs[11] = np.random.rand()
            self_obs[12] = np.random.rand()
            self_obs[13] = np.random.rand()
            self_obs[14] = np.random.rand()

            self_obs[15] = np.random.rand()*40 - 20
            self_obs[16] = np.random.rand()*40 - 20
            self_obs[17] = np.random.rand()*40 - 20


            neigh_obs = np.zeros((6,6))  # np.random.rand(6, 6)
            neigh_obs[0, 0] = np.random.rand()*20 - 10
            neigh_obs[0, 1] = np.random.rand()*20 - 10
            neigh_obs[0, 2] = np.random.rand()*20 - 10

            neigh_obs[0, 3] = np.random.rand()*6 - 3
            neigh_obs[0, 4] = np.random.rand()*6 - 3
            neigh_obs[0, 5] = np.random.rand()*6 - 3

            neigh_obs[1, 0] = np.random.rand() * 20 - 10
            neigh_obs[1, 1] = np.random.rand() * 20 - 10
            neigh_obs[1, 2] = np.random.rand() * 20 - 10

            neigh_obs[1, 3] = np.random.rand() * 6 - 3
            neigh_obs[1, 4] = np.random.rand() * 6 - 3
            neigh_obs[1, 5] = np.random.rand() * 6 - 3

            neigh_obs[2, 0] = np.random.rand() * 20 - 10
            neigh_obs[2, 1] = np.random.rand() * 20 - 10
            neigh_obs[2, 2] = np.random.rand() * 20 - 10

            neigh_obs[2, 3] = np.random.rand() * 6 - 3
            neigh_obs[2, 4] = np.random.rand() * 6 - 3
            neigh_obs[2, 5] = np.random.rand() * 6 - 3

            neigh_obs[3, 0] = np.random.rand() * 20 - 10
            neigh_obs[3, 1] = np.random.rand() * 20 - 10
            neigh_obs[3, 2] = np.random.rand() * 20 - 10

            neigh_obs[3, 3] = np.random.rand() * 6 - 3
            neigh_obs[3, 4] = np.random.rand() * 6 - 3
            neigh_obs[3, 5] = np.random.rand() * 6 - 3

            neigh_obs[4, 0] = np.random.rand() * 20 - 10
            neigh_obs[4, 1] = np.random.rand() * 20 - 10
            neigh_obs[4, 2] = np.random.rand() * 20 - 10

            neigh_obs[4, 3] = np.random.rand() * 6 - 3
            neigh_obs[4, 4] = np.random.rand() * 6 - 3
            neigh_obs[4, 5] = np.random.rand() * 6 - 3

            neigh_obs[5, 0] = np.random.rand() * 20 - 10
            neigh_obs[5, 1] = np.random.rand() * 20 - 10
            neigh_obs[5, 2] = np.random.rand() * 20 - 10

            neigh_obs[5, 3] = np.random.rand() * 6 - 3
            neigh_obs[5, 4] = np.random.rand() * 6 - 3
            neigh_obs[5, 5] = np.random.rand() * 6 - 3


            # self ff
            out_self  =  np.tanh(actor['actor_encoder.self_encoder.0.weight']@self_obs + actor['actor_encoder.self_encoder.0.bias'])
            out_self  = np.tanh(actor['actor_encoder.self_encoder.2.weight']@out_self + actor['actor_encoder.self_encoder.2.bias'])

            neighb_res = np.zeros(neighb_neu)
            B_kq = np.zeros((neighb_neu,6))

            for k in range(6):
                neighb_inp = np.concatenate((self_obs, neigh_obs[k, :]))
                out_neighb = np.tanh(actor['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'] @ neighb_inp + actor['actor_encoder.neighbor_encoder.embedding_mlp.0.bias'])
                out_neighb = np.tanh(actor['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'] @ out_neighb + actor['actor_encoder.neighbor_encoder.embedding_mlp.2.bias'])
                B_kq[:,k] = out_neighb
                neighb_res = neighb_res + out_neighb


            neighb_res = neighb_res/6
            meanMat = np.ones(B_kq.shape)*np.array([neighb_res]).T
            C_inp = np.vstack((B_kq, meanMat))

            out_C_0 = np.zeros(16)
            out_C_1 = np.zeros(16)
            out_C_2 = np.zeros((6,1))
            for k in range(6):
                out_C_0 = np.tanh(actor['actor_encoder.neighbor_encoder.attention_mlp.0.weight'] @ C_inp[:,k] + actor['actor_encoder.neighbor_encoder.attention_mlp.0.bias'])
                out_C_1 = np.tanh(actor['actor_encoder.neighbor_encoder.attention_mlp.2.weight'] @ out_C_0 + actor['actor_encoder.neighbor_encoder.attention_mlp.2.bias'])
                out_C_2[k] = sigmoid(actor['actor_encoder.neighbor_encoder.attention_mlp.4.weight'] @ out_C_1 + actor['actor_encoder.neighbor_encoder.attention_mlp.4.bias'])

            out_C_2 = softmax_local(out_C_2)

            out_D_0 = np.zeros(16)
            out_D_1 = np.zeros((16,6))
            for k in range(6):
                out_D_0 = np.tanh(actor['actor_encoder.neighbor_encoder.neighbor_value_mlp.0.weight'] @ B_kq[:, k] + actor[
                    'actor_encoder.neighbor_encoder.neighbor_value_mlp.0.bias'])
                out_D_1[:,k] = np.tanh(actor['actor_encoder.neighbor_encoder.neighbor_value_mlp.2.weight'] @ out_D_0 + actor[
                    'actor_encoder.neighbor_encoder.neighbor_value_mlp.2.bias'])

            out_D_1_final = out_D_1[:,0]*out_C_2[0] + out_D_1[:,1]*out_C_2[1] + out_D_1[:,2]*out_C_2[2] + out_D_1[:,3]*out_C_2[3] + out_D_1[:,4]*out_C_2[4] + out_D_1[:,5]*out_C_2[5]
            inp_ff = np.concatenate((out_self, out_D_1_final))
            out_mlp = np.tanh(actor['actor_encoder.feed_forward.0.weight'] @ inp_ff + actor['actor_encoder.feed_forward.0.bias'])
            out_mlp = actor['action_parameterization.distribution_linear.weight'] @ out_mlp + actor['action_parameterization.distribution_linear.bias']

            out_mlp = 0.5*(np.clip(out_mlp, -1, 1) + 1)
            #print(out_mlp)
#exit()
            ############################################################################ fixpoint arithmetic ######################################################################

            actor_fix = {}
            for k in modelKeys:
                if 'actor_encoder' in k.split('.'):
                    actor_fix[k] = model['model'][k].detach().cpu().clone().numpy()
                    actor_fix[k] = np.array(actor_fix[k]*(2**n), dtype=np.int32)

            actor_fix['action_parameterization.distribution_linear.weight'] = model['model'][
                'action_parameterization.distribution_linear.weight'].detach().cpu().clone().numpy()
            actor_fix['action_parameterization.distribution_linear.bias'] = model['model'][
                'action_parameterization.distribution_linear.bias'].detach().cpu().clone().numpy()

            actor_fix['action_parameterization.distribution_linear.weight'] = actor_fix[
                'action_parameterization.distribution_linear.weight'].astype(dtype=float)
            actor_fix['action_parameterization.distribution_linear.bias'] = actor_fix[
                'action_parameterization.distribution_linear.bias'].astype(dtype=float)

            actor_fix['action_parameterization.distribution_linear.weight'] = ((2**n)*actor_fix[
                'action_parameterization.distribution_linear.weight']).astype(dtype=np.int32)
            actor_fix['action_parameterization.distribution_linear.bias'] = ((2**n)*actor_fix[
                'action_parameterization.distribution_linear.bias']).astype(dtype=np.int32)

            self_obs_fix = (self_obs*(2**n)).astype(dtype=np.int32)

            # self ff
            out_self_fix = np.zeros(16, dtype=np.int32)
            out_self_fix_0 = np.zeros(16, dtype=np.int32)
            out_self_fix_1 = np.zeros(16, dtype=np.int32)

            for k in range(out_self_fix_0.shape[0]):
                for u in range(self_obs_fix.shape[0]):
                    out_self_fix_0[k] = ((actor_fix['actor_encoder.self_encoder.0.weight'][k][u]*self_obs_fix[u])*(2**-n)).astype(np.int32) + out_self_fix_0[k]
                out_self_fix_0[k] = np.int32(out_self_fix_0[k]) + actor_fix['actor_encoder.self_encoder.0.bias'][k]
                out_self_fix_0[k] = tanh_fix_approx_johan_heinrich(out_self_fix_0[k],n)


            for k in range(out_self_fix_1.shape[0]):
                for u in range(out_self_fix_0.shape[0]):
                    out_self_fix_1[k] = ((actor_fix['actor_encoder.self_encoder.2.weight'][k][u]*out_self_fix_0[u])*(2**-n)).astype(np.int32) + out_self_fix_1[k]
                out_self_fix_1[k] = np.int32(out_self_fix_1[k]) + actor_fix['actor_encoder.self_encoder.2.bias'][k]
                out_self_fix_1[k] = tanh_fix_approx_johan_heinrich(out_self_fix_1[k],n)


            out_self_fix = out_self_fix_1
            ##########################            ################################               ####################
            neigh_obs_fix = (neigh_obs * (2 ** n)).astype(dtype=np.int32)
            neighb_res_fix = np.zeros(16)
            neighb_res_fix = neighb_res_fix.astype(dtype=np.int32)
            B_K_fix = np.zeros((16,6)).astype(dtype=np.int32)

            for q in range(6):
                inp_B = np.concatenate((self_obs_fix, neigh_obs_fix[q,:]))

                neighb_out_0 = np.zeros(16)
                neighb_out_0 = neighb_out_0.astype(dtype=np.int32)

                neighb_out_1 = np.zeros(16)
                neighb_out_1 = neighb_out_1.astype(dtype=np.int32)

                for k in range(neighb_out_0.shape[0]):
                    for u in range(inp_B.shape[0]):
                        neighb_out_0[k] = ((actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'][k][u] * inp_B[
                            u]) * (2 ** -n)).astype(np.int32) + neighb_out_0[k]
                    neighb_out_0[k] = neighb_out_0[k] + actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.0.bias'][k]
                    neighb_out_0[k] = tanh_fix_approx_johan_heinrich(neighb_out_0[k],n)

                for k in range(neighb_out_1.shape[0]):
                    for u in range(16):
                        neighb_out_1[k] = (actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'][k][u] * neighb_out_0[u] * (2 ** -n)).astype(np.int32) + neighb_out_1[k]
                    neighb_out_1[k] = neighb_out_1[k] + actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.bias'][k]
                    neighb_out_1[k] = tanh_fix_approx_johan_heinrich(neighb_out_1[k],n)

                B_K_fix[:,q] = neighb_out_1
                neighb_res_fix = neighb_res_fix + neighb_out_1


            divider = np.int32((1/6)*(2**n))
            neighb_res_fix = (neighb_res_fix*divider*(2**-n)).astype(np.int32)




            out_C_0_fix = np.zeros((16,6),dtype=np.int32)
            out_C_1_fix = np.zeros((16,6),dtype=np.int32)
            out_C_2_fix = np.zeros((1,6),dtype=np.int32)

            mean_mat = np.vstack([[neighb_res_fix],[neighb_res_fix],[neighb_res_fix],[neighb_res_fix],[neighb_res_fix],[neighb_res_fix]]).T
            inp_C = np.concatenate((B_K_fix, mean_mat))

            for q in range(6):
                for k in range(out_C_0_fix.shape[0]):
                    for u in range(6):
                        out_C_0_fix[k][q] = ((actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'][k][u] * inp_C[q][
                            u]) * (2 ** -n)).astype(np.int32) + out_C_0_fix[k][q]
                    out_C_0_fix[k][q] = out_C_0_fix[k][q] + actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.0.bias'][k]
                    out_C_0_fix[k][q] = tanh_fix_approx_johan_heinrich(out_C_0_fix[k][q],n)

                for k in range(out_C_1_fix.shape[0]):
                    for u in range(16):
                        out_C_1_fix[k][q] = (actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'][k][u] * out_C_0_fix[k][q] * (2 ** -n)).astype(np.int32) + out_C_1_fix[k][q]
                    out_C_1_fix[k][q] = out_C_1_fix[k][q] + actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.bias'][k]
                    out_C_1_fix[k][q] = tanh_fix_approx_johan_heinrich(out_C_1_fix[k][q],n)

                for k in range(out_C_2_fix.shape[0]):
                    for u in range(16):
                        out_C_2_fix[k][q] = (actor_fix['actor_encoder.neighbor_encoder.attention_mlp.4.weight'][k][u] *
                                             out_C_1_fix[k][q] * (2 ** -n)).astype(np.int32) + out_C_2_fix[k][q]
                    out_C_2_fix[k][q] = out_C_2_fix[k][q] + \
                                        actor_fix['actor_encoder.neighbor_encoder.attention_mlp.4.bias'][k]
                    out_C_2_fix[k][q] = sigmoid_approx_fix(out_C_2_fix[k][q],n)

            out_C_2_fix = softmax_fix_approx(out_C_2_fix,n)


            out_D_0_fix = np.zeros((16, 6), dtype=np.int32)
            out_D_1_fix = np.zeros((16, 6), dtype=np.int32)

            for q in range(6):

                for k in range(out_C_0_fix.shape[0]):
                    for u in range(6):
                        out_D_0_fix[k][q] = ((actor_fix['actor_encoder.neighbor_encoder.neighbor_value_mlp.0.weight'][k][u] *
                                              B_K_fix[q][
                                                  u]) * (2 ** -n)).astype(np.int32) + out_D_0_fix[k][q]
                    out_D_0_fix[k][q] = out_D_0_fix[k][q] + \
                                        actor_fix['actor_encoder.neighbor_encoder.neighbor_value_mlp.0.bias'][k]
                    out_D_0_fix[k][q] = tanh_fix_approx_johan_heinrich(out_D_0_fix[k][q],n)

                for k in range(out_D_1_fix.shape[0]):
                    for u in range(16):
                        out_D_1_fix[k][q] = (actor_fix['actor_encoder.neighbor_encoder.neighbor_value_mlp.2.weight'][k][u] *
                                             out_D_0_fix[k][q] * (2 ** -n)).astype(np.int32) + out_D_1_fix[k][q]
                    out_D_1_fix[k][q] = out_D_1_fix[k][q] + \
                                        actor_fix['actor_encoder.neighbor_encoder.neighbor_value_mlp.2.bias'][k]
                    out_D_1_fix[k][q] = tanh_fix_approx_johan_heinrich(out_D_1_fix[k][q],n)



            weighted_vec = np.zeros(16, dtype=np.int32)
            for q in range(16):
                for k in range(6):
                    weighted_vec[q] += out_D_1_fix[q][k]*out_C_2_fix[0][k]*(2**(-n)) + weighted_vec[q]


            stacked = np.concatenate([out_self_fix, weighted_vec])

            out_final_fix_0   = np.zeros(32, dtype=np.int32)
            out_final_fix_1 = np.zeros(4, dtype=np.int32)

            for k in range(out_final_fix_0.shape[0]):
                for u in range(stacked.shape[0]):
                    out_final_fix_0[k] = (actor_fix['actor_encoder.feed_forward.0.weight'][k][u] * stacked[
                        u] * (2 ** -n)).astype(np.int32) + out_final_fix_0[k]
                out_final_fix_0[k] = out_final_fix_0[k] + actor_fix['actor_encoder.feed_forward.0.bias'][k]
                if out_final_fix_0[k] < 0:
                    out_final_fix_0[k] = 0

            #print( out_final_fix_0*(2**-n))

            for k in range(out_final_fix_1.shape[0]):
                for u in range(out_final_fix_0.shape[0]):
                    out_final_fix_1[k] = (actor_fix['action_parameterization.distribution_linear.weight'][k][u] * out_final_fix_0[
                        u] * (2 ** -n)).astype(np.int32) + out_final_fix_1[k]
                out_final_fix_1[k] = out_final_fix_1[k] + actor_fix['action_parameterization.distribution_linear.bias'][k]

            one_fix  = np.int32(2**n)
            half_fix = np.int32(2**(n-1))

            for t in range(4):
                if one_fix < out_final_fix_1[t]:
                    out_final_fix_1[t] = one_fix
                elif -one_fix > out_final_fix_1[t]:
                     out_final_fix_1[t] = -one_fix

                out_final_fix_1[t] = out_final_fix_1[t] + one_fix
                out_final_fix_1[t] = out_final_fix_1[t]*half_fix*(2**-n)


            #nerr_arr[n-1] = np.sqrt( np.sum( ( out_final_fix_1*(2**-n) - out_mlp)**2 ) )
            cc[n-1, iter-1] = nerr_arr[n-1]
            if nerr_arr[n-1] <  np.sqrt( np.sum( ( out_final_fix_1*(2**-n) - out_mlp)**2 ) )/4:
                nerr_arr[n-1] = np.sqrt( np.sum( ( out_final_fix_1*(2**-n) - out_mlp)**2 ) )/4

            print(n)
            print(out_final_fix_1*(2**(-n)))
            print(out_mlp)


    #print(np.mean(cc, axis=1))

    #dff = np.abs(np.min(nerr_arr) - np.max(nerr_arr))*0.09
    #plt.plot(range(1,14), nerr_arr[0:13], 'o', markersize=8)
    #plt.plot(14, nerr_arr[13], 'ro', markersize=8)
    #plt.plot(15, nerr_arr[14], 'ro', markersize=8)
    #plt.plot([1, 15], np.array([1, 1])*np.min(nerr_arr) , '--',color='0.5')
    #plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    #plt.yticks(np.arange(np.min(nerr_arr), np.max(nerr_arr), dff))
    #plt.xlabel('$n$ bits')
    #plt.ylabel('$|| 2^{-n}\\mathbf{a}_{n} - \\mathbf{a} ||_2$')
    #plt.ylim([np.min(nerr_arr) - dff*0.2, np.max(nerr_arr) + dff*0.2])
    #plt.grid()

    #plt.hist(cc[12,:])

    #plt.show()


