import numpy as np
import torch
import matplotlib.pyplot as plt
#plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 13})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

SHIFT = 12

MULT1732 = np.int32((17/32)*2**(SHIFT))
def tanh_approx(val):
    if val < -15*(2**(SHIFT-3)): # in kleiner -1,875
        output_local = -2**SHIFT # output = -1
    elif val > 15 * (2**(SHIFT - 3)):  # in größer 1,875
        output_local = 2**SHIFT  # output = 1
    elif val < -9*(2**(SHIFT-3)): # in kleiner -1,125
        output_local = val*(2**(-2))  - MULT1732  # output = x/4 - 17/32
    elif val > 9*(2**(SHIFT-3)): # in größer 1,125
        output_local = val*(2**(-2)) + MULT1732 # output = x/4 + 17/32
    elif val < -(2**(SHIFT-1)): # in kleiner -0,5
        output_local = val*(2**(-1)) - (2**(SHIFT-2))  # output = in/2 - 0,25
    elif val > (2**(SHIFT-1)): # in größer 0,5
        output_local = val*(2**(-1)) + (2**(SHIFT-2))  # output = in/2 + 0,25
    else:
        output_local = val # In der Mitte: ouput = in
    return np.int32(output_local)





#else:
#    output_local = val # In der Mitte: ouput = in


if __name__ == '__main__':
    np.random.seed(0)
    # pths = ['/home/saz/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-pos-20240109T17-31-56.csv',
    #         '/home/saz/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-vel-20240109T17-31-54.csv',
    #         '/home/saz/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-R1-20240109T17-31-55.csv',
    #         '/home/saz/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-R2-20240109T17-31-55.csv',
    #         '/home/saz/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-R3-20240109T17-31-54.csv',
    #         '/home/saz/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-w-20240109T17-32-01.csv',
    #         '/home/saz/quad-swarm-rl/logged_cfclient/20240109T17-30-41/nnthrust-trhst-20240109T17-31-58.csv',
    #         ]

    pths = ['/home/saz/quad-swarm-rl/logged_cfclient/20240118T22-48-42/slfOb-pos-20240118T22-56-13.csv',
            '/home/saz/quad-swarm-rl/logged_cfclient/20240118T22-48-42/slfOb-vel-20240118T22-56-16.csv',
            '/home/saz/quad-swarm-rl/logged_cfclient/20240118T22-48-42/slfOb-R1-20240118T22-56-14.csv',
            '/home/saz/quad-swarm-rl/logged_cfclient/20240118T22-48-42/slfOb-R2-20240118T22-56-15.csv',
            '/home/saz/quad-swarm-rl/logged_cfclient/20240118T22-48-42/slfOb-R3-20240118T22-56-15.csv',
            '/home/saz/quad-swarm-rl/logged_cfclient/20240118T22-48-42/slfOb-w-20240118T22-56-17.csv',
            '/home/saz/quad-swarm-rl/logged_cfclient/20240118T22-48-42/nnthrust-trhst-20240118T22-56-13.csv',
            ]

    sz = 1100
    out_fix_final = np.zeros((sz, 4))
    out_final = np.zeros((sz, 4))

    arr_pos = np.zeros((sz, 3))
    arr_vel = np.zeros((sz, 3))
    arr_R1 = np.zeros((sz, 3))
    arr_R2 = np.zeros((sz, 3))
    arr_R3 = np.zeros((sz, 3))
    arr_w = np.zeros((sz, 3))
    arr_thr = np.zeros((sz, 4))

    u = 0
    for p in range(pths.__len__()):
        with open(pths[p]) as fl:
            k = 0
            for row in fl:
                if k == 0:
                    k = k + 1
                    continue

                rw = row
                rw = rw.split(',')


                if u == 0:
                    rw = np.array(rw[1:4]).astype(float)
                    arr_pos[k - 1, :] = rw
                elif u == 1:
                    rw = np.array(rw[1:4]).astype(float)
                    arr_vel[k - 1, :] = rw
                elif u == 2:
                    rw = np.array(rw[1:4]).astype(float)
                    arr_R1[k - 1, :] = rw
                elif u == 3:
                    rw = np.array(rw[1:4]).astype(float)
                    arr_R2[k - 1, :] = rw
                elif u == 4:
                    rw = np.array(rw[1:4]).astype(float)
                    arr_R3[k - 1, :] = rw
                elif u == 5:
                    rw = np.array(rw[1:4]).astype(float)
                    arr_w[k - 1, :] = rw
                elif u == 6:
                    rw = np.array(rw[1:5]).astype(float)
                    arr_thr[k - 1, 0] = rw[3]
                    arr_thr[k - 1, 1] = rw[0]
                    arr_thr[k - 1, 2] = rw[1]
                    arr_thr[k - 1, 3] = rw[2]

                k = k + 1

        u = u + 1


    model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8/checkpoint_p0/best_000599384_613769216_reward_1.236.pth", map_location='cpu')
    modelKeys = model['model'].keys()

    actor  = {}
    n= 12

    for k in modelKeys:
        if 'actor_encoder' in k.split('.'):
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


    for c in range(0,sz):
        self_obs  = np.zeros(18) #  np.random.rand(18)

        self_obs[0] =  np.clip(arr_pos[c,0],-10,10)
        self_obs[1] =  np.clip(arr_pos[c,1],-10,10)
        self_obs[2] =  np.clip(arr_pos[c,2],-10,10)

        self_obs[3] = np.clip(arr_vel[c,0],-3,3)
        self_obs[4] = np.clip(arr_vel[c,1],-3,3)
        self_obs[5] = np.clip(arr_vel[c,2],-3,3)

        self_obs[6]  = np.clip(arr_R1[c,0],-1,1)
        self_obs[7]  = np.clip(arr_R1[c,1],-1,1)
        self_obs[8]  = np.clip(arr_R1[c,2],-1,1)

        self_obs[9]  = np.clip(arr_R2[c,0],-1,1)
        self_obs[10] = np.clip(arr_R2[c,1],-1,1)
        self_obs[11] = np.clip(arr_R2[c,2],-1,1)

        self_obs[12] = np.clip(arr_R3[c,0],-1,1)
        self_obs[13] = np.clip(arr_R3[c,1],-1,1)
        self_obs[14] = np.clip(arr_R3[c,2],-1,1)

        self_obs[15] = np.clip(arr_w[c,0],-40,40)
        self_obs[16] = np.clip(arr_w[c,1],-40,40)
        self_obs[17] = np.clip(arr_w[c,2],-40,40)


        neigh_obs = np.zeros((6,6))  # np.random.rand(6, 6)
        neigh_obs[0, 0] = 8
        neigh_obs[0, 1] = 0

        neigh_obs[1, 0] = 8
        neigh_obs[1, 1] = 1.5

        neigh_obs[2, 0] = 8
        neigh_obs[2, 1] = 3

        neigh_obs[3, 0] = 8
        neigh_obs[3, 1] = 4.5

        neigh_obs[4, 0] = 8
        neigh_obs[4, 1] = 6

        neigh_obs[5, 0] = 8
        neigh_obs[5, 1] = 7.5

        self_neurons  = 16
        # self f
        out_self = np.zeros(self_neurons)
        out_self_1 = np.zeros(self_neurons)

        for k in range(self_neurons):
            for u in range(18):
                out_self[k] = actor['actor_encoder.self_encoder.0.weight'][k,u]*self_obs[u] + out_self[k]
            out_self[k] = out_self[k] + actor['actor_encoder.self_encoder.0.bias'][k]
            out_self[k] = np.tanh(out_self[k]).astype(dtype=float)

        for k in range(self_neurons):
            for u in range(self_neurons):
                out_self_1[k] = actor['actor_encoder.self_encoder.2.weight'][k,u] * out_self[u] + out_self_1[k]
            out_self_1[k] = out_self_1[k] + actor['actor_encoder.self_encoder.2.bias'][k]
            out_self_1[k] = np.tanh(out_self[k]).astype(dtype=float)


        neighb_res = np.zeros(8)
        neihb_neurons = 8

        for z in range(6):
            neighb_out_0 = np.zeros(neihb_neurons)
            neighb_out_1 = np.zeros(neihb_neurons)

            for k in range(neihb_neurons):
                for u in range(6):
                    neighb_out_0[k] = actor['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'][k,u] * neigh_obs[z,u] + neighb_out_0[k]
                neighb_out_0[k]     = neighb_out_0[k] + actor['actor_encoder.neighbor_encoder.embedding_mlp.0.bias'][k]
                neighb_out_0[k]     = np.tanh(neighb_out_0[k]).astype(dtype=float)

            #print(out_neighb)
            for k in range(neihb_neurons):
                for u in range(neihb_neurons):
                    neighb_out_1[k] = actor['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'][k,u] * neighb_out_0[u] + neighb_out_1[k]
                neighb_out_1[k]     = neighb_out_1[k] + actor['actor_encoder.neighbor_encoder.embedding_mlp.2.bias'][k]
                neighb_out_1[k]     = np.tanh(neighb_out_0[k]).astype(dtype=float)


            neighb_res = neighb_res + neighb_out_1

        neighb_res = neighb_res/6.0
        #print(neighb_res)

        inp_ff = np.concatenate((out_self, neighb_res))


        mlp_neurons = 32
        mlp_out_0 = np.zeros(32)
        mlp_out_1 = np.zeros(4)

        for k in range(mlp_neurons):
            for u in range(24):
                mlp_out_0[k] = actor['actor_encoder.feed_forward.0.weight'][k,u] * inp_ff[
                    u] + mlp_out_0[k]
            mlp_out_0[k] = mlp_out_0[k] + actor['actor_encoder.feed_forward.0.bias'][k]
            mlp_out_0[k] = np.tanh(mlp_out_0[k]).astype(dtype=float)

        for k in range(4):
            for u in range(mlp_neurons):
                mlp_out_1[k] = actor['action_parameterization.distribution_linear.weight'][k,u] * mlp_out_0[u] + mlp_out_1[k]
            mlp_out_1[k] = mlp_out_1[k] + actor['action_parameterization.distribution_linear.bias'][k]
            #mlp_out_1[k] = np.tanh(mlp_out_1[k]).astype(dtype=float)


        #print(out_mlp)
        out_mlp = mlp_out_1
        out_mlp = 0.5*(np.clip(out_mlp, -1, 1) + 1)
        out_final[c,0] =  out_mlp[0]
        out_final[c,1] =  out_mlp[1]
        out_final[c,2] =  out_mlp[2]
        out_final[c,3] =  out_mlp[3]


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

        #self_obs_fix = np.random.rand(18)
        self_obs_fix = (self_obs*(2**n)).astype(dtype=np.int32)

        # self ff
        out_self_fix = np.zeros(16, dtype=np.int32)
        out_self_fix_0 = np.zeros(16, dtype=np.int32)
        out_self_fix_1 = np.zeros(16, dtype=np.int32)

        for k in range(out_self_fix_0.shape[0]):
            for u in range(self_obs_fix.shape[0]):
                out_self_fix_0[k] = ((actor_fix['actor_encoder.self_encoder.0.weight'][k][u]*self_obs_fix[u])*(2**-n)).astype(np.int32) + out_self_fix_0[k]
            out_self_fix_0[k] = out_self_fix_0[k] + actor_fix['actor_encoder.self_encoder.0.bias'][k]
            out_self_fix_0[k] = tanh_approx(out_self_fix_0[k])


        for k in range(out_self_fix_1.shape[0]):
            for u in range(out_self_fix_0.shape[0]):
                out_self_fix_1[k] = ((actor_fix['actor_encoder.self_encoder.2.weight'][k][u]*out_self_fix_0[u])*(2**-n)).astype(np.int32) + out_self_fix_1[k]
            out_self_fix_1[k] = out_self_fix_1[k] + actor_fix['actor_encoder.self_encoder.2.bias'][k]
            out_self_fix_1[k] = tanh_approx(out_self_fix_1[k])

        out_self_fix = out_self_fix_1
        #print(out_self_fix*(2**-n))
        #print(out_self)

        #neigh_obs_fix = np.random.rand(6, 6)
        neigh_obs_fix = (neigh_obs*(2**n)).astype(dtype=np.int32)

        neighb_res_fix = np.zeros(8)
        neighb_res_fix = neighb_res_fix.astype(dtype=np.int32)

        for q in range(6):
            neighb_out_0 = np.zeros(8)
            neighb_out_0 = neighb_out_0.astype(dtype=np.int32)

            neighb_out_1 = np.zeros(8)
            neighb_out_1 = neighb_out_1.astype(dtype=np.int32)

            for k in range(neighb_out_0.shape[0]):
                for u in range(6):
                    neighb_out_0[k] = ((actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'][k][u] * neigh_obs_fix[q][
                        u]) * (2 ** -n)).astype(np.int32) + neighb_out_0[k]
                neighb_out_0[k] = neighb_out_0[k] + actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.0.bias'][k]
                neighb_out_0[k] = tanh_approx(neighb_out_0[k])
            #print(neighb_out_0*(2**-n))

            for k in range(neighb_out_1.shape[0]):
                for u in range(8):
                    neighb_out_1[k] = (actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'][k][u] * neighb_out_0[u] * (2 ** -n)).astype(np.int32) + neighb_out_1[k]
                neighb_out_1[k] = neighb_out_1[k] + actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.bias'][k]
                neighb_out_1[k] = tanh_approx(neighb_out_1[k])

            #print(neighb_out_1 * (2 ** -n))

            neighb_res_fix = neighb_res_fix + neighb_out_1


        divider = np.int32((1/6)*(2**n))
        neighb_res_fix = (neighb_res_fix*divider*(2**-n)).astype(np.int32)
        #print(neighb_res_fix * (2 ** -n))

        #print(neighb_res)

        stacked = np.concatenate([out_self_fix, neighb_res_fix])

        #print("stacked", stacked , "float stacked", inp_ff)

        out_final_fix_0   = np.zeros(32, dtype=np.int32)
        out_final_fix_1 = np.zeros(4, dtype=np.int32)

        for k in range(out_final_fix_0.shape[0]):
            for u in range(stacked.shape[0]):
                out_final_fix_0[k] = (actor_fix['actor_encoder.feed_forward.0.weight'][k][u] * stacked[
                    u] * (2 ** -n)).astype(np.int32) + out_final_fix_0[k]
            out_final_fix_0[k] = out_final_fix_0[k] + actor_fix['actor_encoder.feed_forward.0.bias'][k]

            out_final_fix_0[k] = tanh_approx(out_final_fix_0[k])

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


        out_fix_final[c,0] = out_final_fix_1[0]
        out_fix_final[c,1] = out_final_fix_1[1]
        out_fix_final[c,2] = out_final_fix_1[2]
        out_fix_final[c,3] = out_final_fix_1[3]
        #print("fdghefdgh", out_final_fix_1*2**-n)
        #print(out_mlp)


    #plt.plot(out_fix_final*2**-n)
    #print(out_mlp)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(out_fix_final*(2**-n))
    ax[1].plot(out_final)
    ax[2].plot(arr_thr)
    plt.show()
    #print(out_fix_final[0,:])
    #print(out_fix_final[1,:])
    #print(out_fix_final[200,:])
