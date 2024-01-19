import numpy as np
import torch
import matplotlib.pyplot as plt
#plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 13})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

if __name__ == '__main__':

    np.random.seed(0)
    #model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8_RELU/checkpoint_p0/best_000371996_380923904_reward_1.737.pth")
    #model = torch.load("/home/saz/quad-swarm-rl/train_dir/mean_embed_16_8/checkpoint_p0/best_000599384_613769216_reward_1.236.pth")
    model = torch.load("/home/saz/Desktop/qsrl_test/train_dir/mean_embed_16_8_relu/checkpoint_p0/best_000602426_616884224_reward_1.538.pth", map_location='cpu')
    #model = torch.load("/home/saz/Documents/mean_embed_16_8/checkpoint_p0/checkpoint_001269534_1300002816.pth")
    #model = torch.load("/home/saz/Documents/mean_embed_16_8/checkpoint_p0/checkpoint_001269535_1300003840.pth")
    #print(model['model'].keys())
    #print(type(model['model']['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'][0][0]))
    #print((model['model']['actor_encoder.neighbor_encoder.embedding_mlp.0.weight']))

    modelKeys = model['model'].keys()

    actor  = {}
    z = 15

    nerr_arr = np.array([None]*z)
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

    nerr_arr = np.zeros(z)*10**6
    iter = 10
    kk = 0
    while kk < iter:
        kk = kk+1
        for n in range(1,z+1):



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
            out_self  =  (
                        actor['actor_encoder.self_encoder.0.weight']
                        @ self_obs + actor['actor_encoder.self_encoder.0.bias'])\
                         *(actor['actor_encoder.self_encoder.0.weight']@
                           self_obs + actor['actor_encoder.self_encoder.0.bias'] >= 0)*np.ones(16)

            out_self = (
                               actor['actor_encoder.self_encoder.2.weight'] @ out_self + actor[
                           'actor_encoder.self_encoder.2.bias']) * (
                                   actor['actor_encoder.self_encoder.2.weight'] @ out_self + actor[
                               'actor_encoder.self_encoder.2.bias'] >= 0) * np.ones(16)

            neighb_res = np.zeros(8)

            for k in range(6):
                out_neighb = (
                                   actor['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'] @ neigh_obs[k,:] + actor[
                               'actor_encoder.neighbor_encoder.embedding_mlp.0.bias']) * (
                                       actor['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'] @ neigh_obs[k,:] + actor[
                                   'actor_encoder.neighbor_encoder.embedding_mlp.0.bias'] >= 0) * np.ones(8)

                #print(out_neighb)

                out_neighb = (
                                     actor['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'] @ out_neighb + actor[
                                 'actor_encoder.neighbor_encoder.embedding_mlp.2.bias']) * (
                                     actor['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'] @ out_neighb + actor[
                                 'actor_encoder.neighbor_encoder.embedding_mlp.2.bias'] >= 0) * np.ones(8)

                #print(out_neighb)

                neighb_res = neighb_res + out_neighb

            neighb_res = neighb_res/6
            #print(neighb_res)

            inp_ff = np.concatenate((out_self, neighb_res))

            out_mlp = (
                                 actor['actor_encoder.feed_forward.0.weight'] @ inp_ff + actor[
                             'actor_encoder.feed_forward.0.bias']) * (
                                 actor['actor_encoder.feed_forward.0.weight'] @ inp_ff + actor[
                             'actor_encoder.feed_forward.0.bias'] >= 0) * np.ones(32)

            #print(out_mlp)

            out_mlp = (
                    actor['action_parameterization.distribution_linear.weight'] @ out_mlp + actor[
                'action_parameterization.distribution_linear.bias'])

            out_mlp = 0.5*(np.clip(out_mlp, -1, 1) + 1)

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
                if out_self_fix_0[k] < 0:
                    out_self_fix_0[k] = 0


            for k in range(out_self_fix_1.shape[0]):
                for u in range(out_self_fix_0.shape[0]):
                    out_self_fix_1[k] = ((actor_fix['actor_encoder.self_encoder.2.weight'][k][u]*out_self_fix_0[u])*(2**-n)).astype(np.int32) + out_self_fix_1[k]
                out_self_fix_1[k] = out_self_fix_1[k] + actor_fix['actor_encoder.self_encoder.2.bias'][k]
                if out_self_fix_1[k] < 0:
                    out_self_fix_1[k] = 0

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
                    if neighb_out_0[k] < 0:
                        neighb_out_0[k] = 0
                #print(neighb_out_0*(2**-n))

                for k in range(neighb_out_1.shape[0]):
                    for u in range(8):
                        neighb_out_1[k] = (actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.weight'][k][u] * neighb_out_0[u] * (2 ** -n)).astype(np.int32) + neighb_out_1[k]
                    neighb_out_1[k] = neighb_out_1[k] + actor_fix['actor_encoder.neighbor_encoder.embedding_mlp.2.bias'][k]
                    if neighb_out_1[k] < 0:
                        neighb_out_1[k] = 0

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
            if nerr_arr[n-1] <  np.sqrt( np.sum( ( out_final_fix_1*(2**-n) - out_mlp)**2 ) )/4:
                nerr_arr[n-1] = np.sqrt( np.sum( ( out_final_fix_1*(2**-n) - out_mlp)**2 ) )/4

            #print()
            #print(out_final_fix_1)
            #print(out_mlp)


    dff = np.abs(np.min(nerr_arr) - np.max(nerr_arr))*0.09
    plt.plot(range(1,14), nerr_arr[0:13], 'o')
    plt.plot(14, nerr_arr[13], 'ro')
    plt.plot(15, nerr_arr[14], 'ro')
   # plt.plot(16, nerr_arr[15], 'ro')
    plt.plot([1, 15], np.array([1, 1])*np.min(nerr_arr) , '--',color='0.5')
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    plt.yticks(np.arange(np.min(nerr_arr), np.max(nerr_arr), dff))
    plt.xlabel('$n$ bits')
    plt.ylabel('Error')
    plt.ylim([np.min(nerr_arr) - dff*0.2, np.max(nerr_arr) + dff*0.2])
    plt.grid()

    plt.show()


