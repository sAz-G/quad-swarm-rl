import numpy as np
import torch

if __name__ == '__main__':
    PRNT = True
    model = torch.load("/home/saz/GitHub/quadswarmsharif/train_dir/mean_embed_16_8_RELU/checkpoint_p0/best_000371996_380923904_reward_1.737.pth")
    #model = torch.load("/home/saz/GitHub/quadswarmsharif/train_dir/mean_embed_16_8/checkpoint_p0/best_000599384_613769216_reward_1.236.pth")
    #print(model)

    act_nn ={'action_parameterization.distribution_linear.weight' : np.array(model['model']['action_parameterization.distribution_linear.weight'].cpu(), dtype=np.float32),
               'action_parameterization.distribution_linear.bias':np.array(model['model']['action_parameterization.distribution_linear.bias'].cpu(), dtype=np.float32)}


    for k in model['model'].keys():
        if 'actor_encoder' in k.split('.'):
            act_nn[k] = np.array(model['model'][k].cpu(), dtype=np.float32)


    neighbor_nn_w    = {0 : act_nn['actor_encoder.neighbor_encoder.embedding_mlp.0.weight'],
                        1 : act_nn['actor_encoder.neighbor_encoder.embedding_mlp.2.weight']}

    neighbor_nn_b = {0: act_nn['actor_encoder.neighbor_encoder.embedding_mlp.0.bias'],
                     1: act_nn['actor_encoder.neighbor_encoder.embedding_mlp.2.bias']}

    self_nn_w        = {0 : act_nn['actor_encoder.self_encoder.0.weight'],
                        1 : act_nn['actor_encoder.self_encoder.2.weight']}

    self_nn_b        = {0 : act_nn['actor_encoder.self_encoder.0.bias'],
                        1 : act_nn['actor_encoder.self_encoder.2.bias']}

    feedforward_nn_w = {0 : act_nn['actor_encoder.feed_forward.0.weight'],
                        1 : act_nn['action_parameterization.distribution_linear.weight']}

    feedforward_nn_b = {0 : act_nn['actor_encoder.feed_forward.0.bias'],
                        1 : act_nn['action_parameterization.distribution_linear.bias']}


    samps = 10**5
    self_inp = np.array(np.random.rand(18,samps),  dtype= np.float32)
    self_inp[0,:] = np.array(np.random.rand(samps)*20.0 -10.0,  dtype= np.float32)
    self_inp[1,:] = np.array(np.random.rand(samps)*20.0 -10.0,  dtype= np.float32)
    self_inp[2,:] = np.array(np.random.rand(samps)*20.0 -10.0,  dtype= np.float32)

    self_inp[3, :] = np.array(np.random.rand(samps) * 6.0 - 3.0, dtype=np.float32)
    self_inp[4, :] = np.array(np.random.rand(samps) * 6.0 - 3.0, dtype=np.float32)
    self_inp[5, :] = np.array(np.random.rand(samps) * 6.0 - 3.0, dtype=np.float32)

    self_inp[6, :] = np.array(np.random.rand(samps)*2.0 - 1.0, dtype=np.float32)
    self_inp[7, :] = np.array(np.random.rand(samps)*2.0 - 1.0, dtype=np.float32)
    self_inp[8, :] = np.array(np.random.rand(samps)*2.0 - 1.0, dtype=np.float32)

    self_inp[9, :]  = np.array(np.random.rand(samps) *2.0 - 1.0, dtype=np.float32)
    self_inp[10, :] = np.array(np.random.rand(samps) *2.0 - 1.0, dtype=np.float32)
    self_inp[11, :] = np.array(np.random.rand(samps) *2.0 - 1.0, dtype=np.float32)

    self_inp[12, :] = np.array(np.random.rand(samps)*2.0 - 1.0, dtype=np.float32)
    self_inp[13, :] = np.array(np.random.rand(samps)*2.0 - 1.0, dtype=np.float32)
    self_inp[14, :] = np.array(np.random.rand(samps)*2.0 - 1.0, dtype=np.float32)

    self_inp[15, :] = np.array(np.random.rand(samps)*80.0 -40.0, dtype=np.float32)
    self_inp[16, :] = np.array(np.random.rand(samps)*80.0 -40.0, dtype=np.float32)
    self_inp[17, :] = np.array(np.random.rand(samps)*80.0 -40.0, dtype=np.float32)

    # get max weight
    max_weight_self_abs = 0
    for v in neighbor_nn_w.values():
        if max_weight_self_abs < np.max(v):
            max_weight_self_abs = np.max(v)

    # get max addition result
    max_mult_self      = 0
    max_add_self_0     = 0 #np.max(out_self_0)
    max_out_0          = 0
    max_addmult_self_0 = 0  # np.max(out_self_0)

    max_addmult_self_1 = 0  # np.max(out_self_1)
    max_out_1          = 0
    max_add_self_1     = 0 #np.max(out_self_1)
    max_mult_self_1    = 0
    # get max multiplication

    for k in range(samps):
        # check max multiplication
        mult_vec = np.expand_dims(self_inp[:,k], axis=0)
        if max_mult_self < np.max(self_nn_w[0]*mult_vec):
            max_mult_self = np.max(self_nn_w[0]*mult_vec)


        local_self_inp  = self_inp[:,k]

        out_mult_0 = self_nn_w[0] @ local_self_inp
        if max_addmult_self_0 < np.max(np.abs(out_mult_0)):
            max_addmult_self_0 = np.max(np.abs(out_mult_0))

        out_add_0 = out_mult_0 + self_nn_b[0].reshape(self_nn_b[0].shape[0], 1)

        if max_add_self_0 < np.max(np.abs(out_add_0)):
            max_add_self_0 = np.max(np.abs(out_add_0))

        out_self_0 = np.maximum(out_add_0, 0)

        if max_out_0 < np.max(np.abs(out_self_0)):
            max_out_0 = np.max(np.abs(out_self_0))


########################################################################################################################


        out_mult_1 = self_nn_w[1] * out_self_0
        if max_mult_self_1 < np.max(out_mult_1):
            max_mult_self_1 = np.max(out_mult_1)

        out_multadd_1 = self_nn_w[1] @ out_self_0
        if max_addmult_self_1 < np.max(out_multadd_1):
            max_addmult_self_1 = np.max(out_multadd_1)

        out_add_1 = out_multadd_1 + self_nn_b[1].reshape(16, 1)
        if max_add_self_1 < np.max(out_add_1):
            max_add_self_1 = np.max(out_add_1)


        out_self_1 = np.maximum(out_multadd_1 + self_nn_b[1].reshape(16, 1), 0)
        if max_out_1 < np.max(out_self_1):
            max_out_1 = np.max(out_self_1)

    if PRNT:
        print("Self information")
        print(max_add_self_0)
        print(max_out_0)
        print(max_addmult_self_0)
        print(max_mult_self)

        print()
        print(max_add_self_1)
        print(max_addmult_self_1)
        print(max_out_1)
        print(max_mult_self_1)

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

    neighb_inp        = np.array(np.random.rand(6, samps), dtype=np.float32)
    neighb_inp[0, :] = np.array(np.random.rand(samps)*20.0-10.0, dtype=np.float32)
    neighb_inp[1, :] = np.array(np.random.rand(samps)*20.0-10.0, dtype=np.float32)
    neighb_inp[2, :] = np.array(np.random.rand(samps)*20.0-10.0, dtype=np.float32)

    neighb_inp[3, :] = np.array(np.random.rand(samps) * 12.0 - 6.0, dtype=np.float32)
    neighb_inp[4, :] = np.array(np.random.rand(samps) * 12.0 - 6.0, dtype=np.float32)
    neighb_inp[5, :] = np.array(np.random.rand(samps) * 12.0 - 6.0, dtype=np.float32)


    # get max addition result
    max_mult_neighb = 0
    max_add_neighb_0 = 0  # np.max(out_self_0)
    max_out_neighb_0 = 0
    max_addmult_neighb_0 = 0  # np.max(out_self_0)

    max_mult_neighb_1 = 0
    max_addmult_neighb_1 = 0  # np.max(out_self_1)
    max_out_neighb_1 = 0
    max_add_neighb_1 = 0  # np.max(out_self_1)
    # get max multiplication

    for k in range(samps):
        # check max multiplication
        mult_vec_neighb = np.expand_dims(neighb_inp[:, k], axis=0)

        if max_mult_neighb < np.max(neighbor_nn_w[0] * mult_vec_neighb):
            max_mult_neighb = np.max(neighbor_nn_w[0] * mult_vec_neighb)

        local_neighb_inp = neighb_inp[:, k]

        out_mult_neighb_0 = neighbor_nn_w[0] @ local_neighb_inp
        if max_addmult_neighb_0 < np.max(np.abs(out_mult_neighb_0)):
            max_addmult_neighb_0 = np.max(np.abs(out_mult_neighb_0))

        out_add_neighb_0 = out_mult_neighb_0 + neighbor_nn_b[0]#.reshape(neighbor_nn_b[0].shape[0], 1)

        if max_add_neighb_0 < np.max(np.abs(out_add_neighb_0)):
            max_add_neighb_0 = np.max(np.abs(out_add_neighb_0))

        out_neighb_0 = np.maximum(out_add_neighb_0, 0)

        if max_out_neighb_0 < np.max(np.abs(out_neighb_0)):
            max_out_neighb_0 = np.max(np.abs(out_neighb_0))






        mult_neighb_1 = neighbor_nn_w[1] * out_neighb_0
        if max_mult_neighb_1 < np.max(np.abs(mult_neighb_1)):
            max_mult_neighb_1 = np.max(np.abs(mult_neighb_1))

        out_multadd_1 = neighbor_nn_w[1] @ out_neighb_0
        if max_addmult_neighb_1 < np.max(np.abs(out_multadd_1)):
            max_addmult_neighb_1 = np.max(np.abs(out_multadd_1))

        out_mult_neighb_1 = neighbor_nn_w[1] @ out_neighb_0 + neighbor_nn_b[1]
        if max_add_neighb_1 < np.max(np.abs(out_mult_neighb_1)):
            max_add_neighb_1 = np.max(np.abs(out_mult_neighb_1))

        out_multadd_1 = np.max(neighbor_nn_w[1] @ out_neighb_0 + neighbor_nn_b[1], 0)
        if max_out_neighb_1 < np.max(np.abs(out_multadd_1)):
            max_out_neighb_1 = np.max(np.abs(out_multadd_1))



    if PRNT:
        print("############################################################################################################")
        print("neighbor information")
        print()
        print(max_mult_neighb)
        print(max_add_neighb_0)
        print(max_out_neighb_0)
        print(max_addmult_neighb_0)
        print()
        print(max_addmult_neighb_1)
        print(max_out_neighb_1)
        print(max_add_neighb_1)
        print()

    out_self_0 = np.maximum(self_nn_w[0]@self_inp + self_nn_b[0].reshape(self_nn_b[0].shape[0],1), 0)
    out_self_1 = np.maximum(self_nn_w[1]@out_self_0 + self_nn_b[1].reshape(16,1),0)

    out_neighb_0 = np.maximum(neighbor_nn_w[0] @ neighb_inp + neighbor_nn_b[0].reshape(neighbor_nn_b[0].shape[0], 1), 0)
    out_neighb_1 = np.maximum(neighbor_nn_w[1] @ out_neighb_0 + neighbor_nn_b[1].reshape(8, 1), 0)

    inp_ff   = np.concatenate((out_self_1,out_neighb_1))
    out_ff_1 = np.maximum(feedforward_nn_w[0]@inp_ff + np.expand_dims(feedforward_nn_b[0],axis=1), 0)
    out_ff_2 = feedforward_nn_w[1]@out_ff_1 + np.expand_dims(feedforward_nn_b[1],axis=1)

    print(np.max(np.abs(out_ff_2)))




